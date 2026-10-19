"""Run one service engine in-process, inside its own image, over the parity corpus.

This file is standalone. It imports only the standard library at module level,
and numpy and torch inside main(). That is because it is mounted into service
images, which carry their own numpy/torch versions and an older deepmarkpy.
The host imports it only for case_bits() and case_seed(). Typical use:

    python /parity/runner.py --op apply --inputs /corpus/16000/wm_audioseal \\
        --out /out/encodec/gpu --device cuda --repeat 2

What it records:
- outputs: .npy (float64) for audio, .json for detect results;
- results.jsonl: per case, the time, output length, finiteness, and whether a
  repeat run was bit-identical;
- env.json: torch/CUDA/cuDNN versions, the device and slice name, numerics
  flags, where the engine's parameters actually live, and peak CUDA memory.

Peak CUDA memory is what lets the host refuse a "GPU" run that never touched
the GPU. Under MIG, SE2 silently ran on the CPU in the GPU study.
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import random
import sys
import time


def case_seed(base_seed: int, stem: str) -> int:
    """Seed for one case, derived from its name: independent of order and of the other cases."""
    digest = hashlib.sha256(f"{base_seed}:{stem}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def case_bits(stem: str, size: int):
    """The watermark embedded for one case: fixed per (case, payload size)."""
    import numpy as np

    rng = np.random.default_rng(case_seed(1234, f"bits:{size}:{stem}"))
    return rng.integers(0, 2, size=size, dtype=np.int32)


def _configure_numerics(strict: bool):
    import torch

    jit_fusion_disabled = False
    if strict:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # TorchScript's profiling executor runs a scripted function unfused on
        # its first call(s), then switches to a fused GPU kernel that rounds
        # differently. DAC's @torch.jit.script snake activation made its first
        # GPU call differ from every later one by up to 1.8e-6, so a service's
        # output depended on its request history. Unfused, every call
        # matches, and is just as close to the CPU.
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        jit_fusion_disabled = True
    return {
        "jit_fusion_disabled": jit_fusion_disabled,
        "strict": strict,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _placement(engine, depth=4):
    """Count parameters per device anywhere in the engine's object graph."""
    import torch

    counts, seen = {}, set()

    def walk(obj, level):
        if id(obj) in seen or level > depth:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            for p in obj.parameters():
                counts[str(p.device)] = counts.get(str(p.device), 0) + p.numel()
            return
        if isinstance(obj, dict):
            children = obj.values()
        elif isinstance(obj, (list, tuple)):
            children = obj
        elif hasattr(obj, "__dict__"):
            children = vars(obj).values()
        else:
            return
        for child in children:
            walk(child, level + 1)

    walk(engine, 0)
    return counts


def _load_engine_class(stage: str, override: str):
    from deepmarkpy.core.inference import BaseAttackEngine, BaseModelEngine

    # Explicit path: neural_vocoder's working directory holds BigVGAN's own
    # inference.py, which a bare `import inference` would pick up instead.
    spec = importlib.util.spec_from_file_location("plugin_inference", os.path.join(stage, "inference.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if override:
        # Study engines (`path.py:Class`) subclass the service's engine via `import inference`.
        sys.modules["inference"] = module
        path, cls = override.rsplit(":", 1)
        ospec = importlib.util.spec_from_file_location("engine_override", path)
        omod = importlib.util.module_from_spec(ospec)
        ospec.loader.exec_module(omod)
        return getattr(omod, cls)
    found = [
        obj for obj in vars(module).values()
        if isinstance(obj, type) and obj.__module__ == module.__name__
        and issubclass(obj, (BaseAttackEngine, BaseModelEngine))
    ]
    if len(found) != 1:
        raise SystemExit(f"expected one engine class in {stage}/inference.py, found {found}")
    return found[0]


def _normalise_detect(result):
    """Detect returns bits, (bits, confidence), a scalar, or None depending on the model."""
    import numpy as np

    message, confidence = result, None
    if isinstance(result, tuple) and len(result) == 2:
        message, confidence = result
    if message is not None:
        arr = np.asarray(message, dtype=np.float64)
        message = arr.ravel().tolist() if arr.ndim else [float(arr)]
    return {"message": message, "confidence": None if confidence is None else float(confidence)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="parity runner (in-image)")
    ap.add_argument("--stage", default="/app", help="directory holding the plugin's inference.py")
    ap.add_argument("--cwd", default=None, help="working directory (default: --stage)")
    ap.add_argument("--op", choices=("embed", "detect", "apply"), required=True)
    ap.add_argument("--inputs", required=True, help="directory of <case>.npy inputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), required=True)
    ap.add_argument("--numerics", choices=("strict", "torch-default"), default="strict")
    ap.add_argument("--sampling-rate", type=int, default=16000)
    ap.add_argument("--params", default="{}", help="JSON of request-level params for apply")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=2, help="runs per case; >1 checks bit-identity")
    ap.add_argument("--engine-override", default="", help="path.py:Class to use instead of the shipped engine")
    ap.add_argument("--cases", default="", help="comma-separated case names (default: all)")
    a = ap.parse_args(argv)

    cwd = a.cwd or a.stage
    os.chdir(cwd)
    sys.path[:0] = [a.stage, cwd]
    if a.numerics == "strict":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import numpy as np
    import torch

    numerics = _configure_numerics(a.numerics == "strict")
    with open(os.path.join(cwd, "config.json")) as fh:
        config = json.load(fh)
    engine_cls = _load_engine_class(a.stage, a.engine_override)
    t0 = time.perf_counter()
    engine = engine_cls(config, device=a.device)
    load_s = time.perf_counter() - t0

    params = json.loads(a.params)
    files = sorted(glob.glob(os.path.join(a.inputs, "*.npy")))
    wanted = set(filter(None, a.cases.split(",")))
    if wanted:
        files = [f for f in files if os.path.basename(f)[:-4] in wanted]
    if not files:
        raise SystemExit(f"no inputs in {a.inputs}")
    os.makedirs(a.out, exist_ok=True)
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def call(stem, audio):
        seed = case_seed(a.seed, stem)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if a.op == "embed":
            return engine.embed(audio, case_bits(stem, int(config["watermark_size"])), a.sampling_rate)
        if a.op == "detect":
            return _normalise_detect(engine.detect(audio, a.sampling_rate))
        return engine.apply(audio, a.sampling_rate, **params)

    with open(os.path.join(a.out, "results.jsonl"), "w") as results:
        for path in files:
            stem = os.path.basename(path)[:-4]
            audio = np.load(path)
            outputs, times = [], []
            for _ in range(max(1, a.repeat)):
                t = time.perf_counter()
                out = call(stem, audio.copy())
                if a.device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
                outputs.append(out)
            first = outputs[0]
            record = {"case": stem, "ms": [round(x * 1000, 3) for x in times]}
            if a.op == "detect":
                with open(os.path.join(a.out, f"{stem}.json"), "w") as fh:
                    json.dump(first, fh)
                identical = all(o == first for o in outputs[1:])
            else:
                arr = np.asarray(first, dtype=np.float64).ravel()
                np.save(os.path.join(a.out, f"{stem}.npy"), arr)
                identical = all(
                    np.array_equal(arr, np.asarray(o, dtype=np.float64).ravel()) for o in outputs[1:]
                )
                record.update(out_len=int(arr.size), finite=bool(np.all(np.isfinite(arr))))
            record["repeat_identical"] = identical if len(outputs) > 1 else None
            results.write(json.dumps(record) + "\n")

    env = {
        "op": a.op,
        "device_requested": a.device,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numerics": numerics,
        "placement": _placement(engine),
        "cuda_max_allocated_mib": (
            round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0.0
        ),
        "engine": f"{engine_cls.__module__}.{engine_cls.__name__}",
        "engine_override": a.engine_override or None,
        "params": params,
        "seed": a.seed,
        "repeat": a.repeat,
        "load_s": round(load_s, 3),
        "n_cases": len(files),
        "image_id": os.environ.get("DEEPMARK_IMAGE_ID"),
    }
    with open(os.path.join(a.out, "env.json"), "w") as fh:
        json.dump(env, fh, indent=1)
    print(json.dumps({k: env[k] for k in ("device_requested", "torch", "placement", "cuda_max_allocated_mib", "n_cases")}))


if __name__ == "__main__":
    main()
