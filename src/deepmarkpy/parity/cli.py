"""deepmark-parity: measure whether a service computes the same numbers on GPU as on CPU.

    deepmark-parity corpus --sources DIR                          # once per sampling rate
    deepmark-parity run audioseal embed --variant cpu --label cpu
    deepmark-parity run audioseal embed --variant gpu --label gpu
    deepmark-parity wm audioseal --label cpu                      # attack inputs: the CPU-watermarked corpus
    deepmark-parity run encodec apply --variant cpu --label cpu
    deepmark-parity run encodec apply --variant gpu --label gpu
    deepmark-parity judge encodec apply --ref cpu --cand gpu      # writes parity/verdicts/..., exit 0 = PASS

Configuration:
- Images come from build/images.json, written by scripts/build_images.py.
- Gates come from parity/gates.json.
- Runs and the corpus live under $DEEPMARK_PARITY_ROOT (default build/parity).
- The container engine is $CONTAINER_ENGINE (default docker).
- GPU runs use only $DEEPMARK_GPU_DEVICE, vetted by deepmarkpy.utils.gpu_slice first.

Two checks protect the result:
- A GPU run that allocated no CUDA memory is an error. It ran on the CPU,
  and comparing CPU with CPU proves nothing about the GPU.
- A CPU reference that does not repeat itself fails the gate. Nothing can be
  judged against a moving target.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from deepmarkpy.parity import runner as runner_module
from deepmarkpy.parity.compare import bit_flips, masked_diff, waveform_diff
from deepmarkpy.parity.corpus import _sha256, build_corpus, load_manifest, verify_inputs
from deepmarkpy.parity.tiers import TIERS, CaseEvidence, judge
from deepmarkpy.parity.verdicts import write_verdict
from deepmarkpy.utils.gpu_slice import SliceUnavailable, check_slice

logger = logging.getLogger("deepmark-parity")

REPO = Path(__file__).resolve().parents[3]
METRICS = ("pesq", "visqol", "stoi")


class ParityError(RuntimeError):
    pass


def parity_root() -> Path:
    return Path(os.environ.get("DEEPMARK_PARITY_ROOT") or REPO / "build" / "parity")


def load_gates() -> Dict:
    return json.loads((REPO / "parity" / "gates.json").read_text())


def load_images() -> Dict:
    path = REPO / "build" / "images.json"
    if not path.exists():
        raise ParityError(f"{path} not found; build images with scripts/build_images.py first")
    return json.loads(path.read_text())


def git_rev() -> str:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    return rev + ("-dirty" if dirty else "")


def _run_dir(service, op, label, inputs) -> Path:
    return parity_root() / "runs" / service / op / label / inputs


def _container(image_tag, runner_args, *, gpu_uuid=None, cpu_threads=8, extra_mounts=(), env=None):
    engine = os.environ.get("CONTAINER_ENGINE", "docker")
    root = parity_root()
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "cache" / "home").mkdir(parents=True, exist_ok=True)
    cmd = [engine, "run", "--rm", "--entrypoint", "python", "--cpus", str(cpu_threads)]
    if gpu_uuid:
        cmd += ["--device", f"nvidia.com/gpu={gpu_uuid}"]
    else:
        cmd += ["-e", "CUDA_VISIBLE_DEVICES="]
    threads = {k: str(cpu_threads) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for key, value in {**threads, **(env or {})}.items():
        cmd += ["-e", f"{key}={value}"]
    mounts = [
        (Path(runner_module.__file__), "/parity/runner.py", "ro"),
        (root, "/corpus", "ro"),
        (root / "runs", "/runs", "rw"),
        (root / "cache" / "home", "/root/.cache", "rw"),
        *extra_mounts,
    ]
    for host, ctr, mode in mounts:
        cmd += ["-v", f"{host}:{ctr}:{mode}"]
    cmd += [image_tag, "/parity/runner.py", *runner_args]
    return cmd


def _execute(cmd, log_path: Path):
    logger.info("$ %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = log_path.read_text().splitlines()[-15:]
        raise ParityError(f"container exited {result.returncode}; {log_path}:\n" + "\n".join(tail))


def cmd_corpus(a):
    manifest = build_corpus(Path(a.sources), parity_root(), a.sr)
    print(json.dumps({"cases": len(manifest["cases"]), "corpus_sha256": manifest["corpus_sha256"],
                      "root": str(parity_root() / str(a.sr))}))
    return 0


def cmd_wm(a):
    """Make the CPU reference embed of the corpus the input set every attack is measured on."""
    root = parity_root()
    src = _run_dir(a.model, "embed", a.label, "clean")
    run = json.loads((src / "run.json").read_text())
    if run["variant"] != "cpu":
        raise ParityError(f"{src} is a {run['variant']} run; attack inputs must be the CPU reference")
    dst = root / str(a.sr) / f"wm_{a.model}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    entries = {}
    for npy in sorted(src.glob("*.npy")):
        shutil.copy2(npy, dst / npy.name)
        entries[npy.stem] = {"sha256": _sha256(np.load(npy)), "samples": int(np.load(npy).size)}
    (root / str(a.sr) / f"wm_{a.model}.json").write_text(json.dumps(
        {"from_run": run, "cases": entries}, indent=1) + "\n")
    print(json.dumps({"inputs": f"wm_{a.model}", "cases": len(entries), "image_id": run["image_id"]}))
    return 0


def _input_entries(inputs: str, sr: int) -> Dict:
    root = parity_root()
    if inputs == "clean":
        return load_manifest(root, sr)["cases"]
    return json.loads((root / str(sr) / f"{inputs}.json").read_text())["cases"]


def cmd_run(a):
    gate = load_gates()[a.service]
    spec = gate["ops"][a.op]
    image = load_images()[a.service][a.variant]
    stage = gate.get("stage", "/app")
    cwd = gate.get("cwd", stage)

    slice_info = None
    if a.variant == "gpu":
        status = check_slice(min_free_mib=int(a.min_free_gib * 1024))
        slice_info = {"uuid": status.uuid, "name": status.name, "used_mib_before": status.used_mib}

    extra_mounts = []
    for name, ctr in gate.get("mounts", {}).items():
        host = parity_root() / "cache" / a.service / name
        host.mkdir(parents=True, exist_ok=True)
        extra_mounts.append((host, ctr, "rw"))
    override_arg = []
    if a.engine_override:
        path, cls = a.engine_override.rsplit(":", 1)
        extra_mounts.append((Path(path).resolve(), f"/override/{Path(path).name}", "ro"))
        override_arg = ["--engine-override", f"/override/{Path(path).name}:{cls}"]

    env = {"DEEPMARK_IMAGE_ID": image["image_id"]}
    if a.variant == "gpu" and a.numerics == "strict":
        env.update(NVIDIA_TF32_OVERRIDE="0", STUDY_TF32="0")

    for inputs in a.inputs or spec["inputs"]:
        entries = _input_entries(inputs, a.sr)
        verify_inputs(parity_root() / str(a.sr) / inputs, entries)
        out = _run_dir(a.service, a.op, a.label, inputs)
        if out.exists():
            shutil.rmtree(out)
        out_rel = out.relative_to(parity_root() / "runs")
        args = ["--stage", stage, "--cwd", cwd, "--op", a.op,
                "--inputs", f"/corpus/{a.sr}/{inputs}", "--out", f"/runs/{out_rel}",
                "--device", "cuda" if a.variant == "gpu" else "cpu", "--numerics", a.numerics,
                "--sampling-rate", str(a.sr), "--params", json.dumps(spec.get("params", {})),
                "--seed", str(a.seed), "--repeat", str(a.repeat), *override_arg]
        if a.cases:
            args += ["--cases", a.cases]
        cmd = _container(image["tag"], args, gpu_uuid=slice_info and slice_info["uuid"],
                         cpu_threads=a.cpu_threads, extra_mounts=extra_mounts, env=env)
        _execute(cmd, out.parent / f"{inputs}.container.log")

        run_env = json.loads((out / "env.json").read_text())
        if a.variant == "gpu" and not run_env.get("cuda_max_allocated_mib"):
            raise ParityError(f"{a.service} {a.op}: the GPU run allocated no CUDA memory, so it ran on "
                              f"the CPU (placement {run_env.get('placement')}). Fix device placement first.")
        record = {"service": a.service, "op": a.op, "label": a.label, "variant": a.variant,
                  "inputs": inputs, "tag": image["tag"], "image_id": image["image_id"],
                  "cpu_threads": a.cpu_threads, "numerics": a.numerics, "seed": a.seed,
                  "repeat": a.repeat, "engine_override": a.engine_override or None,
                  "gpu_slice": slice_info, "git_rev": git_rev(), "env": run_env}
        (out / "run.json").write_text(json.dumps(record, indent=1) + "\n")
        print(json.dumps({"run": str(out_rel), "placement": run_env["placement"],
                          "cuda_max_allocated_mib": run_env["cuda_max_allocated_mib"]}))
    return 0


def _results(run_dir: Path) -> Dict[str, Dict]:
    with open(run_dir / "results.jsonl") as fh:
        return {r["case"]: r for r in map(json.loads, fh)}


def _detect_outputs(detector: str, run_dir: Path, a) -> Dict[str, Dict]:
    """Decode every output in ``run_dir`` with the detector's CPU engine (cached per run)."""
    out = run_dir / f"detect_{detector}"
    if not (out / "env.json").exists():
        image = load_images()[detector]["cpu"]
        rel_in = run_dir.relative_to(parity_root() / "runs")
        args = ["--op", "detect", "--inputs", f"/runs/{rel_in}", "--out", f"/runs/{rel_in}/detect_{detector}",
                "--device", "cpu", "--sampling-rate", str(a.sr), "--repeat", "1"]
        _execute(_container(image["tag"], args, cpu_threads=a.cpu_threads), run_dir / f"detect_{detector}.log")
    return {p.stem: json.loads(p.read_text()) for p in out.glob("*.json") if p.name != "env.json"}


def _metric_deltas(clean, ref, cand, sr) -> Dict[str, float]:
    from deepmarkpy.utils.metrics import compute_metrics

    r = compute_metrics(clean, ref, sr, metrics=set(METRICS))
    c = compute_metrics(clean, cand, sr, metrics=set(METRICS))
    deltas = {}
    for m in METRICS:
        if r.get(m) is None and c.get(m) is None:
            deltas[m] = 0.0  # undefined for this case on both devices (e.g. PESQ on silence)
        elif r.get(m) is not None and c.get(m) is not None:
            deltas[m] = abs(float(c[m]) - float(r[m]))
        # one side only: left out, which the judgement reports as not compared
    return deltas


def cmd_judge(a):
    spec = load_gates()[a.service]["ops"][a.op]
    tier_name = a.tier or spec["tier"]
    if tier_name not in TIERS:
        raise ParityError(f"{a.service} {a.op} is {tier_name}: judged on distributions, not by a tier")
    tier, evidence = TIERS[tier_name], spec["evidence"]
    clean_dir = parity_root() / str(a.sr) / "clean"

    cases: List[CaseEvidence] = []
    extra_failures: List[str] = []
    runs = {}
    for inputs in spec["inputs"]:
        ref_dir = _run_dir(a.service, a.op, a.ref, inputs)
        cand_dir = _run_dir(a.service, a.op, a.cand, inputs)
        runs[inputs] = {"reference": json.loads((ref_dir / "run.json").read_text()),
                        "candidate": json.loads((cand_dir / "run.json").read_text())}
        ref_res, cand_res = _results(ref_dir), _results(cand_dir)
        if set(ref_res) != set(cand_res):
            extra_failures.append(f"{inputs}: reference and candidate ran different cases")
        for stem, r in ref_res.items():
            if r.get("repeat_identical") is False:
                extra_failures.append(f"{inputs}/{stem}: the reference itself does not repeat")

        if a.op == "detect":
            for stem in sorted(ref_res):
                ref = json.loads((ref_dir / f"{stem}.json").read_text())
                cand = json.loads((cand_dir / f"{stem}.json").read_text())
                conf = (abs(cand["confidence"] - ref["confidence"])
                        if ref["confidence"] is not None and cand["confidence"] is not None else None)
                cases.append(CaseEvidence(
                    case=f"{inputs}/{stem}", repeat_identical=cand_res.get(stem, {}).get("repeat_identical"),
                    bit_flips=bit_flips(ref["message"], cand["message"]), conf_delta=conf))
            continue

        detected_ref = detected_cand = None
        if "bits" in evidence:
            detected_ref = _detect_outputs(spec["detector"], ref_dir, a)
            detected_cand = _detect_outputs(spec["detector"], cand_dir, a)
        for stem in sorted(ref_res):
            ref, cand = np.load(ref_dir / f"{stem}.npy"), np.load(cand_dir / f"{stem}.npy")
            sig = waveform_diff(ref, cand, a.sr, frame_atol=tier.flip_frame_atol)
            outside = (masked_diff(ref, cand, a.sr, sig.frames_over)
                       if tier.flip_frame_atol is not None and sig.frames_over else None)
            flips = None
            if detected_ref is not None:
                flips = bit_flips(detected_ref.get(stem, {}).get("message"),
                                  detected_cand.get(stem, {}).get("message"))
            deltas = (_metric_deltas(np.load(clean_dir / f"{stem}.npy"), ref, cand, a.sr)
                      if "metrics" in evidence else {})
            cases.append(CaseEvidence(
                case=f"{inputs}/{stem}", signal=sig, signal_outside_flips=outside,
                repeat_identical=cand_res.get(stem, {}).get("repeat_identical"),
                bit_flips=flips, metric_deltas=deltas))

    judgement = judge(cases, tier, evidence)
    judgement.failures = extra_failures + judgement.failures
    judgement.passed = not judgement.failures
    first = runs[spec["inputs"][0]]
    manifest = load_manifest(parity_root(), a.sr)
    record = {
        "service": a.service, "op": a.op, "tier": tier_name, "evidence": evidence,
        "passed": judgement.passed, "failures": judgement.failures, "summary": judgement.to_dict(),
        "reference": {k: first["reference"][k] for k in ("label", "variant", "tag", "image_id", "env", "cpu_threads")},
        "candidate": {k: first["candidate"][k] for k in ("label", "variant", "tag", "image_id", "env", "gpu_slice", "engine_override")},
        "corpus": {"sha256": manifest["corpus_sha256"], "sampling_rate": a.sr, "inputs": spec["inputs"]},
        "cases": [{**c.__dict__, "signal": c.signal and c.signal.to_dict(),
                   "signal_outside_flips": c.signal_outside_flips and c.signal_outside_flips.to_dict()}
                  for c in cases],
        "git_rev": git_rev(),
    }
    path = write_verdict(record)
    s = judgement
    print(f"{a.service} {a.op}: {'PASS' if s.passed else 'FAIL'} ({tier_name}, {s.n_cases} cases) "
          f"rel.L2 median {s.rel_l2_median} worst {s.rel_l2_worst} max|d| {s.max_abs} flip frames {s.flip_frames}")
    for failure in s.failures[:25]:
        print(f"  - {failure}")
    if len(s.failures) > 25:
        print(f"  ... {len(s.failures) - 25} more in {path}")
    print(f"verdict: {path}")
    return 0 if s.passed else 1


def main(argv=None) -> int:
    from deepmarkpy.utils.utils import load_env_file

    parser = argparse.ArgumentParser(prog="deepmark-parity", description=__doc__.split("\n\n")[0])
    parser.add_argument("--sr", type=int, default=16000, help="corpus sampling rate")
    parser.add_argument("--cpu-threads", type=int, default=8,
                        help="CPU threads per container (CPU torch results depend on it; recorded)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("corpus", help="build the parity corpus")
    p.add_argument("--sources", required=True, help="directory of speech clips")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("run", help="run one service op over its inputs, CPU or GPU")
    p.add_argument("service")
    p.add_argument("op", choices=("embed", "detect", "apply"))
    p.add_argument("--variant", choices=("cpu", "gpu"), required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--inputs", nargs="*", help="input sets (default: the gate's)")
    p.add_argument("--numerics", choices=("strict", "torch-default"), default="strict")
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cases", default="", help="comma-separated subset (debugging only)")
    p.add_argument("--engine-override", default="", help="path.py:Class (study engines)")
    p.add_argument("--min-free-gib", type=float, default=4.0)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("wm", help="use a model's CPU reference embed as the attacks' input set")
    p.add_argument("model")
    p.add_argument("--label", required=True)
    p.set_defaults(func=cmd_wm)

    p = sub.add_parser("judge", help="judge a candidate run against a reference run; write the verdict")
    p.add_argument("service")
    p.add_argument("op", choices=("embed", "detect", "apply"))
    p.add_argument("--ref", required=True)
    p.add_argument("--cand", required=True)
    p.add_argument("--tier", default=None, help="override the gate's tier (e.g. EXACT)")
    p.set_defaults(func=cmd_judge)

    a = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_env_file(str(REPO / ".env"))
    try:
        return a.func(a)
    except (ParityError, SliceUnavailable, KeyError, FileNotFoundError, ValueError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
