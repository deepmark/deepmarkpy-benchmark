"""Prove a quantising codec's CPU/GPU differences are tie-breaks and nothing else.

A codec chooses a discrete code per frame: a codebook entry (EnCodec and DAC
residual VQ) or a rounding level (XCodec2's FSQ). When the latent sits almost
exactly between two choices, rounding noise of 1e-6 can pick the other one,
and the decoded audio then differs locally by far more than any float
tolerance. That is a tie-break, not an error. This probe makes the claim
checkable rather than assumed. It is mounted into the codec's image next to
runner.py, and its hooks replicate each library's selection code exactly:

    record   run the engine as is; save each layer's chosen codes, the
             near-tie margin of every choice, and the quantiser's input
    inject   run the engine with the codes taken from another run (the GPU's),
             so everything else is computed on this device

The host then requires:
- every frame's first differing code to have been a near-tie on the CPU
  (margin < 1e-4);
- the GPU output to match the CPU decoding of the GPU's own codes at the
  NUMERIC bounds.

Together these show the two devices disagree only where the codec's own
decision was a coin toss.

Margins:
- RVQ: (second-best distance − best) / |best|.
- FSQ: distance of the bounded latent to the nearest rounding boundary, in
  level units, minimised over dimensions.
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


class _State:
    def __init__(self):
        self.codes, self.margins, self.inputs = [], [], []
        self.inject = None  # list of per-layer code arrays to substitute, consumed in call order
        self.codebooks = {}  # call index within a case -> codebook (N, D), as the selection used it

    def reset(self, inject=None):
        self.codes, self.margins, self.inputs = [], [], []
        self.inject = list(inject) if inject is not None else None

    def take(self):
        if not self.inject:
            raise RuntimeError("more quantiser calls than injected layers")
        return self.inject.pop(0)


STATE = _State()


def _hook_encodec():
    import encodec.quantization.core_vq as cv
    import torch

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices  # exactly as the library
        STATE.inputs.append(x.detach().float().cpu().numpy())
        STATE.codebooks.setdefault(len(STATE.inputs) - 1, self.embed.detach().float().cpu().numpy())
        top = dist.topk(2, dim=-1).values
        best, second = -top[:, 0], -top[:, 1]
        STATE.margins.append(((second - best) / best.abs().clamp_min(1e-12)).double().cpu().numpy())
        if STATE.inject is not None:
            embed_ind = torch.as_tensor(STATE.take(), device=embed_ind.device, dtype=embed_ind.dtype)
        STATE.codes.append(embed_ind.cpu().numpy())
        return embed_ind

    cv.EuclideanCodebook.quantize = quantize


def _hook_dac():
    import dac.nn.quantize as q
    import torch
    import torch.nn.functional as F
    from einops import rearrange

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))  # exactly as the library
        STATE.inputs.append(encodings.detach().float().cpu().numpy())
        STATE.codebooks.setdefault(len(STATE.inputs) - 1, codebook.detach().float().cpu().numpy())
        top = (-dist).topk(2, dim=-1).values
        best, second = -top[:, 0], -top[:, 1]
        STATE.margins.append(((second - best) / best.abs().clamp_min(1e-12)).double().cpu().numpy())
        if STATE.inject is not None:
            indices = torch.as_tensor(STATE.take(), device=indices.device, dtype=indices.dtype).view_as(indices)
        STATE.codes.append(indices.reshape(-1).cpu().numpy())
        z_q = self.decode_code(indices)
        return z_q, indices

    q.VectorQuantize.decode_latents = decode_latents


def _hook_fsq():
    import torch
    import vector_quantize_pytorch.finite_scalar_quantization as fsq

    round_ste = fsq.round_ste

    def quantize(self, z):
        bounded = self.bound(z)
        quantized = round_ste(bounded)  # exactly as the library
        STATE.inputs.append(bounded.detach().reshape(-1, bounded.shape[-1]).float().cpu().numpy())
        to_boundary = 0.5 - (bounded - torch.round(bounded)).abs()
        STATE.margins.append(to_boundary.reshape(-1, to_boundary.shape[-1]).min(dim=-1).values.double().cpu().numpy())
        if STATE.inject is not None:
            levels = torch.as_tensor(STATE.take(), device=quantized.device, dtype=quantized.dtype)
            quantized = levels.view_as(quantized)
        STATE.codes.append(quantized.detach().reshape(-1, quantized.shape[-1]).cpu().numpy())
        half_width = self._levels // 2
        return quantized / half_width

    fsq.FSQ.quantize = quantize


HOOKS = {"encodec": _hook_encodec, "dac": _hook_dac, "fsq": _hook_fsq}


def main(argv=None):
    ap = argparse.ArgumentParser(description="codec tie-break probe (in-image)")
    ap.add_argument("--codec", choices=sorted(HOOKS), required=True)
    ap.add_argument("--mode", choices=("record", "inject"), required=True)
    ap.add_argument("--codes-from", default=None, help="inject: directory of <case>.codes.npz from a record run")
    ap.add_argument("--stage", default="/app")
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), required=True)
    ap.add_argument("--sampling-rate", type=int, default=16000)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)

    sys.path.insert(0, HERE)
    import runner  # the parity runner mounted alongside: same engine loading and numerics

    cwd = a.cwd or a.stage
    os.chdir(cwd)
    sys.path[:0] = [a.stage, cwd]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import random

    import numpy as np
    import torch

    numerics = runner._configure_numerics(True)
    HOOKS[a.codec]()
    with open(os.path.join(cwd, "config.json")) as fh:
        config = json.load(fh)
    engine = runner._load_engine_class(a.stage, "")(config, device=a.device)
    params = json.loads(a.params)
    os.makedirs(a.out, exist_ok=True)

    summary = {}
    for path in sorted(glob.glob(os.path.join(a.inputs, "*.npy"))):
        stem = os.path.basename(path)[:-4]
        inject = None
        if a.mode == "inject":
            with np.load(os.path.join(a.codes_from, f"{stem}.codes.npz")) as saved:
                inject = [saved[f"layer{i}"] for i in range(len(saved.files))]
        STATE.reset(inject)
        seed = runner.case_seed(a.seed, stem)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        out = np.asarray(engine.apply(np.load(path), a.sampling_rate, **params), dtype=np.float64).ravel()
        if a.mode == "inject" and STATE.inject:
            raise RuntimeError(f"{stem}: {len(STATE.inject)} injected layers were never used")
        np.save(os.path.join(a.out, f"{stem}.npy"), out)
        np.savez(os.path.join(a.out, f"{stem}.codes.npz"), **{f"layer{i}": c for i, c in enumerate(STATE.codes)})
        np.savez(os.path.join(a.out, f"{stem}.margins.npz"), **{f"layer{i}": m for i, m in enumerate(STATE.margins)})
        np.savez(os.path.join(a.out, f"{stem}.inputs.npz"), **{f"layer{i}": x for i, x in enumerate(STATE.inputs)})
        summary[stem] = {"layers": len(STATE.codes)}
    np.savez(os.path.join(a.out, "codebooks.npz"), **{f"layer{i}": c for i, c in sorted(STATE.codebooks.items())})
    with open(os.path.join(a.out, "probe.json"), "w") as fh:
        json.dump({"codec": a.codec, "mode": a.mode, "device": a.device, "numerics": numerics,
                   "torch": torch.__version__, "cases": summary}, fh, indent=1)
    print(json.dumps({"mode": a.mode, "device": a.device, "cases": len(summary)}))


if __name__ == "__main__":
    main()
