"""Tolerance tiers, and the judgement of one component's parity evidence.

The bounds are the ones agreed in the GPU plan, §3.2 of
scripts/gpu_study/PLAN-gpu-implementation.md. They are derived from what the
GPU study measured with TF32 off.

- **EXACT**: no difference at all. For changes that must not alter a single
  bit: transport, caching, scheduling.
- **NUMERIC**: a deterministic float path on another device.
  - max |d| ≤ 1e-4 (−80 dBFS);
  - relative L2 ≤ 1e-5 median and ≤ 3e-5 worst;
  - no decoded-bit flips, |Δconfidence| ≤ 1e-4;
  - small downstream metric deltas;
  - the candidate repeats bit for bit.
- **QUANTIZED**: a codec whose quantiser can resolve a near-tie the other
  way. A frame where |d| exceeds 1e-4 is a flip frame. At most 1% of frames
  may flip, and every flip frame must be explained: evidence from the
  reference run shows its codebook margin was a near-tie. Outside the flip
  frames, the NUMERIC bounds apply.

Stochastic components are judged on distributions, not here.

Evidence that a tier needs but a run did not produce counts as a failure.
A verdict cannot pass because a check was skipped.
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from deepmarkpy.parity.compare import WaveformDiff

# Kinds of evidence a judgement can require.
SIGNAL = "signal"          # waveform output compared with the reference
REPEAT = "repeat"          # candidate run twice, bit-identical
BITS = "bits"              # decoded watermark bits compared (CPU detector, or the op itself)
CONFIDENCE = "confidence"  # detector confidence compared
METRICS = "metrics"        # PESQ / ViSQOL / STOI against the clean clip compared


@dataclass(frozen=True)
class Tier:
    name: str
    max_abs: Optional[float]
    rel_l2_median: Optional[float]
    rel_l2_worst: Optional[float]
    max_bit_flips: int = 0
    max_conf_delta: Optional[float] = None
    max_metric_delta: Mapping[str, float] = field(default_factory=dict)
    # QUANTIZED only: a frame whose max |d| exceeds flip_frame_atol is a flip.
    flip_frame_atol: Optional[float] = None
    max_flip_fraction: float = 0.0


EXACT = Tier(
    "EXACT", max_abs=0.0, rel_l2_median=0.0, rel_l2_worst=0.0,
    max_bit_flips=0, max_conf_delta=0.0,
    max_metric_delta={"pesq": 0.0, "visqol": 0.0, "stoi": 0.0},
)
NUMERIC = Tier(
    "NUMERIC", max_abs=1e-4, rel_l2_median=1e-5, rel_l2_worst=3e-5,
    max_bit_flips=0, max_conf_delta=1e-4,
    max_metric_delta={"pesq": 1e-3, "visqol": 1e-3, "stoi": 1e-4},
)
QUANTIZED = Tier(
    "QUANTIZED", max_abs=1e-4, rel_l2_median=1e-5, rel_l2_worst=3e-5,
    max_bit_flips=0, max_conf_delta=1e-4,
    max_metric_delta={"pesq": 5e-3, "visqol": 5e-3, "stoi": 5e-4},
    flip_frame_atol=1e-4, max_flip_fraction=0.01,
)
TIERS = {t.name: t for t in (EXACT, NUMERIC, QUANTIZED)}


@dataclass
class CaseEvidence:
    """Everything measured for one corpus case, reference vs candidate."""

    case: str
    signal: Optional[WaveformDiff] = None
    # QUANTIZED: the same comparison with the flip frames zeroed in both.
    signal_outside_flips: Optional[WaveformDiff] = None
    # QUANTIZED: frames the reference run shows were codebook near-ties.
    explained_frames: Optional[Tuple[int, ...]] = None
    repeat_identical: Optional[bool] = None
    bit_flips: Optional[int] = None
    conf_delta: Optional[float] = None
    metric_deltas: Dict[str, float] = field(default_factory=dict)


@dataclass
class Judgement:
    tier: str
    passed: bool
    failures: List[str]
    n_cases: int
    rel_l2_median: Optional[float]
    rel_l2_worst: Optional[float]
    max_abs: Optional[float]
    flip_frames: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _fmt(x: float) -> str:
    return "inf" if math.isinf(x) else f"{x:.3g}"


def judge(cases: Sequence[CaseEvidence], tier: Tier, required: Sequence[str]) -> Judgement:
    """Judge ``cases`` against ``tier``. Every kind in ``required`` must be present for every case."""
    failures: List[str] = []
    rel_l2s: List[float] = []
    max_abs_seen = 0.0
    flip_frames = 0

    if not cases:
        failures.append("no cases: an empty corpus proves nothing")

    for ev in cases:
        where = ev.case
        if SIGNAL in required:
            sig = ev.signal
            if sig is None:
                failures.append(f"{where}: no waveform comparison")
            elif not sig.same_length:
                failures.append(f"{where}: length {sig.cand_len} vs reference {sig.ref_len}")
            elif not sig.finite:
                failures.append(f"{where}: candidate has NaN or Inf")
            else:
                judged = sig
                if tier.flip_frame_atol is not None:
                    flips = set(sig.frames_over)
                    flip_frames += len(flips)
                    if len(flips) > tier.max_flip_fraction * sig.n_frames:
                        failures.append(
                            f"{where}: {len(flips)} of {sig.n_frames} frames flipped, "
                            f"limit {tier.max_flip_fraction:.0%}"
                        )
                    if flips:
                        if ev.explained_frames is None:
                            failures.append(f"{where}: flip frames {sorted(flips)} have no near-tie evidence")
                        else:
                            unexplained = sorted(flips - set(ev.explained_frames))
                            if unexplained:
                                failures.append(f"{where}: flip frames {unexplained} are not near-ties")
                        if ev.signal_outside_flips is None:
                            failures.append(f"{where}: no comparison outside the flip frames")
                        else:
                            judged = ev.signal_outside_flips
                max_abs_seen = max(max_abs_seen, judged.max_abs)
                if tier.max_abs is not None and judged.max_abs > tier.max_abs:
                    failures.append(f"{where}: max |d| {_fmt(judged.max_abs)} > {tier.max_abs:g}")
                # A near-silent reference is judged on max |d| alone (see compare.REL_L2_MIN_REF_RMS).
                if sig.relative_applies:
                    rel_l2s.append(judged.rel_l2)
                    if tier.rel_l2_worst is not None and judged.rel_l2 > tier.rel_l2_worst:
                        failures.append(f"{where}: relative L2 {_fmt(judged.rel_l2)} > {tier.rel_l2_worst:g}")

        if REPEAT in required and ev.repeat_identical is not True:
            failures.append(
                f"{where}: candidate did not repeat bit for bit" if ev.repeat_identical is False
                else f"{where}: no repeat run"
            )
        if BITS in required:
            if ev.bit_flips is None:
                failures.append(f"{where}: decoded bits not compared")
            elif ev.bit_flips > tier.max_bit_flips:
                failures.append(f"{where}: {ev.bit_flips} decoded bit(s) flipped")
        if CONFIDENCE in required:
            if ev.conf_delta is None:
                failures.append(f"{where}: confidence not compared")
            elif tier.max_conf_delta is not None and ev.conf_delta > tier.max_conf_delta:
                failures.append(f"{where}: |Δconfidence| {_fmt(ev.conf_delta)} > {tier.max_conf_delta:g}")
        if METRICS in required:
            for metric, bound in tier.max_metric_delta.items():
                delta = ev.metric_deltas.get(metric)
                if delta is None:
                    failures.append(f"{where}: {metric} not compared")
                elif delta > bound:
                    failures.append(f"{where}: |Δ{metric}| {_fmt(delta)} > {bound:g}")

    median = statistics.median(rel_l2s) if rel_l2s else None
    if SIGNAL in required and median is not None and tier.rel_l2_median is not None and median > tier.rel_l2_median:
        failures.append(f"median relative L2 {_fmt(median)} > {tier.rel_l2_median:g}")

    return Judgement(
        tier=tier.name,
        passed=not failures,
        failures=failures,
        n_cases=len(cases),
        rel_l2_median=median,
        rel_l2_worst=max(rel_l2s) if rel_l2s else None,
        max_abs=max_abs_seen if cases else None,
        flip_frames=flip_frames,
    )
