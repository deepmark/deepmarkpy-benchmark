"""How far a candidate output is from its reference, element-wise and frame-wise.

Correlation is not used as an acceptance measure. With cuDNN's TF32 on, the
GPU study saw AudioSeal's full-signal PCC read 0.999997 while the watermark
residual dropped to 23 dB and decoded bits flipped. The measures here are:

- the size of the difference itself (max |d|, relative L2, SNR of d);
- where it sits (per 20 ms frame);
- whether a difference exists at all.
"""

import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

FRAME_SECONDS = 0.02

# Floor for the reference's RMS when forming the relative error, so a silent
# reference (a valid corpus case) gives a finite number.
REL_L2_RMS_FLOOR = 1e-6

# An RMS difference at or below this counts as no difference for the
# relative checks: -120 dBFS, about 8 float32 steps at full scale. A case
# passes when its RMS error is within max(bound x reference RMS, this floor),
# so a quiet reference is not asked for agreement below what float32
# networks resolve. Decided after wave 1: the DAC "silence" case differed by
# 4e-8 (-148 dBFS), yet failed a relative bound of 3e-5 at -78 dBFS.
ABS_RMS_FLOOR = 1e-6


@dataclass(frozen=True)
class WaveformDiff:
    """One reference/candidate pair. Lengths must match for any of it to count."""

    ref_len: int
    cand_len: int
    ref_rms: float
    err_rms: float
    finite: bool
    identical: bool
    max_abs: float
    rel_l2: float
    snr_db: float
    frame_len: int
    n_frames: int
    # Frames whose max |d| exceeds the frame tolerance passed to
    # waveform_diff(); empty when it was not given.
    frames_over: Tuple[int, ...] = ()

    @property
    def same_length(self) -> bool:
        return self.ref_len == self.cand_len

    @property
    def below_floor(self) -> bool:
        """Whether the RMS difference is within the absolute floor (counts as none)."""
        return self.err_rms <= ABS_RMS_FLOOR

    def within_relative(self, bound: float) -> bool:
        """RMS error within max(bound x reference RMS, ABS_RMS_FLOOR)."""
        return self.err_rms <= max(bound * self.ref_rms, ABS_RMS_FLOOR)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["frames_over"] = list(self.frames_over)
        out["same_length"] = self.same_length
        out["below_floor"] = self.below_floor
        return out


def _as_signal(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def waveform_diff(ref, cand, sampling_rate: int, frame_atol: Optional[float] = None) -> WaveformDiff:
    """Compare ``cand`` against ``ref``. Nothing is trimmed: a length mismatch is a result."""
    ref, cand = _as_signal(ref), _as_signal(cand)
    frame_len = max(1, int(round(FRAME_SECONDS * sampling_rate)))
    finite = bool(np.all(np.isfinite(cand)))

    if ref.shape != cand.shape or not finite:
        return WaveformDiff(
            ref_len=ref.size, cand_len=cand.size, ref_rms=float(np.sqrt(np.mean(ref * ref))) if ref.size else 0.0,
            err_rms=math.inf, finite=finite, identical=False,
            max_abs=math.inf, rel_l2=math.inf, snr_db=-math.inf,
            frame_len=frame_len, n_frames=math.ceil(ref.size / frame_len),
        )

    d = cand - ref
    identical = bool(np.array_equal(ref, cand))
    max_abs = float(np.max(np.abs(d))) if d.size else 0.0
    err = float(np.sqrt(np.mean(d * d))) if d.size else 0.0
    ref_rms = float(np.sqrt(np.mean(ref * ref))) if ref.size else 0.0
    rel_l2 = err / max(ref_rms, REL_L2_RMS_FLOOR)
    snr_db = math.inf if err == 0.0 else 20.0 * math.log10(max(ref_rms, REL_L2_RMS_FLOOR) / err)

    n_frames = math.ceil(ref.size / frame_len)
    frames_over: Tuple[int, ...] = ()
    if frame_atol is not None and not identical:
        padded = np.zeros(n_frames * frame_len)
        padded[: d.size] = np.abs(d)
        per_frame = padded.reshape(n_frames, frame_len).max(axis=1)
        frames_over = tuple(int(i) for i in np.flatnonzero(per_frame > frame_atol))

    return WaveformDiff(
        ref_len=ref.size, cand_len=cand.size, ref_rms=ref_rms, err_rms=err, finite=finite, identical=identical,
        max_abs=max_abs, rel_l2=rel_l2, snr_db=snr_db,
        frame_len=frame_len, n_frames=n_frames, frames_over=frames_over,
    )


def masked_diff(ref, cand, sampling_rate: int, frames) -> WaveformDiff:
    """:func:`waveform_diff` with ``frames`` zeroed in both signals.

    Used for quantised codecs: after the explained flip frames are removed,
    what remains must meet the NUMERIC bounds.
    """
    ref, cand = _as_signal(ref).copy(), _as_signal(cand).copy()
    if ref.shape == cand.shape:
        frame_len = max(1, int(round(FRAME_SECONDS * sampling_rate)))
        for i in frames:
            ref[i * frame_len:(i + 1) * frame_len] = 0.0
            cand[i * frame_len:(i + 1) * frame_len] = 0.0
    return waveform_diff(ref, cand, sampling_rate)


def bit_flips(ref_bits, cand_bits) -> Optional[int]:
    """Count differing bits, or None when the two cannot be compared bit for bit."""
    if ref_bits is None or cand_bits is None:
        return None
    a = np.rint(np.asarray(ref_bits, dtype=np.float64)).astype(np.int64).ravel()
    b = np.rint(np.asarray(cand_bits, dtype=np.float64)).astype(np.int64).ravel()
    if a.shape != b.shape:
        return None
    return int(np.sum(a != b))
