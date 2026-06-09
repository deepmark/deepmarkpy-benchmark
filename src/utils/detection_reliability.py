"""Detection reliability — false positive / false negative measurements.

Lives in its own module so the main benchmark loop stays focused on
accuracy. The flow is:

  Without attacks (always when --detection_reliability is set):
    1. detect() on the clean audio                  -> false_positive_no_attack
    2. embed() then detect() on watermarked         -> false_negative_no_attack

  With attacks (only when at least one attack is provided):
    3. attack() on the clean audio, then detect()   -> false_positive_with_attack
    4. attack() on the watermarked audio, then detect() -> false_negative_with_attack

Restricted to zero-bit models for now: those return a binary 0/1 from
detect() so the FP/FN bookkeeping is unambiguous. Multi-bit models
require a threshold on bit-accuracy that we don't want to bake in here.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from utils.metrics import ALL_METRICS, compute_metrics
from utils.utils import load_audio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class DetectionReliabilityResult(dict):
    """Plain dict subclass; documents the expected shape.

    Structure::

        {
            "model_name": str,
            "is_zero_bit": True,
            "n_files": int,
            "no_attack": {
                "false_positive_count": int,    # detected on clean audio
                "false_negative_count": int,    # missed on watermarked audio
            },
            "attacks": {
                attack_name: {
                    "accuracy_mean": float,
                    "always_on_metrics": {"pesq": float, "visqol": float, "stoi": float},
                    "false_positive_count": int,
                    "false_negative_count": int,
                },
                ...
            },
        }
    """


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_is_positive(detect_output: Any) -> bool:
    """Return True when a zero-bit detector says 'watermark detected'.

    Perth returns 0 or 1 (with ``round=True``) and StariVigil returns the
    same shape; both collapse to a scalar after ``.item()`` / ``tolist()``.
    Accept either int or numpy array; treat any non-zero value as positive.
    """
    if isinstance(detect_output, np.ndarray):
        detect_output = detect_output.tolist()
    if isinstance(detect_output, list):
        # Some zero-bit models may return a single-element list.
        return bool(detect_output[0]) if detect_output else False
    return bool(detect_output)


def _detect(model_instance, audio: np.ndarray, sampling_rate: int,
            returns_confidence: bool) -> bool:
    """Run detect() and reduce to a positive/negative boolean."""
    if returns_confidence:
        detected, _confidence = model_instance.detect(audio, sampling_rate)
    else:
        detected = model_instance.detect(audio, sampling_rate)
    return _detect_is_positive(detected)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_detection_reliability(
    benchmark,
    filepaths: List[str],
    wm_model: str,
    attack_types: Optional[Iterable[str]] = None,
    sampling_rate: Optional[int] = None,
    verbose: bool = False,
    **attack_kwargs,
) -> DetectionReliabilityResult:
    """Run the detection-reliability pass on ``filepaths``.

    Args:
        benchmark: ``Benchmark`` instance (already plugin-loaded).
        filepaths: list of audio file paths.
        wm_model: name of a zero-bit watermarking model.
        attack_types: optional list of attack class names to evaluate.
            When non-empty, FP/FN are also reported per attack and the
            three always-on metrics (PESQ, ViSQOL, STOI) are recorded
            for the per-attack table.
        sampling_rate: defaults to the model config's sampling rate.
        verbose: extra per-file logging.
        **attack_kwargs: forwarded to attack ``apply()`` calls (CLI
            overrides for attack-specific parameters).

    Returns:
        ``DetectionReliabilityResult`` with no-attack and per-attack
        FP/FN counts plus accuracy / always-on metric means.

    Raises:
        ValueError: if ``wm_model`` isn't a zero-bit model.
    """
    if wm_model not in benchmark.models:
        raise ValueError(
            f"Model '{wm_model}' not found. "
            f"Available: {list(benchmark.models.keys())}"
        )

    model_config = benchmark.models[wm_model]["config"] or {}
    is_zero_bit = model_config.get("is_zero_bit", False)
    if not is_zero_bit:
        raise ValueError(
            "--detection_reliability is currently supported only for "
            f"zero-bit models. '{wm_model}' is not zero-bit."
        )
    returns_confidence = model_config.get("returns_confidence", False)

    if sampling_rate is None:
        sampling_rate = model_config["sampling_rate"]
        logger.info(
            f"Using default sampling rate {sampling_rate} for model {wm_model}"
        )

    model_cls = benchmark.models[wm_model]["class"]
    model_instance = model_cls()

    attack_types = list(attack_types or [])
    n_files = len(filepaths)

    # ------------------------------------------------------------------
    # Per-file accumulators
    # ------------------------------------------------------------------
    fp_no_attack = 0
    fn_no_attack = 0

    # Per-attack accumulators
    attack_state: Dict[str, Dict[str, Any]] = {
        a: {
            "accuracy": [],
            "metrics": {m: [] for m in benchmark.ALWAYS_ON_METRICS},
            "fp_count": 0,
            "fn_count": 0,
        }
        for a in attack_types
    }

    for filepath in filepaths:
        if verbose:
            logger.info(f"Processing file: {filepath}")

        audio, sr = load_audio(filepath, target_sr=sampling_rate)

        # --- Step 1: FP without attack (detect on clean audio) ---
        if _detect(model_instance, audio, sr, returns_confidence):
            fp_no_attack += 1

        # --- Step 2: embed + detect (FN without attack) ---
        watermark = model_instance.generate_watermark()
        watermarked_audio = model_instance.embed(
            audio=audio, watermark_data=watermark, sampling_rate=sr,
        )
        if not _detect(model_instance, watermarked_audio, sr, returns_confidence):
            fn_no_attack += 1

        # --- Steps 3 + 4: per-attack FP and FN ---
        for attack_name in attack_types:
            attack_instance = benchmark.attacks[attack_name]["class"]()

            kw_for_attack = {
                **attack_kwargs,
                "model": model_instance,
                "watermark_data": watermark,
                "sampling_rate": sr,
                "models": benchmark.models,
                "orig_audio": audio,
            }

            # Step 3: attack the clean audio, then detect.
            try:
                attacked_clean = _apply_attack(
                    attack_instance, attack_name, audio, kw_for_attack,
                )
            except Exception as e:  # pragma: no cover -- log and skip file
                logger.warning(
                    f"Attack {attack_name} on clean audio failed for "
                    f"{filepath}: {e}. Skipping this attack for this file."
                )
                continue

            if _detect(model_instance, attacked_clean, sr, returns_confidence):
                attack_state[attack_name]["fp_count"] += 1

            # Step 4: attack the watermarked audio, then detect.
            try:
                attacked_wm = _apply_attack(
                    attack_instance, attack_name, watermarked_audio, kw_for_attack,
                )
            except Exception as e:
                logger.warning(
                    f"Attack {attack_name} on watermarked audio failed for "
                    f"{filepath}: {e}. Skipping this attack for this file."
                )
                continue

            wm_detected = _detect(
                model_instance, attacked_wm, sr, returns_confidence,
            )
            if not wm_detected:
                attack_state[attack_name]["fn_count"] += 1

            # Per-attack accuracy mirrors what the basic report shows
            # for every other attack: per-file 100/0 based on whether
            # the watermark survived, averaged across files.
            attack_state[attack_name]["accuracy"].append(
                100.0 if wm_detected else 0.0
            )

            quality = compute_metrics(
                audio, attacked_wm, sr,
                metrics=set(benchmark.ALWAYS_ON_METRICS),
            )
            for m in benchmark.ALWAYS_ON_METRICS:
                v = quality.get(m)
                if v is not None:
                    attack_state[attack_name]["metrics"][m].append(v)

    # ------------------------------------------------------------------
    # Aggregate per-attack means
    # ------------------------------------------------------------------
    attacks_summary: Dict[str, Dict[str, Any]] = {}
    for attack_name, state in attack_state.items():
        accuracies = state["accuracy"]
        attacks_summary[attack_name] = {
            "accuracy_mean": (
                float(np.mean(accuracies)) if accuracies else None
            ),
            "always_on_metrics": {
                m: (float(np.mean(vals)) if vals else None)
                for m, vals in state["metrics"].items()
            },
            "false_positive_count": state["fp_count"],
            "false_negative_count": state["fn_count"],
        }

    return DetectionReliabilityResult(
        model_name=wm_model,
        is_zero_bit=True,
        n_files=n_files,
        no_attack={
            "false_positive_count": fp_no_attack,
            "false_negative_count": fn_no_attack,
        },
        attacks=attacks_summary,
    )


# ---------------------------------------------------------------------------
# Attack dispatch (mirrors the special cases in benchmark.run)
# ---------------------------------------------------------------------------

def _apply_attack(attack_instance, attack_name, audio, attack_kwargs):
    """Apply ``attack_instance`` to ``audio`` honoring the model's quirks.

    Mirrors the dispatch in ``benchmark.run`` for the attacks that
    return tuples (``CrossModelAttack``) or need extra kwargs
    (``ZeroBitCollusionAttack``). Anything else takes the default path.
    """
    if attack_name == "CrossModelAttack":
        attacked_audio, _diff_watermark = attack_instance.apply(
            audio, **attack_kwargs,
        )
    elif attack_name == "ZeroBitCollusionAttack":
        kwargs = {**attack_kwargs, "original_audio_collusion": audio}
        attacked_audio = attack_instance.apply(audio, **kwargs)
    else:
        attacked_audio = attack_instance.apply(audio, **attack_kwargs)

    if isinstance(attacked_audio, np.ndarray):
        attacked_audio = np.squeeze(attacked_audio)
    return attacked_audio
