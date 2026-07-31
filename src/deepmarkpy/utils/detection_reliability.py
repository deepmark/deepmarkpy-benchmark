"""Detection reliability — false positive / false negative measurements.

Lives in its own module so the main benchmark loop stays focused on
accuracy. The flow is:

  Without attacks (always when --detection_reliability is set):
    1. detect() on the clean audio                  -> false_positive_no_attack
    2. embed() then detect() on watermarked         -> false_negative_no_attack

  With attacks (only when at least one attack is provided):
    3. attack() on the clean audio, then detect()   -> false_positive_with_attack
    4. attack() on the watermarked audio, then detect() -> false_negative_with_attack

Each model that supports this mode must implement ``is_watermarked()``
which takes the raw output of ``detect()`` and returns a boolean
indicating whether a watermark is present.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf

from deepmarkpy.benchmark import apply_attack, expand_attacks
from deepmarkpy.utils.metrics import ALL_METRICS, compute_metrics
from deepmarkpy.utils.attack_groups import get_metrics_for_attack
from deepmarkpy.utils.utils import load_audio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class DetectionReliabilityResult(dict):
    """Plain dict subclass; documents the expected shape.

    Structure::

        {
            "model_name": str,
            "is_zero_bit": bool,
            "detection_threshold": float | None,
            "n_files": int,
            "no_attack": {
                "false_positive_count": int,
                "false_negative_count": int,
                "metrics": {...} | absent,
            },
            "attacks": {
                attack_name: {
                    "accuracy_mean": float | None,
                    "metrics": {metric_name: float | None, ...},
                    "false_positive_count": int,
                    "false_positive_attempts": int,
                    "false_negative_count": int,
                    "false_negative_attempts": int,
                },
                ...
            },
        }
    """


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------

def _detect(model_instance, audio: np.ndarray, sampling_rate: int) -> bool:
    """Run detect() and delegate the decision to the model's is_watermarked()."""
    detect_output = model_instance.detect(audio, sampling_rate)
    return model_instance.is_watermarked(detect_output)


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
    calculate_quality_metrics: bool = False,
    save_audio: bool = False,
    output_dir: Optional[str] = None,
    **attack_kwargs,
) -> DetectionReliabilityResult:
    """Run the detection-reliability pass on ``filepaths``.

    Args:
        benchmark: ``Benchmark`` instance (already plugin-loaded).
        filepaths: list of audio file paths.
        wm_model: name of a zero-bit or confidence-based watermarking model.
        attack_types: optional list of attack class names to evaluate.
            When non-empty, FP/FN are also reported per attack with
            group-specific quality metrics (via ``get_metrics_for_attack``).
        sampling_rate: defaults to the model config's sampling rate.
        verbose: extra per-file logging.
        calculate_quality_metrics: when True, computes ALL_METRICS for the
            no-attack case (original vs watermarked).
        **attack_kwargs: forwarded to attack ``apply()`` calls (CLI
            overrides for attack-specific parameters).

    Returns:
        ``DetectionReliabilityResult`` with no-attack and per-attack
        FP/FN counts plus accuracy and quality metric means.

    Raises:
        ValueError: if ``wm_model`` does not implement ``is_watermarked()``.
    """
    if wm_model not in benchmark.models:
        raise ValueError(
            f"Model '{wm_model}' not found. "
            f"Available: {list(benchmark.models.keys())}"
        )

    model_config = benchmark.models[wm_model]["config"] or {}
    is_zero_bit = model_config.get("is_zero_bit", False)
    detection_threshold = model_config.get("detection_threshold", None)

    model_cls = benchmark.models[wm_model]["class"]
    model_instance = model_cls()

    if not hasattr(model_instance, "is_watermarked"):
        raise ValueError(
            f"Model '{wm_model}' does not implement is_watermarked(). "
            f"Cannot use --detection_reliability with this model."
        )
    from deepmarkpy.core.base_model import BaseModel
    if isinstance(model_instance, BaseModel) and type(model_instance).is_watermarked is BaseModel.is_watermarked:
        raise ValueError(
            f"Model '{wm_model}' does not implement is_watermarked(). "
            f"Cannot use --detection_reliability with this model."
        )

    if sampling_rate is None:
        sampling_rate = model_config["sampling_rate"]
        logger.info(
            f"Using default sampling rate {sampling_rate} for model {wm_model}"
        )

    attack_types = list(attack_types or [])
    # Same expansion (and therefore the same row labels) as benchmark.run.
    expanded_attacks = expand_attacks(attack_types, benchmark.attacks)
    n_files = len(filepaths)

    if save_audio and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-file accumulators
    # ------------------------------------------------------------------
    fp_no_attack = 0
    fn_no_attack = 0
    no_attack_metrics: Dict[str, List[float]] = {
        m: [] for m in ALL_METRICS
    }

    # Per-attack accumulators — full group metrics when calculate_quality_metrics,
    # otherwise just always-on (pesq, visqol, stoi).
    ALWAYS_ON = ["pesq", "visqol", "stoi"]

    def _metrics_for(attack_name):
        if calculate_quality_metrics:
            return get_metrics_for_attack(attack_name)
        return ALWAYS_ON

    attack_state: Dict[str, Dict[str, Any]] = {
        a: {
            "accuracy": [],
            "metrics": {m: [] for m in _metrics_for(a)},
            "fp_count": 0,
            "fp_attempts": 0,
            "fn_count": 0,
            "fn_attempts": 0,
        }
        for _, a, _ in expanded_attacks
    }

    for filepath in filepaths:
        if verbose:
            logger.info(f"Processing file: {filepath}")

        audio, sr = load_audio(filepath, target_sr=sampling_rate)

        # --- Step 1: FP without attack (detect on clean audio) ---
        if _detect(model_instance, audio, sr):
            fp_no_attack += 1

        # --- Step 2: embed + detect (FN without attack) ---
        watermark = model_instance.generate_watermark()
        watermarked_audio = model_instance.embed(
            audio=audio, watermark_data=watermark, sampling_rate=sr,
        )
        if not _detect(model_instance, watermarked_audio, sr):
            fn_no_attack += 1

        if save_audio and output_dir:
            base = os.path.splitext(os.path.basename(filepath))[0]
            sf.write(
                os.path.join(output_dir, f"{base}_watermarked.wav"),
                watermarked_audio, sr,
            )

        if calculate_quality_metrics:
            quality = compute_metrics(
                audio, watermarked_audio, sr,
                metrics=set(ALL_METRICS),
            )
            for m in ALL_METRICS:
                v = quality.get(m)
                if v is not None:
                    no_attack_metrics[m].append(v)

        # --- Steps 3 + 4: per-attack FP and FN ---
        for attack_class_name, attack_name, attack_overrides in expanded_attacks:
            attack_instance = benchmark.attacks[attack_class_name]["class"]()

            kw_for_attack = {
                **attack_kwargs,
                **attack_overrides,
                "model": model_instance,
                "watermark_data": watermark,
                "sampling_rate": sr,
                "models": benchmark.models,
                "orig_audio": audio,
            }

            # Step 3: attack the clean audio, then detect.
            try:
                attacked_clean, _ = apply_attack(
                    attack_instance, attack_class_name,
                    target_audio=audio, clean_audio=audio,
                    attack_kwargs=kw_for_attack,
                )
            except Exception as e:  # pragma: no cover -- log and skip file
                logger.warning(
                    f"Attack {attack_name} on clean audio failed for "
                    f"{filepath}: {e}. Skipping this attack for this file."
                )
                continue

            if save_audio and output_dir:
                base = os.path.splitext(os.path.basename(filepath))[0]
                sf.write(
                    os.path.join(output_dir, f"{base}_{attack_name}_clean.wav"),
                    attacked_clean, sr,
                )

            # Step 4: attack the watermarked audio, then detect.
            try:
                attacked_wm, _ = apply_attack(
                    attack_instance, attack_class_name,
                    target_audio=watermarked_audio, clean_audio=audio,
                    attack_kwargs=kw_for_attack,
                )
            except Exception as e:
                logger.warning(
                    f"Attack {attack_name} on watermarked audio failed for "
                    f"{filepath}: {e}. Skipping this attack for this file."
                )
                continue

            if save_audio and output_dir:
                base = os.path.splitext(os.path.basename(filepath))[0]
                sf.write(
                    os.path.join(output_dir, f"{base}_{attack_name}.wav"),
                    attacked_wm, sr,
                )

            # Both attacks succeeded for this file, so both rates are counted
            # over the same file set.
            attack_state[attack_name]["fp_attempts"] += 1
            if _detect(model_instance, attacked_clean, sr):
                attack_state[attack_name]["fp_count"] += 1

            attack_state[attack_name]["fn_attempts"] += 1
            wm_detected = _detect(model_instance, attacked_wm, sr)
            if not wm_detected:
                attack_state[attack_name]["fn_count"] += 1

            # Per-attack accuracy mirrors what the basic report shows
            # for every other attack: per-file 100/0 based on whether
            # the watermark survived, averaged across files.
            attack_state[attack_name]["accuracy"].append(
                100.0 if wm_detected else 0.0
            )

            attack_metrics = _metrics_for(attack_name)
            quality = compute_metrics(
                audio, attacked_wm, sr,
                metrics=set(attack_metrics),
            )
            for m in attack_metrics:
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
            "metrics": {
                m: (float(np.mean(vals)) if vals else None)
                for m, vals in state["metrics"].items()
            },
            "false_positive_count": state["fp_count"],
            "false_positive_attempts": state["fp_attempts"],
            "false_negative_count": state["fn_count"],
            "false_negative_attempts": state["fn_attempts"],
        }

    no_attack_result = {
        "false_positive_count": fp_no_attack,
        "false_negative_count": fn_no_attack,
    }
    if calculate_quality_metrics:
        no_attack_result["metrics"] = {
            m: (float(np.mean(vals)) if vals else None)
            for m, vals in no_attack_metrics.items()
        }

    return DetectionReliabilityResult(
        model_name=wm_model,
        is_zero_bit=is_zero_bit,
        detection_threshold=detection_threshold,
        n_files=n_files,
        no_attack=no_attack_result,
        attacks=attacks_summary,
    )


