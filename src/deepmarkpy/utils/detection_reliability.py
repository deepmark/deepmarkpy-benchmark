"""Detection reliability — false positive / false negative measurements.

Lives in its own module so the main benchmark loop stays focused on
accuracy. The flow is:

  Without attacks (always, in detection_reliability mode):
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
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf

from deepmarkpy.benchmark import (
    apply_attack,
    audio_filename_label,
    expand_attacks,
    instantiate_attack,
    require_attacks_available,
    resolve_cross_model_name,
    _BENCHMARK_INTERNAL_KEYS,
)
from deepmarkpy.utils.metrics import compute_metrics
from deepmarkpy.utils.metric_resolver import (
    EFFICIENCY_METRICS,
    PER_FILE_EFFICIENCY_METRICS,
    MetricResolver,
    compute_statistics,
)
from deepmarkpy.utils import efficiency
from deepmarkpy.utils.utils import load_audio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class DetectionReliabilityResult(dict):
    """The dict ``run_detection_reliability`` returns.

    Keys: ``model_name``, ``is_zero_bit``, ``detection_threshold``,
    ``n_files``, ``no_attack``, ``attacks`` and ``per_file``.
    """


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------

def _detect(model_instance, audio: np.ndarray, sampling_rate: int,
            record=None) -> bool:
    """Run detect() and return the model's is_watermarked() decision.

    Times detect() into ``record`` when given; callers pass it only for
    watermarked audio, the call the other modes time.
    """
    if record is None:
        detect_output = model_instance.detect(audio, sampling_rate)
    else:
        with efficiency.measure(record, "detect_latency"):
            detect_output = model_instance.detect(audio, sampling_rate)
    return bool(model_instance.is_watermarked(detect_output))


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
    metric_resolver: Optional[MetricResolver] = None,
    attack_parameters=None,
    extra_attack_versions=None,
    **attack_kwargs,
) -> DetectionReliabilityResult:
    """Run the detection-reliability pass on ``filepaths``.

    Args:
        benchmark: ``Benchmark`` instance (already plugin-loaded).
        filepaths: list of audio file paths.
        wm_model: name of a zero-bit or confidence-based watermarking model.
        attack_types: optional list of attack class names to evaluate.
            When non-empty, FP/FN are also reported per attack, with the
            quality metrics that attack's group enables.
        sampling_rate: defaults to the model config's sampling rate.
        verbose: extra per-file logging.
        calculate_quality_metrics: passed through to the resolver when one
            is not supplied; see ``MetricResolver``.
        metric_resolver: decides which metrics each attack's group gets and
            which statistics they are reduced to. Defaults to the built-in
            matrix declared by ``ATTACK_GROUPS``.
        attack_parameters: ``(attack, version) -> params``, applied per
            expanded entry; see ``expand_attacks``.
        extra_attack_versions: versions the config defines that the plugin
            does not; see ``expand_attacks``.
        **attack_kwargs: extra keyword arguments forwarded to every
            attack's ``apply()``; per-attack overrides come through
            ``attack_parameters``.

    Returns:
        ``DetectionReliabilityResult`` with no-attack and per-attack
        FP/FN counts and the configured statistics of each metric.

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

    from deepmarkpy.core.base_model import implements_is_watermarked
    if not implements_is_watermarked(model_instance):
        raise ValueError(
            f"Model '{wm_model}' does not implement is_watermarked(). "
            f"Cannot use detection_reliability mode with this model."
        )

    if sampling_rate is None:
        sampling_rate = model_config["sampling_rate"]
        logger.info(
            f"Using default sampling rate {sampling_rate} for model {wm_model}"
        )

    all_attack_config_keys = set()
    for atk_entry in benchmark.attacks.values():
        if atk_entry.get("config"):
            all_attack_config_keys.update(atk_entry["config"].keys())
    attack_kwargs = {
        k: v for k, v in attack_kwargs.items()
        if k in all_attack_config_keys or k in _BENCHMARK_INTERNAL_KEYS
    }

    attack_types = list(attack_types or [])
    # expand_attacks passes unknown names straight through, and the per-file
    # loop below indexes benchmark.attacks with them, so an unavailable attack
    # would surface as a KeyError partway through the run.
    require_attacks_available(
        attack_types, benchmark.attacks,
        getattr(getattr(benchmark, "plugin_manager", None), "failed", None),
    )
    # Same expansion (and therefore the same row labels) as benchmark.run.
    expanded_attacks = expand_attacks(
        attack_types, benchmark.attacks,
        parameters=attack_parameters,
        extra_versions=extra_attack_versions,
    )
    # CrossModelAttack's second model, resolved once before any audio as
    # benchmark.run does: the attack has no config.json fallback for it.
    expanded_attacks = [
        (cls, name, {**overrides, "different_model_name_cross_model":
                     resolve_cross_model_name(
                         {**attack_kwargs, **overrides},
                         benchmark.attacks, benchmark.models,
                     )}, version)
        if cls == "CrossModelAttack" else (cls, name, overrides, version)
        for cls, name, overrides, version in expanded_attacks
    ]
    n_files = len(filepaths)

    if save_audio and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-file accumulators
    # ------------------------------------------------------------------
    fp_no_attack = 0
    fn_no_attack = 0
    resolver = metric_resolver or MetricResolver.from_attack_groups(
        calculate_quality_metrics=calculate_quality_metrics,
    )
    # The no-attack row is read against every group's tables, so it carries
    # whatever any group asks for -- not just what metrics.defaults enables,
    # which a config that states its metrics per group leaves empty.
    baseline_metrics = resolver.all_signal_metrics()
    # Only the timings the config enables are measured.
    timed = {m: resolver.is_enabled(None, m) for m in EFFICIENCY_METRICS}
    no_attack_metrics: Dict[str, List[float]] = {
        m: [] for m in baseline_metrics
    }

    def _metrics_for(attack_name):
        """Metrics this attack's group asked for, per the config file."""
        return resolver.metrics_for_attack(attack_name)

    attack_state: Dict[str, Dict[str, Any]] = {
        a: {
            "accuracy": [],
            "metrics": {m: [] for m in _metrics_for(a)},
            "timings": {},
            "fp_count": 0,
            "fp_attempts": 0,
            "fn_count": 0,
            "fn_attempts": 0,
        }
        for _, a, _, _ in expanded_attacks
    }

    per_file_records: Dict[str, Dict[str, Any]] = {}

    for filepath in filepaths:
        if verbose:
            logger.info(f"Processing file: {filepath}")

        audio, sr = load_audio(filepath, target_sr=sampling_rate)

        file_record: Dict[str, Any] = {"attacks": {}}

        # --- Step 1: FP without attack (detect on clean audio) ---
        fp_this = _detect(model_instance, audio, sr)
        if fp_this:
            fp_no_attack += 1
        file_record["no_attack_fp"] = fp_this

        # --- Step 2: embed + detect (FN without attack) ---
        watermark = model_instance.generate_watermark()
        file_timings: Dict[str, float] = {}
        with efficiency.measure(file_timings, "embed_latency",
                                timed["embed_latency"]):
            watermarked_audio = model_instance.embed(
                audio=audio, watermark_data=watermark, sampling_rate=sr,
            )
        fn_this = not _detect(
            model_instance, watermarked_audio, sr,
            record=file_timings if timed["detect_latency"] else None,
        )
        if fn_this:
            fn_no_attack += 1
        file_record["no_attack_fn"] = fn_this
        file_record.update(file_timings)

        if save_audio and output_dir:
            base = os.path.splitext(os.path.basename(filepath))[0]
            sf.write(
                os.path.join(output_dir, f"{base}_watermarked.wav"),
                watermarked_audio, sr,
            )

        if baseline_metrics:
            quality = compute_metrics(
                audio, watermarked_audio, sr,
                metrics=set(baseline_metrics),
            )
            for m in baseline_metrics:
                v = quality.get(m)
                if v is not None:
                    no_attack_metrics[m].append(v)
            file_record["no_attack_metrics"] = quality

        # --- Steps 3 + 4: per-attack FP and FN ---
        for attack_class_name, attack_name, attack_overrides, attack_version in expanded_attacks:
            attack_instance = instantiate_attack(
                benchmark.attacks[attack_class_name]["class"],
                attack_class_name, attack_version,
            )

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

            label = audio_filename_label(attack_name)
            if save_audio and output_dir:
                base = os.path.splitext(os.path.basename(filepath))[0]
                sf.write(
                    os.path.join(output_dir, f"{base}_{label}_clean.wav"),
                    attacked_clean, sr,
                )

            # Step 4: attack the watermarked audio, then detect.
            attack_timings: Dict[str, float] = {}
            try:
                with efficiency.measure(attack_timings, "attack_latency",
                                        timed["attack_latency"]):
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
                    os.path.join(output_dir, f"{base}_{label}.wav"),
                    attacked_wm, sr,
                )

            # Both attacks succeeded for this file, so both rates are counted
            # over the same file set.
            attack_state[attack_name]["fp_attempts"] += 1
            fp_detected = _detect(model_instance, attacked_clean, sr)
            if fp_detected:
                attack_state[attack_name]["fp_count"] += 1

            attack_state[attack_name]["fn_attempts"] += 1
            wm_detected = _detect(
                model_instance, attacked_wm, sr,
                record=attack_timings if timed["detect_latency"] else None,
            )
            if not wm_detected:
                attack_state[attack_name]["fn_count"] += 1

            # After the detect above, which fills detect_latency. Only the
            # per-file metrics come from file_timings: its detect_latency
            # times the un-attacked signal, the baseline's, not this attack's.
            for m, value in attack_timings.items():
                attack_state[attack_name]["timings"].setdefault(m, []).append(value)
            for m in PER_FILE_EFFICIENCY_METRICS:
                if m in file_timings:
                    attack_state[attack_name]["timings"].setdefault(
                        m, []).append(file_timings[m])

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

            file_record["attacks"][attack_name] = {
                "fp": fp_detected,
                "fn": not wm_detected,
                "accuracy": 100.0 if wm_detected else 0.0,
                "metrics": quality,
                # Embedding happens once per file; carried here so the
                # per-attack aggregate can reach it. Only the per-file keys:
                # file_timings' detect_latency belongs to the baseline.
                **{m: file_timings[m] for m in PER_FILE_EFFICIENCY_METRICS
                   if m in file_timings},
                **attack_timings,
            }

        per_file_records[filepath] = file_record

    # ------------------------------------------------------------------
    # Aggregate: each value reduced to the statistics its group configures
    # ------------------------------------------------------------------
    attacks_summary: Dict[str, Dict[str, Any]] = {}
    for attack_name, state in attack_state.items():
        group_key = resolver.group_for_attack(attack_name)
        accuracies = state["accuracy"]

        accuracy_stats = compute_statistics(
            accuracies, resolver.statistics_for(group_key, "accuracy"),
        ) or {}
        entry = {
            f"accuracy_{name}": value
            for name, value in accuracy_stats.items()
        }
        entry["accuracy_n"] = len(accuracies)

        if resolver.is_enabled(group_key, "emr"):
            exact = sum(1 for a in accuracies if a == 100.0)
            entry["emr_count"] = exact
            entry["emr_rate"] = (
                float(exact / len(accuracies)) if accuracies else 0.0
            )

        entry["metrics"] = {
            m: compute_statistics(
                vals, resolver.statistics_for(group_key, m), m,
            )
            for m, vals in state["metrics"].items()
        }
        # Timings are measured rather than computed from two signals, so
        # they are kept apart from the quality metrics all the way through.
        entry["timings"] = {
            m: compute_statistics(
                vals, resolver.statistics_for(group_key, m), m,
            )
            for m, vals in state.get("timings", {}).items()
            if resolver.is_enabled(group_key, m)
        }
        entry.update(
            false_positive_count=state["fp_count"],
            false_positive_attempts=state["fp_attempts"],
            false_negative_count=state["fn_count"],
            false_negative_attempts=state["fn_attempts"],
        )
        attacks_summary[attack_name] = entry

    baseline_timings: Dict[str, list] = {}
    for record in per_file_records.values():
        for m in EFFICIENCY_METRICS:
            if record.get(m) is not None and resolver.is_enabled(None, m):
                baseline_timings.setdefault(m, []).append(record[m])

    no_attack_result = {
        "false_positive_count": fp_no_attack,
        "false_negative_count": fn_no_attack,
        "emr_count": n_files - fn_no_attack,
        "emr_rate": float((n_files - fn_no_attack) / n_files) if n_files else 0.0,
    }
    if baseline_metrics:
        no_attack_result["metrics"] = {
            m: compute_statistics(vals, resolver.statistics_for(None, m), m)
            for m, vals in no_attack_metrics.items()
        }
    if baseline_timings:
        no_attack_result["timings"] = {
            m: compute_statistics(vals, resolver.statistics_for(None, m), m)
            for m, vals in baseline_timings.items()
        }

    return DetectionReliabilityResult(
        model_name=wm_model,
        is_zero_bit=is_zero_bit,
        detection_threshold=detection_threshold,
        n_files=n_files,
        no_attack=no_attack_result,
        attacks=attacks_summary,
        per_file=per_file_records,
    )


