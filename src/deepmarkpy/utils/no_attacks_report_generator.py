"""Report generator for the ``no_attacks`` mode.

Whether each model reads its own watermark back with nothing in between,
and what embedding costs the audio: a detection table per model family
(zero-bit detection rates and multi-bit bit agreement are never mixed),
with a "Detected" count for models that implement ``is_watermarked()``,
then the configured quality metrics. Every metric and statistic comes from
``metrics.defaults``.
"""

import logging
import os

import numpy as np

from deepmarkpy.utils.latex_helpers import (
    compact_header,
    compile_latex,
    container_section,
    duration_label_tex,
    efficiency_tables,
    format_emr_cell,
    format_metric_cell,
    grid_table,
    make_preamble,
    part_heading,
    slugify,
    metric_label,
    stat_header,
)
from deepmarkpy.utils.metric_resolver import (
    INTELLIGIBILITY_METRICS,
    PER_MODEL_EFFICIENCY_METRICS,
    MetricResolver,
    NISQA_METRICS,
    QUALITY_METRICS,
    compute_statistics,
)

logger = logging.getLogger(__name__)

# Metric families, each tabled separately.
_METRIC_SECTIONS = (
    ("quality", "Audio quality of the watermarked signal (no attack).",
     QUALITY_METRICS),
    ("intelligibility",
     "Speech intelligibility of the watermarked signal (no attack).",
     INTELLIGIBILITY_METRICS),
    ("nisqa",
     "NISQA non-intrusive quality dimensions of the watermarked signal "
     "(no attack).", NISQA_METRICS),
)


def _compute_stats(values, statistics=None, metric="accuracy"):
    """``compute_statistics`` plus the sample count ``n``, or None without values."""
    stats = compute_statistics(values, statistics, metric)
    if stats is not None:
        stats["n"] = len(values)
    return stats


def _summarize_model(results, resolver):
    """Compute summary stats for a single model's no-attacks results."""
    is_zero_bit = results.get("is_zero_bit", False)
    returns_confidence = results.get("returns_confidence", False)
    files = results.get("files", [])

    accuracies = [f["accuracy"] for f in files if f.get("accuracy") is not None]
    accuracy_stats = _compute_stats(
        accuracies, resolver.statistics_for(None, "accuracy"),
    )

    summary = {
        "is_zero_bit": is_zero_bit,
        "returns_confidence": returns_confidence,
        "n_files": len(files),
        "accuracy_stats": accuracy_stats,
    }

    # The model's own is_watermarked() answer, recorded per file by
    # run_no_attacks; without it there is no count.
    if results.get("supports_detection"):
        # Over the files it decided: one whose is_watermarked() raised has
        # no answer.
        decided = [f for f in files if "detected" in f]
        summary["supports_detection"] = True
        summary["positive_detections"] = sum(1 for f in decided if f["detected"])
        summary["detection_n"] = len(decided)

    if resolver.is_enabled(None, "emr") and accuracies:
        exact = sum(1 for a in accuracies if a == 100.0)
        summary["emr_count"] = exact
        summary["emr_rate"] = exact / len(accuracies)

    if resolver.is_enabled(None, "ber") and accuracies:
        summary["ber_stats"] = _compute_stats(
            [1.0 - a / 100.0 for a in accuracies],
            resolver.statistics_for(None, "ber"), "ber",
        )

    if returns_confidence:
        confidences = [
            f["confidence"] for f in files if f.get("confidence") is not None
        ]
        if confidences:
            summary["mean_confidence"] = float(np.mean(confidences))

    # Timings sit on the file entry, not in its quality dict.
    timing_stats = {}
    for metric in resolver.metrics_for_group(None, bucket="efficiency"):
        values = [f[metric] for f in files if f.get(metric) is not None]
        if values:
            timing_stats[metric] = _compute_stats(
                values, resolver.statistics_for(None, metric), metric,
            )
    if timing_stats:
        summary["efficiency_stats"] = timing_stats

    quality_files = [
        f.get("watermarked_audio_quality") for f in files
        if f.get("watermarked_audio_quality")
    ]
    if quality_files:
        quality_stats = {}
        for metric in resolver.signal_metrics_for_group(None):
            values = [
                q.get(metric) for q in quality_files
                if q.get(metric) is not None
            ]
            if values:
                quality_stats[metric] = _compute_stats(
                    values, resolver.statistics_for(None, metric), metric,
                )
        if quality_stats:
            summary["quality_metrics_stats"] = quality_stats

    return summary


def _short_model_name(name):
    """Strip common suffixes for display."""
    for suffix in ("Model", "Watermark"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _accuracy_table(models_data, resolver, is_zero_bit, label):
    """Detection performance for one model family.

    "Detected" appears for the models that answer ``is_watermarked()``,
    whatever their family.
    """
    statistics = resolver.statistics_for(None, "accuracy")
    show_ber = resolver.is_enabled(None, "ber") and not is_zero_bit
    ber_statistics = resolver.statistics_for(None, "ber") if show_ber else []
    show_emr = resolver.is_enabled(None, "emr")
    has_confidence = any("mean_confidence" in d for d in models_data.values())
    show_detected = any(
        d.get("supports_detection") for d in models_data.values()
    )

    headers = [stat_header(s) for s in statistics]
    if show_detected:
        headers.append("Detected")
    if show_ber and len(ber_statistics) == 1:
        headers.append(metric_label("ber"))
    if show_emr:
        headers.append(metric_label("emr"))
    if has_confidence:
        headers.append("Confidence")

    rows = []
    for model_name, data in models_data.items():
        stats = data.get("accuracy_stats") or {}
        cells = [format_metric_cell("accuracy", stats.get(s), "--")
                 for s in statistics]
        if show_detected:
            cells.append(
                f"{data.get('positive_detections', 0)}/"
                f"{data.get('detection_n', data['n_files'])}"
                if data.get("supports_detection") else "N/A"
            )
        if show_ber and len(ber_statistics) == 1:
            ber = (data.get("ber_stats") or {}).get(ber_statistics[0])
            cells.append(format_metric_cell("ber", ber, "--"))
        if show_emr:
            cells.append(format_emr_cell(
                data.get("emr_count"), stats.get("n"), data.get("emr_rate"),
            ))
        if has_confidence:
            confidence = data.get("mean_confidence")
            cells.append(f"{confidence:.4f}" if confidence is not None else "N/A")
        rows.append((_short_model_name(model_name), cells))

    family = "zero-bit" if is_zero_bit else "multi-bit"
    return grid_table(
        "Model", headers, rows,
        f"Baseline detection performance --- {family} models.", label,
    )


def _ber_table(models_data, resolver, label):
    """BER with two or more statistics, one row per model."""
    statistics = resolver.statistics_for(None, "ber")
    rows = [
        (_short_model_name(model_name), [
            format_metric_cell("ber", (data.get("ber_stats") or {}).get(s), "--")
            for s in statistics
        ])
        for model_name, data in models_data.items()
    ]
    return grid_table(
        "Model", [stat_header(s) for s in statistics], rows,
        "Bit error rate of the watermarked signal (no attack).", label,
    )


def _metric_table(models_data, columns, headers, caption, label):
    """One row per model, one column per ``(metric, statistic)``."""
    rows = [
        (_short_model_name(model_name), [
            format_metric_cell(metric, (
                (data.get("quality_metrics_stats") or {}).get(metric) or {}
            ).get(statistic))
            for metric, statistic in columns
        ])
        for model_name, data in models_data.items()
    ]
    return grid_table("Model", headers, rows, caption, label)


def _efficiency_table(summaries, resolver, label):
    """Per-model timing tables; container memory has a section of its own."""
    metrics = [
        m for m in resolver.metrics_for_group(None, bucket="efficiency")
        if m not in PER_MODEL_EFFICIENCY_METRICS
        and any((d.get("efficiency_stats") or {}).get(m)
                for d in summaries.values())
    ]
    return efficiency_tables(
        "Model",
        [(_short_model_name(name), data.get("efficiency_stats") or {})
         for name, data in summaries.items()],
        metrics, lambda metric: resolver.statistics_for(None, metric),
        "per model, with no attack applied", label,
        note=(
            " These depend on the machine and on whether the model runs "
            "natively or in a container, so they do not reproduce across runs "
            "the way the measurements above do."
        ),
    )


def _build_body(summaries, resolver, label_suffix=""):
    """Build the report body for a set of model summaries."""
    suffix = f"_{label_suffix}" if label_suffix else ""
    zero_bit = {m: s for m, s in summaries.items() if s["is_zero_bit"]}
    multi_bit = {m: s for m, s in summaries.items() if not s["is_zero_bit"]}

    tables = []
    if multi_bit:
        tables.append(_accuracy_table(
            multi_bit, resolver, False, f"tab:no_attacks_multibit{suffix}",
        ))
        if resolver.is_enabled(None, "ber") and \
                len(resolver.statistics_for(None, "ber")) > 1:
            tables.append(_ber_table(
                multi_bit, resolver, f"tab:no_attacks_ber{suffix}",
            ))
    if zero_bit:
        tables.append(_accuracy_table(
            zero_bit, resolver, True, f"tab:no_attacks_zerobit{suffix}",
        ))

    quality_tables = []
    silent = []
    for section_key, caption, family in _METRIC_SECTIONS:
        enabled = [
            m for m in resolver.signal_metrics_for_group(None) if m in family
        ]
        if not enabled:
            continue

        with_data = [
            m for m in enabled
            if any((s.get("quality_metrics_stats") or {}).get(m)
                   for s in summaries.values())
        ]
        silent += [m for m in enabled if m not in with_data]

        single = []
        for metric in with_data:
            statistics = resolver.statistics_for(None, metric)
            if len(statistics) > 1:
                # Only the models with a value for this metric get a row.
                present = {
                    name: data for name, data in summaries.items()
                    if (data.get("quality_metrics_stats") or {}).get(metric)
                }
                quality_tables.append(_metric_table(
                    present, [(metric, s) for s in statistics],
                    [stat_header(s) for s in statistics],
                    f"{metric_label(metric)} of the watermarked signal "
                    "(no attack).",
                    f"tab:no_attacks_{metric}{suffix}",
                ))
            else:
                single.append((metric, statistics[0]))
        if single:
            quality_tables.append(_metric_table(
                summaries, single, [compact_header(m, s) for m, s in single],
                caption, f"tab:no_attacks_{section_key}{suffix}",
            ))

    body = "\n\n".join(tables)
    if quality_tables:
        body += (
            "\n\n\\subsection{Watermark Audio Quality}\n\n"
            + "\n\n".join(quality_tables)
        )

    timings = _efficiency_table(
        summaries, resolver, f"tab:no_attacks_efficiency{suffix}",
    )
    if timings:
        body += "\n\n\\subsection{Processing Time}\n\n" + timings
    if silent:
        names = ", ".join(metric_label(m) for m in silent)
        body += (
            "\n\n{\\noindent\\footnotesize Enabled in the configuration but not "
            f"reported, because no value was produced for any model: {names}. "
            "This usually means the metric's service or optional package was "
            "unavailable.}\n"
        )
    return body


def generate_no_attacks_report(all_results, report_dir="report",
                               duration_partitions=None, resolver=None,
                               containers=None):
    """Generate a baseline fidelity report for one or more models.

    Args:
        all_results: Dict of {model_name: results_from_run_no_attacks}.
        report_dir: Directory for output files.
        duration_partitions: Optional list of (label, file_list) tuples.
            When provided, generates per-duration-group sections.
        resolver: metric/statistic configuration. Defaults to the built-in
            matrix.

    Returns:
        Path to the generated .tex file.
    """
    os.makedirs(report_dir, exist_ok=True)
    has_cls = os.path.exists(os.path.join(report_dir, "deepmark.cls"))
    resolver = resolver or MetricResolver.from_attack_groups()

    summaries = {
        model: _summarize_model(res, resolver)
        for model, res in all_results.items()
    }

    n_models = len(summaries)
    model_word = "model" if n_models == 1 else "models"
    n_files = next(iter(summaries.values()))["n_files"] if summaries else 0

    preamble = make_preamble(
        title="Baseline Fidelity Report",
        author="DeepMark Benchmark System",
        has_deepmark_cls=has_cls,
    )

    abstract = (
        f"\\begin{{abstract}}\n"
        f"This report evaluates the baseline detection fidelity of "
        f"{n_models} watermarking {model_word}. The watermark is "
        f"embedded and immediately detected without any intermediate attacks, "
        f"measuring each model's inherent accuracy across {n_files} "
        f"{'file' if n_files == 1 else 'files'}.\n"
        f"\\end{{abstract}}\n\n"
    )

    if duration_partitions:
        sections = []
        for group_label, group_files in duration_partitions:
            group_file_set = set(group_files)
            group_results = {}
            for model_name, model_res in all_results.items():
                files = [
                    f for f in model_res.get("files", [])
                    if f.get("filepath") in group_file_set
                ]
                if files:
                    group_results[model_name] = {**model_res, "files": files}

            if not group_results:
                continue
            group_summaries = {
                model: _summarize_model(res, resolver)
                for model, res in group_results.items()
            }
            safe_label = duration_label_tex(group_label)
            n_group = next(iter(group_summaries.values()))["n_files"]
            sections.append(
                part_heading(safe_label, f"{n_group} files")
                + "\\section{Baseline Detection Performance}\n\n"
                + _build_body(group_summaries, resolver,
                              label_suffix=slugify(group_label))
            )
        body = "\n\n".join(sections)
    else:
        body = (
            "\\section{Baseline Detection Performance}\n\n"
            + _build_body(summaries, resolver)
        )

    latex_content = (f"{preamble}\n\n" + abstract + body
                     + "\n\n" + container_section(containers or [])
                     + "\n\n\\end{document}")

    tex_path = os.path.join(report_dir, "no_attacks_report.tex")
    with open(tex_path, "w") as f:
        f.write(latex_content)
    logger.info(f"No-attacks LaTeX report saved to {tex_path}")

    compile_latex(report_dir, "no_attacks_report")
    return tex_path
