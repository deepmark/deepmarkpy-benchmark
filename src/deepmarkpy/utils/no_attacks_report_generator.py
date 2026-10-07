"""Report generator for the ``no_attacks`` mode.

Shows what each model does to audio it watermarks, and whether it can
read its own watermark back with nothing in between:

* a detection table per model family -- a zero-bit score is a detection
  rate and a multi-bit score is bit agreement, and the two are never mixed
  into one column. A "Detected" count is added for the models that
  implement ``is_watermarked()``, which is the only thing entitled to say
  whether a watermark was found;
* one table per configured quality metric, with the statistics that
  metric's configuration asks for.

No attacks run in this mode, so there are no attack groups: every metric
and statistic comes from ``metrics.defaults`` via the resolver.
"""

import logging
import os

import numpy as np

from deepmarkpy.utils.latex_helpers import (
    build_longtable,
    compile_latex,
    container_section,
    duration_label_tex,
    format_emr_cell,
    format_metric_cell,
    make_preamble,
    part_heading,
    slugify,
    metric_label,
    stat_header,
)
from deepmarkpy.utils.metric_resolver import (
    ALL_STATISTICS,
    INTELLIGIBILITY_METRICS,
    PER_MODEL_EFFICIENCY_METRICS,
    MetricResolver,
    NISQA_METRICS,
    QUALITY_METRICS,
    worst_case_of,
)

logger = logging.getLogger(__name__)

# The three metric families get their own tables so no single table has to
# carry thirteen columns.
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
    """Statistics for a list of values, limited to ``statistics``.

    ``metric`` decides which end of the range "worst case" means.
    """
    if not values:
        return None
    wanted = set(ALL_STATISTICS if statistics is None else statistics)
    arr = np.array(values)
    available = {
        "mean": lambda: float(np.mean(arr)),
        "std": lambda: float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": lambda: float(np.median(arr)),
        "p5": lambda: float(np.percentile(arr, 5)),
        "p10": lambda: float(np.percentile(arr, 10)),
        "p95": lambda: float(np.percentile(arr, 95)),
        "p99": lambda: float(np.percentile(arr, 99)),
        "worst_case": lambda: worst_case_of(arr, metric),
    }
    result = {
        name: compute() for name, compute in available.items()
        if name in wanted
    }
    result["n"] = len(arr)
    return result


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

    # "Was a watermark found" is the model's own answer, recorded per file
    # by run_no_attacks when the model implements is_watermarked(). Without
    # it there is no honest count to print: a threshold on detect() output
    # would be this report guessing what the output means.
    if results.get("supports_detection"):
        # Counted over the files the model actually decided. A file whose
        # is_watermarked() raised has no answer, and counting it in the
        # denominator reported each failure as "not detected".
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

    # Timings sit on the file entry beside accuracy, not inside the
    # quality dict: they are not a comparison of two signals.
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

    Zero-bit and multi-bit models are tabled separately: a zero-bit score
    is the share of files in which anything was detected, a multi-bit
    score is bit agreement, and a shared column would invite reading one
    as the other.

    The "Detected" column appears for the models that answer
    ``is_watermarked()``, whatever family they are in -- a multi-bit model
    that can say yes or no gets the count too. Models that cannot are left
    with their accuracy columns alone.
    """
    statistics = resolver.statistics_for(None, "accuracy")
    show_ber = resolver.is_enabled(None, "ber") and not is_zero_bit
    ber_statistics = resolver.statistics_for(None, "ber") if show_ber else []
    show_emr = resolver.is_enabled(None, "emr")
    has_confidence = any("mean_confidence" in d for d in models_data.values())
    show_detected = any(
        d.get("supports_detection") for d in models_data.values()
    )

    headers = ["Model"] + [stat_header(s) for s in statistics]
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
        cells = [_short_model_name(model_name)]
        stats = data.get("accuracy_stats") or {}
        for statistic in statistics:
            value = stats.get(statistic)
            cells.append(
                "--" if value is None else format_metric_cell("accuracy", value)
            )
        if show_detected:
            cells.append(
                f"{data.get('positive_detections', 0)}/"
                f"{data.get('detection_n', data['n_files'])}"
                if data.get("supports_detection") else "N/A"
            )
        if show_ber and len(ber_statistics) == 1:
            ber = (data.get("ber_stats") or {}).get(ber_statistics[0])
            cells.append("--" if ber is None else format_metric_cell("ber", ber))
        if show_emr:
            cells.append(format_emr_cell(
                data.get("emr_count"), stats.get("n"), data.get("emr_rate"),
            ))
        if has_confidence:
            confidence = data.get("mean_confidence")
            cells.append(f"{confidence:.4f}" if confidence is not None else "N/A")
        rows.append("    " + " & ".join(cells) + " \\\\")

    family = "zero-bit" if is_zero_bit else "multi-bit"
    return build_longtable(
        "l" + "c" * (len(headers) - 1),
        " & ".join(headers),
        rows,
        f"Baseline detection performance --- {family} models.",
        label,
    )


def _ber_table(models_data, resolver, label):
    """BER with two or more statistics, which does not fit inline."""
    statistics = resolver.statistics_for(None, "ber")
    headers = ["Model"] + [stat_header(s) for s in statistics]
    rows = []
    for model_name, data in models_data.items():
        stats = data.get("ber_stats") or {}
        cells = [_short_model_name(model_name)] + [
            "--" if stats.get(s) is None else format_metric_cell("ber", stats[s])
            for s in statistics
        ]
        rows.append("    " + " & ".join(cells) + " \\\\")
    return build_longtable(
        "l" + "c" * len(statistics), " & ".join(headers), rows,
        "Bit error rate of the watermarked signal (no attack).", label,
    )


def _metric_table(models_data, resolver, metric, label):
    """One metric, one row per model, one column per configured statistic."""
    statistics = resolver.statistics_for(None, metric)
    present = {
        name: (data.get("quality_metrics_stats") or {})[metric]
        for name, data in models_data.items()
        if (data.get("quality_metrics_stats") or {}).get(metric)
    }
    if not present:
        return ""

    headers = ["Model"] + [stat_header(s) for s in statistics]
    rows = []
    for model_name, stats in present.items():
        cells = [_short_model_name(model_name)] + [
            "N/A" if stats.get(s) is None else format_metric_cell(metric, stats[s])
            for s in statistics
        ]
        rows.append("    " + " & ".join(cells) + " \\\\")

    return build_longtable(
        "l" + "c" * len(statistics), " & ".join(headers), rows,
        f"{metric_label(metric)} of the watermarked signal (no attack).",
        label,
    )


def _compact_metric_table(models_data, resolver, metrics, caption, label):
    """Metrics reduced to one statistic each, one column per metric."""
    headers = ["Model"]
    for metric in metrics:
        statistic = resolver.statistics_for(None, metric)[0]
        header = metric_label(metric)
        if statistic != "mean":
            header += f" [{stat_header(statistic)}]"
        headers.append(header)

    rows = []
    for model_name, data in models_data.items():
        stats_by_metric = data.get("quality_metrics_stats") or {}
        cells = [_short_model_name(model_name)]
        for metric in metrics:
            statistic = resolver.statistics_for(None, metric)[0]
            value = (stats_by_metric.get(metric) or {}).get(statistic)
            cells.append(
                "N/A" if value is None else format_metric_cell(metric, value)
            )
        rows.append("    " + " & ".join(cells) + " \\\\")

    return build_longtable(
        "l" + "c" * len(metrics), " & ".join(headers), rows, caption, label,
    )


def _efficiency_table(summaries, resolver, label):
    """What embedding and detection cost in time, per model.

    Its own tables below the quality ones. Those describe the model and
    reproduce from a seed; this describes the machine that ran it and
    does not, so the two never share a row. Split the same way the
    quality tables are, so several statistics do not make one wide table.
    """
    metrics = [
        m for m in resolver.metrics_for_group(None, bucket="efficiency")
        if m not in PER_MODEL_EFFICIENCY_METRICS
        and any((d.get("efficiency_stats") or {}).get(m)
                for d in summaries.values())
    ]
    if not metrics:
        return ""

    note = (
        " These depend on the machine and on whether the model runs "
        "natively or in a container, so they do not reproduce across runs "
        "the way the measurements above do."
    )

    tables = []
    compact = []
    for metric in metrics:
        statistics = resolver.statistics_for(None, metric)
        if len(statistics) > 1:
            tables.append(_timing_table(
                summaries, [(metric, s) for s in statistics],
                [stat_header(s) for s in statistics],
                f"{metric_label(metric)} per model, with no attack "
                f"applied.{note}",
                f"{label}_{metric}",
            ))
        elif statistics:
            compact.append((metric, statistics[0]))

    if compact:
        tables.append(_timing_table(
            summaries, compact, [metric_label(m) for m, _ in compact],
            f"Processing time per model, with no attack applied.{note}",
            label,
        ))

    return "\n\n".join(t for t in tables if t)


def _timing_table(summaries, columns, headers, caption, label):
    """One timing table: a row per model, a column per (metric, statistic)."""
    rows = []
    for name, data in summaries.items():
        stats = data.get("efficiency_stats") or {}
        cells = [_short_model_name(name)]
        for metric, statistic in columns:
            value = (stats.get(metric) or {}).get(statistic)
            cells.append("--" if value is None else f"{float(value):.4f}")
        rows.append("    " + " & ".join(cells) + " \\\\")

    return build_longtable(
        "l" + "c" * len(columns), " & ".join(["Model"] + headers),
        rows, caption, label,
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

        multi = [m for m in with_data
                 if len(resolver.statistics_for(None, m)) > 1]
        single = [m for m in with_data if m not in multi]

        for metric in multi:
            table = _metric_table(
                summaries, resolver, metric,
                f"tab:no_attacks_{metric}{suffix}",
            )
            if table:
                quality_tables.append(table)
        if single:
            quality_tables.append(_compact_metric_table(
                summaries, resolver, single, caption,
                f"tab:no_attacks_{section_key}{suffix}",
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
            matrix so the generator stays usable as a library.

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
