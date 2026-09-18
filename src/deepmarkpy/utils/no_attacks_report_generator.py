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

Figures appear only when two or more models are compared, and are left
out of a single-model run. Every quality metric here is already a
before/after measurement -- original against watermarked -- so there is
no separate "before" number to plot one against; the only axis left is
the model, and one model is one bar, which the table states better.

Each metric's figure sits under that metric's own table and is drawn on
that metric's declared range (``METRIC_RANGES``), so a PESQ of 2.0 shows
as a bar reaching a third of the way up rather than filling the axis the
way an auto-fitted scale would.

Accuracy is the exception to "one figure per configured statistic": only
mean and worst case are charted, and only when the configuration lists
them (``CHARTED_ACCURACY_STATISTICS``). A config that names all eight
would otherwise turn one table into eight figures.

No attacks run in this mode, so there are no attack groups: every metric
and statistic comes from ``metrics.defaults`` via the resolver.
"""

import logging
import os

import numpy as np

from deepmarkpy.utils import report_charts
from deepmarkpy.utils.latex_helpers import (
    NARROW_FIGURE_WIDTH,
    build_longtable,
    compile_latex,
    container_section,
    figure_block,
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
    LOWER_IS_BETTER_METRICS,
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
        # The raw list, kept for the distribution figure: the statistics
        # above are whichever ones the config asked for, and a box plot
        # needs the values themselves.
        "accuracy_values": [float(a) for a in accuracies],
    }

    # "Was a watermark found" is the model's own answer, recorded per file
    # by run_no_attacks when the model implements is_watermarked(). Without
    # it there is no honest count to print: a threshold on detect() output
    # would be this report guessing what the output means.
    if results.get("supports_detection"):
        summary["supports_detection"] = True
        summary["positive_detections"] = sum(
            1 for f in files if f.get("detected")
        )

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
                f"{data.get('positive_detections', 0)}/{data['n_files']}"
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


def _accuracy_spread_figure(models_data, resolver, report_dir, is_zero_bit,
                            suffix):
    """How each model's files split by outcome, before any attack.

    A model that reads its own watermark back on 99 files out of 100 and
    fails on the hundredth has almost the same mean as one that scores
    99\\% on every file, and a very different failure mode.
    """
    if report_dir is None:
        return ""
    distributions = {
        _short_model_name(name): data.get("accuracy_values") or []
        for name, data in models_data.items()
    }
    family = "zerobit" if is_zero_bit else "multibit"
    filename = f"no_attacks_accuracy_{family}{suffix}.png"
    drawn = report_charts.per_file_outcome_bars(
        distributions, os.path.join(report_dir, filename),
        title="Baseline outcome with no attack applied",
        chance_floor=0.0 if is_zero_bit else 50.0,
        is_zero_bit=is_zero_bit,
    )
    if not drawn:
        return ""
    return figure_block(
        filename,
        "Share of files by detection outcome before any attack, with file "
        "counts inside the bars. Anything outside the best band is a file "
        "the model could not fully read back from its own untouched output, "
        "which bounds everything the attack results can show.",
        f"fig:no_attacks_accuracy_{family}{suffix}",
    )


def _family_figures(models_data, resolver, report_dir, is_zero_bit, suffix):
    """The outcome figure belonging under one model family's accuracy table."""
    if len(models_data) < 2:
        return ""
    figure = _accuracy_spread_figure(
        models_data, resolver, report_dir, is_zero_bit, suffix,
    )
    return f"\n\n{figure}" if figure else ""


# The only accuracy statistics worth a chart: the typical case and the
# floor under it. The percentiles between them describe a distribution,
# which the table states more precisely than bars can, and a standard
# deviation is a spread that does not belong on the 0--100 axis at all.
# A config naming all eight therefore still gets one figure of two bars.
CHARTED_ACCURACY_STATISTICS = ("mean", "worst_case")


def _accuracy_figures(summaries, resolver, report_dir, suffix):
    """Accuracy per model, for whichever of mean and worst case is configured.

    One figure over every model, not one per family. The tables split
    zero-bit from multi-bit because the two score different things, but
    splitting the figure the same way leaves a run of one model per
    family -- the common case -- with no figure at all. Following the
    comparative report, the zero-bit bars are marked rather than hidden,
    and the caption says what the marker means.
    """
    if report_dir is None or len(summaries) < 2:
        return ""

    # The configuration's order, not this module's, as everywhere else.
    statistics = [
        statistic for statistic in resolver.statistics_for(None, "accuracy")
        if statistic in CHARTED_ACCURACY_STATISTICS
    ]
    if not statistics:
        return ""

    models = list(summaries)
    zero_bit = [bool(summaries[name].get("is_zero_bit")) for name in models]
    labels = [
        _short_model_name(name) + ("\\textsuperscript{0}" if zb else "")
        for name, zb in zip(models, zero_bit)
    ]

    series = [
        (stat_header(statistic),
         [(summaries[name].get("accuracy_stats") or {}).get(statistic)
          for name in models])
        for statistic in statistics
    ]

    # The floor is a multi-bit notion: a failed zero-bit detection scores
    # 0, which is the axis origin and needs no line.
    floor = 50.0 if not all(zero_bit) else None
    filename = f"no_attacks_accuracy_values{suffix}.png"
    drawn = report_charts.accuracy_by_model(
        series, os.path.join(report_dir, filename), models=labels,
        statistic_labels=statistics, chance_floor=floor,
        title="Baseline accuracy by model",
    )
    if not drawn:
        return ""

    named = ", ".join(stat_header(s).lower() for s in statistics)
    marker = (
        " \\textsuperscript{0}Zero-bit model: the score is the share of files "
        "in which a watermark was detected, so its floor is 0\\%. Multi-bit "
        "scores are bit agreement, whose floor is chance ($\\sim$50\\%). The "
        "two families are not comparable with each other."
        if any(zero_bit) and not all(zero_bit) else ""
    )
    others = [
        s for s in resolver.statistics_for(None, "accuracy")
        if s not in CHARTED_ACCURACY_STATISTICS
    ]
    note = (
        " The remaining statistics the configuration asks for are in the "
        "tables above; they describe the distribution rather than the level, "
        "and are read more precisely as numbers."
        if others else ""
    )
    return "\n\n" + figure_block(
        filename,
        f"Baseline detection accuracy by model before any attack ({named}). "
        f"The axis is the full 0--100\\% range.{marker}{note}",
        f"fig:no_attacks_accuracy_values{suffix}",
    )


def _metric_figure(summaries, resolver, metric, report_dir, suffix):
    """One metric, one bar per model, drawn on that metric's own scale.

    Emitted next to that metric's table rather than collected into one
    panel of many: a figure of PESQ read three tables away from the PESQ
    numbers is a figure the reader has to re-anchor.
    """
    if report_dir is None:
        return ""

    statistics = resolver.statistics_for(None, metric)
    if not statistics:
        return ""
    statistic = statistics[0]

    values = {
        _short_model_name(name):
            (data.get("quality_metrics_stats") or {}).get(metric, {})
            .get(statistic)
        for name, data in summaries.items()
    }
    if not any(v is not None for v in values.values()):
        return ""

    higher_is_better = metric not in LOWER_IS_BETTER_METRICS
    filename = f"no_attacks_{metric}{suffix}.png"
    drawn = report_charts.metric_by_model(
        values, os.path.join(report_dir, filename), metric,
        metric_label=report_charts.direction_hint(
            metric_label(metric), higher_is_better,
        ),
        higher_is_better=higher_is_better,
    )
    if not drawn:
        return ""

    from deepmarkpy.utils.metrics import METRIC_RANGES
    if metric in METRIC_RANGES:
        low, high = METRIC_RANGES[metric]
        scale_note = (
            f" The axis spans the metric's full range ({low:g}--{high:g}), so "
            f"the bar height is the share of the scale the model reached."
        )
    else:
        scale_note = (
            " This metric has no defined ceiling, so the axis is fitted to "
            "the values and bar heights compare the models to each other, "
            "not to an absolute best."
        )

    return figure_block(
        filename,
        f"{metric_label(metric)} of the watermarked signal against the "
        f"original ({stat_header(statistic).lower()}).{scale_note}",
        f"fig:no_attacks_{metric}{suffix}",
        width=NARROW_FIGURE_WIDTH,
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


def _build_body(summaries, resolver, label_suffix="", report_dir=None):
    """Build the report body for a set of model summaries."""
    suffix = f"_{label_suffix}" if label_suffix else ""
    zero_bit = {m: s for m, s in summaries.items() if s["is_zero_bit"]}
    multi_bit = {m: s for m, s in summaries.items() if not s["is_zero_bit"]}

    tables = []
    if multi_bit:
        tables.append(_accuracy_table(
            multi_bit, resolver, False, f"tab:no_attacks_multibit{suffix}",
        ) + _family_figures(multi_bit, resolver, report_dir, False, suffix))
        if resolver.is_enabled(None, "ber") and \
                len(resolver.statistics_for(None, "ber")) > 1:
            tables.append(_ber_table(
                multi_bit, resolver, f"tab:no_attacks_ber{suffix}",
            ))
    if zero_bit:
        tables.append(_accuracy_table(
            zero_bit, resolver, True, f"tab:no_attacks_zerobit{suffix}",
        ) + _family_figures(zero_bit, resolver, report_dir, True, suffix))

    # One figure over every model, after the last accuracy table, because
    # a run with one model per family would otherwise get none.
    accuracy_figure = _accuracy_figures(summaries, resolver, report_dir, suffix)
    if accuracy_figure and tables:
        tables[-1] += accuracy_figure

    # A bar chart of one bar is a table with worse resolution, so the
    # per-metric figures wait until there are models to compare.
    def show_figure(metric):
        if len(summaries) < 2:
            return ""
        figure = _metric_figure(summaries, resolver, metric, report_dir, suffix)
        return f"\n\n{figure}" if figure else ""

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
                quality_tables.append(table + show_figure(metric))
        if single:
            # A single-statistic metric has no table of its own, so its
            # figure follows the compact table that carries its column.
            quality_tables.append(_compact_metric_table(
                summaries, resolver, single, caption,
                f"tab:no_attacks_{section_key}{suffix}",
            ) + "".join(show_figure(m) for m in single))

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
            safe_label = group_label.replace("<", "$<$").replace(">", "$>$")
            n_group = next(iter(group_summaries.values()))["n_files"]
            sections.append(
                part_heading(safe_label, f"{n_group} files")
                + "\\section{Baseline Detection Performance}\n\n"
                + _build_body(group_summaries, resolver,
                              label_suffix=slugify(group_label),
                              report_dir=report_dir)
            )
        body = "\n\n".join(sections)
    else:
        body = (
            "\\section{Baseline Detection Performance}\n\n"
            + _build_body(summaries, resolver, report_dir=report_dir)
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
