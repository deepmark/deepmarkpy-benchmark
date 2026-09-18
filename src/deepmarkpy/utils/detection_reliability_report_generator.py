"""LaTeX report for the ``detection_reliability`` mode.

Structure:

* **No-Attack Baseline** (always): false-positive and false-negative
  counts on untouched audio, plus the quality of the watermarked signal
  the detector was asked to judge.
* **One section per attack family** (when attacks were selected): FP/FN
  per attack, accuracy statistics, and the metric tables that family's
  configuration asks for.

Every column comes from the ``MetricResolver`` the config file built.
``other`` -- the attacks belonging to no declared family -- is a
configurable group like any other.

False positives and false negatives are counts over attempts, not
distributions, so no statistic applies to them and none is configurable.

Each attack section carries a figure putting both error rates side by
side against the no-attack rates. The two failures trade off against each
other, and the point of the mode is whether an attack moves the detector
off the operating point it already had -- which a two-column table makes
the reader reconstruct.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from deepmarkpy.utils import report_charts
from deepmarkpy.utils.attack_groups import (
    GROUP_ORDER,
    OTHER_GROUP_KEY,
    group_attacks,
    group_label,
)
from deepmarkpy.utils.latex_helpers import (
    build_longtable,
    compile_latex,
    container_section,
    display_attack_name,
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
    EFFICIENCY_METRICS,
    INTELLIGIBILITY_METRICS,
    PER_FILE_EFFICIENCY_METRICS,
    MetricResolver,
    NISQA_METRICS,
    QUALITY_METRICS,
)

logger = logging.getLogger(__name__)

# Timings that belong to an attack rather than to the file.
PER_ATTACK_TIMINGS = tuple(
    m for m in EFFICIENCY_METRICS if m not in PER_FILE_EFFICIENCY_METRICS
)

_METRIC_SECTIONS = (
    ("qual", "Audio quality", QUALITY_METRICS),
    ("intell", "Speech intelligibility", INTELLIGIBILITY_METRICS),
    ("nisqa", "NISQA non-intrusive quality", NISQA_METRICS),
)


def _format_count(count: int, total: int) -> str:
    if total <= 0:
        return "N/A"
    return f"{count}/{total}"


def _format_pct(count: int, total: int) -> str:
    if total <= 0:
        return "N/A"
    return f"{100.0 * count / total:.1f}\\%"


def _short_model_name(name: str) -> str:
    for suffix in ("Model", "Watermark"):
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("_", "\\_").replace("&", "\\&").replace("#", "\\#")


def _stat_value(entry, statistic):
    """One statistic out of a ``{statistic: value}`` dict.

    A bare number is read as the mean: some result files store a single
    value per metric rather than a statistics dict.
    """
    if isinstance(entry, (int, float)) and not isinstance(entry, bool):
        return float(entry) if statistic == "mean" else None
    if not isinstance(entry, dict):
        return None
    return entry.get(statistic)


# ---------------------------------------------------------------------------
# No-attack tables
# ---------------------------------------------------------------------------

def _no_attack_reliability_table(result: Dict[str, Any],
                                 suffix: str = "") -> str:
    n = int(result["n_files"])
    fp = int(result["no_attack"]["false_positive_count"])
    fn = int(result["no_attack"]["false_negative_count"])

    rows = [
        f"    False Positive & {_format_count(fp, n)} & {_format_pct(fp, n)} \\\\",
        f"    False Negative & {_format_count(fn, n)} & {_format_pct(fn, n)} \\\\",
    ]
    return build_longtable(
        col_spec="lcc",
        header="Metric & Count & Rate",
        rows=rows,
        caption=(
            "Detection reliability without attacks. "
            "False positive: detection on the clean original. "
            "False negative: missed detection on the watermarked signal."
        ),
        label=f"tab:dr_no_attack{suffix}",
    )


def _no_attack_quality_tables(result: Dict[str, Any],
                              resolver: MetricResolver,
                              part: str = "") -> str:
    """Quality of the watermarked signal, per configured metric."""
    metrics = result.get("no_attack", {}).get("metrics") or {}
    if not metrics:
        return ""

    blocks = []
    silent = []
    for section_key, section_title, family in _METRIC_SECTIONS:
        enabled = [
            m for m in resolver.signal_metrics_for_group(None) if m in family
        ]
        with_data = [
            m for m in enabled
            if _stat_value(metrics.get(m), "mean") is not None
            or (isinstance(metrics.get(m), dict) and any(
                v is not None for v in metrics[m].values()))
        ]
        silent += [m for m in enabled if m not in with_data]
        if not with_data:
            continue

        # One row per metric: the columns are statistics, and every metric
        # in a family that shares a statistic list shares a table.
        by_statistics = {}
        for metric in with_data:
            key = tuple(resolver.statistics_for(None, metric))
            by_statistics.setdefault(key, []).append(metric)

        for index, (statistics, members) in enumerate(by_statistics.items()):
            headers = ["Metric"] + [stat_header(s) for s in statistics]
            rows = []
            for metric in members:
                cells = [metric_label(metric)]
                for statistic in statistics:
                    value = _stat_value(metrics[metric], statistic)
                    cells.append(
                        "N/A" if value is None
                        else format_metric_cell(metric, value)
                    )
                rows.append("    " + " & ".join(cells) + " \\\\")
            suffix = f"_{index}" if len(by_statistics) > 1 else ""
            blocks.append(build_longtable(
                "l" + "c" * len(statistics), " & ".join(headers), rows,
                f"{section_title} of the watermarked audio compared to the "
                f"original (no attack applied).",
                f"tab:dr_no_attack_{section_key}{part}{suffix}",
            ))

    body = "\n\n".join(blocks)
    return body + _silent_note(silent)


# ---------------------------------------------------------------------------
# Per-attack tables
# ---------------------------------------------------------------------------

def _fp_fn_table(attacks: Dict[str, Any], attack_names: List[str],
                 n_files: int, caption: str, label: str) -> str:
    """FP/FN counts and rates for a list of attacks."""
    rows = []
    for name in attack_names:
        if name not in attacks:
            continue
        data = attacks[name]
        fp = int(data.get("false_positive_count", 0))
        fp_n = int(data.get("false_positive_attempts", n_files))
        fn = int(data.get("false_negative_count", 0))
        fn_n = int(data.get("false_negative_attempts", n_files))
        rows.append(
            f"    {display_attack_name(name)} & {_format_count(fp, fp_n)} "
            f"& {_format_pct(fp, fp_n)} & {_format_count(fn, fn_n)} "
            f"& {_format_pct(fn, fn_n)} \\\\"
        )

    if not rows:
        return ""

    return build_longtable(
        col_spec="lcccc",
        header="Attack & FP Count & FP Rate & FN Count & FN Rate",
        rows=rows,
        caption=caption,
        label=label,
    )


def _accuracy_table(attacks: Dict[str, Any], attack_names: List[str],
                    group_key: Optional[str], resolver: MetricResolver,
                    caption: str, label: str) -> str:
    """Accuracy statistics per attack, with columns from the config."""
    present = [name for name in attack_names if name in attacks]
    if not present:
        return ""

    statistics = resolver.statistics_for(group_key, "accuracy")
    show_emr = resolver.is_enabled(group_key, "emr") and any(
        attacks[name].get("emr_count") is not None for name in present
    )

    headers = ["Attack"] + [stat_header(s) for s in statistics]
    if show_emr:
        headers.append(metric_label("emr"))

    rows = []
    for name in present:
        data = attacks[name]
        cells = [display_attack_name(name)]
        for statistic in statistics:
            value = data.get(f"accuracy_{statistic}")
            cells.append(
                "--" if value is None
                else format_metric_cell("accuracy", value)
            )
        if show_emr:
            cells.append(format_emr_cell(
                data.get("emr_count"), data.get("accuracy_n"),
                data.get("emr_rate"),
            ))
        rows.append("    " + " & ".join(cells) + " \\\\")

    return build_longtable(
        "l" + "c" * (len(headers) - 1), " & ".join(headers), rows,
        caption, label,
    )


def _metric_tables(attacks: Dict[str, Any], attack_names: List[str],
                   group_key: Optional[str], resolver: MetricResolver,
                   caption: str, label: str) -> str:
    """Every configured metric table for one attack group.

    Metrics with two or more configured statistics get their own table;
    the rest are collected into one column-per-metric table.
    """
    present = [name for name in attack_names if name in attacks]
    if not present:
        return ""

    resolver = resolver or MetricResolver.from_attack_groups()
    blocks = []
    silent = []

    for section_key, section_title, family in _METRIC_SECTIONS:
        enabled = [
            m for m in resolver.signal_metrics_for_group(group_key)
            if m in family
        ]
        if not enabled:
            continue

        with_data = [
            m for m in enabled
            if any(
                (attacks[name].get("metrics") or {}).get(m) is not None
                for name in present
            )
        ]
        silent += [m for m in enabled if m not in with_data]

        multi = [m for m in with_data
                 if len(resolver.statistics_for(group_key, m)) > 1]
        single = [m for m in with_data if m not in multi]

        for metric in multi:
            blocks.append(_single_metric_table(
                attacks, present, metric,
                resolver.statistics_for(group_key, metric),
                f"{metric_label(metric)} --- {caption}",
                f"{label}_{metric}",
            ))

        if single:
            blocks.append(_compact_metric_table(
                attacks, present, single, group_key, resolver,
                f"{section_title} --- {caption}", f"{label}_{section_key}",
            ))

    return "\n\n".join(b for b in blocks if b) + _silent_note(silent)


def _single_metric_table(attacks, present, metric, statistics, caption, label):
    """One metric, one column per configured statistic."""
    headers = ["Attack"] + [stat_header(s) for s in statistics]
    rows = []
    for name in present:
        entry = (attacks[name].get("metrics") or {}).get(metric)
        cells = [display_attack_name(name)]
        for statistic in statistics:
            value = _stat_value(entry, statistic)
            if value is None:
                cells.append("N/A")
                continue
            cells.append(format_metric_cell(metric, value))
        rows.append("    " + " & ".join(cells) + " \\\\")

    return build_longtable(
        "l" + "c" * len(statistics), " & ".join(headers), rows, caption, label,
    )


def _compact_metric_table(attacks, present, metrics, group_key, resolver,
                          caption, label):
    """Metrics reduced to one statistic each, one column per metric."""
    headers = ["Attack"]
    for metric in metrics:
        statistic = resolver.statistics_for(group_key, metric)[0]
        header = metric_label(metric)
        if statistic != "mean":
            header += f" [{stat_header(statistic)}]"
        headers.append(header)

    rows = []
    for name in present:
        entry_metrics = attacks[name].get("metrics") or {}
        cells = [display_attack_name(name)]
        for metric in metrics:
            statistic = resolver.statistics_for(group_key, metric)[0]
            entry = entry_metrics.get(metric)
            # Older result files stored a bare float per metric rather than
            # a statistics dict; read that as the mean it was.
            if isinstance(entry, (int, float)):
                value = entry if statistic == "mean" else None
            else:
                value = _stat_value(entry, statistic)
            if value is None:
                cells.append("N/A")
                continue
            cells.append(format_metric_cell(metric, value))
        rows.append("    " + " & ".join(cells) + " \\\\")

    return build_longtable(
        "l" + "c" * len(metrics), " & ".join(headers), rows, caption, label,
    )


def _silent_note(metrics):
    """Footnote naming metrics that were enabled but produced nothing."""
    if not metrics:
        return ""
    names = ", ".join(metric_label(m) for m in dict.fromkeys(metrics))
    return (
        "\n\n{\\noindent\\footnotesize Enabled in the configuration but not "
        f"reported here, because no value was produced: {names}. This usually "
        "means the metric's service or optional package was unavailable.}\n"
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _error_rate_figure(attacks, attack_names, n_files, label_text, label_key,
                       report_dir, baseline) -> str:
    """Both error rates per attack, against the rates before any attack.

    This mode measures two failures that trade off against each other, and
    a table asks the reader to hold both columns in their head while
    scanning. Side by side, and against the no-attack line, the question
    "did this attack move the detector, and in which direction" is one
    look.
    """
    if report_dir is None:
        return ""

    rows = []
    for name in attack_names:
        data = attacks.get(name)
        if not data:
            continue
        fp_n = int(data.get("false_positive_attempts", n_files) or 0)
        fn_n = int(data.get("false_negative_attempts", n_files) or 0)
        rows.append((
            display_attack_name(name),
            100.0 * int(data.get("false_positive_count", 0)) / fp_n if fp_n else None,
            100.0 * int(data.get("false_negative_count", 0)) / fn_n if fn_n else None,
        ))

    filename = f"dr_error_rates_{label_key}.png"
    drawn = report_charts.false_positive_negative_bars(
        rows, os.path.join(report_dir, filename),
        baseline_fp=baseline[0], baseline_fn=baseline[1],
    )
    if not drawn:
        return ""

    note = ""
    if baseline[0] is not None or baseline[1] is not None:
        note = (
            " The dashed lines are the same two rates with no attack applied: "
            "an attack matters insofar as it moves the detector off the "
            "operating point it already had."
        )
    return figure_block(
        filename,
        f"Detector error rates --- {label_text}. A false negative is a "
        f"watermark that was there and was missed; a false positive is one "
        f"claimed on clean audio.{note}",
        f"fig:dr_error_rates_{label_key}",
    )


def _baseline_rates(result: Dict[str, Any]):
    """No-attack FP and FN as percentages, for the figure's reference lines."""
    n = int(result.get("n_files", 0) or 0)
    if n <= 0:
        return (None, None)
    no_attack = result.get("no_attack") or {}
    return (
        100.0 * int(no_attack.get("false_positive_count", 0)) / n,
        100.0 * int(no_attack.get("false_negative_count", 0)) / n,
    )


def _accuracy_figure(attacks, attack_names, group_key, resolver, label_text,
                     label_key, report_dir) -> str:
    """The accuracy table above, ranked worst-first and coloured by tier."""
    if report_dir is None or len(attack_names) < 3:
        # With one or two bars the table already reads as a ranking.
        return ""

    statistics = resolver.statistics_for(group_key, "accuracy")
    statistic = statistics[0] if statistics else "mean"
    values = {}
    for name in attack_names:
        value = attacks[name].get(f"accuracy_{statistic}")
        if value is not None:
            values[display_attack_name(name)] = float(value)
    if len(values) < 3:
        return ""

    filename = f"dr_accuracy_{label_key}.png"
    drawn = report_charts.accuracy_ranking(
        values, os.path.join(report_dir, filename),
        statistic_label=stat_header(statistic),
        title=f"{label_text} ranked by accuracy ({stat_header(statistic)})",
    )
    if not drawn:
        return ""
    return figure_block(
        filename,
        f"Attacks in {label_text.lower()} ranked by detection accuracy "
        f"({stat_header(statistic).lower()}), worst first. Bar colour is the "
        f"robustness tier.",
        f"fig:dr_accuracy_{label_key}",
    )


def _efficiency_tables(attacks, attack_names, group_key, resolver, label_text,
                       label_key) -> str:
    """Processing time for one attack group, in its own tables.

    Split the way the quality tables are: a metric with two or more
    statistics gets its own table, the single-statistic ones share one.
    Embedding is left out -- it happens once per file, not once per
    attack, and is stated for the run instead.
    """
    present = [a for a in attack_names if a in attacks]
    if not present:
        return ""

    metrics = [
        m for m in resolver.metrics_for_group(None, bucket="efficiency")
        if m not in PER_FILE_EFFICIENCY_METRICS
        and any((attacks[a].get("timings") or {}).get(m) for a in present)
    ]
    if not metrics:
        return ""

    note = (
        " These depend on the machine and on whether the plugin ran "
        "natively or over HTTP, so they do not reproduce across runs the "
        "way the measurements above do."
    )

    def table(columns, headers, caption, label):
        rows = []
        for name in present:
            timings = attacks[name].get("timings") or {}
            cells = [display_attack_name(name)]
            for metric, statistic in columns:
                value = (timings.get(metric) or {}).get(statistic)
                cells.append("--" if value is None else f"{float(value):.4f}")
            rows.append("    " + " & ".join(cells) + " \\\\")
        return build_longtable(
            "l" + "c" * len(columns), " & ".join(["Attack"] + headers),
            rows, caption, label,
        )

    blocks = []
    compact = []
    for metric in metrics:
        statistics = resolver.statistics_for(group_key, metric)
        if len(statistics) > 1:
            blocks.append(table(
                [(metric, s) for s in statistics],
                [stat_header(s) for s in statistics],
                f"{metric_label(metric)} --- {label_text}.{note}",
                f"tab:dr_efficiency_{label_key}_{metric}",
            ))
        elif statistics:
            compact.append((metric, statistics[0]))

    if compact:
        blocks.append(table(
            compact, [metric_label(m) for m, _ in compact],
            f"Processing time --- {label_text}.{note}",
            f"tab:dr_efficiency_{label_key}",
        ))

    return "\n\n" + "\n\n".join(b for b in blocks if b) if blocks else ""


def _embedding_cost_line(result, resolver) -> str:
    """Embedding time, stated once for the run rather than per attack."""
    metric = "embed_latency"
    if not resolver.is_enabled(None, metric):
        return ""
    timings = ((result.get("no_attack") or {}).get("timings") or {}).get(metric)
    if not timings:
        return ""

    parts = []
    for statistic in resolver.statistics_for(None, metric):
        value = timings.get(statistic)
        if value is not None:
            parts.append(f"{float(value):.4f}\\,s ({stat_header(statistic).lower()})")
    if not parts:
        return ""
    return (
        f"\\noindent\\textbf{{Embedding cost per file:}} {', '.join(parts)}\n"
        "\\\\{\\footnotesize Measured once per file, before any attack, so it "
        "does not vary by attack. Like every timing it depends on this machine "
        "and does not reproduce across runs.}\n\n"
    )


def _build_group_section(attacks, attack_names, group_key, label_text,
                         n_files, resolver, report_dir=None, suffix="",
                         baseline=(None, None)) -> str:
    """Build a full section for one attack group."""
    present = [a for a in attack_names if a in attacks]
    if not present:
        return ""

    label_key = f"{group_key}{suffix}"
    section = f"\\section{{{label_text}}}\n\n"
    section += _fp_fn_table(
        attacks, present, n_files,
        caption=f"False positive and false negative rates --- {label_text}.",
        label=f"tab:dr_fpfn_{label_key}",
    )
    section += "\n\n"
    section += _error_rate_figure(
        attacks, present, n_files, label_text, label_key, report_dir, baseline,
    )
    section += _accuracy_table(
        attacks, present, group_key, resolver,
        caption=f"Detection accuracy statistics --- {label_text}.",
        label=f"tab:dr_acc_{label_key}",
    )
    section += "\n\n"
    section += _accuracy_figure(
        attacks, present, group_key, resolver, label_text, label_key,
        report_dir,
    )
    section += _metric_tables(
        attacks, present, group_key, resolver,
        caption=f"{label_text}.", label=f"tab:dr_{label_key}",
    )
    section += _efficiency_tables(
        attacks, present, group_key, resolver, label_text, label_key,
    )
    return section + "\n\n"


def _build_sections(result: Dict[str, Any], resolver: MetricResolver,
                    report_dir: Optional[str] = None,
                    suffix: str = "") -> List[str]:
    """Build report body sections from a result dict."""
    n_files = int(result.get("n_files", 0))
    sections = []

    baseline = "\\section{No-Attack Baseline}\n\n"
    baseline += _embedding_cost_line(result, resolver)
    baseline += _no_attack_reliability_table(result, suffix)
    quality = _no_attack_quality_tables(result, resolver, suffix)
    if quality:
        baseline += "\n\n" + quality
    sections.append(baseline)

    attacks = result.get("attacks") or {}
    if attacks:
        grouped = group_attacks(list(attacks))
        ordered = [k for k in GROUP_ORDER if k in grouped]
        if OTHER_GROUP_KEY in grouped:
            ordered.append(OTHER_GROUP_KEY)

        rates = _baseline_rates(result)
        for group_key in ordered:
            section = _build_group_section(
                attacks, grouped[group_key]["attacks"], group_key,
                group_label(group_key, grouped[group_key]["label"]),
                n_files, resolver, report_dir=report_dir, suffix=suffix,
                baseline=rates,
            )
            if section:
                sections.append(section)

    return sections


def generate_detection_reliability_report(
    result: Dict[str, Any], report_dir: str = "report",
    resolver: Optional[MetricResolver] = None,
    duration_partitions: Optional[List] = None,
    containers=None,
) -> str:
    """Write the detection-reliability LaTeX report and compile to PDF.

    Args:
        result: Detection reliability result dict.
        report_dir: Output directory.
        resolver: metric/statistic configuration for every table. Defaults
            to the built-in matrix so the generator stays usable as a
            library.
        duration_partitions: Optional list of (label, file_list) tuples.
            When provided, generates per-duration-group sections.
    """
    os.makedirs(report_dir, exist_ok=True)
    has_cls = os.path.exists(os.path.join(report_dir, "deepmark.cls"))
    resolver = resolver or MetricResolver.from_attack_groups()

    model_name = result.get("model_name", "DeepMark")
    short_name = _short_model_name(model_name)
    n_files = int(result.get("n_files", 0))
    has_attacks = bool(result.get("attacks"))

    preamble = make_preamble(
        title=f"Detection Reliability Report: {short_name}",
        author="DeepMark Benchmark System",
        has_deepmark_cls=has_cls,
    )

    abstract = (
        f"\\begin{{abstract}}\n"
        f"This report measures detection reliability for the "
        f"{short_name} watermarking model across {n_files} "
        f"{'file' if n_files == 1 else 'files'}. "
    )
    threshold = result.get("detection_threshold")
    if threshold is not None:
        abstract += f"Detection threshold: {threshold} (confidence-based model). "
    abstract += (
        "False positives are detections on the clean (non-watermarked) input; "
        "false negatives are missed detections on the watermarked input."
    )
    if has_attacks:
        abstract += (
            " Per-attack results are grouped by attack category with "
            "FP/FN rates and quality metrics."
        )
    abstract += "\n\\end{abstract}\n\n"

    if duration_partitions:
        sections = _build_grouped_dr_sections(
            result, duration_partitions, resolver, report_dir,
        )
    else:
        sections = _build_sections(result, resolver, report_dir)

    latex_content = (
        f"{preamble}\n\n" + abstract + "\n\n".join(sections)
        + "\n\n" + container_section(containers or [])
        + "\n\n\\end{document}"
    )

    tex_path = os.path.join(report_dir, "detection_reliability_report.tex")
    with open(tex_path, "w") as f:
        f.write(latex_content)
    logger.info(f"Detection reliability LaTeX report saved to {tex_path}")

    compile_latex(report_dir, "detection_reliability_report")
    return tex_path


def _build_grouped_dr_sections(
    result: Dict[str, Any],
    duration_partitions: List,
    resolver: MetricResolver,
    report_dir: Optional[str] = None,
) -> List[str]:
    """Build per-duration-group sections for the detection reliability report.

    Each group gets its own ``\\part`` with the subset of files that fall
    into that duration bucket.
    """
    per_file = result.get("per_file", {})
    sections = []

    for group_label_text, group_files in duration_partitions:
        group_file_set = set(group_files)
        group_per_file = {
            fp: data for fp, data in per_file.items() if fp in group_file_set
        }
        safe_label = group_label_text.replace("<", "$<$").replace(">", "$>$")
        if not group_per_file:
            sections.append(
                part_heading(safe_label, f"{len(group_files)} files")
                + "No results available for this duration group.\n"
            )
            continue

        group_result = _aggregate_per_file_to_result(
            group_per_file, result, len(group_files), resolver,
        )
        slug = slugify(group_label_text)
        sections.append(
            part_heading(safe_label, f"{len(group_per_file)} files")
            + "\n\n".join(_build_sections(
                group_result, resolver, report_dir, suffix=f"_{slug}",
            ))
        )

    return sections


def _aggregate_per_file_to_result(
    per_file: Dict[str, Any], original_result: Dict[str, Any], n_files: int,
    resolver: MetricResolver,
) -> Dict[str, Any]:
    """Rebuild a result-shaped dict from a subset of per-file data.

    Uses the per-file records to recompute FP/FN counts and metrics for
    the subset, reducing each to the statistics its group configured so
    the duration sections and the overall ones agree on their columns.
    """
    from deepmarkpy.utils.detection_reliability import _compute_metric_stats

    grouped = {}
    no_attack_fp = 0
    no_attack_fn = 0
    no_attack_metrics = {}

    for _, file_data in per_file.items():
        if file_data.get("no_attack_fp"):
            no_attack_fp += 1
        if file_data.get("no_attack_fn"):
            no_attack_fn += 1

        for metric, value in (file_data.get("no_attack_metrics") or {}).items():
            if value is not None:
                no_attack_metrics.setdefault(metric, []).append(value)

        for attack_name, atk_data in (file_data.get("attacks") or {}).items():
            state = grouped.setdefault(attack_name, {
                "fp": 0, "fn": 0, "fp_n": 0, "fn_n": 0,
                "accuracies": [], "metrics": {}, "timings": {},
            })
            if atk_data.get("fp"):
                state["fp"] += 1
            state["fp_n"] += 1
            if atk_data.get("fn"):
                state["fn"] += 1
            state["fn_n"] += 1
            if atk_data.get("accuracy") is not None:
                state["accuracies"].append(atk_data["accuracy"])
            for metric, value in (atk_data.get("metrics") or {}).items():
                if value is not None:
                    state["metrics"].setdefault(metric, []).append(value)
            for metric in PER_ATTACK_TIMINGS:
                value = atk_data.get(metric)
                if value is not None:
                    state["timings"].setdefault(metric, []).append(value)

    no_attack = {
        "false_positive_count": no_attack_fp,
        "false_negative_count": no_attack_fn,
    }
    if no_attack_metrics:
        no_attack["metrics"] = {
            metric: _compute_metric_stats(
                values, resolver.statistics_for(None, metric), metric,
            )
            for metric, values in no_attack_metrics.items()
        }

    attacks = {}
    for attack_name, state in grouped.items():
        group_key = resolver.group_for_attack(attack_name)
        accuracies = state["accuracies"]
        stats = _compute_metric_stats(
            accuracies, resolver.statistics_for(group_key, "accuracy"),
            "accuracy",
        ) or {}
        entry = {f"accuracy_{k}": v for k, v in stats.items()}
        entry["accuracy_n"] = len(accuracies)

        if resolver.is_enabled(group_key, "emr"):
            exact = sum(1 for a in accuracies if a == 100.0)
            entry["emr_count"] = exact
            entry["emr_rate"] = float(exact / len(accuracies)) if accuracies else 0.0

        entry["metrics"] = {
            metric: _compute_metric_stats(
                values, resolver.statistics_for(group_key, metric), metric,
            )
            for metric, values in state["metrics"].items()
        }
        entry["timings"] = {
            metric: _compute_metric_stats(
                values, resolver.statistics_for(group_key, metric), metric,
            )
            for metric, values in state["timings"].items()
        }
        entry.update(
            false_positive_count=state["fp"],
            false_positive_attempts=state["fp_n"],
            false_negative_count=state["fn"],
            false_negative_attempts=state["fn_n"],
        )
        attacks[attack_name] = entry

    return {
        "model_name": original_result.get("model_name", "DeepMark"),
        "is_zero_bit": original_result.get("is_zero_bit", False),
        "detection_threshold": original_result.get("detection_threshold"),
        "n_files": n_files,
        "no_attack": no_attack,
        "attacks": attacks or None,
    }
