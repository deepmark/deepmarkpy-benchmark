"""LaTeX report for the ``detection_reliability`` mode.

A no-attack baseline (false positives and negatives on untouched audio,
and the watermarked signal's quality), then, when attacks ran, one section
per attack family: FP/FN per attack, accuracy, and the metric tables the
config asks for. FP and FN are counts over attempts, so no statistic
applies to them.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from deepmarkpy.utils.attack_groups import (
    GROUP_ORDER,
    OTHER_GROUP_KEY,
    group_attacks,
    group_label,
)
from deepmarkpy.utils.latex_helpers import (
    MetricCaveats,
    build_longtable,
    compact_header,
    compile_latex,
    container_section,
    display_attack_name,
    duration_label_tex,
    efficiency_tables,
    embedding_cost_line,
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
    EFFICIENCY_METRICS,
    INTELLIGIBILITY_METRICS,
    PER_FILE_EFFICIENCY_METRICS,
    MetricResolver,
    NISQA_METRICS,
    QUALITY_METRICS,
    compute_statistics,
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
    """One statistic from a ``{statistic: value}`` dict; a bare number is the mean."""
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
            m for m in resolver.all_signal_metrics() if m in family
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

        # A row per metric and a column per statistic; metrics sharing a
        # statistic list share a table.
        by_statistics = {}
        for metric in with_data:
            key = tuple(resolver.statistics_for(None, metric))
            by_statistics.setdefault(key, []).append(metric)

        for index, (statistics, members) in enumerate(by_statistics.items()):
            rows = [
                (metric_label(metric), [
                    format_metric_cell(metric, _stat_value(metrics[metric], s))
                    for s in statistics
                ])
                for metric in members
            ]
            suffix = f"_{index}" if len(by_statistics) > 1 else ""
            blocks.append(grid_table(
                "Metric", [stat_header(s) for s in statistics], rows,
                f"{section_title} of the watermarked audio compared to the "
                f"original (no attack applied).",
                f"tab:dr_no_attack_{section_key}{part}{suffix}",
            ))

    return "\n\n".join(blocks) + _silent_note(silent)


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

    headers = [stat_header(s) for s in statistics]
    if show_emr:
        headers.append(metric_label("emr"))

    rows = []
    for name in present:
        data = attacks[name]
        cells = [format_metric_cell("accuracy", data.get(f"accuracy_{s}"), "--")
                 for s in statistics]
        if show_emr:
            cells.append(format_emr_cell(
                data.get("emr_count"), data.get("accuracy_n"),
                data.get("emr_rate"),
            ))
        rows.append((display_attack_name(name), cells))

    return grid_table("Attack", headers, rows, caption, label)


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

        single = []
        for metric in with_data:
            statistics = resolver.statistics_for(group_key, metric)
            if len(statistics) > 1:
                blocks.append(_metric_table(
                    attacks, present, [(metric, s) for s in statistics],
                    [stat_header(s) for s in statistics],
                    f"{metric_label(metric)} --- {caption}",
                    f"{label}_{metric}",
                ))
            else:
                single.append((metric, statistics[0]))

        if single:
            blocks.append(_metric_table(
                attacks, present, single,
                [compact_header(m, s) for m, s in single],
                f"{section_title} --- {caption}", f"{label}_{section_key}",
            ))

    return "\n\n".join(blocks) + _silent_note(silent)


def _metric_table(attacks, present, columns, headers, caption, label):
    """One row per attack, one column per ``(metric, statistic)``.

    A metric's first statistic carries the caveat mark.
    """
    rows = []
    caveats = MetricCaveats()
    for name in present:
        entry_metrics = attacks[name].get("metrics") or {}
        first = {}
        cells = []
        for metric, statistic in columns:
            first.setdefault(metric, statistic)
            value = _stat_value(entry_metrics.get(metric), statistic)
            if value is None:
                cells.append("N/A")
                continue
            cell = format_metric_cell(metric, value)
            if statistic == first[metric]:
                cell += caveats.mark(name, metric)
            cells.append(cell)
        rows.append((display_attack_name(name), cells))

    if caveats.any_flagged:
        caption += " " + caveats.footnote().strip()
    return grid_table("Attack", headers, rows, caption, label)


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

def _efficiency_tables(attacks, attack_names, group_key, resolver, label_text,
                       label_key) -> str:
    """Timing tables for one attack group, after a blank line, or ''.

    Embedding happens once per file, so the baseline states it instead.
    """
    present = [a for a in attack_names if a in attacks]
    metrics = [
        m for m in resolver.metrics_for_group(None, bucket="efficiency")
        if m not in PER_FILE_EFFICIENCY_METRICS
        and any((attacks[a].get("timings") or {}).get(m) for a in present)
    ]
    tables = efficiency_tables(
        "Attack",
        [(display_attack_name(a), attacks[a].get("timings") or {})
         for a in present],
        metrics, lambda metric: resolver.statistics_for(group_key, metric),
        f"--- {label_text}", f"tab:dr_efficiency_{label_key}",
    )
    return "\n\n" + tables if tables else ""


def _embedding_cost_line(result, resolver) -> str:
    """Embedding time, stated once for the run rather than per attack."""
    metric = "embed_latency"
    if not resolver.is_enabled(None, metric):
        return ""
    timings = ((result.get("no_attack") or {}).get("timings") or {}).get(metric)
    return embedding_cost_line(timings or {},
                               resolver.statistics_for(None, metric))


def _build_group_section(attacks, attack_names, group_key, label_text,
                         n_files, resolver, suffix="") -> str:
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
    section += _accuracy_table(
        attacks, present, group_key, resolver,
        caption=f"Detection accuracy statistics --- {label_text}.",
        label=f"tab:dr_acc_{label_key}",
    )
    section += "\n\n"
    section += _metric_tables(
        attacks, present, group_key, resolver,
        caption=f"{label_text}.", label=f"tab:dr_{label_key}",
    )
    section += _efficiency_tables(
        attacks, present, group_key, resolver, label_text, label_key,
    )
    return section + "\n\n"


def _build_sections(result: Dict[str, Any], resolver: MetricResolver,
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

        for group_key in ordered:
            section = _build_group_section(
                attacks, grouped[group_key]["attacks"], group_key,
                group_label(group_key, grouped[group_key]["label"]),
                n_files, resolver, suffix=suffix,
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
            to the built-in matrix.
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
            result, duration_partitions, resolver,
        )
    else:
        sections = _build_sections(result, resolver)

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
        safe_label = duration_label_tex(group_label_text)
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
                group_result, resolver, suffix=f"_{slug}",
            ))
        )

    return sections


def _aggregate_per_file_to_result(
    per_file: Dict[str, Any], original_result: Dict[str, Any], n_files: int,
    resolver: MetricResolver,
) -> Dict[str, Any]:
    """A result-shaped dict recomputed from a subset of the per-file records.

    Each value is reduced to the statistics its group configures.
    """
    grouped = {}
    no_attack_fp = 0
    no_attack_fn = 0
    no_attack_metrics = {}
    no_attack_timings = {}

    for _, file_data in per_file.items():
        if file_data.get("no_attack_fp"):
            no_attack_fp += 1
        if file_data.get("no_attack_fn"):
            no_attack_fn += 1

        for metric, value in (file_data.get("no_attack_metrics") or {}).items():
            if value is not None:
                no_attack_metrics.setdefault(metric, []).append(value)
        # The baseline's timings sit at the top of the per-file record.
        for metric in EFFICIENCY_METRICS:
            value = file_data.get(metric)
            if value is not None and resolver.is_enabled(None, metric):
                no_attack_timings.setdefault(metric, []).append(value)

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
            metric: compute_statistics(
                values, resolver.statistics_for(None, metric), metric,
            )
            for metric, values in no_attack_metrics.items()
        }
    if no_attack_timings:
        no_attack["timings"] = {
            metric: compute_statistics(
                values, resolver.statistics_for(None, metric), metric,
            )
            for metric, values in no_attack_timings.items()
        }

    attacks = {}
    for attack_name, state in grouped.items():
        group_key = resolver.group_for_attack(attack_name)
        accuracies = state["accuracies"]
        stats = compute_statistics(
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
            metric: compute_statistics(
                values, resolver.statistics_for(group_key, metric), metric,
            )
            for metric, values in state["metrics"].items()
        }
        entry["timings"] = {
            metric: compute_statistics(
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
