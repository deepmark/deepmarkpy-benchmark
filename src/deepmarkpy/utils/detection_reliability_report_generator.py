"""LaTeX report for the --detection_reliability mode.

Layout depends on whether attacks are provided:

  Without attacks:
    Section 1 — No-Attack Reliability (FP/FN table)
    Section 2 — Watermarked Audio Quality (if --calculate_quality_metrics)

  With attacks (no --no_attacks):
    Per-group sections, each containing:
      - Accuracy + FP/FN table
      - Always-on metrics table (PESQ, ViSQOL, STOI) by default
      - OR three separate tables (quality, intelligibility, NISQA)
        when --calculate_quality_metrics is set
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from deepmarkpy.utils.attack_groups import (
    ATTACK_GROUPS,
    group_attacks,
    get_group_for_attack,
)
from deepmarkpy.utils.latex_helpers import (
    build_longtable,
    compile_latex,
    display_attack_name,
    make_preamble,
)
from deepmarkpy.utils.metrics import METRIC_LABELS, NISQA_METRICS

logger = logging.getLogger(__name__)

ALWAYS_ON_METRICS = ["pesq", "visqol", "stoi"]

GROUP_ORDER = [
    "process_disruption",
    "audio_editing",
    "audio_distortion",
    "desynchronization",
    "ai_attacks",
    "transmission",
]


def _format_count(count: int, total: int) -> str:
    if total <= 0:
        return "N/A"
    return f"{count}/{total}"


def _format_pct(count: int, total: int) -> str:
    if total <= 0:
        return "N/A"
    pct = 100.0 * count / total
    return f"{pct:.1f}\\%"


def _format_metric(value):
    if value is None or value == "N/A":
        return "N/A"
    return f"{float(value):.2f}"


def _short_model_name(name: str) -> str:
    for suffix in ("Model", "Watermark"):
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("_", "\\_").replace("&", "\\&").replace("#", "\\#")


def _metric_label(metric_name: str) -> str:
    return METRIC_LABELS.get(metric_name, metric_name.upper().replace("_", " "))


# ---------------------------------------------------------------------------
# No-attack tables
# ---------------------------------------------------------------------------

def _no_attack_reliability_table(result: Dict[str, Any]) -> str:
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
        label="tab:dr_no_attack",
    )


def _no_attack_quality_table(result: Dict[str, Any]) -> str:
    metrics = result.get("no_attack", {}).get("metrics")
    if not metrics:
        return ""

    rows = []
    for metric_name, value in metrics.items():
        rows.append(f"    {_metric_label(metric_name)} & {_format_metric(value)} \\\\")

    return build_longtable(
        col_spec="lc",
        header="Metric & Value",
        rows=rows,
        caption=(
            "Audio quality metrics for watermarked audio compared to the "
            "original (no attack applied)."
        ),
        label="tab:dr_no_attack_quality",
    )


# ---------------------------------------------------------------------------
# Per-attack tables (with attacks)
# ---------------------------------------------------------------------------

def _accuracy_fp_fn_table(attacks: Dict[str, Any], attack_names: List[str],
                          n_files: int, caption: str, label: str) -> str:
    """Accuracy + FP/FN for a list of attacks."""
    rows = []
    for name in attack_names:
        if name not in attacks:
            continue
        data = attacks[name]
        display = display_attack_name(name)
        acc = (
            "N/A"
            if data.get("accuracy_mean") is None
            else f"{data['accuracy_mean']:.2f}\\%"
        )
        fp = int(data.get("false_positive_count", 0))
        fp_n = int(data.get("false_positive_attempts", n_files))
        fn = int(data.get("false_negative_count", 0))
        fn_n = int(data.get("false_negative_attempts", n_files))
        rows.append(
            f"    {display} & {acc} & {_format_count(fp, fp_n)} & {_format_pct(fp, fp_n)} "
            f"& {_format_count(fn, fn_n)} & {_format_pct(fn, fn_n)} \\\\"
        )

    if not rows:
        return ""

    return build_longtable(
        col_spec="lccccc",
        header="Attack & Accuracy & FP Count & FP Rate & FN Count & FN Rate",
        rows=rows,
        caption=caption,
        label=label,
    )


def _always_on_table(attacks: Dict[str, Any], attack_names: List[str],
                     caption: str, label: str) -> str:
    """PESQ, ViSQOL, STOI table for a list of attacks."""
    rows = []
    for name in attack_names:
        if name not in attacks:
            continue
        data = attacks[name]
        display = display_attack_name(name)
        q = data.get("metrics") or {}
        cols = " & ".join(_format_metric(q.get(m)) for m in ALWAYS_ON_METRICS)
        rows.append(f"    {display} & {cols} \\\\")

    if not rows:
        return ""

    headers = " & ".join(_metric_label(m) for m in ALWAYS_ON_METRICS)
    return build_longtable(
        col_spec="l" + "c" * len(ALWAYS_ON_METRICS),
        header=f"Attack & {headers}",
        rows=rows,
        caption=caption,
        label=label,
    )


def _metrics_table(attacks: Dict[str, Any], attack_names: List[str],
                   metric_keys: List[str], caption: str, label: str) -> str:
    """Generic metrics table for a list of attacks and metric keys."""
    if not metric_keys:
        return ""

    rows = []
    for name in attack_names:
        if name not in attacks:
            continue
        data = attacks[name]
        display = display_attack_name(name)
        q = data.get("metrics") or {}
        cols = " & ".join(_format_metric(q.get(m)) for m in metric_keys)
        rows.append(f"    {display} & {cols} \\\\")

    if not rows:
        return ""

    headers = " & ".join(_metric_label(m) for m in metric_keys)
    return build_longtable(
        col_spec="l" + "c" * len(metric_keys),
        header=f"Attack & {headers}",
        rows=rows,
        caption=caption,
        label=label,
    )


# ---------------------------------------------------------------------------
# Group section builder
# ---------------------------------------------------------------------------

def _build_group_section(attacks: Dict[str, Any], attack_names: List[str],
                         group_key: str, group_label: str, n_files: int,
                         calculate_quality_metrics: bool) -> str:
    """Build a full section for one attack group."""
    present = [a for a in attack_names if a in attacks]
    if not present:
        return ""

    section = f"\\section{{{group_label}}}\n\n"

    # Accuracy + FP/FN table (always present)
    section += _accuracy_fp_fn_table(
        attacks, present, n_files,
        caption=f"Detection accuracy and reliability --- {group_label}.",
        label=f"tab:dr_acc_{group_key}",
    )
    section += "\n\n"

    if calculate_quality_metrics:
        group_def = ATTACK_GROUPS.get(group_key, {})
        q_metrics = group_def.get("quality_metrics", [])
        i_metrics = group_def.get("intelligibility_metrics", [])
        n_metrics = group_def.get("nisqa_metrics", [])

        # Quality metrics (without NISQA)
        q_no_nisqa = [m for m in q_metrics if m not in NISQA_METRICS]
        if q_no_nisqa:
            section += _metrics_table(
                attacks, present, q_no_nisqa,
                caption=f"Audio quality --- {group_label}.",
                label=f"tab:dr_qual_{group_key}",
            )
            section += "\n\n"

        # Intelligibility metrics
        if i_metrics:
            section += _metrics_table(
                attacks, present, i_metrics,
                caption=f"Speech intelligibility --- {group_label}.",
                label=f"tab:dr_intell_{group_key}",
            )
            section += "\n\n"

        # NISQA metrics
        if n_metrics:
            section += _metrics_table(
                attacks, present, n_metrics,
                caption=f"NISQA non-intrusive quality --- {group_label}.",
                label=f"tab:dr_nisqa_{group_key}",
            )
            section += "\n\n"
    else:
        # Always-on metrics only
        section += _always_on_table(
            attacks, present,
            caption=f"Audio quality (always-on) --- {group_label}.",
            label=f"tab:dr_ao_{group_key}",
        )
        section += "\n\n"

    return section


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_detection_reliability_report(
    result: Dict[str, Any], report_dir: str = "report",
) -> str:
    """Write the detection-reliability LaTeX report and compile to PDF."""
    os.makedirs(report_dir, exist_ok=True)
    has_cls = os.path.exists(os.path.join(report_dir, "deepmark.cls"))

    model_name = result.get("model_name", "DeepMark")
    short_name = _short_model_name(model_name)
    n_files = int(result.get("n_files", 0))
    has_attacks = bool(result.get("attacks"))
    has_quality_metrics = bool(
        result.get("no_attack", {}).get("metrics")
        or any(
            len(d.get("metrics", {})) > 3
            for d in (result.get("attacks") or {}).values()
        )
    )

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
        abstract += (
            f"Detection threshold: {threshold} (confidence-based model). "
        )
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

    sections = []

    if not has_attacks:
        # No attacks: show FP/FN + quality metrics
        sections.append(
            "\\section{No-Attack Reliability}\n\n"
            + _no_attack_reliability_table(result)
        )
        quality = _no_attack_quality_table(result)
        if quality:
            sections.append(
                "\\section{Watermarked Audio Quality}\n\n" + quality
            )
    else:
        # With attacks: grouped sections
        attacks = result.get("attacks", {})
        attack_names = list(attacks.keys())
        grouped = group_attacks(attack_names)

        for group_key in GROUP_ORDER:
            if group_key not in grouped:
                continue
            group_info = grouped[group_key]
            group_label = ATTACK_GROUPS.get(group_key, {}).get(
                "label", group_info["label"]
            )
            section = _build_group_section(
                attacks, group_info["attacks"], group_key, group_label,
                n_files, has_quality_metrics,
            )
            if section:
                sections.append(section)

        # Handle attacks not in any known group
        if "other" in grouped:
            other_attacks = grouped["other"]["attacks"]
            section = _build_group_section(
                attacks, other_attacks, "other", "Other Attacks",
                n_files, has_quality_metrics,
            )
            if section:
                sections.append(section)

    latex_content = (
        f"{preamble}\n\n"
        + abstract
        + "\n\n".join(sections)
        + "\n\n\\end{document}"
    )

    tex_path = os.path.join(report_dir, "detection_reliability_report.tex")
    with open(tex_path, "w") as f:
        f.write(latex_content)
    logger.info(f"Detection reliability LaTeX report saved to {tex_path}")

    compile_latex(report_dir, "detection_reliability_report")
    return tex_path
