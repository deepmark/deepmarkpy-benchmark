"""LaTeX report for the --detection_reliability mode.

The report is its own .tex/.pdf so the existing benchmark/no-attacks
reports stay untouched. Layout:

  Section 1 — No-Attack Reliability
    One-row-per-metric table: false positives on clean audio, false
    negatives on watermarked audio. Shows count and percentage.

  Section 2 — Per-Attack Accuracy & Quality (only if attacks were run)
    One row per attack: accuracy + the three always-on quality metrics
    (PESQ, ViSQOL, STOI), matching the basic benchmark report.

  Section 3 — Per-Attack Reliability (only if attacks were run)
    One row per attack: FP and FN with the attack applied to clean and
    watermarked audio respectively. Counts and percentages.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from utils.latex_helpers import (
    build_longtable,
    compile_latex,
    display_attack_name,
    make_preamble,
)

logger = logging.getLogger(__name__)


def _format_count(count: int, total: int) -> str:
    """Render an FP/FN count as ``count/total`` for the Count column."""
    if total <= 0:
        return "N/A"
    return f"{count}/{total}"


def _format_pct(count: int, total: int) -> str:
    """Render an FP/FN ratio as ``pct%`` for the Rate column."""
    if total <= 0:
        return "N/A"
    pct = 100.0 * count / total
    return f"{pct:.1f}\\%"


def _format_metric(value):
    return "N/A" if value is None else f"{float(value):.2f}"


def _short_model_name(name: str) -> str:
    for suffix in ("Model", "Watermark"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _no_attack_table(result: Dict[str, Any]) -> str:
    """Two-row table: false positives + false negatives without attack."""
    n = int(result["n_files"])
    fp = int(result["no_attack"]["false_positive_count"])
    fn = int(result["no_attack"]["false_negative_count"])

    rows = [
        f"    False Positive & {_format_count(fp, n)} & {_format_pct(fp, n)} \\\\",
        f"    False Negative & {_format_count(fn, n)} & {_format_pct(fn, n)} \\\\",
    ]
    header = "Metric & Count & Rate"
    caption = (
        "Detection reliability without attacks. "
        "False positive: detection on the clean original. "
        "False negative: missed detection on the watermarked signal."
    )
    return build_longtable(
        col_spec="lcc",
        header=header,
        rows=rows,
        caption=caption,
        label="tab:detection_reliability_no_attack",
    )


def _per_attack_quality_table(result: Dict[str, Any]) -> str:
    """One row per attack: accuracy + always-on quality metrics."""
    attacks = result.get("attacks") or {}
    if not attacks:
        return ""

    rows = []
    for attack_name in sorted(attacks.keys()):
        data = attacks[attack_name]
        display = display_attack_name(attack_name)
        acc = (
            "N/A"
            if data.get("accuracy_mean") is None
            else f"{data['accuracy_mean']:.2f}\\%"
        )
        q = data.get("always_on_metrics") or {}
        rows.append(
            f"    {display} & {acc} & {_format_metric(q.get('pesq'))} "
            f"& {_format_metric(q.get('visqol'))} "
            f"& {_format_metric(q.get('stoi'))} \\\\"
        )

    header = "Attack & Accuracy & PESQ & ViSQOL & STOI"
    caption = (
        "Watermark detection accuracy and always-on audio quality "
        "metrics per attack. Accuracy is the share of files where the "
        "watermark survived the attack."
    )
    return build_longtable(
        col_spec="lcccc",
        header=header,
        rows=rows,
        caption=caption,
        label="tab:detection_reliability_per_attack_quality",
    )


def _per_attack_reliability_table(result: Dict[str, Any]) -> str:
    """One row per attack: FP and FN counts with attack applied."""
    attacks = result.get("attacks") or {}
    if not attacks:
        return ""

    n = int(result["n_files"])
    rows = []
    for attack_name in sorted(attacks.keys()):
        data = attacks[attack_name]
        display = display_attack_name(attack_name)
        fp = int(data.get("false_positive_count", 0))
        fn = int(data.get("false_negative_count", 0))
        rows.append(
            f"    {display} & {_format_count(fp, n)} & {_format_pct(fp, n)} "
            f"& {_format_count(fn, n)} & {_format_pct(fn, n)} \\\\"
        )

    header = "Attack & FP Count & FP Rate & FN Count & FN Rate"
    caption = (
        "Detection reliability with attacks. False positive: detection "
        "after applying the attack to the clean original. False "
        "negative: missed detection after applying the attack to the "
        "watermarked signal."
    )
    return build_longtable(
        col_spec="lcccc",
        header=header,
        rows=rows,
        caption=caption,
        label="tab:detection_reliability_per_attack",
    )


def generate_detection_reliability_report(
    result: Dict[str, Any], report_dir: str = "report",
) -> str:
    """Write the detection-reliability LaTeX report and compile to PDF."""
    os.makedirs(report_dir, exist_ok=True)
    has_cls = os.path.exists(os.path.join(report_dir, "deepmark.cls"))

    model_name = result.get("model_name", "DeepMark")
    short_name = _short_model_name(model_name)
    n_files = int(result.get("n_files", 0))

    preamble = make_preamble(
        title=f"Detection Reliability Report: {short_name}",
        author="DeepMark Benchmark System",
        has_deepmark_cls=has_cls,
    )

    has_attacks = bool(result.get("attacks"))

    abstract = (
        f"\\begin{{abstract}}\n"
        f"This report measures detection reliability for the "
        f"{short_name} watermarking model across {n_files} "
        f"{'file' if n_files == 1 else 'files'}. False positives are "
        f"detections on the clean (non-watermarked) input; false "
        f"negatives are missed detections on the watermarked input. "
    )
    if has_attacks:
        abstract += (
            "When attacks are configured, the same two metrics are also "
            "reported with each attack applied -- to the clean original "
            "for the false-positive measurement, and to the watermarked "
            "signal for the false-negative measurement.\n"
        )
    else:
        abstract += "\n"
    abstract += "\\end{abstract}\n\n"

    sections = [
        "\\section{No-Attack Reliability}\n\n"
        + _no_attack_table(result)
    ]

    if has_attacks:
        per_attack_quality = _per_attack_quality_table(result)
        if per_attack_quality:
            sections.append(
                "\\section{Per-Attack Accuracy and Audio Quality}\n\n"
                + per_attack_quality
            )

        per_attack_reliability = _per_attack_reliability_table(result)
        if per_attack_reliability:
            sections.append(
                "\\section{Per-Attack Reliability}\n\n"
                + per_attack_reliability
            )

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
