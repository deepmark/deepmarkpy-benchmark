"""Report generator for --no_attacks mode.

Produces a LaTeX report with comparative tables showing baseline
embed+detect fidelity across one or more models. Models are split
into zero-bit and multi-bit groups, each with appropriate columns:

Multi-bit:  Model | Mean Accuracy (%) | Mean Confidence
Zero-bit:   Model | Mean Accuracy (%) | Positive Detections | Mean Confidence

When the benchmark was run with ``--calculate_quality_metrics``, three
extra tables are appended (one row per model): Audio quality, Speech
intelligibility, and NISQA. They mirror the per-group layout used by
the detailed report so the no-attacks report stays visually consistent.
"""

import logging
import os

import numpy as np

from utils.latex_helpers import compile_latex, make_preamble
from utils.metrics import (
    INTELLIGIBILITY_METRICS,
    METRIC_LABELS,
    NISQA_METRICS,
    QUALITY_METRICS,
)

logger = logging.getLogger(__name__)

# Quality metrics minus NISQA -- NISQA gets its own table to keep the
# audio-quality table narrow enough to fit the page.
_QUALITY_AUDIO_METRICS = [m for m in QUALITY_METRICS if m not in NISQA_METRICS]


def _summarize_model(results):
    """Compute summary stats for a single model's no-attacks results."""
    is_zero_bit = results.get("is_zero_bit", False)
    returns_confidence = results.get("returns_confidence", False)
    files = results.get("files", [])
    n_files = len(files)

    accuracies = [f["accuracy"] for f in files if f.get("accuracy") is not None]
    mean_accuracy = float(np.mean(accuracies)) if accuracies else 0.0

    summary = {
        "is_zero_bit": is_zero_bit,
        "returns_confidence": returns_confidence,
        "n_files": n_files,
        "mean_accuracy": mean_accuracy,
    }

    if is_zero_bit:
        detected_count = sum(
            1 for f in files
            if f.get("accuracy") is not None and float(f["accuracy"]) > 50.0
        )
        summary["positive_detections"] = detected_count

    if returns_confidence:
        confidences = [
            f["confidence"] for f in files if f.get("confidence") is not None
        ]
        if confidences:
            summary["mean_confidence"] = float(np.mean(confidences))

    # Aggregate per-file quality metrics into a single mean per metric
    # when --calculate_quality_metrics produced them; otherwise leave the
    # field absent so the report falls back to the original layout.
    quality_files = [f.get("watermarked_audio_quality") for f in files]
    quality_files = [q for q in quality_files if q]
    if quality_files:
        all_keys = set()
        for q in quality_files:
            all_keys.update(q.keys())
        means = {}
        for key in all_keys:
            vals = [q.get(key) for q in quality_files]
            vals = [v for v in vals if v is not None]
            if vals:
                means[key] = float(np.mean(vals))
        if means:
            summary["quality_metrics"] = means

    return summary


def _short_model_name(name):
    """Strip common suffixes for display."""
    for suffix in ("Model", "Watermark"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _build_multibit_table(models_data):
    """Build LaTeX table for multi-bit models."""
    has_confidence = any(
        "mean_confidence" in data for data in models_data.values()
    )

    if has_confidence:
        col_spec = "lcc"
        header = "    Model & Mean Accuracy & Mean Confidence \\\\"
    else:
        col_spec = "lc"
        header = "    Model & Mean Accuracy \\\\"

    rows = []
    for model_name, data in models_data.items():
        display = _short_model_name(model_name)
        acc = data["mean_accuracy"]
        row = f"    {display} & {acc:.2f}\\%"
        if has_confidence:
            conf = data.get("mean_confidence")
            row += f" & {conf:.4f}" if conf is not None else " & N/A"
        row += " \\\\"
        rows.append(row)

    return (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Baseline detection performance -- multi-bit models.}\n"
        "\\label{tab:no_attacks_multibit}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "    \\toprule\n"
        f"{header}\n"
        "    \\midrule\n"
        + "\n".join(rows) + "\n"
        "    \\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def _build_zerobit_table(models_data):
    """Build LaTeX table for zero-bit models."""
    has_confidence = any(
        "mean_confidence" in data for data in models_data.values()
    )

    if has_confidence:
        col_spec = "lccc"
        header = "    Model & Mean Accuracy & Positive Detections & Mean Confidence \\\\"
    else:
        col_spec = "lcc"
        header = "    Model & Mean Accuracy & Positive Detections \\\\"

    rows = []
    for model_name, data in models_data.items():
        display = _short_model_name(model_name)
        acc = data["mean_accuracy"]
        pos = data.get("positive_detections", 0)
        n = data["n_files"]
        row = f"    {display} & {acc:.2f}\\% & {pos}/{n}"
        if has_confidence:
            conf = data.get("mean_confidence")
            row += f" & {conf:.4f}" if conf is not None else " & N/A"
        row += " \\\\"
        rows.append(row)

    return (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Baseline detection performance -- zero-bit models.}\n"
        "\\label{tab:no_attacks_zerobit}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "    \\toprule\n"
        f"{header}\n"
        "    \\midrule\n"
        + "\n".join(rows) + "\n"
        "    \\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def _format_metric_value(metric, value):
    """Format a metric value for the LaTeX table; ``N/A`` if missing."""
    if value is None:
        return "N/A"
    if metric in ("stoi", "sii", "ncm"):
        return f"{value:.4f}"
    return f"{value:.2f}"


def _build_metric_table(models_data, metric_keys, caption, label):
    """Build a one-row-per-model table for the given metric subset.

    Returns "" if no model has any of the requested metrics, so callers
    can drop empty tables without an empty section header.
    """
    rows_data = []
    for model_name, data in models_data.items():
        q = data.get("quality_metrics") or {}
        if any(k in q for k in metric_keys):
            rows_data.append((model_name, q))
    if not rows_data:
        return ""

    col_spec = "l" + "c" * len(metric_keys)
    header_cells = " & ".join(METRIC_LABELS.get(m, m) for m in metric_keys)
    header = f"    Model & {header_cells} \\\\"

    rows = []
    for model_name, q in rows_data:
        display = _short_model_name(model_name)
        cells = " & ".join(
            _format_metric_value(m, q.get(m)) for m in metric_keys
        )
        rows.append(f"    {display} & {cells} \\\\")

    return (
        "\\begin{table}[H]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "    \\toprule\n"
        f"{header}\n"
        "    \\midrule\n"
        + "\n".join(rows) + "\n"
        "    \\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def generate_no_attacks_report(all_results, report_dir="report"):
    """Generate a baseline fidelity report for one or more models.

    Args:
        all_results: Dict of {model_name: results_from_run_no_attacks}.
        report_dir: Directory for output files.

    Returns:
        Path to the generated .tex file.
    """
    os.makedirs(report_dir, exist_ok=True)
    has_cls = os.path.exists(os.path.join(report_dir, "deepmark.cls"))

    summaries = {
        model: _summarize_model(res) for model, res in all_results.items()
    }

    zero_bit = {m: s for m, s in summaries.items() if s["is_zero_bit"]}
    multi_bit = {m: s for m, s in summaries.items() if not s["is_zero_bit"]}

    n_models = len(summaries)
    model_word = "model" if n_models == 1 else "models"
    n_files = next(iter(summaries.values()))["n_files"] if summaries else 0

    preamble = make_preamble(
        title="Baseline Fidelity Report",
        author="DeepMark Benchmark System",
        has_deepmark_cls=has_cls,
    )

    tables = []
    if multi_bit:
        tables.append(_build_multibit_table(multi_bit))
    if zero_bit:
        tables.append(_build_zerobit_table(zero_bit))

    # Quality metric tables only appear when --calculate_quality_metrics
    # populated them. They mirror the per-group layout of the detailed
    # report (audio quality + intelligibility + NISQA, separate tables)
    # so the no-attacks report stays visually consistent.
    has_quality = any("quality_metrics" in s for s in summaries.values())
    quality_section = ""
    if has_quality:
        quality_tables = []
        audio_table = _build_metric_table(
            summaries, _QUALITY_AUDIO_METRICS,
            "Audio quality of the watermarked signal (no attack).",
            "tab:no_attacks_quality",
        )
        if audio_table:
            quality_tables.append(audio_table)
        intel_table = _build_metric_table(
            summaries, list(INTELLIGIBILITY_METRICS),
            "Speech intelligibility of the watermarked signal (no attack).",
            "tab:no_attacks_intelligibility",
        )
        if intel_table:
            quality_tables.append(intel_table)
        nisqa_table = _build_metric_table(
            summaries, list(NISQA_METRICS),
            "NISQA non-intrusive quality dimensions of the "
            "watermarked signal (no attack).",
            "tab:no_attacks_nisqa",
        )
        if nisqa_table:
            quality_tables.append(nisqa_table)
        if quality_tables:
            quality_section = (
                "\n\n\\section{Watermark Audio Quality (No Attack)}\n\n"
                + "\n\n".join(quality_tables)
            )

    latex_content = (
        f"{preamble}\n\n"
        f"\\begin{{abstract}}\n"
        f"This report evaluates the baseline detection fidelity of "
        f"{n_models} watermarking {model_word}. The watermark is "
        f"embedded and immediately detected without any intermediate attacks, "
        f"measuring each model's inherent accuracy across {n_files} "
        f"{'file' if n_files == 1 else 'files'}.\n"
        f"\\end{{abstract}}\n\n"
        f"\\section{{Baseline Detection Performance}}\n\n"
        + "\n\n".join(tables)
        + quality_section
        + "\n\n\\end{document}"
    )

    tex_path = os.path.join(report_dir, "no_attacks_report.tex")
    with open(tex_path, "w") as f:
        f.write(latex_content)
    logger.info(f"No-attacks LaTeX report saved to {tex_path}")

    compile_latex(report_dir, "no_attacks_report")
    return tex_path
