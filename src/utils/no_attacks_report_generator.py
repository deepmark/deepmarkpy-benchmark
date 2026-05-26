"""Report generator for --no_attacks mode.

Produces a LaTeX report with comparative tables showing baseline
embed+detect fidelity across one or more models. Models are split
into zero-bit and multi-bit groups, each with appropriate columns:

Multi-bit:  Model | Mean Accuracy (%) | Mean Confidence
Zero-bit:   Model | Mean Accuracy (%) | Positive Detections | Mean Confidence
"""

import logging
import os

import numpy as np

from utils.latex_helpers import compile_latex, make_preamble

logger = logging.getLogger(__name__)


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
        + "\n\n".join(tables) + "\n\n"
        f"\\end{{document}}"
    )

    tex_path = os.path.join(report_dir, "no_attacks_report.tex")
    with open(tex_path, "w") as f:
        f.write(latex_content)
    logger.info(f"No-attacks LaTeX report saved to {tex_path}")

    compile_latex(report_dir, "no_attacks_report")
    return tex_path
