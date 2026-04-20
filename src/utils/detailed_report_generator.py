import json
import os
import logging
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from utils.attack_groups import group_attacks, ATTACK_GROUPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-group metric relevance
# ---------------------------------------------------------------------------

GROUP_QUALITY_METRICS = {
    "process_disruption": ["pesq", "psnr", "si_sdr", "mcd", "visqol"],
    "audio_distortion": ["pesq", "psnr", "si_sdr", "visqol"],
    "desynchronization": ["mcd", "visqol"],
    "ai_attacks": ["mcd", "pesq", "visqol"],
    "transmission": ["pesq", "psnr", "si_sdr", "mcd", "visqol"],
}

GROUP_INTELLIGIBILITY_METRICS = {
    "process_disruption": [],
    "audio_distortion": ["stoi", "sii", "ncm"],
    "desynchronization": [],
    "ai_attacks": ["stoi", "sii", "ncm"],
    "transmission": ["stoi", "sii", "ncm"],
}

# Audio Editing is handled via sub-groups instead of top-level metrics
AUDIO_EDITING_SUBGROUPS = {
    "frequency_filtering": {
        "label": "Frequency Filtering",
        "attacks": [
            "LowpassFilterAttack",
            "HighpassFilterAttack",
            "BandstopFilterAttack",
            "EqualizerAttack",
        ],
        "quality_metrics": ["mcd", "visqol", "pesq"],
        "intelligibility_metrics": ["stoi", "sii", "ncm"],
        "description": (
            "Frequency-domain modifications that selectively attenuate "
            "or boost spectral content."
        ),
    },
    "temporal_editing": {
        "label": "Temporal Editing",
        "attacks": [
            "CutSamplesAttack",
            "CropBeginningAttack",
            "CropRandomAttack",
        ],
        "quality_metrics": [],
        "intelligibility_metrics": [],
        "description": (
            "Attacks that modify the temporal structure of the audio signal "
            "by removing or rearranging samples. Quality and intelligibility "
            "metrics are not reported for these attacks, as changes in signal "
            "length make direct metric comparison unreliable."
        ),
    },
    "audio_effects": {
        "label": "Audio Effects",
        "attacks": [
            "WaveletAttack",
            "SmoothingAttack",
            "ChorusAttack",
            "FlangerAttack",
            "EchoAttack",
            "MixingAttack",
        ],
        "quality_metrics": ["pesq", "visqol", "si_sdr", "mcd"],
        "intelligibility_metrics": [],
        "description": (
            "Common audio processing effects that alter signal characteristics "
            "while preserving perceptual quality."
        ),
    },
    "compression_quantization": {
        "label": "Compression \\& Quantization",
        "attacks": [
            "QuantizationAttack",
            "STFTQuantizationAttack",
            "PCMQuantizationAttack",
            "Mp3CompressionAttack",
            "EncodecAttack",
            "DescriptAudioCodecAttack",
            "ResamplingPolyAttack",
        ],
        "quality_metrics": ["pesq", "visqol", "psnr", "mcd"],
        "intelligibility_metrics": ["stoi", "sii"],
        "description": (
            "Lossy compression and bit-depth reduction operations commonly "
            "encountered in audio distribution pipelines."
        ),
    },
}

GROUP_DESCRIPTIONS = {
    "process_disruption": (
        "These attacks attempt to disrupt the watermarking process itself, "
        "including cross-model interference, collusion between multiple "
        "watermarked copies, and same-model re-watermarking."
    ),
    "audio_editing": (
        "Audio editing attacks simulate common audio processing operations "
        "that may be applied to watermarked content, ranging from filtering "
        "and effects to compression and temporal modifications."
    ),
    "audio_distortion": (
        "These attacks introduce various forms of noise and signal distortion, "
        "testing the watermark's resilience to additive interference and "
        "signal corruption."
    ),
    "desynchronization": (
        "Desynchronization attacks alter the temporal alignment of the audio "
        "signal through time-scaling, pitch shifting, and sample-level "
        "manipulations."
    ),
    "ai_attacks": (
        "AI-based attacks leverage neural networks and machine learning models "
        "to process the watermarked audio, potentially removing or degrading "
        "the embedded watermark."
    ),
    "transmission": (
        "These attacks simulate real-world audio transmission scenarios, "
        "including acoustic replay through speakers/microphones and "
        "network-based audio transmission."
    ),
}

GROUP_ORDER = [
    "process_disruption",
    "audio_editing",
    "audio_distortion",
    "desynchronization",
    "ai_attacks",
    "transmission",
]


class DetailedReportGenerator:
    """Generate detailed LaTeX reports with full quality metrics analysis."""

    BASE_QUALITY_METRICS = ["pesq", "psnr", "si_sdr", "mcd", "visqol"]
    INTELLIGIBILITY_METRICS = ["stoi", "sii", "ncm"]
    ALL_METRIC_LABELS = {
        "pesq": "PESQ (1--4.5)",
        "psnr": "PSNR (dB)",
        "si_sdr": "SI-SDR (dB)",
        "mcd": "MCD (dB)",
        "visqol": "ViSQOL (1--5)",
        "stoi": "STOI (0--1)",
        "sii": "SII (0--1)",
        "ncm": "NCM (0--1)",
    }

    def __init__(self, report_dir="report"):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self._has_deepmark_cls = os.path.exists(
            os.path.join(self.report_dir, "deepmark.cls")
        )

    # ------------------------------------------------------------------
    # LaTeX helpers
    # ------------------------------------------------------------------

    def _preamble(self, title, author):
        """Generate LaTeX preamble with deepmark class fallback."""
        if self._has_deepmark_cls:
            return (
                f"\\documentclass{{deepmark}}\n"
                f"\\usepackage{{float}}\n"
                f"\\usepackage{{longtable}}\n"
                f"\\usepackage{{needspace}}\n\n"
                f"\\title{{{title}}}\n"
                f"\\author{{{author}}}\n\n"
                f"\\begin{{document}}\n"
                f"\\thispagestyle{{firststyle}}\n"
                f"\\maketitle"
            )
        return (
            f"\\documentclass{{article}}\n"
            f"\\usepackage{{booktabs}}\n"
            f"\\usepackage{{graphicx}}\n"
            f"\\usepackage{{amsmath}}\n"
            f"\\usepackage{{cleveref}}\n"
            f"\\usepackage{{float}}\n"
            f"\\usepackage{{longtable}}\n"
            f"\\usepackage{{needspace}}\n"
            f"\\usepackage{{geometry}}\n"
            f"\\geometry{{margin=2.5cm}}\n\n"
            f"\\title{{{title}}}\n"
            f"\\author{{{author}}}\n"
            f"\\date{{\\today}}\n\n"
            f"\\begin{{document}}\n"
            f"\\maketitle"
        )

    def _format_val(self, stats):
        """Format mean value for display."""
        if stats is None or stats.get("mean") is None:
            return "N/A"
        return f"{stats['mean']:.2f}"

    def _display_name(self, attack_name):
        """Convert attack class name to readable display name."""
        return attack_name.replace("Attack", "")

    # ------------------------------------------------------------------
    # Data aggregation
    # ------------------------------------------------------------------

    def aggregate_results(self, results):
        """
        Aggregate per-file results into per-attack means.

        Args:
            results: Raw benchmark results dict (per file, per attack)

        Returns:
            dict with structure:
            {
                "quality_metrics": [...],
                "intelligibility_metrics": [...],
                "watermarked_audio_quality": {metric: {"mean": ...}},
                "watermark_intelligibility": {metric: {"mean": ...}},
                "attacks": {
                    attack_name: {
                        "accuracy": {"mean": ...},
                        "attacked_audio_quality_wm": {metric: {"mean": ...}},
                        "attack_intelligibility": {metric: {"mean": ...}},
                    }
                }
            }
        """
        quality_metrics = list(self.BASE_QUALITY_METRICS)
        intelligibility_metrics = list(self.INTELLIGIBILITY_METRICS)
        all_metrics = quality_metrics + intelligibility_metrics

        watermark_values = {m: [] for m in all_metrics}
        attack_data = {}

        for filepath, attacks in results.items():
            for attack_name, data in attacks.items():
                if attack_name not in attack_data:
                    attack_data[attack_name] = {
                        "accuracy": [],
                        "confidence": [],
                        "metrics": {m: [] for m in all_metrics},
                    }

                attack_data[attack_name]["accuracy"].append(data["accuracy"])

                if "confidence" in data:
                    attack_data[attack_name]["confidence"].append(data["confidence"])

                # Watermark quality (same for all attacks within a file)
                if data.get("watermarked_audio_quality"):
                    for m in all_metrics:
                        val = data["watermarked_audio_quality"].get(m)
                        if val is not None:
                            watermark_values[m].append(val)

                if data.get("attacked_audio_quality_wm"):
                    for m in all_metrics:
                        val = data["attacked_audio_quality_wm"].get(m)
                        if val is not None:
                            attack_data[attack_name]["metrics"][m].append(val)

        def mean_std(values):
            if not values:
                return {"mean": None, "std": None}
            return {"mean": float(np.mean(values)), "std": float(np.std(values))}

        aggregated = {
            "quality_metrics": quality_metrics,
            "intelligibility_metrics": intelligibility_metrics,
            "watermarked_audio_quality": {
                m: mean_std(watermark_values[m]) for m in quality_metrics
            },
            "watermark_intelligibility": {
                m: mean_std(watermark_values[m]) for m in intelligibility_metrics
            },
            "attacks": {},
        }

        for attack_name, data in attack_data.items():
            entry = {
                "accuracy": mean_std(data["accuracy"]),
                "attacked_audio_quality_wm": {
                    m: mean_std(data["metrics"][m]) for m in quality_metrics
                },
                "attack_intelligibility": {
                    m: mean_std(data["metrics"][m]) for m in intelligibility_metrics
                },
            }
            if data["confidence"]:
                entry["confidence"] = mean_std(data["confidence"])

            aggregated["attacks"][attack_name] = entry

        return aggregated

    # ------------------------------------------------------------------
    # Generic table builders
    # ------------------------------------------------------------------

    def _accuracy_table(self, aggregated, attacks, caption, label):
        """Generate accuracy longtable for a subset of attacks."""
        available = sorted([a for a in attacks if a in aggregated["attacks"]])
        if not available:
            return ""

        has_confidence = any(
            "confidence" in aggregated["attacks"][a] for a in available
        )

        rows = []
        for a in available:
            data = aggregated["attacks"][a]
            display = self._display_name(a)
            acc = self._format_val(data["accuracy"])
            if has_confidence:
                conf = (
                    self._format_val(data["confidence"])
                    if "confidence" in data
                    else "---"
                )
                rows.append(f"    {display} & {acc} & {conf} \\\\")
            else:
                rows.append(f"    {display} & {acc} \\\\")

        if has_confidence:
            col_spec = "lcc"
            header = "Attack & Accuracy (\\%) & Confidence"
        else:
            col_spec = "lc"
            header = "Attack & Accuracy (\\%)"

        return (
            f"\\begin{{longtable}}{{{col_spec}}}\n"
            f"    \\caption{{{caption}}}\n"
            f"    \\label{{{label}}} \\\\\n"
            f"    \\toprule\n"
            f"    {header} \\\\\n"
            f"    \\midrule\n"
            f"    \\endfirsthead\n"
            f"    \\toprule\n"
            f"    {header} \\\\\n"
            f"    \\midrule\n"
            f"    \\endhead\n"
            f"    \\bottomrule\n"
            f"    \\endlastfoot\n"
            + "\n".join(rows) + "\n"
            "\\end{longtable}"
        )

    def _metrics_table(self, aggregated, attacks, metrics, data_key,
                       caption, label, include_baseline=True,
                       baseline_data=None):
        """Generate metrics longtable for a subset of attacks and metrics.

        Args:
            aggregated: Aggregated results
            attacks: List of attack names to include
            metrics: List of metric keys to include
            data_key: Key in attack data ('attacked_audio_quality_wm' or
                      'attack_intelligibility')
            caption: Table caption
            label: Table label
            include_baseline: Whether to include watermark-only baseline row
            baseline_data: Baseline data dict (e.g. watermarked_audio_quality)
        """
        if not metrics:
            return ""

        available = sorted([a for a in attacks if a in aggregated["attacks"]])
        if not available:
            return ""

        headers = [self.ALL_METRIC_LABELS[m] for m in metrics]
        header_str = " & ".join(headers)
        col_spec = "l" + "c" * len(metrics)

        rows = []
        if include_baseline and baseline_data:
            baseline_cols = " & ".join(
                self._format_val(baseline_data.get(m, {"mean": None}))
                for m in metrics
            )
            rows.append(
                f"    No Attack (watermark only) & {baseline_cols} \\\\"
            )
            rows.append("    \\midrule")

        for a in available:
            data = aggregated["attacks"][a][data_key]
            display = self._display_name(a)
            cols = " & ".join(
                self._format_val(data.get(m, {"mean": None}))
                for m in metrics
            )
            rows.append(f"    {display} & {cols} \\\\")

        return (
            f"\\begin{{longtable}}{{{col_spec}}}\n"
            f"    \\caption{{{caption}}}\n"
            f"    \\label{{{label}}} \\\\\n"
            f"    \\toprule\n"
            f"    Condition & {header_str} \\\\\n"
            f"    \\midrule\n"
            f"    \\endfirsthead\n"
            f"    \\toprule\n"
            f"    Condition & {header_str} \\\\\n"
            f"    \\midrule\n"
            f"    \\endhead\n"
            f"    \\bottomrule\n"
            f"    \\endlastfoot\n"
            + "\n".join(rows) + "\n"
            "\\end{longtable}"
        )

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def create_radar_chart(self, aggregated, output_path, model_name="DeepMark"):
        """Create radar chart of accuracy per attack with legend."""
        attacks = sorted(aggregated["attacks"].keys())
        accuracies = []
        codes = []
        legend_entries = []

        for i, a in enumerate(attacks):
            acc = aggregated["attacks"][a]["accuracy"]["mean"]
            accuracies.append(acc if acc is not None else 0)
            code = f"A{i + 1}"
            codes.append(code)
            legend_entries.append(f"{code}  --  {self._display_name(a)}")

        if not accuracies:
            logger.warning("No data available for radar chart, skipping.")
            return

        n = len(accuracies)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        accuracies_closed = accuracies + [accuracies[0]]
        angles_closed = angles + [angles[0]]

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor("white")

        ax = fig.add_axes([0.05, 0.08, 0.55, 0.82], polar=True)
        ax.set_facecolor("white")
        ax.plot(angles_closed, accuracies_closed, color="#469CA9",
                linewidth=2, alpha=0.85)
        ax.fill(angles_closed, accuracies_closed, color="#469CA9", alpha=0.15)

        ax.set_xticks(angles)
        ax.set_xticklabels(codes, fontsize=9, color="#333333")
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"],
                           fontsize=8, color="#777777")
        ax.set_rlabel_position(0)
        ax.grid(color="#dddddd", linewidth=0.5)
        ax.set_title(
            "Watermark Detection Accuracy by Attack Type",
            fontsize=14, fontweight="700", pad=20, color="#2c3e50",
        )

        legend_ax = fig.add_axes([0.63, 0.08, 0.35, 0.82])
        legend_ax.axis("off")
        legend_ax.set_title("Attack Legend", fontsize=12, fontweight="700",
                            color="#2c3e50", loc="left", pad=10)

        col_size = (n + 1) // 2
        for i, entry in enumerate(legend_entries):
            col = i // col_size
            row = i % col_size
            x = 0.0 + col * 0.5
            y = 1.0 - (row + 1) * (1.0 / (col_size + 1))
            legend_ax.text(x, y, entry, fontsize=8, color="#555555",
                           transform=legend_ax.transAxes,
                           verticalalignment="center")

        plt.savefig(output_path, dpi=300, bbox_inches="tight",
                    facecolor="white")
        plt.close()
        logger.info(f"Radar chart saved to {output_path}")

    # ------------------------------------------------------------------
    # Report section builders
    # ------------------------------------------------------------------

    def _generate_audio_editing_section(self, aggregated, group_attack_list):
        """Generate the Audio Editing section with sub-group subsections."""
        sections = ""

        # Sub-sections
        for sub_key, sub_info in AUDIO_EDITING_SUBGROUPS.items():
            sub_attacks = [
                a for a in sub_info["attacks"]
                if a in aggregated["attacks"]
            ]
            if not sub_attacks:
                continue

            sections += "\\needspace{5\\baselineskip}\n"
            sections += f"\\subsection{{{sub_info['label']}}}\n\n"
            sections += f"{sub_info['description']}\n\n"

            q_metrics = sub_info.get("quality_metrics", [])
            i_metrics = sub_info.get("intelligibility_metrics", [])

            if q_metrics:
                sections += self._metrics_table(
                    aggregated, sub_attacks, q_metrics,
                    "attacked_audio_quality_wm",
                    f"Audio quality --- {sub_info['label']}.",
                    f"tab:qual_{sub_key}",
                    include_baseline=True,
                    baseline_data=aggregated["watermarked_audio_quality"],
                )
                sections += "\n\n"

            if i_metrics:
                sections += self._metrics_table(
                    aggregated, sub_attacks, i_metrics,
                    "attack_intelligibility",
                    f"Speech intelligibility --- {sub_info['label']}.",
                    f"tab:intell_{sub_key}",
                    include_baseline=True,
                    baseline_data=aggregated["watermark_intelligibility"],
                )
                sections += "\n\n"

        return sections

    def _generate_grouped_sections(self, aggregated):
        """Generate per-group sections with relevant metrics."""
        attack_names = list(aggregated["attacks"].keys())
        grouped = group_attacks(attack_names)

        sections = ""
        for group_key in GROUP_ORDER:
            if group_key not in grouped:
                continue

            group_info = grouped[group_key]
            group_label = ATTACK_GROUPS.get(group_key, {}).get(
                "label", group_info["label"]
            )
            group_attack_list = group_info["attacks"]
            description = GROUP_DESCRIPTIONS.get(group_key, "")

            sections += "\\needspace{5\\baselineskip}\n"
            sections += f"\\section{{{group_label}}}\n\n"
            if description:
                sections += f"{description}\n\n"

            if group_key == "audio_editing":
                sections += self._generate_audio_editing_section(
                    aggregated, group_attack_list
                )
            else:
                # Quality and intelligibility metrics for this group
                q_metrics = GROUP_QUALITY_METRICS.get(group_key, [])
                if q_metrics:
                    sections += self._metrics_table(
                        aggregated, group_attack_list, q_metrics,
                        "attacked_audio_quality_wm",
                        f"Audio quality --- {group_label}.",
                        f"tab:qual_{group_key}",
                        include_baseline=True,
                        baseline_data=aggregated["watermarked_audio_quality"],
                    )
                    sections += "\n\n"

                i_metrics = GROUP_INTELLIGIBILITY_METRICS.get(group_key, [])
                if i_metrics:
                    sections += self._metrics_table(
                        aggregated, group_attack_list, i_metrics,
                        "attack_intelligibility",
                        f"Speech intelligibility --- {group_label}.",
                        f"tab:intell_{group_key}",
                        include_baseline=True,
                        baseline_data=aggregated[
                            "watermark_intelligibility"
                        ],
                    )
                    sections += "\n\n"

        # Handle attacks not in any known group
        if "other" in grouped:
            other_attacks = grouped["other"]["attacks"]
            sections += "\\needspace{5\\baselineskip}\n"
            sections += "\\section{Other Attacks}\n\n"
            sections += self._metrics_table(
                aggregated, other_attacks, self.BASE_QUALITY_METRICS,
                "attacked_audio_quality_wm",
                "Audio quality --- other attacks.",
                "tab:qual_other",
                include_baseline=True,
                baseline_data=aggregated["watermarked_audio_quality"],
            )
            sections += "\n\n"
            sections += self._metrics_table(
                aggregated, other_attacks, self.INTELLIGIBILITY_METRICS,
                "attack_intelligibility",
                "Speech intelligibility --- other attacks.",
                "tab:intell_other",
                include_baseline=True,
                baseline_data=aggregated["watermark_intelligibility"],
            )
            sections += "\n\n"

        return sections

    # ------------------------------------------------------------------
    # Main report assembly
    # ------------------------------------------------------------------

    def generate_latex_report(self, aggregated, model_name="DeepMark"):
        """Generate complete LaTeX document.

        Args:
            aggregated: Aggregated results from aggregate_results()
            model_name: Name of the watermarking model
        """
        num_attacks = len(aggregated["attacks"])
        all_attacks = list(aggregated["attacks"].keys())

        preamble = self._preamble(
            f"Detailed Benchmark Report: {model_name}",
            "DeepMark Benchmark System",
        )

        abstract = (
            f"\\begin{{abstract}}\n"
            f"This report presents a detailed evaluation of the "
            f"{model_name} watermarking model across {num_attacks} attack "
            f"types. It covers watermark detection robustness, audio quality "
            f"impact analysis, and speech intelligibility measures, comparing "
            f"the effects of watermark embedding and adversarial attacks on "
            f"the audio signal.\n"
            f"\\end{{abstract}}\n"
        )

        # Overall accuracy table + bar chart at the top
        robustness = "\\section{Watermark Detection Robustness}\n\n"
        robustness += (
            "The following table reports the mean detection accuracy "
            "per attack type.\n\n"
        )
        robustness += self._accuracy_table(
            aggregated, all_attacks,
            "Watermark detection robustness per attack type.",
            "tab:robustness",
        )
        robustness += "\n\n"

        chart_path = os.path.join(self.report_dir, "benchmark_chart.png")
        if os.path.exists(chart_path):
            robustness += (
                "\\begin{figure}[H]\n"
                "    \\centering\n"
                "    \\includegraphics[width=\\linewidth]"
                "{benchmark_chart.png}\n"
                "    \\caption{Watermark detection accuracy by attack "
                "type.}\n"
                "    \\label{fig:accuracy_chart}\n"
                "\\end{figure}\n\n"
            )

        # Per-group metric sections (only groups with attacks are shown)
        body = self._generate_grouped_sections(aggregated)

        return (
            f"{preamble}\n\n"
            f"{abstract}\n"
            f"{robustness}"
            f"{body}"
            f"\\end{{document}}"
        )

    def generate_full_report(self, results, model_name="DeepMark",
                             total_attacks=None):
        """
        Generate complete detailed report from raw benchmark results.

        Args:
            results: Raw benchmark results dict from Benchmark.run()
            model_name: Name of the watermarking model
            total_attacks: Total number of available attacks (unused,
                           kept for API compatibility)

        Returns:
            Tuple of (latex_path, None)
        """
        aggregated = self.aggregate_results(results)

        latex_content = self.generate_latex_report(aggregated, model_name)
        latex_path = os.path.join(self.report_dir, "detailed_report.tex")
        with open(latex_path, "w") as f:
            f.write(latex_content)

        logger.info(f"Detailed report saved to {latex_path}")

        if shutil.which("pdflatex"):
            try:
                pdflatex_cmd = [
                    "pdflatex", "-interaction=nonstopmode",
                    "detailed_report.tex",
                ]
                subprocess.run(
                    pdflatex_cmd, cwd=self.report_dir,
                    capture_output=True, timeout=60,
                )
                subprocess.run(
                    pdflatex_cmd, cwd=self.report_dir,
                    capture_output=True, timeout=60,
                )
                pdf_path = os.path.join(
                    self.report_dir, "detailed_report.pdf"
                )
                if os.path.exists(pdf_path):
                    logger.info(f"PDF report generated: {pdf_path}")
                    for ext in [".aux", ".log", ".out"]:
                        aux = os.path.join(
                            self.report_dir, f"detailed_report{ext}"
                        )
                        if os.path.exists(aux):
                            os.remove(aux)
            except Exception as e:
                logger.warning(f"PDF compilation failed: {e}")

        return latex_path, None
