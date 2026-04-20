import os
import logging
import shutil
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from utils.attack_groups import get_group_for_attack
from utils.detailed_report_generator import (
    GROUP_QUALITY_METRICS,
    GROUP_INTELLIGIBILITY_METRICS,
    AUDIO_EDITING_SUBGROUPS,
)

logger = logging.getLogger(__name__)

# Rank text colors (best -> worst), readable on white background
RANK_COLORS = [
    ("green!40!black", "best"),
    ("blue", "second-best"),
    ("yellow!40!black", "third-best"),
    ("orange!70!black", "second-worst"),
    ("red!60!black", "worst"),
]


class ComparativeReportGenerator:
    """Generate comparative LaTeX reports across multiple watermarking models."""

    QUALITY_METRICS = ["pesq", "psnr", "si_sdr", "mcd", "visqol"]
    INTELLIGIBILITY_METRICS = ["stoi", "sii", "ncm"]
    METRIC_LABELS = {
        "pesq": "PESQ (1--4.5)",
        "psnr": "PSNR (dB)",
        "si_sdr": "SI-SDR (dB)",
        "mcd": "MCD (dB)",
        "visqol": "ViSQOL (1--5)",
        "stoi": "STOI (0--1)",
        "sii": "SII (0--1)",
        "ncm": "NCM (0--1)",
    }
    # Higher is better for most metrics; MCD is lower-is-better
    HIGHER_IS_BETTER = {
        "pesq": True, "psnr": True, "si_sdr": True,
        "mcd": False, "visqol": True,
        "stoi": True, "sii": True, "ncm": True,
    }

    def __init__(self, report_dir="results/comparison"):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self._has_deepmark_cls = os.path.exists(
            os.path.join(self.report_dir, "deepmark.cls")
        )

    def _preamble(self, title, author):
        """Generate LaTeX preamble with deepmark class fallback."""
        if self._has_deepmark_cls:
            return (
                f"\\documentclass{{deepmark}}\n"
                f"\\usepackage{{float}}\n"
                f"\\usepackage{{longtable}}\n"
                f"\\usepackage{{colortbl}}\n"
                f"\\usepackage{{xcolor}}\n"
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
            f"\\usepackage{{colortbl}}\n"
            f"\\usepackage{{xcolor}}\n"
            f"\\usepackage{{needspace}}\n"
            f"\\usepackage{{geometry}}\n"
            f"\\geometry{{margin=2.5cm}}\n\n"
            f"\\title{{{title}}}\n"
            f"\\author{{{author}}}\n"
            f"\\date{{\\today}}\n\n"
            f"\\begin{{document}}\n"
            f"\\maketitle"
        )

    @staticmethod
    def _display_name(attack_name):
        return attack_name.replace("Attack", "")

    @staticmethod
    def _short_model_name(model_name):
        return model_name.replace("Model", "")

    @staticmethod
    def _attacks_for_metric(attacks, metric_key):
        """Return the subset of attacks for which metric_key is relevant."""
        relevant = []
        for attack in attacks:
            group = get_group_for_attack(attack)
            if group is None:
                # Unknown group — keep all metrics
                relevant.append(attack)
                continue
            if group == "audio_editing":
                for sub_info in AUDIO_EDITING_SUBGROUPS.values():
                    if attack in sub_info["attacks"]:
                        sub_metrics = (
                            sub_info.get("quality_metrics", [])
                            + sub_info.get("intelligibility_metrics", [])
                        )
                        if metric_key in sub_metrics:
                            relevant.append(attack)
                        break
            else:
                group_metrics = (
                    GROUP_QUALITY_METRICS.get(group, [])
                    + GROUP_INTELLIGIBILITY_METRICS.get(group, [])
                )
                if metric_key in group_metrics:
                    relevant.append(attack)
        return relevant

    def _rank_color(self, value, all_values, higher_is_better=True):
        """Return rank color name for a value among all_values, or None.

        Ties receive the same color. When fewer than five distinct values
        appear in a row, only the applicable ranks are colored.
        """
        valid = [v for v in all_values if v is not None]
        if value is None or not valid:
            return None
        sorted_vals = sorted(set(valid), reverse=higher_is_better)
        n_distinct = len(sorted_vals)
        if n_distinct <= 1:
            return None
        try:
            rank = sorted_vals.index(value)
        except ValueError:
            return None
        if rank < len(RANK_COLORS) and rank < n_distinct:
            return RANK_COLORS[rank][0]
        return None

    def _colored_val(self, value, all_values, higher_is_better=True,
                     fmt=".2f"):
        """Format a numeric value with rank-based text color."""
        if value is None:
            return "N/A"
        color = self._rank_color(value, all_values,
                                 higher_is_better=higher_is_better)
        formatted = f"{value:{fmt}}"
        if color:
            return f"\\textcolor{{{color}}}{{\\textbf{{{formatted}}}}}"
        return formatted

    @staticmethod
    def _color_legend_text():
        """Return LaTeX color legend explanation."""
        return (
            "\\noindent\\small\\textit{Colors are assigned row-wise "
            "using distinct performance levels with ties receiving "
            "the same color: "
            "\\textcolor{green!40!black}{\\textbf{best}}, "
            "\\textcolor{blue}{\\textbf{second-best}}, "
            "\\textcolor{yellow!40!black}{\\textbf{third-best}}, "
            "\\textcolor{orange!70!black}{\\textbf{second-worst}}, and "
            "\\textcolor{red!60!black}{\\textbf{worst}}. "
            "When fewer than five distinct values appear in a row, "
            "only the applicable ranks are colored.}"
        )

    def aggregate_stats(self, all_stats):
        """Get sorted attack list and model names from stats."""
        model_names = list(all_stats.keys())
        all_attacks = set()
        for stats in all_stats.values():
            all_attacks.update(stats.keys())
        attacks = sorted(all_attacks)
        return model_names, attacks

    def aggregate_metrics(self, all_results):
        """Aggregate per-file results into per-model, per-attack metric means."""
        aggregated = {}
        for model_name, results in all_results.items():
            aggregated[model_name] = {}
            attack_data = {}
            for filepath, attacks in results.items():
                for attack_name, data in attacks.items():
                    if attack_name not in attack_data:
                        attack_data[attack_name] = {
                            m: [] for m in self.QUALITY_METRICS + self.INTELLIGIBILITY_METRICS
                        }
                    quality = data.get("attacked_audio_quality_wm")
                    if quality:
                        for m in self.QUALITY_METRICS + self.INTELLIGIBILITY_METRICS:
                            val = quality.get(m)
                            if val is not None:
                                attack_data[attack_name][m].append(val)

            for attack_name, metrics in attack_data.items():
                aggregated[model_name][attack_name] = {
                    m: float(np.mean(vals)) if vals else None
                    for m, vals in metrics.items()
                }
        return aggregated

    # ----------------------------------------------------------------
    # Accuracy comparison table
    # ----------------------------------------------------------------
    def generate_accuracy_table(self, all_stats):
        """Generate accuracy comparison table (attack x model) with rank colors."""
        model_names, attacks = self.aggregate_stats(all_stats)
        n_models = len(model_names)
        short_names = [self._short_model_name(m) for m in model_names]

        header = "Attack Type & " + " & ".join(short_names)
        col_spec = "l" + "c" * n_models

        rows = []
        for attack in attacks:
            display = self._display_name(attack)
            values = [all_stats[m].get(attack) for m in model_names]
            cells = [
                self._colored_val(v, values, higher_is_better=True)
                for v in values
            ]
            rows.append(f"    {display} & " + " & ".join(cells) + " \\\\")

        return (
            f"\\begin{{longtable}}{{{col_spec}}}\n"
            "    \\caption{Watermark detection accuracy (\\%) by attack type "
            "for all tested models.}\n"
            "    \\label{tab:comparison_accuracy} \\\\\n"
            "    \\toprule\n"
            f"    {header} \\\\\n"
            "    \\midrule\n"
            "    \\endfirsthead\n"
            "    \\toprule\n"
            f"    {header} \\\\\n"
            "    \\midrule\n"
            "    \\endhead\n"
            "    \\bottomrule\n"
            "    \\endlastfoot\n"
            + "\n".join(rows) + "\n"
            "\\end{longtable}"
        )

    # ----------------------------------------------------------------
    # Metric comparison table
    # ----------------------------------------------------------------
    def generate_metric_table(self, aggregated_metrics, model_names, attacks,
                              metric_key, caption, label):
        """Generate a single metric comparison table (attack x model)."""
        if not attacks:
            return ""
        n_models = len(model_names)
        short_names = [self._short_model_name(m) for m in model_names]
        header = "Attack Type & " + " & ".join(short_names)
        col_spec = "l" + "c" * n_models
        higher = self.HIGHER_IS_BETTER.get(metric_key, True)

        rows = []
        for attack in attacks:
            display = self._display_name(attack)
            values = [
                aggregated_metrics[m].get(attack, {}).get(metric_key)
                for m in model_names
            ]
            cells = [
                self._colored_val(v, values, higher_is_better=higher)
                for v in values
            ]
            rows.append(f"    {display} & " + " & ".join(cells) + " \\\\")

        return (
            f"\\begin{{longtable}}{{{col_spec}}}\n"
            f"    \\caption{{{caption}}}\n"
            f"    \\label{{{label}}} \\\\\n"
            "    \\toprule\n"
            f"    {header} \\\\\n"
            "    \\midrule\n"
            "    \\endfirsthead\n"
            "    \\toprule\n"
            f"    {header} \\\\\n"
            "    \\midrule\n"
            "    \\endhead\n"
            "    \\bottomrule\n"
            "    \\endlastfoot\n"
            + "\n".join(rows) + "\n"
            "\\end{longtable}"
        )

    # ----------------------------------------------------------------
    # Charts
    # ----------------------------------------------------------------
    def create_radar_chart(self, all_stats, output_path):
        """Create radar chart comparing multiple models across attacks."""
        model_names, attacks = self.aggregate_stats(all_stats)
        n = len(attacks)
        if n < 5:
            logger.warning("Too few attacks for radar chart, skipping.")
            return False

        codes = [f"A{i+1}" for i in range(n)]
        legend_entries = [
            f"{codes[i]}  --  {self._display_name(attacks[i])}"
            for i in range(n)
        ]
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles_closed = angles + [angles[0]]

        colors = ["#039FAC", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6",
                  "#1ABC9C", "#E67E22", "#3498DB"]

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.05, 0.08, 0.55, 0.82], polar=True)
        ax.set_facecolor("white")

        for idx, model in enumerate(model_names):
            values = [all_stats[model].get(a, 0) or 0 for a in attacks]
            values_closed = values + [values[0]]
            color = colors[idx % len(colors)]
            ax.plot(angles_closed, values_closed, color=color,
                    linewidth=2, alpha=0.85,
                    label=self._short_model_name(model))
            ax.fill(angles_closed, values_closed, color=color, alpha=0.08)

        ax.set_xticks(angles)
        ax.set_xticklabels(codes, fontsize=8, color="#333333")
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"],
                           fontsize=8, color="#777777")
        ax.set_rlabel_position(0)
        ax.grid(color="#dddddd", linewidth=0.5)
        ax.set_title("Watermark Detection Accuracy by Attack Type",
                     fontsize=14, fontweight="700", pad=20, color="#2c3e50")

        legend_ax = fig.add_axes([0.63, 0.08, 0.35, 0.82])
        legend_ax.axis("off")

        # Model legend at top — two columns, compact
        n_models = len(model_names)
        legend_ax.text(0.0, 1.0, "Models", fontsize=14, fontweight="700",
                       color="#2c3e50", transform=legend_ax.transAxes,
                       verticalalignment="top")
        model_col_size = (n_models + 1) // 2
        model_row_spacing = 0.04
        for idx, model in enumerate(model_names):
            color = colors[idx % len(colors)]
            short = self._short_model_name(model)
            col = idx // model_col_size
            row = idx % model_col_size
            x = col * 0.5
            y = 1.0 - (row + 1) * model_row_spacing - 0.01
            legend_ax.plot([x, x + 0.04], [y, y], color=color,
                           linewidth=2.5, transform=legend_ax.transAxes,
                           clip_on=False)
            legend_ax.text(x + 0.06, y, short, fontsize=11, color=color,
                           fontweight="600",
                           transform=legend_ax.transAxes,
                           verticalalignment="center")

        # Attack legend below model legend
        model_rows = model_col_size
        attack_top = 1.0 - (model_rows + 1) * model_row_spacing - 0.04
        legend_ax.text(0.0, attack_top, "Attack Legend",
                       fontsize=14, fontweight="700", color="#2c3e50",
                       transform=legend_ax.transAxes,
                       verticalalignment="top")

        col_size = (n + 1) // 2
        available_height = attack_top - 0.05
        row_spacing = min(0.032, available_height / (col_size + 1))
        for i, entry in enumerate(legend_entries):
            col = i // col_size
            row = i % col_size
            x = col * 0.5
            y = attack_top - 0.06 - row * row_spacing
            legend_ax.text(x, y, entry, fontsize=10, color="#555555",
                          transform=legend_ax.transAxes,
                          verticalalignment="center")

        plt.savefig(output_path, dpi=300, bbox_inches="tight",
                    facecolor="white")
        plt.close()
        logger.info(f"Comparative radar chart saved to {output_path}")
        return True

    # ----------------------------------------------------------------
    # Full LaTeX report
    # ----------------------------------------------------------------
    def generate_latex_report(self, all_stats, all_results,
                              include_radar=True,
                              calculate_quality_metrics=False):
        """Generate complete comparative LaTeX document."""
        model_names, attacks = self.aggregate_stats(all_stats)
        num_models = len(model_names)
        num_attacks = len(attacks)

        preamble = self._preamble(
            "Comparative Benchmark Report",
            "DeepMark Benchmark System",
        )

        # Model listing for the abstract
        short_names = [self._short_model_name(m) for m in model_names]
        model_list = ", ".join(
            f"\\textbf{{{name}}}" for name in short_names
        )

        accuracy_table = self.generate_accuracy_table(all_stats)
        color_legend = self._color_legend_text()

        # Radar chart right after accuracy table
        radar_figure = ""
        if include_radar:
            radar_figure = (
                "\\begin{figure}[H]\n"
                "    \\centering\n"
                "    \\includegraphics[width=\\linewidth]"
                "{radar_chart.png}\n"
                "    \\caption{Detection accuracy comparison "
                "across all attacks.}\n"
                "    \\label{fig:comp_radar}\n"
                "\\end{figure}\n"
            )

        # Metric sections — only attacks where each metric is relevant
        metrics_sections = ""
        if calculate_quality_metrics:
            aggregated_metrics = self.aggregate_metrics(all_results)

            quality_tables = []
            for m in self.QUALITY_METRICS:
                relevant = self._attacks_for_metric(attacks, m)
                if not relevant:
                    continue
                label = self.METRIC_LABELS.get(m, m)
                table = self.generate_metric_table(
                    aggregated_metrics, model_names, relevant,
                    metric_key=m,
                    caption=f"{label} comparison across models "
                            f"(post-attack).",
                    label=f"tab:comp_{m}",
                )
                if table:
                    quality_tables.append(table)

            intell_tables = []
            for m in self.INTELLIGIBILITY_METRICS:
                relevant = self._attacks_for_metric(attacks, m)
                if not relevant:
                    continue
                label = self.METRIC_LABELS.get(m, m)
                table = self.generate_metric_table(
                    aggregated_metrics, model_names, relevant,
                    metric_key=m,
                    caption=f"{label} comparison across models "
                            f"(post-attack).",
                    label=f"tab:comp_{m}",
                )
                if table:
                    intell_tables.append(table)

            if quality_tables:
                metrics_sections += (
                    "\\needspace{5\\baselineskip}\n"
                    "\\section{Audio Quality Comparison}\n\n"
                    "The following tables compare audio quality metrics "
                    "across models for each attack type. Only attacks for "
                    "which each metric is relevant are included. Colors "
                    "indicate row-wise rank using the same scheme as the "
                    "accuracy table.\n\n"
                    + "\n\n".join(quality_tables)
                )

            if intell_tables:
                metrics_sections += (
                    "\n\n\\needspace{5\\baselineskip}\n"
                    "\\section{Speech Intelligibility Comparison}\n\n"
                    "The following tables compare speech intelligibility "
                    "metrics across models for each attack type.\n\n"
                    + "\n\n".join(intell_tables)
                )

        return (
            f"{preamble}\n\n"
            f"\\begin{{abstract}}\n"
            f"This report compares {num_models} watermarking models "
            f"across {num_attacks} attack types: {model_list}. "
            f"Detection accuracy is compared per attack, with "
            f"row-wise color ranking highlighting relative performance.\n"
            f"\\end{{abstract}}\n\n"
            f"\\section{{Accuracy Comparison}}\n\n"
            f"{accuracy_table}\n\n"
            f"{color_legend}\n\n"
            f"{radar_figure}\n"
            f"{metrics_sections}\n\n"
            f"\\end{{document}}"
        )

    def generate_full_report(self, all_results, all_stats,
                              calculate_quality_metrics=False):
        """Generate complete comparative report.

        Args:
            all_results: Dict of {model_name: raw benchmark results}
            all_stats: Dict of {model_name: {attack: accuracy_mean}}
            calculate_quality_metrics: Whether quality metrics are available
        """
        # Radar chart
        radar_path = os.path.join(self.report_dir, "radar_chart.png")
        include_radar = self.create_radar_chart(all_stats, radar_path)

        latex_content = self.generate_latex_report(
            all_stats, all_results,
            include_radar=include_radar,
            calculate_quality_metrics=calculate_quality_metrics,
        )

        latex_path = os.path.join(self.report_dir, "comparative_report.tex")
        with open(latex_path, "w") as f:
            f.write(latex_content)
        logger.info(f"Comparative report saved to {latex_path}")

        if shutil.which("pdflatex"):
            try:
                pdflatex_cmd = [
                    "pdflatex", "-interaction=nonstopmode",
                    "comparative_report.tex",
                ]
                subprocess.run(pdflatex_cmd, cwd=self.report_dir,
                              capture_output=True, timeout=60)
                subprocess.run(pdflatex_cmd, cwd=self.report_dir,
                              capture_output=True, timeout=60)
                pdf_path = os.path.join(
                    self.report_dir, "comparative_report.pdf"
                )
                if os.path.exists(pdf_path):
                    logger.info(f"PDF report generated: {pdf_path}")
                    for ext in [".aux", ".log", ".out"]:
                        aux = os.path.join(
                            self.report_dir, f"comparative_report{ext}"
                        )
                        if os.path.exists(aux):
                            os.remove(aux)
            except Exception as e:
                logger.warning(f"PDF compilation failed: {e}")

        return latex_path
