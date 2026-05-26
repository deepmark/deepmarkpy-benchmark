import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from utils.latex_helpers import (
    build_longtable,
    compile_latex,
    display_attack_name,
    make_preamble,
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
    """Generate comparative LaTeX reports across multiple watermarking models.

    The report compares only watermark detection accuracy across models.
    Per-model quality and intelligibility breakdowns stay in the
    individual detailed reports; mixing model-level metrics here would
    compare different models on different signals and is not meaningful.
    """

    def __init__(self, report_dir="report/comparison"):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self._has_deepmark_cls = os.path.exists(
            os.path.join(self.report_dir, "deepmark.cls")
        )

    def _preamble(self, title, author):
        return make_preamble(
            title, author, self._has_deepmark_cls,
            extra_packages=("colortbl", "xcolor"),
        )

    @staticmethod
    def _display_name(attack_name):
        return display_attack_name(attack_name)

    @staticmethod
    def _short_model_name(model_name):
        return model_name.replace("Model", "")

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

        return build_longtable(
            col_spec,
            header,
            rows,
            caption="Watermark detection accuracy (\\%) by attack type "
                    "for all tested models.",
            label="tab:comparison_accuracy",
        )

    # ----------------------------------------------------------------
    # Charts
    # ----------------------------------------------------------------
    _MODEL_COLORS = [
        "#039FAC", "#E74C3C", "#2ECC71", "#F39C12",
        "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB",
    ]

    def create_radar_chart(self, all_stats, output_path):
        """Create radar chart comparing multiple models across attacks.

        The figure is split into a polar plot on the left and a legend
        panel on the right. Each piece is drawn by a dedicated helper to
        keep this method focused on orchestration.
        """
        model_names, attacks = self.aggregate_stats(all_stats)
        n = len(attacks)
        if n < 2:
            logger.warning("Too few attacks for radar chart, skipping.")
            return False

        codes = [f"A{i+1}" for i in range(n)]
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor("white")
        polar_ax = fig.add_axes([0.05, 0.08, 0.55, 0.82], polar=True)
        legend_ax = fig.add_axes([0.63, 0.08, 0.35, 0.82])

        self._draw_radar_axes(polar_ax, all_stats, model_names,
                              attacks, angles, codes)
        self._draw_model_legend(legend_ax, model_names)
        self._draw_attack_legend(legend_ax, attacks, codes,
                                 n_models=len(model_names))

        plt.savefig(output_path, dpi=300, bbox_inches="tight",
                    facecolor="white")
        plt.close()
        logger.info(f"Comparative radar chart saved to {output_path}")
        return True

    def _draw_radar_axes(self, ax, all_stats, model_names,
                         attacks, angles, codes):
        """Draw the polar plot itself: one coloured polygon per model."""
        angles_closed = angles + [angles[0]]
        ax.set_facecolor("white")

        for idx, model in enumerate(model_names):
            values = [all_stats[model].get(a, 0) or 0 for a in attacks]
            values_closed = values + [values[0]]
            color = self._MODEL_COLORS[idx % len(self._MODEL_COLORS)]
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

    def _draw_model_legend(self, legend_ax, model_names):
        """Draw the "Models" block at the top of the legend panel."""
        legend_ax.axis("off")
        n_models = len(model_names)
        legend_ax.text(0.0, 1.0, "Models", fontsize=14, fontweight="700",
                       color="#2c3e50", transform=legend_ax.transAxes,
                       verticalalignment="top")
        col_size = (n_models + 1) // 2
        row_spacing = 0.04
        for idx, model in enumerate(model_names):
            color = self._MODEL_COLORS[idx % len(self._MODEL_COLORS)]
            short = self._short_model_name(model)
            col = idx // col_size
            row = idx % col_size
            x = col * 0.5
            y = 1.0 - (row + 1) * row_spacing - 0.01
            legend_ax.plot([x, x + 0.04], [y, y], color=color,
                           linewidth=2.5, transform=legend_ax.transAxes,
                           clip_on=False)
            legend_ax.text(x + 0.06, y, short, fontsize=11, color=color,
                           fontweight="600",
                           transform=legend_ax.transAxes,
                           verticalalignment="center")

    def _draw_attack_legend(self, legend_ax, attacks, codes, n_models):
        """Draw the "Attack Legend" block below the model legend."""
        legend_entries = [
            f"{codes[i]}  --  {self._display_name(attacks[i])}"
            for i in range(len(attacks))
        ]
        model_col_size = (n_models + 1) // 2
        model_row_spacing = 0.04
        attack_top = 1.0 - (model_col_size + 1) * model_row_spacing - 0.04
        legend_ax.text(0.0, attack_top, "Attack Legend",
                       fontsize=14, fontweight="700", color="#2c3e50",
                       transform=legend_ax.transAxes,
                       verticalalignment="top")

        n = len(attacks)
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

    # ----------------------------------------------------------------
    # Full LaTeX report
    # ----------------------------------------------------------------
    def generate_latex_report(self, all_stats, include_radar=True):
        """Generate complete comparative LaTeX document.

        The comparative report compares detection accuracy only. Per-
        model quality and intelligibility breakdowns stay in each
        model's own detailed report.
        """
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

        attack_word = "attack type" if num_attacks == 1 else "attack types"
        model_word = "watermarking model" if num_models == 1 else "watermarking models"

        return (
            f"{preamble}\n\n"
            f"\\begin{{abstract}}\n"
            f"This report compares {num_models} {model_word} "
            f"({model_list}) "
            f"across {num_attacks} {attack_word}. "
            f"Detection accuracy is compared per attack, with "
            f"row-wise color ranking highlighting relative performance.\n"
            f"\\end{{abstract}}\n\n"
            f"\\section{{Accuracy Comparison}}\n\n"
            f"{accuracy_table}\n\n"
            f"{color_legend}\n\n"
            f"{radar_figure}\n"
            f"\\end{{document}}"
        )

    def generate_full_report(self, all_results, all_stats,
                              calculate_quality_metrics=False):
        """Generate complete comparative report.

        Args:
            all_results: Dict of {model_name: raw benchmark results}.
                Kept in the signature for API compatibility with callers;
                the comparative report now only uses accuracy stats.
            all_stats: Dict of {model_name: {attack: accuracy_mean}}
            calculate_quality_metrics: Kept for API compatibility;
                ignored — per-model quality details live in each
                model's detailed report, not the comparative one.
        """
        del all_results, calculate_quality_metrics  # unused, see docstring

        # Radar chart
        radar_path = os.path.join(self.report_dir, "radar_chart.png")
        include_radar = self.create_radar_chart(all_stats, radar_path)

        latex_content = self.generate_latex_report(
            all_stats, include_radar=include_radar,
        )

        latex_path = os.path.join(self.report_dir, "comparative_report.tex")
        with open(latex_path, "w") as f:
            f.write(latex_content)
        logger.info(f"Comparative report saved to {latex_path}")

        compile_latex(self.report_dir, "comparative_report")
        return latex_path
