import json
import os
import logging
import shutil
import subprocess
from typing import Dict
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class BenchmarkReportGenerator:
    """Generate LaTeX reports for benchmark results with visualizations."""

    def __init__(self, report_dir: str = "report"):
        self.report_dir = report_dir
        self.ensure_report_dir()
        self._has_deepmark_cls = os.path.exists(os.path.join(self.report_dir, "deepmark.cls"))

    def ensure_report_dir(self):
        """Ensure the report directory exists."""
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    def _preamble(self, title, author):
        """Generate LaTeX preamble with deepmark class fallback."""
        if self._has_deepmark_cls:
            return (
                f"\\documentclass{{deepmark}}\n"
                f"        \\usepackage{{float}}\n"
                f"        \\usepackage{{longtable}}\n\n"
                f"        \\title{{{title}}}\n"
                f"        \\author{{{author}}}\n\n"
                f"        \\begin{{document}}\n"
                f"        \\thispagestyle{{firststyle}}\n"
                f"        \\maketitle"
            )
        return (
            f"\\documentclass{{article}}\n"
            f"        \\usepackage{{booktabs}}\n"
            f"        \\usepackage{{graphicx}}\n"
            f"        \\usepackage{{cleveref}}\n"
            f"        \\usepackage{{float}}\n"
            f"        \\usepackage{{longtable}}\n\n"
            f"        \\title{{{title}}}\n"
            f"        \\author{{{author}}}\n"
            f"        \\date{{\\today}}\n\n"
            f"        \\begin{{document}}\n"
            f"        \\maketitle"
        )

    def create_gradient_bar_chart(self, stats: Dict[str, float], output_path: str):
        """
        Create a modern bar chart with consistent color scheme.

        Args:
            stats: Dictionary with attack names as keys and accuracy values
            output_path: Path to save the chart image
        """
        sorted_attacks = sorted(stats.items())
        attack_names = [name for name, _ in sorted_attacks]
        accuracies = [acc for _, acc in sorted_attacks]

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor('white')

        bar_color = '#469CA9'

        bars = ax.bar(range(len(attack_names)), accuracies,
                     color=bar_color, alpha=0.85,
                     edgecolor='white', linewidth=1.5,
                     width=0.7)

        ax.set_facecolor('#fafafa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')

        ax.set_xlabel('Attack Types', fontsize=14, fontweight='600', color='#333333')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='600', color='#333333')
        ax.set_title('Watermark Detection Accuracy by Attack Type',
                    fontsize=16, fontweight='700', pad=25, color='#2c3e50')

        ax.set_xticks(range(len(attack_names)))
        ax.set_xticklabels(attack_names, rotation=45, ha='right',
                          fontsize=11, color='#555555')

        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.5, color='#dddddd')
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', colors='#555555', labelsize=11)


        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"Bar chart saved to {output_path}")

    def generate_latex_table(self, stats: Dict[str, float]) -> str:
        """
        Generate LaTeX table code for the benchmark results.

        Args:
            stats: Dictionary with attack names as keys and accuracy values

        Returns:
            LaTeX table code as string
        """
        sorted_attacks = sorted(stats.items())

        table_rows = []
        for attack_name, accuracy in sorted_attacks:
            display_name = attack_name.replace("Attack", "").strip()
            display_name = ''.join([' ' + c if c.isupper() and i > 0 else c for i, c in enumerate(display_name)]).strip()

            acc_percent = accuracy
            table_rows.append(f"    {display_name} & {acc_percent:.2f}\\% \\\\")

        table_code = (
            "\\begin{longtable}{lc}\n"
            "    \\caption{Watermark detection accuracy for different attack types.}\n"
            "    \\label{tab:benchmark_results} \\\\\n"
            "    \\toprule\n"
            "    Attack Type & Accuracy \\\\\n"
            "    \\midrule\n"
            "    \\endfirsthead\n"
            "    \\toprule\n"
            "    Attack Type & Accuracy \\\\\n"
            "    \\midrule\n"
            "    \\endhead\n"
            "    \\bottomrule\n"
            "    \\endlastfoot\n"
            + "\n".join(table_rows) + "\n"
            "\\end{longtable}"
        )

        return table_code

    def calculate_mean_accuracy(self, stats: Dict[str, float]) -> float:
        """Calculate overall mean accuracy across all attacks."""
        if not stats:
            return 0.0
        return sum(stats.values()) / len(stats)

    def generate_latex_report(self, stats: Dict[str, float], model_name: str = "DeepMark",
                            chart_filename: str = "benchmark_chart.png") -> str:
        """
        Generate complete LaTeX report content.

        Args:
            stats: Dictionary with attack names as keys and accuracy values
            model_name: Name of the watermarking model
            chart_filename: Filename of the generated chart

        Returns:
            Complete LaTeX document as string
        """
        mean_accuracy = self.calculate_mean_accuracy(stats)
        table_code = self.generate_latex_table(stats)

        preamble = self._preamble(
            f"Benchmark Report: {model_name}",
            "DeepMark Benchmark System",
        )

        latex_content = f"""{preamble}

        % -------------------- Abstract --------------------
        \\begin{{abstract}}
        This report presents the benchmark results for the {model_name} watermarking model across various attack scenarios. The evaluation covers {len(stats)} different attack types, measuring the robustness of watermark detection under adversarial conditions using the DeepMark benchmark framework.
        \\end{{abstract}}

        % -------------------- Results --------------------
        \\section{{Benchmark Results}}

        Table~\\ref{{tab:benchmark_results}} reports per-attack watermark detection accuracy (attacks are listed alphabetically).

        {table_code}

        \\vspace{{1em}}
        \\noindent\\textbf{{Overall Mean Accuracy:}} {mean_accuracy:.2f}\\%

        \\vspace{{1em}}
        \\noindent Figure~\\ref{{fig:benchmark_chart}} complements the table by visualizing the same results, enabling quicker inspection of relative differences and overall trends across attacks.

        \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=\\linewidth]{{{chart_filename}}}
        \\caption{{Watermark detection accuracy by attack type.}}
        \\label{{fig:benchmark_chart}}
        \\end{{figure}}

        % -------------------- Analysis --------------------
        \\section{{Performance Analysis}}

        The watermarking model demonstrates following levels of robustness across different attack types:

        \\begin{{itemize}}
        """

        excellent = [name for name, acc in stats.items() if acc >= 95]
        good = [name for name, acc in stats.items() if 85 <= acc < 95]
        fair = [name for name, acc in stats.items() if 70 <= acc < 85]
        poor = [name for name, acc in stats.items() if acc < 70]

        if excellent:
            attack_word = "attack" if len(excellent) == 1 else "attacks"
            latex_content += f"  \\item \\textbf{{Excellent Performance ($\\geq$95\\%):}} {len(excellent)} {attack_word}\n"
        if good:
            attack_word = "attack" if len(good) == 1 else "attacks"
            latex_content += f"  \\item \\textbf{{Good Performance (85-95\\%):}} {len(good)} {attack_word}\n"
        if fair:
            attack_word = "attack" if len(fair) == 1 else "attacks"
            latex_content += f"  \\item \\textbf{{Fair Performance (70-85\\%):}} {len(fair)} {attack_word}\n"
        if poor:
            attack_word = "attack" if len(poor) == 1 else "attacks"
            latex_content += f"  \\item \\textbf{{Poor Performance ($<$70\\%):}} {len(poor)} {attack_word}\n"

        latex_content += """\\end{itemize}

        \\end{document}"""

        return latex_content

    def generate_full_report(self, stats_file: str = "benchmark_stats.json",
                           model_name: str = "DeepMark"):
        """
        Generate complete benchmark report with chart and LaTeX document.

        Args:
            stats_file: Path to the benchmark statistics JSON file
            model_name: Name of the watermarking model
        """
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)

            logger.info(f"Loaded benchmark statistics for {len(stats)} attacks")

            chart_path = os.path.join(self.report_dir, "benchmark_chart.png")
            self.create_gradient_bar_chart(stats, chart_path)

            latex_content = self.generate_latex_report(stats, model_name, "benchmark_chart.png")

            latex_path = os.path.join(self.report_dir, "benchmark_report.tex")
            with open(latex_path, 'w') as f:
                f.write(latex_content)

            logger.info(f"LaTeX report saved to {latex_path}")

            if shutil.which("pdflatex"):
                try:
                    pdflatex_cmd = ["pdflatex", "-interaction=nonstopmode", "benchmark_report.tex"]
                    # Run twice to resolve cross-references
                    subprocess.run(pdflatex_cmd, cwd=self.report_dir, capture_output=True, timeout=60)
                    subprocess.run(pdflatex_cmd, cwd=self.report_dir, capture_output=True, timeout=60)
                    pdf_path = os.path.join(self.report_dir, "benchmark_report.pdf")
                    if os.path.exists(pdf_path):
                        logger.info(f"PDF report generated: {pdf_path}")
                        for ext in [".aux", ".log", ".out"]:
                            aux_file = os.path.join(self.report_dir, f"benchmark_report{ext}")
                            if os.path.exists(aux_file):
                                os.remove(aux_file)
                except Exception as e:
                    logger.warning(f"PDF compilation failed: {e}")

            return latex_path, chart_path

        except FileNotFoundError:
            logger.error(f"Benchmark statistics file not found: {stats_file}")
            raise
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise


def generate_benchmark_report(stats_file: str = "benchmark_stats.json",
                            model_name: str = "DeepMark",
                            report_dir: str = "report"):
    """
    Convenience function to generate a complete benchmark report.

    Args:
        stats_file: Path to the benchmark statistics JSON file
        model_name: Name of the watermarking model
        report_dir: Directory to save the report files

    Returns:
        Tuple of (latex_path, chart_path)
    """
    generator = BenchmarkReportGenerator(report_dir)
    return generator.generate_full_report(stats_file, model_name)
