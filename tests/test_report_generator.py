"""Tests for src/utils/report_generator.py — LaTeX report and chart generation."""

import json
import os

import pytest

from utils.report_generator import BenchmarkReportGenerator, generate_benchmark_report


# ---------------------------------------------------------------------------
# Performance tier classification (the threshold bug fix)
# ---------------------------------------------------------------------------
class TestPerformanceTiers:
    def test_excellent_threshold(self, sample_stats):
        gen = BenchmarkReportGenerator()
        report = gen.generate_latex_report(sample_stats)
        # 100.0 should be excellent, 97.5 should be excellent
        assert "Excellent Performance" in report

    def test_poor_threshold(self, sample_stats):
        gen = BenchmarkReportGenerator()
        report = gen.generate_latex_report(sample_stats)
        # 29.0 and 41.0 and 53.1 are < 70 → poor
        assert "Poor Performance" in report

    def test_fair_threshold(self, sample_stats):
        gen = BenchmarkReportGenerator()
        report = gen.generate_latex_report(sample_stats)
        # 82.0 is 70-85 → fair
        assert "Fair Performance" in report

    def test_thresholds_are_percentage_not_decimal(self):
        """Regression test: thresholds must be 95/85/70, not 0.95/0.85/0.70."""
        stats = {"PerfectAttack": 100.0, "TerribleAttack": 50.0}
        gen = BenchmarkReportGenerator()
        report = gen.generate_latex_report(stats)
        assert "Poor Performance" in report
        assert "Excellent Performance" in report


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------
class TestLatexTable:
    def test_contains_all_attacks(self, sample_stats):
        gen = BenchmarkReportGenerator()
        table = gen.generate_latex_table(sample_stats)
        for name in sample_stats:
            # The generator removes "Attack" and adds spaces before uppercase letters
            display = name.replace("Attack", "").strip()
            display = ''.join([' ' + c if c.isupper() and i > 0 else c for i, c in enumerate(display)]).strip()
            assert display in table, f"'{display}' not found in table"

    def test_contains_accuracy_values(self, sample_stats):
        gen = BenchmarkReportGenerator()
        table = gen.generate_latex_table(sample_stats)
        for acc in sample_stats.values():
            assert f"{acc:.2f}" in table

    def test_has_tabular_structure(self, sample_stats):
        gen = BenchmarkReportGenerator()
        table = gen.generate_latex_table(sample_stats)
        assert "\\begin{longtable}" in table
        assert "\\end{longtable}" in table
        assert "\\toprule" in table
        assert "\\bottomrule" in table


# ---------------------------------------------------------------------------
# Mean accuracy calculation
# ---------------------------------------------------------------------------
class TestCalculateMeanAccuracy:
    def test_correct_mean(self, sample_stats):
        gen = BenchmarkReportGenerator()
        mean = gen.calculate_mean_accuracy(sample_stats)
        expected = sum(sample_stats.values()) / len(sample_stats)
        assert mean == pytest.approx(expected)

    def test_empty_stats_returns_zero(self):
        gen = BenchmarkReportGenerator()
        assert gen.calculate_mean_accuracy({}) == 0.0

    def test_single_attack(self):
        gen = BenchmarkReportGenerator()
        assert gen.calculate_mean_accuracy({"A": 75.0}) == 75.0


# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------
class TestFullReport:
    def test_generates_latex_and_chart(self, sample_stats, tmp_path):
        stats_file = tmp_path / "stats.json"
        stats_file.write_text(json.dumps(sample_stats))

        gen = BenchmarkReportGenerator(str(tmp_path))
        latex_path, chart_path = gen.generate_full_report(
            str(stats_file), model_name="TestModel"
        )

        assert os.path.exists(latex_path)
        assert os.path.exists(chart_path)
        assert latex_path.endswith(".tex")
        assert chart_path.endswith(".png")

    def test_latex_fallback_without_deepmark_cls(self, sample_stats, tmp_path):
        """Without deepmark.cls in report dir, should fall back to article."""
        stats_file = tmp_path / "stats.json"
        stats_file.write_text(json.dumps(sample_stats))

        gen = BenchmarkReportGenerator(str(tmp_path))
        gen.generate_full_report(str(stats_file), model_name="TestModel")

        latex_path = tmp_path / "benchmark_report.tex"
        content = latex_path.read_text()
        assert "\\documentclass{article}" in content

    def test_latex_uses_deepmark_cls_when_available(self, sample_stats, tmp_path):
        """With deepmark.cls in report dir, should use deepmark class."""
        stats_file = tmp_path / "stats.json"
        stats_file.write_text(json.dumps(sample_stats))
        (tmp_path / "deepmark.cls").write_text("")

        gen = BenchmarkReportGenerator(str(tmp_path))
        gen.generate_full_report(str(stats_file), model_name="TestModel")

        latex_path = tmp_path / "benchmark_report.tex"
        content = latex_path.read_text()
        assert "\\documentclass{deepmark}" in content

    def test_missing_stats_file_raises(self, tmp_path):
        gen = BenchmarkReportGenerator(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            gen.generate_full_report("/nonexistent/stats.json")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
class TestConvenienceFunction:
    def test_generate_benchmark_report(self, sample_stats, tmp_path):
        stats_file = tmp_path / "stats.json"
        stats_file.write_text(json.dumps(sample_stats))

        latex_path, chart_path = generate_benchmark_report(
            str(stats_file), model_name="TestModel", report_dir=str(tmp_path)
        )
        assert os.path.exists(latex_path)
        assert os.path.exists(chart_path)
