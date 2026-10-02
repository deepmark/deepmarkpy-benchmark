"""Every figure a report references exists, and says what the config asked.

Two things can go wrong with a generated figure that a table cannot go
wrong with. The ``.tex`` can point ``\\includegraphics`` at a file that
was never written -- which fails at compile time, long after the run --
and a chart can plot a metric the configuration turned off, contradicting
the tables beside it. Both are checked here for all five reports.

Chart drawing is real: the PNGs are written and their existence asserted.
Only ``pdflatex`` is stubbed out.
"""

import json
import re

import pytest

from deepmarkpy.benchmark import Benchmark
from deepmarkpy.config import load_configs
from deepmarkpy.utils import report_charts
from deepmarkpy.utils.comparative_report_generator import ComparativeReportGenerator
from deepmarkpy.utils.latex_helpers import (
    metric_label,
    part_heading,
    slugify,
)
from deepmarkpy.utils.detailed_report_generator import DetailedReportGenerator
from deepmarkpy.utils.detection_reliability_report_generator import (
    generate_detection_reliability_report,
)
from deepmarkpy.utils.no_attacks_report_generator import generate_no_attacks_report
from deepmarkpy.utils.report_generator import BenchmarkReportGenerator

SIGNAL_METRICS = [
    "pesq", "psnr", "si_sdr", "mcd", "visqol", "stoi", "sii", "ncm",
]

# Two versions of one attack so the strength-curve figure has a ladder,
# plus enough distinct attacks for the scatter's three-point minimum.
ATTACKS = [
    "GaussianNoiseAttack (mild)",
    "GaussianNoiseAttack (severe)",
    "PinkNoiseAttack",
    "SignInversionAttack",
    "AdditiveNoiseAttack",
    "LowpassFilterAttack",
    "EchoAttack",
    "TimeStretchAttack",
]

# Figures live in the section whose tables they draw, so their filenames
# carry the group. audio_distortion holds the version ladder and enough
# attacks for the scatter; audio_editing holds two, which is below the
# scatter's minimum.
SCATTER = "robustness_quality_audio_distortion.png"
LADDER = "attack_strength_audio_distortion.png"


@pytest.fixture(autouse=True)
def _skip_pdflatex(monkeypatch):
    for module in ("report_generator", "detailed_report_generator",
                   "no_attacks_report_generator",
                   "detection_reliability_report_generator",
                   "comparative_report_generator"):
        monkeypatch.setattr(
            f"deepmarkpy.utils.{module}.compile_latex",
            lambda *a, **k: None, raising=False,
        )
    monkeypatch.setattr("deepmarkpy.utils.latex_helpers.compile_latex",
                        lambda *a, **k: None)


def make_results(n_files=6):
    """Per-file results whose accuracy varies, so a distribution exists."""
    results = {}
    for index in range(n_files):
        quality = {m: 2.0 + 0.2 * index for m in SIGNAL_METRICS}
        results[f"f{index}.wav"] = {
            "watermarked_audio_quality": {m: 4.5 for m in SIGNAL_METRICS},
            "attacks": {
                name: {
                    "accuracy": 55.0 + 7 * index + 3 * position,
                    "detection_valid": True,
                    "attacked_audio_quality_wm": dict(quality),
                }
                for position, name in enumerate(ATTACKS)
            },
        }
    return results


def write_config(tmp_path, **overrides):
    data = {"mode": "benchmark", "models": ["AudioSealModel"],
            "calculate_quality_metrics": True}
    data.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return load_configs([str(path)])[0]


def included_figures(tex):
    """Filenames the document asks LaTeX to include."""
    return re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)


def assert_figures_exist(tex, directory):
    """No figure is referenced that was not written."""
    referenced = included_figures(tex)
    assert referenced, "the report includes no figure at all"
    missing = [f for f in referenced if not (directory / f).exists()]
    assert not missing, f"referenced but never drawn: {missing}"


def _no_attack_files(offset=0.0, n=5, supports_detection=False,
                     is_zero_bit=False):
    """One model's no-attacks result block."""
    files = []
    for index in range(n):
        entry = {
            "filepath": f"f{index}.wav",
            "accuracy": 100.0 if is_zero_bit and index else 92.0 + index,
            "watermarked_audio_quality": {
                m: 4.0 - offset for m in SIGNAL_METRICS
            },
        }
        if supports_detection:
            entry["detected"] = index != 0
        files.append(entry)
    return {"is_zero_bit": is_zero_bit, "files": files,
            "supports_detection": supports_detection}


def build_basic(tmp_path, config, results=None):
    benchmark = Benchmark.__new__(Benchmark)
    stats = benchmark.compute_mean_accuracy(
        results or make_results(), resolver=config.resolver,
    )
    stats_file = tmp_path / "benchmark_stats.json"
    stats_file.write_text(json.dumps(stats))
    BenchmarkReportGenerator(str(tmp_path), resolver=config.resolver) \
        .generate_full_report(str(stats_file), model_name="TestModel")
    return (tmp_path / "benchmark_report.tex").read_text()


class TestEveryReportReferencesOnlyFiguresItDrew:
    def test_basic_report(self, tmp_path):
        tex = build_basic(tmp_path, write_config(tmp_path))
        assert_figures_exist(tex, tmp_path)

    def test_detailed_report(self, tmp_path):
        config = write_config(tmp_path)
        DetailedReportGenerator(str(tmp_path), resolver=config.resolver) \
            .generate_full_report(make_results(), model_name="TestModel")
        assert_figures_exist((tmp_path / "detailed_report.tex").read_text(),
                             tmp_path)

    def test_no_attacks_report_with_two_models(self, tmp_path):
        config = write_config(tmp_path, mode="no_attacks")
        generate_no_attacks_report(
            {"AudioSealModel": _no_attack_files(0.0),
             "WavMarkModel": _no_attack_files(0.4)},
            report_dir=str(tmp_path), resolver=config.resolver,
        )
        assert_figures_exist((tmp_path / "no_attacks_report.tex").read_text(),
                             tmp_path)

    def test_detection_reliability_report(self, tmp_path):
        config = write_config(tmp_path, mode="detection_reliability")
        result = {
            "model_name": "AudioSealModel", "n_files": 8,
            "no_attack": {"false_positive_count": 1, "false_negative_count": 0},
            "attacks": {
                name: {
                    "false_positive_count": index,
                    "false_positive_attempts": 8,
                    "false_negative_count": 8 - index,
                    "false_negative_attempts": 8,
                    "accuracy_mean": 60.0 + index, "accuracy_n": 8,
                    "metrics": {},
                }
                for index, name in enumerate(ATTACKS)
            },
        }
        generate_detection_reliability_report(
            result, report_dir=str(tmp_path), resolver=config.resolver,
        )
        tex = (tmp_path / "detection_reliability_report.tex").read_text()
        assert_figures_exist(tex, tmp_path)

    def test_comparative_report(self, tmp_path):
        config = write_config(tmp_path)
        benchmark = Benchmark.__new__(Benchmark)
        stats = benchmark.compute_mean_accuracy(
            make_results(), resolver=config.resolver,
        )
        other = {a: dict(v) for a, v in stats.items()}
        generator = ComparativeReportGenerator(
            str(tmp_path), resolver=config.resolver, primary_statistic="mean",
        )
        generator.generate_full_report({"A": stats, "B": other})
        tex = (tmp_path / "comparative_report.tex").read_text()
        assert_figures_exist(tex, tmp_path)
        assert "accuracy_heatmap.png" in tex


class TestFiguresFollowTheConfiguration:
    def test_no_quality_figure_when_every_quality_metric_is_off(self, tmp_path):
        """A figure may not plot a metric the tables are forbidden to show."""
        config = write_config(tmp_path, metrics={"defaults": {
            m: {"enabled": False} for m in SIGNAL_METRICS
        }})
        tex = build_basic(tmp_path, config)
        assert not (tmp_path / SCATTER).exists()
        assert SCATTER not in tex
        # The ranking chart needs no quality metric, so it still appears.
        assert (tmp_path / "benchmark_chart.png").exists()

    def test_quality_metrics_off_leaves_the_always_on_trio_plottable(self, tmp_path):
        """``calculate_quality_metrics: false`` still computes PESQ/ViSQOL/STOI.

        The scatter has to pick from those rather than declaring there is
        no quality data, which would disagree with the tables that show
        the trio.
        """
        config = write_config(tmp_path, calculate_quality_metrics=False)
        tex = build_basic(tmp_path, config)
        assert "ViSQOL" in _figure_block(tex, SCATTER)

    def test_the_scatter_uses_a_configured_metric(self, tmp_path):
        """With ViSQOL off, the scatter falls back to the next enabled metric."""
        config = write_config(tmp_path, metrics={"defaults": {
            "visqol": {"enabled": False}, "pesq": {"enabled": True},
        }})
        tex = build_basic(tmp_path, config)
        figure = _figure_block(tex, SCATTER)
        assert "PESQ" in figure and "ViSQOL" not in figure

    def test_the_chance_floor_follows_the_model_family(self, tmp_path):
        config = write_config(tmp_path)
        stats_file = tmp_path / "benchmark_stats.json"
        benchmark = Benchmark.__new__(Benchmark)
        stats_file.write_text(json.dumps(
            benchmark.compute_mean_accuracy(make_results(),
                                            resolver=config.resolver)
        ))
        for is_zero_bit, expected in ((False, 50.0), (True, 0.0)):
            generator = BenchmarkReportGenerator(
                str(tmp_path), resolver=config.resolver, is_zero_bit=is_zero_bit,
            )
            assert generator._chance_floor == expected
            generator.generate_full_report(str(stats_file))
            tex = (tmp_path / "benchmark_report.tex").read_text()
            assert f"({expected:.0f}\\%)" in tex


def _figure_block(tex, filename):
    for block in tex.split("\\begin{figure}")[1:]:
        if filename in block:
            return block.split("\\end{figure}")[0]
    raise AssertionError(f"no figure including {filename} in:\n{tex}")


class TestStrengthLadder:
    def test_versions_are_grouped_in_declaration_order(self):
        series = report_charts.version_series(
            ["GaussianNoiseAttack (mild)", "GaussianNoiseAttack (severe)",
             "EchoAttack", "LowpassFilterAttack (a)"],
            {"GaussianNoiseAttack (mild)": 90.0,
             "GaussianNoiseAttack (severe)": 60.0,
             "EchoAttack": 70.0,
             "LowpassFilterAttack (a)": 80.0}.get,
        )
        # Only the attack with two or more versions makes a ladder, and the
        # versions keep the order the configuration declared them in.
        assert series == {"GaussianNoise": [("mild", 90.0), ("severe", 60.0)]}

    def test_the_curve_is_drawn_only_for_a_ladder(self, tmp_path):
        config = write_config(tmp_path)
        build_basic(tmp_path, config)
        assert (tmp_path / LADDER).exists()

        single = {
            path: {**data, "attacks": {"EchoAttack": data["attacks"]["EchoAttack"]}}
            for path, data in make_results().items()
        }
        other = tmp_path / "single"
        other.mkdir()
        assert "attack_strength" not in build_basic(
            other, write_config(tmp_path), results=single,
        )


class TestChartsNeverBreakAReport:
    def test_a_failing_chart_returns_false_instead_of_raising(self, tmp_path):
        # An unwritable path is the simplest real failure; the guard has to
        # turn it into a missing figure, not a lost report.
        assert report_charts.accuracy_ranking(
            {"A": 10.0}, str(tmp_path / "nope" / "x.png"),
        ) is False

    def test_empty_data_is_declined_not_drawn(self, tmp_path):
        assert report_charts.accuracy_ranking({}, str(tmp_path / "a.png")) is False
        assert report_charts.per_file_outcome_bars(
            {"A": []}, str(tmp_path / "b.png"), "t",
        ) is False
        assert report_charts.accuracy_heatmap(
            ["a"], ["m"], [[1.0]], str(tmp_path / "c.png"),
        ) is False


class TestLabelsRenderAsText:
    @pytest.mark.parametrize("latex,expected", [
        ("ViSQOL (1--5)", "ViSQOL (1–5)"),
        ("PESQ --- Audio Editing", "PESQ — Audio Editing"),
        ("Codec2Vocoder\\_700", "Codec2Vocoder_700"),
        ("Accuracy (\\%)", "Accuracy (%)"),
    ])
    def test_latex_fragments_become_plain_text(self, latex, expected):
        assert report_charts.plain(latex) == expected

    def test_lower_is_better_metrics_say_so(self):
        assert report_charts.direction_hint("MCD (dB)", False).endswith(
            "lower is better"
        )
        assert report_charts.direction_hint("PESQ", True) == "PESQ"


class TestPerFileOutcomeFigure:
    """The figure a zero-bit model gets has to say something.

    A zero-bit detector scores every file 0 or 100, so a box plot of that
    distribution collapses onto the two ends and shows nothing. The
    outcome bands stay readable, and drop the middle one that could never
    be occupied.
    """

    def test_a_zero_bit_model_gets_two_bands_not_three(self, tmp_path):
        out = tmp_path / "zero.png"
        assert report_charts.per_file_outcome_bars(
            {"Echo": [100.0, 0.0, 100.0, 0.0, 100.0]}, str(out),
            "t", chance_floor=0.0, is_zero_bit=True,
        )
        assert out.exists()

    def test_a_multi_bit_model_splits_three_ways(self, tmp_path):
        out = tmp_path / "multi.png"
        assert report_charts.per_file_outcome_bars(
            {"Echo": [100.0, 82.0, 41.0, 100.0]}, str(out),
            "t", chance_floor=50.0, is_zero_bit=False,
        )
        assert out.exists()

    def test_a_single_file_still_draws_when_the_bands_differ(self, tmp_path):
        """The old box plot needed two values per attack; counts need one."""
        out = tmp_path / "one.png"
        assert report_charts.per_file_outcome_bars(
            {"Echo": [100.0], "Lowpass": [20.0]}, str(out), "t",
        )
        assert out.exists()

    def test_it_declines_when_every_file_is_in_the_same_band(self, tmp_path):
        """Every bar spans the full width, because the bands are a
        composition. Identical full-width bars beside an accuracy table
        read as "everything scored 100%", which is a different claim."""
        out = tmp_path / "flat.png"
        assert report_charts.per_file_outcome_bars(
            {"Echo": [100.0] * 5, "Lowpass": [100.0] * 5}, str(out), "t",
        ) is False
        assert not out.exists()

    def test_it_draws_when_two_rows_sit_in_different_bands(self, tmp_path):
        """Uniform rows still differ in colour, which is worth showing."""
        out = tmp_path / "split.png"
        assert report_charts.per_file_outcome_bars(
            {"Echo": [100.0] * 5, "Lowpass": [80.0] * 5}, str(out), "t",
        )
        assert out.exists()

    def test_the_detailed_report_uses_it_for_a_zero_bit_model(self, tmp_path):
        config = write_config(tmp_path)
        # A zero-bit detector scores 0 or 100. The files have to actually
        # split, or the figure rightly declines as uninformative.
        results = make_results()
        for index, data in enumerate(results.values()):
            for attack in data["attacks"].values():
                attack["accuracy"] = 100.0 if index % 2 else 0.0

        DetailedReportGenerator(str(tmp_path), resolver=config.resolver) \
            .generate_full_report(results, model_name="Zero", is_zero_bit=True)
        tex = (tmp_path / "detailed_report.tex").read_text()
        assert_figures_exist(tex, tmp_path)
        assert "either yielded a detection or did not" in tex


class TestFigurePlacement:
    """A figure belongs under the table it draws, not at the end of a section.

    The scatter plots one quality metric. Emitted after the last quality
    table it would follow a table of a different metric, and the reader has
    to re-anchor it.
    """

    @staticmethod
    def _labels_in_order(tex):
        return re.findall(r"\\label\{((?:tab|fig):[^}]*)\}", tex)

    def test_the_scatter_follows_its_own_metric_table(self, tmp_path):
        tex = build_basic(tmp_path, write_config(tmp_path))
        labels = self._labels_in_order(tex)

        figure = "fig:robustness_quality_audio_distortion"
        assert figure in labels, labels
        before = labels[labels.index(figure) - 1]
        assert before.startswith("tab:"), before

        metric = before.rsplit("_audio_distortion", 1)[0].split("_", 1)[1]
        # The caption names the metric the way the tables do, so the label
        # is what has to match, not the config key.
        assert metric_label(metric) in _figure_block(tex, SCATTER), (
            f"the figure sits under the {metric} table but does not plot it"
        )

    def test_the_detailed_scatter_follows_its_own_metric_table(self, tmp_path):
        config = write_config(tmp_path)
        DetailedReportGenerator(str(tmp_path), resolver=config.resolver) \
            .generate_full_report(make_results(), model_name="TestModel")
        tex = (tmp_path / "detailed_report.tex").read_text()
        labels = self._labels_in_order(tex)

        scatters = [
            (index, label) for index, label in enumerate(labels)
            if label.startswith("fig:scatter_")
        ]
        assert scatters, labels
        for index, label in scatters:
            metric = label.split("fig:scatter_", 1)[1].split("_", 1)[0]
            previous = labels[index - 1]
            assert previous.startswith("tab:"), previous
            assert previous.endswith(f"_{metric}") or metric in previous, (
                f"{label} follows {previous}, which is a different metric"
            )


class TestNoAttacksDetectedColumn:
    """The count of detected files comes from the model, or not at all.

    ``is_watermarked()`` is the only thing that knows what a model's
    ``detect()`` output means. A report that applied its own threshold
    would be guessing, and would disagree with the detection_reliability
    mode on the same file.
    """

    def _tex(self, tmp_path, **kwargs):
        config = write_config(tmp_path, mode="no_attacks")
        generate_no_attacks_report(
            {"PerthModel": _no_attack_files(**kwargs)},
            report_dir=str(tmp_path), resolver=config.resolver,
        )
        return (tmp_path / "no_attacks_report.tex").read_text()

    def test_shown_when_the_model_answers_is_watermarked(self, tmp_path):
        tex = self._tex(tmp_path, supports_detection=True, is_zero_bit=True)
        assert "Detected" in tex
        assert "4/5" in tex, tex

    def test_omitted_when_the_model_does_not(self, tmp_path):
        tex = self._tex(tmp_path, supports_detection=False, is_zero_bit=True)
        assert "Detected" not in tex

    def test_a_multi_bit_model_that_answers_gets_the_column_too(self, tmp_path):
        """The column follows the method, not the zero-bit flag."""
        tex = self._tex(tmp_path, supports_detection=True, is_zero_bit=False)
        assert "Detected" in tex


class TestNoAttacksFiguresNeedSomethingToCompare:
    def test_a_single_model_run_has_no_figures(self, tmp_path):
        config = write_config(tmp_path, mode="no_attacks")
        generate_no_attacks_report(
            {"PerthModel": _no_attack_files()},
            report_dir=str(tmp_path), resolver=config.resolver,
        )
        tex = (tmp_path / "no_attacks_report.tex").read_text()
        assert not included_figures(tex), (
            "one model is one bar; the table says it better"
        )
        assert not list(tmp_path.glob("no_attacks_*.png"))


class TestNoAttacksAccuracyFigures:
    """Accuracy is charted for mean and worst case only, and only if asked.

    Those two are the typical case and the floor under it. The
    percentiles between them describe a distribution, which the table
    states more precisely than bars can, and a config naming all eight
    statistics must not turn into eight charts.
    """

    CHART = "no_attacks_accuracy_values.png"

    def _build(self, tmp_path, statistics=None):
        accuracy = {"enabled": True}
        if statistics is not None:
            accuracy["statistics"] = statistics
        config = write_config(tmp_path, mode="no_attacks",
                              metrics={"defaults": {"accuracy": accuracy}})
        generate_no_attacks_report(
            {"AudioSealModel": _no_attack_files(0.0),
             "WavMarkModel": _no_attack_files(0.3)},
            report_dir=str(tmp_path), resolver=config.resolver,
        )
        return (tmp_path / "no_attacks_report.tex").read_text()

    def _caption(self, tex):
        return _figure_block(tex, self.CHART).lower()

    def test_both_are_charted_when_both_are_configured(self, tmp_path):
        caption = self._caption(self._build(tmp_path, ["mean", "worst_case"]))
        assert "mean" in caption and "worst case" in caption

    def test_only_the_one_that_is_configured_is_charted(self, tmp_path):
        caption = self._caption(self._build(tmp_path, ["mean"]))
        assert "mean" in caption and "worst case" not in caption

    def test_they_keep_the_order_the_configuration_wrote(self, tmp_path):
        caption = self._caption(self._build(tmp_path, ["worst_case", "mean"]))
        assert caption.index("worst case") < caption.index("mean")

    @pytest.mark.parametrize("statistics", [
        ["median", "p95"], ["std"], ["p5", "p10", "p99"],
    ])
    def test_no_chart_when_neither_is_configured(self, tmp_path, statistics):
        tex = self._build(tmp_path, statistics)
        assert not (tmp_path / self.CHART).exists()
        assert self.CHART not in tex

    def test_the_others_are_left_to_the_table(self, tmp_path):
        tex = self._build(tmp_path, ["mean", "std", "p95"])
        caption = self._caption(tex)
        assert "mean" in caption
        assert "p95" not in caption and "std" not in caption
        assert "remaining statistics" in caption, caption

    def test_all_eight_produce_one_chart_of_two_bars(self, tmp_path):
        """The fallback to all eight must not become eight figures."""
        tex = self._build(tmp_path, None)
        charts = list(tmp_path.glob("no_attacks_accuracy_*.png"))
        assert len(charts) <= 2, [c.name for c in charts]
        caption = self._caption(tex)
        assert "mean, worst case" in caption
        for statistic in ("median", "p5", "p10", "p95", "p99", "std"):
            assert statistic not in caption

    def test_the_figure_follows_the_accuracy_tables(self, tmp_path):
        """One figure over every model, so it sits after the last of them."""
        tex = self._build(tmp_path, ["mean"])
        labels = re.findall(r"\\label\{((?:tab|fig):[^}]*)\}", tex)
        figure = labels.index("fig:no_attacks_accuracy_values")
        accuracy_tables = [
            index for index, label in enumerate(labels)
            if label.startswith(("tab:no_attacks_multibit",
                                 "tab:no_attacks_zerobit"))
        ]
        assert accuracy_tables, labels
        assert figure == max(accuracy_tables) + 1, labels

    def test_one_model_per_family_still_gets_a_figure(self, tmp_path):
        """Splitting the figure by family the way the tables are split
        would leave the common mixed-family run with nothing to show."""
        config = write_config(tmp_path, mode="no_attacks")
        generate_no_attacks_report(
            {"AudioSealModel": _no_attack_files(0.0, is_zero_bit=False),
             "PerthModel": _no_attack_files(0.2, is_zero_bit=True)},
            report_dir=str(tmp_path), resolver=config.resolver,
        )
        tex = (tmp_path / "no_attacks_report.tex").read_text()
        assert (tmp_path / self.CHART).exists(), "no figure for a mixed pair"
        assert "Zero-bit model" in _figure_block(tex, self.CHART)

    def test_accuracy_is_drawn_on_the_full_percentage_range(self):
        assert report_charts.metric_axis_range("accuracy", [88.0, 97.0]) == (
            0.0, 100.0
        )


class TestDetectionReliabilityFigures:
    def _result(self, fn_counts):
        attacks = {
            name: {
                "false_positive_count": 0, "false_positive_attempts": 8,
                "false_negative_count": count, "false_negative_attempts": 8,
                "accuracy_mean": 100.0 - count * 12,
                "accuracy_n": 8, "metrics": {},
            }
            for name, count in fn_counts.items()
        }
        return {
            "model_name": "PerthModel", "n_files": 8,
            "no_attack": {"false_positive_count": 0, "false_negative_count": 0},
            "attacks": attacks,
        }

    def _build(self, tmp_path, fn_counts):
        config = write_config(tmp_path, mode="detection_reliability")
        generate_detection_reliability_report(
            self._result(fn_counts), report_dir=str(tmp_path),
            resolver=config.resolver,
        )
        return (tmp_path / "detection_reliability_report.tex").read_text()

    def test_an_all_zero_group_gets_no_error_rate_figure(self, tmp_path):
        """A group the detector never erred on draws a row of empty axes,
        which says less than the table's column of zeros."""
        tex = self._build(tmp_path, {
            "LowpassFilterAttack": 0, "BandstopFilterAttack": 0,
            "EchoAttack": 0, "PCMQuantizationAttack": 0,
        })
        assert "fig:dr_error_rates" not in tex
        assert not list(tmp_path.glob("dr_error_rates_*.png"))

    def test_a_group_with_errors_keeps_its_figure(self, tmp_path):
        tex = self._build(tmp_path, {
            "LowpassFilterAttack": 3, "BandstopFilterAttack": 0,
            "EchoAttack": 1, "PCMQuantizationAttack": 0,
        })
        assert "fig:dr_error_rates" in tex
        assert_figures_exist(tex, tmp_path)

    def test_the_accuracy_figure_follows_its_table(self, tmp_path):
        tex = self._build(tmp_path, {
            "GaussianNoiseAttack": 1, "PinkNoiseAttack": 2,
            "SignInversionAttack": 3, "LPCAttack": 0,
        })
        labels = re.findall(r"\\label\{((?:tab|fig):[^}]*)\}", tex)
        accuracy_table = next(i for i, l in enumerate(labels)
                              if l.startswith("tab:dr_acc_"))
        assert labels[accuracy_table + 1].startswith("fig:dr_accuracy_"), labels


class TestDurationPartsAreSelfContained:
    def test_a_part_restarts_the_section_numbering(self):
        """Each part is a report over its own files, so its sections are
        its first and second, not the document's seventh and eighth."""
        heading = part_heading("$<$ 5.0s", "3 files")
        assert "\\part{$<$ 5.0s (3 files)}" in heading
        assert "\\setcounter{section}{0}" in heading

    def test_the_two_sides_of_one_boundary_get_different_slugs(self):
        """Stripping the comparison as punctuation collapsed them onto one
        slug, so both bins wrote their figures to the same filenames."""
        assert slugify("< 5.0s") != slugify("> 5.0s")

    @pytest.mark.parametrize("label", ["< 4.0s", "4.0-6.0s", "≥ 6.0s",
                                       "Overall", "???"])
    def test_every_label_yields_a_usable_slug(self, label):
        slug = slugify(label)
        assert slug and all(c.isalnum() or c == "_" for c in slug)


class TestTheLastDurationBinIsInclusive:
    """A file exactly on the last boundary goes into the last bin, which
    was labelled ``> b`` -- describing a population that excluded it."""

    def test_a_file_on_the_last_boundary_is_in_a_bin_that_says_so(
        self, tmp_path,
    ):
        import numpy as np
        import soundfile as sf
        from deepmarkpy.utils.utils import partition_files_by_duration

        path = tmp_path / "two_seconds.wav"
        sf.write(str(path), np.zeros(32000, dtype=np.float32), 16000)

        (label, files), = partition_files_by_duration([str(path)], [1.0, 2.0])
        assert files == [str(path)]
        assert label == "≥ 2.0s"

    def test_the_config_names_the_bins_the_run_uses(self):
        from deepmarkpy.config import ModeConfig
        from deepmarkpy.utils.utils import duration_bin_labels

        config = ModeConfig(mode="benchmark", source="c.json",
                            duration_boundaries=[5.0, 10.0])
        assert config.duration_labels() == duration_bin_labels([5.0, 10.0])

    def test_the_two_sides_of_one_boundary_still_get_different_slugs(self):
        assert slugify("< 5.0s") != slugify("≥ 5.0s")

    def test_a_grouped_report_carries_no_raw_sign_pdflatex_cannot_set(
        self, tmp_path,
    ):
        """≥ has no glyph under pdflatex's default input encoding, in the
        heading text or in a \\label name."""
        from deepmarkpy.utils.report_generator import BenchmarkReportGenerator

        stats = {label: {"n_files": 2, "stats": {
            "GaussianNoiseAttack": {"accuracy_mean": 93.0, "accuracy_n": 2},
        }} for label in ("< 5.0s", "≥ 5.0s")}
        stats_file = tmp_path / "stats.json"
        stats_file.write_text(json.dumps(stats))
        BenchmarkReportGenerator(str(tmp_path)).generate_full_report(
            str(stats_file), "TestModel",
        )
        tex = (tmp_path / "benchmark_report.tex").read_text()
        assert "≥" not in tex
        assert "$\\geq$ 5.0s" in tex
