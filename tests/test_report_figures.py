"""Every figure a report references exists, and only three kinds are drawn.

The ``.tex`` can point ``\\includegraphics`` at a file that was never
written, which fails at compile time, long after the run. The basic report
draws the accuracy ranking and the strength curves, the comparative report
draws the radar chart, and the detailed, no_attacks and
detection_reliability reports are tables only. All five are checked here.

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

# Two versions of one attack so the strength-curve figure has a ladder.
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

# The strength curves sit in the section whose accuracy table they draw,
# so their filename carries the group; audio_distortion holds the ladder.
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


def assert_no_figures(tex, directory):
    """A tables-only report references no figure and writes no image."""
    assert not included_figures(tex), included_figures(tex)
    assert not list(directory.glob("*.png"))


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
        assert set(included_figures(tex)) == {"benchmark_chart.png", LADDER}

    def test_detailed_report(self, tmp_path):
        config = write_config(tmp_path)
        DetailedReportGenerator(str(tmp_path), resolver=config.resolver) \
            .generate_full_report(make_results(), model_name="TestModel")
        assert_no_figures((tmp_path / "detailed_report.tex").read_text(),
                          tmp_path)

    def test_no_attacks_report_with_two_models(self, tmp_path):
        config = write_config(tmp_path, mode="no_attacks")
        generate_no_attacks_report(
            {"AudioSealModel": _no_attack_files(0.0),
             "WavMarkModel": _no_attack_files(0.4)},
            report_dir=str(tmp_path), resolver=config.resolver,
        )
        assert_no_figures((tmp_path / "no_attacks_report.tex").read_text(),
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
        assert_no_figures(tex, tmp_path)

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
        assert included_figures(tex) == ["radar_chart.png"]


class TestFiguresFollowTheConfiguration:
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

    @pytest.mark.parametrize("pink", [
        # Differently named versions, the longer ladder second.
        [("default", 98.0), ("light", 95.0), ("heavy", 80.0),
         ("extreme", 55.0)],
        # One naming scheme, but not every attack has every version.
        [("default", 98.0), ("aggressive", 55.0)],
    ])
    def test_each_ladder_is_drawn_against_its_own_version_names(
        self, tmp_path, monkeypatch, pink,
    ):
        """Two attacks in one group need not name their versions alike.

        Every point sits above its own version's name. A shared 0..n-1
        axis under one ladder's names would put the other ladder's points
        on versions it does not have.
        """
        drawn = {}
        save = report_charts._save

        def read_back(fig, path):
            ax = fig.axes[0]
            ticks = [tick.get_text() for tick in ax.get_xticklabels()]
            # The chance floor is an unlabelled line, so only the ladders
            # carry a name of their own.
            for line in ax.get_lines():
                if not line.get_label().startswith("_"):
                    drawn[line.get_label()] = [
                        (ticks[int(x)], float(y))
                        for x, y in zip(line.get_xdata(), line.get_ydata())
                    ]
            return save(fig, path)

        monkeypatch.setattr(report_charts, "_save", read_back)
        series = {
            "GaussianNoise": [("default", 99.0), ("mild", 90.0),
                              ("aggressive", 60.0)],
            "PinkNoise": pink,
        }
        assert report_charts.attack_strength_curves(
            series, str(tmp_path / "strength.png"), chance_floor=50.0,
        )
        assert drawn == series

    def test_a_shorter_ladder_first_keeps_the_longer_one_in_order(
        self, tmp_path, monkeypatch,
    ):
        """Each ladder reads left to right in its own version order.

        With the two-version ladder first, a tick axis in first-seen order
        would put ``mild`` after ``aggressive`` and draw ``GaussianNoise``
        right to left.
        """
        drawn = {}
        save = report_charts._save

        def read_back(fig, path):
            ax = fig.axes[0]
            drawn["ticks"] = [tick.get_text() for tick in ax.get_xticklabels()]
            for line in ax.get_lines():
                if not line.get_label().startswith("_"):
                    drawn[line.get_label()] = [int(x) for x in line.get_xdata()]
            return save(fig, path)

        monkeypatch.setattr(report_charts, "_save", read_back)
        assert report_charts.attack_strength_curves(
            {"PinkNoise": [("default", 98.0), ("aggressive", 55.0)],
             "GaussianNoise": [("default", 99.0), ("mild", 90.0),
                               ("aggressive", 60.0)]},
            str(tmp_path / "strength.png"), chance_floor=50.0,
        )
        assert drawn == {
            "ticks": ["default", "mild", "aggressive"],
            "PinkNoise": [0, 2],
            "GaussianNoise": [0, 1, 2],
        }


class TestChartsNeverBreakAReport:
    def test_a_failing_chart_returns_false_instead_of_raising(self, tmp_path):
        # An unwritable path is the simplest real failure; the guard has to
        # turn it into a missing figure, not a lost report.
        assert report_charts.accuracy_ranking(
            {"A": 10.0}, str(tmp_path / "nope" / "x.png"),
        ) is False

    def test_empty_data_is_declined_not_drawn(self, tmp_path):
        assert report_charts.accuracy_ranking({}, str(tmp_path / "a.png")) is False
        assert report_charts.attack_strength_curves(
            {"Echo": [("mild", 90.0)]}, str(tmp_path / "b.png"),
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
