"""No figure names a metric its report was told not to compute.

The tables are already held to this by ``test_report_config_fidelity``.
A figure is the easier place to break it, because it picks *one* metric
out of a preference order rather than iterating the configured list -- so
a wrong fallback shows a chart of a metric that appears in no table.

The check is deliberately blunt: for every metric the configuration
disables, neither its key nor its label may appear anywhere in the
generated ``.tex`` figure blocks or in the filenames written beside it.
"""

import json
import re

import pytest

from deepmarkpy.benchmark import Benchmark
from deepmarkpy.config import load_configs
from deepmarkpy.utils.detailed_report_generator import DetailedReportGenerator
from deepmarkpy.utils.latex_helpers import metric_label
from deepmarkpy.utils.no_attacks_report_generator import generate_no_attacks_report
from deepmarkpy.utils.report_generator import BenchmarkReportGenerator

SIGNAL_METRICS = [
    "pesq", "psnr", "si_sdr", "mcd", "visqol", "stoi", "sii", "ncm",
    "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud",
]

# calculate_quality_metrics: false leaves exactly these computable,
# whatever the enable flags say.
ALWAYS_ON = {"pesq", "visqol", "stoi"}

ATTACKS = ["GaussianNoiseAttack", "PinkNoiseAttack", "SignInversionAttack",
           "LowpassFilterAttack", "EchoAttack"]


@pytest.fixture(autouse=True)
def _skip_pdflatex(monkeypatch):
    for module in ("report_generator", "detailed_report_generator",
                   "no_attacks_report_generator"):
        monkeypatch.setattr(
            f"deepmarkpy.utils.{module}.compile_latex",
            lambda *a, **k: None, raising=False,
        )
    monkeypatch.setattr("deepmarkpy.utils.latex_helpers.compile_latex",
                        lambda *a, **k: None)


def make_results(n_files=5):
    """Per-file results carrying every metric, so nothing is missing for
    lack of data and a stray figure can only come from the config."""
    quality = {m: 2.5 + 0.1 * i for i, m in enumerate(SIGNAL_METRICS)}
    return {
        f"f{index}.wav": {
            "watermarked_audio_quality": dict(quality),
            "attacks": {
                name: {
                    "accuracy": 60.0 + 7 * index + position,
                    "detection_valid": True,
                    "attacked_audio_quality_wm": dict(quality),
                }
                for position, name in enumerate(ATTACKS)
            },
        }
        for index in range(n_files)
    }


def config_for(tmp_path, mode="benchmark", **overrides):
    data = {"mode": mode, "models": ["AudioSealModel"]}
    data.update(overrides)
    path = tmp_path / f"{mode}.json"
    path.write_text(json.dumps(data))
    return load_configs([str(path)])[0]


def figure_text(tex):
    """Every figure environment in the document, concatenated."""
    return "\n".join(re.findall(r"\\begin\{figure\}.*?\\end\{figure\}",
                                tex, re.DOTALL))


def assert_no_forbidden_metric(tex, directory, allowed):
    """No figure may name, or be filed under, a metric outside ``allowed``."""
    figures = figure_text(tex)
    written = " ".join(p.name for p in directory.glob("*.png"))
    for metric in SIGNAL_METRICS:
        if metric in allowed:
            continue
        assert metric_label(metric) not in figures, (
            f"a figure plots {metric}, which the config disabled"
        )
        assert f"_{metric}_" not in written and not written.count(
            f"_{metric}."
        ), f"a figure file was written for the disabled metric {metric}"


def build_basic(tmp_path, config):
    benchmark = Benchmark.__new__(Benchmark)
    stats = benchmark.compute_mean_accuracy(
        make_results(), resolver=config.resolver,
    )
    stats_file = tmp_path / "benchmark_stats.json"
    stats_file.write_text(json.dumps(stats))
    BenchmarkReportGenerator(str(tmp_path), resolver=config.resolver) \
        .generate_full_report(str(stats_file), model_name="TestModel")
    return (tmp_path / "benchmark_report.tex").read_text()


def build_detailed(tmp_path, config):
    DetailedReportGenerator(str(tmp_path), resolver=config.resolver) \
        .generate_full_report(make_results(), model_name="TestModel")
    return (tmp_path / "detailed_report.tex").read_text()


def build_no_attacks(tmp_path, config, n_models=2):
    quality = {m: 2.5 + 0.1 * i for i, m in enumerate(SIGNAL_METRICS)}
    results = {
        f"Model{index}": {
            "is_zero_bit": False,
            "files": [
                {"filepath": f"f{f}.wav", "accuracy": 95.0,
                 "watermarked_audio_quality": dict(quality)}
                for f in range(4)
            ],
        }
        for index in range(n_models)
    }
    generate_no_attacks_report(results, report_dir=str(tmp_path),
                               resolver=config.resolver)
    return (tmp_path / "no_attacks_report.tex").read_text()


ONE_METRIC = pytest.mark.parametrize("kept", ["pesq", "visqol", "mcd", "ncm"])


class TestBenchmarkReport:
    @ONE_METRIC
    def test_only_the_kept_metric_can_be_plotted(self, tmp_path, kept):
        config = config_for(tmp_path, calculate_quality_metrics=True,
                            metrics={"defaults": {
                                m: {"enabled": m == kept}
                                for m in SIGNAL_METRICS
                            }})
        tex = build_basic(tmp_path, config)
        assert_no_forbidden_metric(tex, tmp_path, {kept})

    def test_quality_metrics_off_leaves_only_the_always_on_trio(self, tmp_path):
        """The enable flags are ignored, so the figures must follow what is
        actually computed rather than what the block asks for."""
        config = config_for(tmp_path, calculate_quality_metrics=False,
                            metrics={"defaults": {
                                m: {"enabled": True} for m in SIGNAL_METRICS
                            }})
        tex = build_basic(tmp_path, config)
        assert_no_forbidden_metric(tex, tmp_path, ALWAYS_ON)

    def test_a_group_exclusion_is_not_undone_by_another_group(self, tmp_path):
        """A per-group figure must use that group's list, not the union.

        ViSQOL is on for audio_editing and off for audio_distortion, so the
        distortion section may not plot it -- the bug a union of every
        group's metrics would reintroduce.
        """
        config = config_for(
            tmp_path, calculate_quality_metrics=True,
            metrics={
                "defaults": {m: {"enabled": False} for m in SIGNAL_METRICS},
                "per_group": {
                    "audio_editing": {"visqol": {"enabled": True}},
                    "audio_distortion": {"pesq": {"enabled": True}},
                },
            },
        )
        tex = build_basic(tmp_path, config)
        distortion = [
            block for block in tex.split("\\section")
            if "Audio Distortion" in block
        ]
        assert distortion, tex
        assert metric_label("visqol") not in figure_text(distortion[0])
        assert metric_label("pesq") in figure_text(distortion[0])


class TestDetailedReport:
    @ONE_METRIC
    def test_only_the_kept_metric_can_be_plotted(self, tmp_path, kept):
        config = config_for(tmp_path, calculate_quality_metrics=True,
                            metrics={"defaults": {
                                m: {"enabled": m == kept}
                                for m in SIGNAL_METRICS
                            }})
        tex = build_detailed(tmp_path, config)
        assert_no_forbidden_metric(tex, tmp_path, {kept})

    def test_quality_metrics_off_leaves_only_the_always_on_trio(self, tmp_path):
        config = config_for(tmp_path, calculate_quality_metrics=False)
        tex = build_detailed(tmp_path, config)
        assert_no_forbidden_metric(tex, tmp_path, ALWAYS_ON)


class TestNoAttacksReport:
    @ONE_METRIC
    def test_only_the_kept_metric_gets_a_figure(self, tmp_path, kept):
        config = config_for(tmp_path, mode="no_attacks",
                            calculate_quality_metrics=True,
                            metrics={"defaults": {
                                m: {"enabled": m == kept}
                                for m in SIGNAL_METRICS
                            }})
        tex = build_no_attacks(tmp_path, config)
        assert_no_forbidden_metric(tex, tmp_path, {kept})
        assert metric_label(kept) in figure_text(tex)

    def test_quality_metrics_off_leaves_only_the_always_on_trio(self, tmp_path):
        config = config_for(tmp_path, mode="no_attacks",
                            calculate_quality_metrics=False)
        tex = build_no_attacks(tmp_path, config)
        assert_no_forbidden_metric(tex, tmp_path, ALWAYS_ON)


class TestNoMetricAtAll:
    def test_the_reports_still_build_without_a_single_quality_figure(
        self, tmp_path,
    ):
        """Accuracy-only figures stay; nothing metric-shaped is drawn."""
        config = config_for(tmp_path, calculate_quality_metrics=True,
                            metrics={"defaults": {
                                m: {"enabled": False} for m in SIGNAL_METRICS
                            }})
        tex = build_basic(tmp_path, config)
        assert_no_forbidden_metric(tex, tmp_path, set())
        # The ranking chart needs no quality metric at all.
        assert "benchmark_chart.png" in tex


class TestUngroupedSection:
    """``generate_latex_table`` without a group must resolve as ``None``.

    Falling back to the union across every group let a section whose own
    tables exclude ViSQOL carry a ViSQOL figure -- the exact failure the
    grouped path is protected from, reachable through the public method's
    default argument.
    """

    def test_the_figure_follows_defaults_not_the_union(self, tmp_path):
        config = config_for(
            tmp_path, calculate_quality_metrics=True,
            metrics={
                "defaults": {"visqol": {"enabled": False},
                             "pesq": {"enabled": True}},
                "per_group": {"audio_editing": {"visqol": {"enabled": True}}},
            },
        )
        resolver = config.resolver
        assert "visqol" in resolver.all_signal_metrics(), (
            "the union must contain it, or the test proves nothing"
        )
        assert "visqol" not in resolver.signal_metrics_for_group(None)

        generator = BenchmarkReportGenerator(str(tmp_path), resolver=resolver)
        stats = {
            f"{name}Attack": {
                "accuracy_n": 4, "accuracy_mean": 80.0 - 10 * index,
                "pesq_mean": 3.0, "visqol_mean": 4.2,
            }
            for index, name in enumerate(
                ["GaussianNoise", "PinkNoise", "SignInversion"]
            )
        }
        body = generator.generate_latex_table(stats)

        assert metric_label("visqol") not in figure_text(body)
        assert metric_label("pesq") in figure_text(body)
