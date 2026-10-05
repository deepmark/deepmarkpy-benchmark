"""A generated report's columns are exactly what its config asked for.

Not "at least" and not "roughly": every table header in the ``.tex`` is
compared against the metric and statistic lists the config file declares.
This is the check that stops a report generator from quietly adding a
column of its own.

The reports are driven end to end from synthetic per-file results, so
the aggregation and the rendering are both covered.
"""

import json
import re

import pytest

from deepmarkpy.benchmark import Benchmark
from deepmarkpy.config import load_configs
from deepmarkpy.utils.detailed_report_generator import DetailedReportGenerator
from deepmarkpy.utils.latex_helpers import metric_label, stat_header
from deepmarkpy.utils.no_attacks_report_generator import generate_no_attacks_report
from deepmarkpy.utils.report_generator import BenchmarkReportGenerator

# One attack per group so several sections are exercised at once.
ATTACKS = {
    "GaussianNoiseAttack": "audio_distortion",
    "TimeStretchAttack": "desynchronization",
    "CropBeginningAttack": "audio_editing",
}

ALL_SIGNAL_METRICS = [
    "pesq", "psnr", "si_sdr", "mcd", "visqol", "stoi", "sii", "ncm",
    "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud",
]


def make_results(n_files=4):
    """Per-file results carrying every metric, so nothing is dropped for
    lack of data and a missing column can only come from the config."""
    quality = {m: 3.0 + i * 0.1 for i, m in enumerate(ALL_SIGNAL_METRICS)}
    results = {}
    for index in range(n_files):
        results[f"f{index}.wav"] = {
            "watermarked_audio_quality": dict(quality),
            "attacks": {
                name: {
                    "accuracy": 80.0 + index * 5,
                    "detection_valid": True,
                    "attacked_audio_quality_wm": dict(quality),
                }
                for name in ATTACKS
            },
        }
    return results


@pytest.fixture(autouse=True)
def _skip_pdflatex(monkeypatch):
    """Assert on the .tex, not on pdflatex.

    These tests are about which columns the generators emit; running
    pdflatex twice per report adds ~25s and checks nothing they claim.
    """
    for module in ("report_generator", "detailed_report_generator",
                   "no_attacks_report_generator",
                   "detection_reliability_report_generator"):
        monkeypatch.setattr(
            f"deepmarkpy.utils.{module}.compile_latex",
            lambda *a, **k: None, raising=False,
        )
    monkeypatch.setattr(
        "deepmarkpy.utils.latex_helpers.compile_latex",
        lambda *a, **k: None,
    )


def write_config(tmp_path, **overrides):
    data = {
        "mode": "benchmark",
        "models": ["AudioSealModel"],
        "calculate_quality_metrics": True,
    }
    data.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return load_configs([str(path)])[0]


def header_rows(tex):
    """Every table header row in the document, split into cells."""
    rows = []
    for line in tex.splitlines():
        stripped = line.strip()
        if not stripped.endswith("\\\\") or "&" not in stripped:
            continue
        # Header rows are the ones directly naming a row-label column.
        if stripped.split("&")[0].strip() in ("Attack Type", "Attack",
                                              "Condition", "Model", "Metric"):
            cells = [c.strip() for c in stripped.rstrip("\\").split("&")]
            if cells not in rows:
                rows.append(cells)
    return rows


def table_for(tex, label):
    """The single table body carrying ``\\label{label}``."""
    blocks = tex.split("\\begin{longtable}")
    for block in blocks[1:]:
        if f"\\label{{{label}}}" in block:
            return block.split("\\end{longtable}")[0]
    raise AssertionError(f"no table labelled {label} in:\n{tex}")


def build_benchmark_tex(tmp_path, config):
    """Run aggregation and rendering exactly as the CLI does."""
    benchmark = Benchmark.__new__(Benchmark)
    stats = benchmark.compute_mean_accuracy(make_results(), resolver=config.resolver)

    stats_file = tmp_path / "benchmark_stats.json"
    stats_file.write_text(json.dumps(stats))

    generator = BenchmarkReportGenerator(str(tmp_path), resolver=config.resolver)
    generator.generate_full_report(str(stats_file), model_name="TestModel")
    return (tmp_path / "benchmark_report.tex").read_text(), stats


class TestAccuracyColumnsMatchTheConfig:
    @pytest.mark.parametrize("statistics", [
        ["mean"],
        ["mean", "worst_case"],
        ["median", "p10", "p99"],
        ["worst_case", "std", "mean", "p5"],
    ])
    def test_exactly_the_configured_statistics_appear(self, tmp_path, statistics):
        config = write_config(tmp_path, statistics=statistics, metrics={
            "defaults": {
                "accuracy": {"enabled": True, "statistics": statistics},
                "ber": {"enabled": False},
                "emr": {"enabled": False},
                **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
            },
        })
        tex, _ = build_benchmark_tex(tmp_path, config)
        table = table_for(tex, "tab:benchmark_accuracy_audio_distortion")
        header = next(l for l in table.splitlines() if "Attack Type" in l)
        cells = [c.strip() for c in header.strip().rstrip("\\").split("&")][1:]

        assert cells == [stat_header(s) for s in statistics], (
            "accuracy columns are not exactly, and only, what the config listed"
        )

    def test_column_order_follows_the_configured_order(self, tmp_path):
        statistics = ["worst_case", "mean", "median"]
        config = write_config(tmp_path, metrics={"defaults": {
            "accuracy": {"enabled": True, "statistics": statistics},
            "ber": {"enabled": False}, "emr": {"enabled": False},
            **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
        }})
        tex, _ = build_benchmark_tex(tmp_path, config)
        table = table_for(tex, "tab:benchmark_accuracy_audio_distortion")
        header = next(l for l in table.splitlines() if "Attack Type" in l)
        assert header.index("Worst Case") < header.index("Mean") < header.index("Median")

    def test_adding_std_makes_a_std_column_appear(self, tmp_path):
        """Requirement 6, stated literally: a statistic a report never showed
        before must appear once the config asks for it."""
        without = write_config(tmp_path, metrics={"defaults": {
            "accuracy": {"enabled": True, "statistics": ["mean"]},
            "ber": {"enabled": False}, "emr": {"enabled": False},
            **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
        }})
        tex_without, _ = build_benchmark_tex(tmp_path, without)
        assert "Std" not in tex_without

        with_std = write_config(tmp_path, metrics={"defaults": {
            "accuracy": {"enabled": True, "statistics": ["mean", "std"]},
            "ber": {"enabled": False}, "emr": {"enabled": False},
            **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
        }})
        tex_with, _ = build_benchmark_tex(tmp_path, with_std)
        assert "Std" in tex_with


class TestMetricTablesMatchTheConfig:
    def test_only_the_enabled_metrics_get_tables(self, tmp_path):
        config = write_config(tmp_path, statistics=["mean", "p95"], metrics={
            "defaults": {
                **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
                "pesq": {"enabled": True},
                "mcd": {"enabled": True},
            },
        })
        tex, _ = build_benchmark_tex(tmp_path, config)

        labels = set(re.findall(r"\\label\{(tab:benchmark_[a-z0-9_]+)\}", tex))
        metric_labels = {
            l for l in labels
            if not l.startswith("tab:benchmark_accuracy")
            and not l.startswith("tab:benchmark_metrics")
        }
        for label in metric_labels:
            assert "_pesq_" in label or "_mcd_" in label, (
                f"a table appeared for a metric the config disabled: {label}"
            )

    def test_a_group_showing_a_metric_another_group_hides(self, tmp_path):
        config = write_config(tmp_path, statistics=["mean", "p95"], metrics={
            "defaults": {
                **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
                "mcd": {"enabled": True},
            },
            "per_group": {"audio_distortion": {"mcd": {"enabled": False}}},
        })
        tex, _ = build_benchmark_tex(tmp_path, config)

        assert "tab:benchmark_mcd_desynchronization" in tex
        assert "tab:benchmark_mcd_audio_distortion" not in tex

    def test_a_single_statistic_collapses_metrics_into_one_table(self, tmp_path):
        config = write_config(tmp_path, statistics=["mean"], metrics={
            "defaults": {
                **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
                "pesq": {"enabled": True},
                "stoi": {"enabled": True},
            },
        })
        tex, _ = build_benchmark_tex(tmp_path, config)

        table = table_for(tex, "tab:benchmark_metrics_audio_distortion")
        header = next(l for l in table.splitlines() if "Attack Type" in l)
        cells = [c.strip() for c in header.strip().rstrip("\\").split("&")][1:]
        assert cells == [metric_label("pesq"), metric_label("stoi")]

    def test_a_non_mean_single_statistic_is_named_in_the_header(self, tmp_path):
        """A bare quality column has always meant the mean, so anything else
        has to say so."""
        config = write_config(tmp_path, metrics={"defaults": {
            **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
            "pesq": {"enabled": True, "statistics": ["worst_case"]},
        }})
        tex, _ = build_benchmark_tex(tmp_path, config)
        table = table_for(tex, "tab:benchmark_metrics_audio_distortion")
        assert "[Worst Case]" in table

    def test_ber_and_emr_appear_only_when_enabled(self, tmp_path):
        off = write_config(tmp_path, statistics=["mean"], metrics={"defaults": {
            "ber": {"enabled": False}, "emr": {"enabled": False},
            **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
        }})
        tex_off, _ = build_benchmark_tex(tmp_path, off)
        assert "BER" not in tex_off and "EMR" not in tex_off

        on = write_config(tmp_path, statistics=["mean"], metrics={"defaults": {
            "ber": {"enabled": True}, "emr": {"enabled": True},
            **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
        }})
        tex_on, _ = build_benchmark_tex(tmp_path, on)
        assert "BER" in tex_on and "EMR" in tex_on


class TestSectionsFollowTheAttackGroups:
    def test_one_section_per_group_present_in_the_results(self, tmp_path):
        config = write_config(tmp_path, statistics=["mean"])
        tex, _ = build_benchmark_tex(tmp_path, config)
        sections = re.findall(r"\\section\{([^}]*)\}", tex)
        assert "Audio Distortion Attacks" in sections
        assert "Desynchronization Attacks" in sections
        assert "Audio Editing Attacks" in sections

    def test_a_group_with_no_attacks_gets_no_section(self, tmp_path):
        config = write_config(tmp_path, statistics=["mean"])
        tex, _ = build_benchmark_tex(tmp_path, config)
        assert "AI Attacks" not in tex


class TestComputationFollowsTheConfig:
    def test_a_disabled_metric_is_never_aggregated(self, tmp_path):
        """Not just hidden: the key must not be in benchmark_stats.json."""
        config = write_config(tmp_path, statistics=["mean"], metrics={
            "defaults": {
                **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
                "pesq": {"enabled": True},
            },
        })
        _, stats = build_benchmark_tex(tmp_path, config)
        entry = stats["GaussianNoiseAttack"]
        assert "pesq_mean" in entry
        for metric in ALL_SIGNAL_METRICS:
            if metric != "pesq":
                assert not any(k.startswith(f"{metric}_") for k in entry), (
                    f"{metric} was aggregated despite being disabled"
                )

    def test_only_the_configured_statistics_are_computed(self, tmp_path):
        config = write_config(tmp_path, metrics={"defaults": {
            "accuracy": {"enabled": True, "statistics": ["mean", "median"]},
            "ber": {"enabled": False}, "emr": {"enabled": False},
            **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
        }})
        _, stats = build_benchmark_tex(tmp_path, config)
        accuracy_keys = {
            k for k in stats["GaussianNoiseAttack"] if k.startswith("accuracy_")
        }
        assert accuracy_keys == {"accuracy_mean", "accuracy_median", "accuracy_n"}


class TestDetailedReportFollowsTheConfig:
    def test_subgroup_metric_tables_use_the_subgroup_configuration(self, tmp_path):
        config = write_config(tmp_path, statistics=["mean", "p95"], metrics={
            "defaults": {
                **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
                "pesq": {"enabled": True},
            },
            "per_group": {"temporal_editing": {"pesq": {"enabled": False}}},
        })
        generator = DetailedReportGenerator(str(tmp_path), resolver=config.resolver)
        generator.generate_full_report(make_results(), model_name="TestModel")
        tex = (tmp_path / "detailed_report.tex").read_text()

        # CropBeginningAttack lives in temporal_editing, which switched pesq
        # off; the subsection must therefore carry no PESQ table.
        assert "Temporal Editing" in tex
        assert "tab:qual_audio_editing_temporal_editing_pesq" not in tex

    def test_every_metric_table_carries_the_no_attack_baseline(self, tmp_path):
        config = write_config(tmp_path, statistics=["mean", "p95"], metrics={
            "defaults": {
                **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
                "pesq": {"enabled": True},
            },
        })
        generator = DetailedReportGenerator(str(tmp_path), resolver=config.resolver)
        generator.generate_full_report(make_results(), model_name="TestModel")
        tex = (tmp_path / "detailed_report.tex").read_text()
        assert "No Attack (watermark only)" in tex

    def test_detailed_statistic_columns_match_the_config(self, tmp_path):
        statistics = ["median", "p99"]
        config = write_config(tmp_path, metrics={"defaults": {
            **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
            "pesq": {"enabled": True, "statistics": statistics},
        }})
        generator = DetailedReportGenerator(str(tmp_path), resolver=config.resolver)
        generator.generate_full_report(make_results(), model_name="TestModel")
        tex = (tmp_path / "detailed_report.tex").read_text()

        for row in header_rows(tex):
            if row[0] == "Condition":
                assert row[1:] == [stat_header(s) for s in statistics]


@pytest.mark.parametrize("is_zero_bit", [False, True])
@pytest.mark.parametrize("statistics", [["mean"], ["mean", "worst_case"]])
def test_ber_requires_payload_bits(tmp_path, is_zero_bit, statistics):
    """Binary detection results must not be presented as bit-error rates."""
    config = write_config(tmp_path, statistics=statistics)
    results = make_results(n_files=2)
    for index, file_data in enumerate(results.values()):
        for entry in file_data["attacks"].values():
            entry["accuracy"] = 100.0 if index else 0.0

    # Include BER in the input to exercise rendering of previously saved
    # statistics as well as newly computed reports.
    benchmark = Benchmark.__new__(Benchmark)
    stats = benchmark.compute_mean_accuracy(results, resolver=config.resolver)
    basic = BenchmarkReportGenerator(
        str(tmp_path), resolver=config.resolver, is_zero_bit=is_zero_bit,
    )
    basic_tex = basic.generate_latex_table(stats, group_key="audio_distortion")

    detailed = DetailedReportGenerator(str(tmp_path), resolver=config.resolver)
    aggregate = detailed.aggregate_results(results, is_zero_bit=is_zero_bit)
    detailed_tex = detailed._accuracy_table(
        aggregate, list(ATTACKS), "audio_distortion", "Robustness", "tab:test",
    )
    for tex in (basic_tex, detailed_tex):
        has_ber = "BER" in tex or "Bit error rate" in tex
        assert has_ber is not is_zero_bit
        assert "GaussianNoise" in tex
    for entry in aggregate["attacks"].values():
        assert ("ber" in entry["accuracy"]) is not is_zero_bit


class TestNoAttacksReportFollowsTheConfig:
    @staticmethod
    def _results(n_files=3):
        quality = {m: 3.0 + i * 0.1 for i, m in enumerate(ALL_SIGNAL_METRICS)}
        return {"AudioSealModel": {
            "is_zero_bit": False,
            "returns_confidence": False,
            "files": [
                {"file": f"f{i}.wav", "filepath": f"f{i}.wav",
                 "accuracy": 90.0 + i, "watermarked_audio_quality": dict(quality)}
                for i in range(n_files)
            ],
        }}

    def test_per_metric_statistics_are_no_longer_discarded(self, tmp_path):
        """Each metric gets the statistics its own configuration asks for."""
        path = tmp_path / "c.json"
        path.write_text(json.dumps({
            "mode": "no_attacks",
            "models": ["AudioSealModel"],
            "calculate_quality_metrics": True,
            "metrics": {"defaults": {
                **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
                "pesq": {"enabled": True,
                         "statistics": ["mean", "std", "worst_case"]},
            }},
        }))
        config = load_configs([str(path)])[0]

        generate_no_attacks_report(
            self._results(), report_dir=str(tmp_path), resolver=config.resolver,
        )
        tex = (tmp_path / "no_attacks_report.tex").read_text()

        table = table_for(tex, "tab:no_attacks_pesq")
        header = next(l for l in table.splitlines() if l.strip().startswith("Model"))
        cells = [c.strip() for c in header.strip().rstrip("\\").split("&")]
        assert cells == ["Model", "Mean", "Std", "Worst Case"]

    def test_only_enabled_metrics_are_tabled(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({
            "mode": "no_attacks",
            "models": ["AudioSealModel"],
            "calculate_quality_metrics": True,
            "metrics": {"defaults": {
                **{m: {"enabled": False} for m in ALL_SIGNAL_METRICS},
                "stoi": {"enabled": True},
            }},
        }))
        config = load_configs([str(path)])[0]
        generate_no_attacks_report(
            self._results(), report_dir=str(tmp_path), resolver=config.resolver,
        )
        tex = (tmp_path / "no_attacks_report.tex").read_text()

        assert metric_label("stoi") in tex
        assert metric_label("pesq") not in tex
        assert metric_label("mcd") not in tex


class TestWorstCaseFollowsTheMetricDirection:
    """"Worst" is the bad end of the range, which is not always the minimum.

    Accuracy and PESQ are worst at their smallest; latency, MCD and BER
    are worst at their largest. Taking the minimum for all of them
    reports the *best* case of every lower-is-better metric under the
    label "worst case".
    """

    @pytest.mark.parametrize("metric,values,expected", [
        ("accuracy", [60.0, 80.0, 95.0], 60.0),
        ("pesq", [2.1, 3.0, 4.0], 2.1),
        ("visqol", [3.0, 4.5], 3.0),
        ("mcd", [1.2, 2.0, 3.4], 3.4),
        ("ber", [0.01, 0.2], 0.2),
        ("embed_latency", [0.41, 0.45], 0.45),
        ("attack_latency", [0.02, 0.09], 0.09),
    ])
    def test_the_worst_value_is_the_bad_end(self, metric, values, expected):
        from deepmarkpy.utils.metric_resolver import worst_case_of

        assert worst_case_of(values, metric) == expected

    def test_the_aggregate_uses_it(self, tmp_path):
        """The rule has to reach the numbers a report prints, not just the
        helper."""
        config = write_config(tmp_path, statistics=["mean", "worst_case"],
                              efficiency={"enabled": True, "metrics": {
                                  "attack_latency": {
                                      "statistics": ["mean", "worst_case"]}}})
        results = make_results()
        for index, data in enumerate(results.values()):
            for attack in data["attacks"].values():
                attack["attack_latency"] = 0.10 + index * 0.05

        benchmark = Benchmark.__new__(Benchmark)
        stats = benchmark.compute_mean_accuracy(results, resolver=config.resolver)
        row = next(iter(stats.values()))

        assert row["attack_latency_worst_case"] > row["attack_latency_mean"], (
            "the slowest run must be the worst case, not the fastest"
        )
