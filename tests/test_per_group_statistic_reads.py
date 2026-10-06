"""A per-group statistic override must not silently read as zero.

``compute_mean_accuracy`` writes each attack's accuracy under the
statistic *that attack's group* configured. Anything reading a single
report-wide statistic off every attack therefore finds nothing on an
attack whose group overrode it -- and a missing accuracy read as 0.0 is
indistinguishable from a watermark that did not survive, so the charts,
the headline mean and the "most damaging" ranking were all wrong without
anything in the run saying so.

The tables were always right: they ask the resolver per section. These
pin the figures and the prose to the same answer.

Nor may a spread stand in for a level. A config may list ``std`` before
``mean``, and read as the accuracy, a std makes the steadiest attack look
the most damaging. Every single-number view therefore reads the first
configured statistic that is not ``std``.
"""

import numpy as np
import pytest

from deepmarkpy.benchmark import Benchmark
from deepmarkpy.config import load_config_data
from deepmarkpy.utils.comparative_report_generator import ComparativeReportGenerator
from deepmarkpy.utils.detailed_report_generator import DetailedReportGenerator
from deepmarkpy.utils.detection_reliability_report_generator import _accuracy_figure
from deepmarkpy.utils.report_generator import BenchmarkReportGenerator

# GaussianNoise is audio_distortion, Lowpass is audio_editing, so one
# group can be overridden while the other keeps the default.
NOISE = "GaussianNoiseAttack"
LOWPASS = "LowpassFilterAttack"

NOISE_ACCURACY = 90.0
LOWPASS_ACCURACY = 70.0

PINK_NOISE = "PinkNoiseAttack"
SIGN_INVERSION = "SignInversionAttack"

# One audio_distortion section whose spreads rank the attacks in the
# opposite order to their means.
SPREAD_STATS = {
    NOISE: {"accuracy_n": 5, "accuracy_mean": 100.0, "accuracy_std": 0.0},
    PINK_NOISE: {"accuracy_n": 5, "accuracy_mean": 82.0, "accuracy_std": 24.9},
    SIGN_INVERSION: {"accuracy_n": 5, "accuracy_mean": 70.0, "accuracy_std": 27.39},
}


def split_statistics_config(default_statistic, overridden_statistic):
    """A config whose audio_editing group uses a different statistic."""
    return load_config_data({
        "mode": "benchmark",
        "models": ["AudioSealModel"],
        "calculate_quality_metrics": True,
        "metrics": {
            "defaults": {
                "accuracy": {"enabled": True, "statistics": [default_statistic]},
            },
            "per_group": {
                "audio_editing": {
                    "accuracy": {
                        "enabled": True,
                        "statistics": [overridden_statistic],
                    },
                },
            },
        },
    }, quiet=True)


def std_first_config():
    """A config that lists the accuracy std before the mean."""
    return load_config_data({
        "mode": "benchmark",
        "models": ["AudioSealModel"],
        "metrics": {
            "defaults": {
                "accuracy": {"enabled": True, "statistics": ["std", "mean"]},
            },
        },
    }, quiet=True)


@pytest.fixture
def ranking_calls(monkeypatch):
    """Every accuracy ranking requested, as ``(values, kwargs)``; none is drawn."""
    calls = []
    monkeypatch.setattr(
        "deepmarkpy.utils.report_charts.accuracy_ranking",
        lambda values, path, **kwargs: calls.append((values, kwargs)) or True,
    )
    return calls


@pytest.fixture
def stats():
    """Per-attack statistics as the run loop writes them."""
    config = split_statistics_config("mean", "median")
    results = {
        f"file{i}.wav": {
            "attacks": {
                NOISE: {"accuracy": NOISE_ACCURACY},
                LOWPASS: {"accuracy": LOWPASS_ACCURACY},
            },
        }
        for i in range(4)
    }
    benchmark = object.__new__(Benchmark)
    computed = Benchmark.compute_mean_accuracy(
        benchmark, results, resolver=config.resolver,
    )
    return config, computed


class TestTheAggregateReallyDiffers:
    """The premise: the two attacks are stored under different keys."""

    def test_each_attack_carries_only_its_own_group_statistic(self, stats):
        _, computed = stats
        assert "accuracy_mean" in computed[NOISE]
        assert "accuracy_mean" not in computed[LOWPASS]
        assert "accuracy_median" in computed[LOWPASS]


class TestBasicReportFigures:
    def test_an_overridden_group_is_read_at_its_own_statistic(self, stats, tmp_path):
        config, computed = stats
        generator = BenchmarkReportGenerator(str(tmp_path), resolver=config.resolver)

        assert generator._accuracy_of(computed[LOWPASS], LOWPASS) == LOWPASS_ACCURACY
        assert generator._accuracy_of(computed[NOISE], NOISE) == NOISE_ACCURACY

    def test_the_headline_mean_covers_both_groups(self, stats, tmp_path):
        config, computed = stats
        generator = BenchmarkReportGenerator(str(tmp_path), resolver=config.resolver)

        expected = (NOISE_ACCURACY + LOWPASS_ACCURACY) / 2
        assert generator.calculate_mean_accuracy(computed) == expected

    def test_the_chart_label_does_not_name_one_groups_statistic(self, stats, tmp_path):
        """No single statistic describes the figure, so none is claimed."""
        config, computed = stats
        generator = BenchmarkReportGenerator(str(tmp_path), resolver=config.resolver)

        label = generator._accuracy_label_for(computed)
        assert "Mean" not in label and "Median" not in label

    def test_one_shared_statistic_is_still_named(self, tmp_path):
        config = split_statistics_config("median", "median")
        generator = BenchmarkReportGenerator(str(tmp_path), resolver=config.resolver)

        label = generator._accuracy_label_for({NOISE: {}, LOWPASS: {}})
        assert label == "Median"

    def test_an_unconfigured_statistic_falls_back_rather_than_reading_zero(
            self, tmp_path):
        """A stats dict from elsewhere still yields the number it holds."""
        config = split_statistics_config("mean", "median")
        generator = BenchmarkReportGenerator(str(tmp_path), resolver=config.resolver)

        assert generator._accuracy_of({"accuracy_p95": 88.0}, NOISE) == 88.0

    def test_a_std_listed_first_is_not_the_headline_accuracy(self, tmp_path):
        """The headline and the most damaging attack are read at the mean."""
        generator = BenchmarkReportGenerator(
            str(tmp_path), resolver=std_first_config().resolver,
        )

        tex = generator.generate_latex_report(SPREAD_STATS, "TestModel")
        assert "Accuracy (Mean) across the 3 attacks run:} 84.00" in tex
        assert "Most damaging:} SignInversion (70.0" in tex

    def test_a_steady_attack_is_not_read_as_a_failed_watermark(self, tmp_path):
        """Every mean here is 70 or more, so no attack is poor or at chance."""
        generator = BenchmarkReportGenerator(
            str(tmp_path), resolver=std_first_config().resolver,
        )

        tex = generator.generate_latex_report(SPREAD_STATS, "TestModel")
        assert "Poor Performance" not in tex
        assert "no better than guessing" not in tex


class TestComparativeReportFigures:
    """The radar and the heatmap draw one number per attack."""

    def test_the_heatmap_reads_each_attack_at_its_groups_statistic(
            self, stats, tmp_path):
        config, computed = stats
        generator = ComparativeReportGenerator(
            str(tmp_path), resolver=config.resolver,
        )

        assert generator._primary_value(computed[LOWPASS], LOWPASS) == LOWPASS_ACCURACY
        assert generator._primary_value(computed[NOISE], NOISE) == NOISE_ACCURACY

    def test_a_statistic_only_a_group_configures_still_gets_a_table(
            self, stats, tmp_path):
        """Otherwise the override is computed and then never shown."""
        config, computed = stats
        generator = ComparativeReportGenerator(
            str(tmp_path), resolver=config.resolver,
        )

        tabled = generator._statistics({"AudioSealModel": computed})
        assert "median" in tabled and "mean" in tabled

    def test_a_per_statistic_table_leaves_absent_cells_absent(
            self, stats, tmp_path):
        """A median table must not quietly print the mean instead."""
        config, computed = stats
        generator = ComparativeReportGenerator(
            str(tmp_path), resolver=config.resolver,
        )

        assert generator._value(computed[NOISE], "median") is None
        assert generator._value(computed[NOISE], "mean") == NOISE_ACCURACY

    def test_a_std_listed_first_is_not_what_the_figures_draw(self, tmp_path):
        """The radar and the heatmap draw the mean, and are labelled so."""
        generator = ComparativeReportGenerator(
            str(tmp_path), resolver=std_first_config().resolver,
        )

        assert generator._primary_value(SPREAD_STATS[PINK_NOISE], PINK_NOISE) == 82.0
        assert generator._primary_label({"AudioSealModel": SPREAD_STATS}) == "Mean"


class TestDetailedReportFigures:
    """Each section's ranking figure draws one number per attack."""

    def test_a_std_listed_first_is_not_what_a_section_ranks(
            self, ranking_calls, tmp_path):
        generator = DetailedReportGenerator(
            str(tmp_path), resolver=std_first_config().resolver,
        )
        aggregated = {"attacks": {
            name: {"accuracy": {"mean": entry["accuracy_mean"],
                                "std": entry["accuracy_std"]}}
            for name, entry in SPREAD_STATS.items()
        }}

        generator._ranking_figure(
            aggregated, list(SPREAD_STATS), "audio_distortion",
            "Audio distortion", "audio_distortion",
        )
        (values, kwargs), = ranking_calls
        assert values["SignInversion"] == 70.0
        assert kwargs["statistic_label"] == "Mean"


class TestDetectionReliabilityFigures:
    """Each group's accuracy figure draws one number per attack."""

    def test_a_std_listed_first_is_not_what_a_section_ranks(
            self, ranking_calls, tmp_path):
        _accuracy_figure(
            SPREAD_STATS, list(SPREAD_STATS), "audio_distortion",
            std_first_config().resolver, "Audio distortion",
            "audio_distortion", str(tmp_path),
        )
        (values, kwargs), = ranking_calls
        assert values["SignInversion"] == 70.0
        assert kwargs["statistic_label"] == "Mean"
