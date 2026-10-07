"""Every single-number view of accuracy reads each attack at its own group's
statistic, so a per-group override never reads as zero, and at the first
configured statistic other than ``std``, so a spread never stands in for it."""

import numpy as np
import pytest

from deepmarkpy.benchmark import Benchmark
from deepmarkpy.config import load_config_data
from deepmarkpy.utils.comparative_report_generator import ComparativeReportGenerator
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
    """The radar draws one number per attack."""

    def test_the_radar_reads_each_attack_at_its_groups_statistic(
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

    def test_a_std_listed_first_is_not_what_the_radar_draws(self, tmp_path):
        """The radar draws the mean."""
        generator = ComparativeReportGenerator(
            str(tmp_path), resolver=std_first_config().resolver,
        )

        assert generator._primary_value(SPREAD_STATS[PINK_NOISE], PINK_NOISE) == 82.0
