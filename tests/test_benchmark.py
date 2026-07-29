"""Tests for src/benchmark.py — compare_watermarks, compute_mean_accuracy."""

import numpy as np
import pytest

from deepmarkpy.benchmark import Benchmark


# ---------------------------------------------------------------------------
# compare_watermarks()
# ---------------------------------------------------------------------------
class TestCompareWatermarks:
    @pytest.fixture(autouse=True)
    def _create_benchmark(self):
        """Create a Benchmark instance; plugin loading is unavoidable."""
        self.bench = Benchmark()

    def test_perfect_match(self):
        wm = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        assert self.bench.compare_watermarks(wm, wm) == 100.0

    def test_no_match(self):
        wm = np.array([1, 1, 1, 1])
        inv = np.array([0, 0, 0, 0])
        assert self.bench.compare_watermarks(wm, inv) == 0.0

    def test_half_match(self):
        wm = np.array([1, 1, 0, 0])
        det = np.array([1, 1, 1, 1])
        assert self.bench.compare_watermarks(wm, det) == 50.0

    def test_none_detected_returns_50(self):
        wm = np.array([1, 0, 1])
        assert self.bench.compare_watermarks(wm, None) == 50.0

    def test_empty_detected_returns_50(self):
        wm = np.array([1, 0, 1])
        assert self.bench.compare_watermarks(wm, np.array([])) == 50.0

    def test_length_mismatch_returns_50(self):
        wm = np.array([1, 0, 1])
        det = np.array([1, 0])
        assert self.bench.compare_watermarks(wm, det) == 50.0

    def test_zero_dim_array_returns_50(self):
        wm = np.array([1])
        det = np.array(0.5)  # 0-d array
        assert self.bench.compare_watermarks(wm, det) == 50.0

    def test_with_list_input(self):
        wm = np.array([1, 0, 1, 0])
        det = [1, 0, 1, 0]
        assert self.bench.compare_watermarks(wm, det) == 100.0


# ---------------------------------------------------------------------------
# compute_mean_accuracy()
# ---------------------------------------------------------------------------
class TestComputeMeanAccuracy:
    @pytest.fixture(autouse=True)
    def _create_benchmark(self):
        self.bench = Benchmark()

    def test_basic_mean(self, sample_results):
        stats = self.bench.compute_mean_accuracy(sample_results)
        assert "GaussianNoiseAttack" in stats
        assert "SignInversionAttack" in stats
        assert stats["GaussianNoiseAttack"]["accuracy_mean"] == pytest.approx(97.5)
        assert stats["SignInversionAttack"]["accuracy_mean"] == pytest.approx(29.0)

    def test_with_confidence(self, sample_results_with_confidence):
        stats = self.bench.compute_mean_accuracy(sample_results_with_confidence)
        assert "Attack1" in stats
        assert stats["Attack1"]["accuracy_mean"] == pytest.approx(92.5)

    def test_with_cross_model(self, sample_results_with_cross_model):
        stats = self.bench.compute_mean_accuracy(sample_results_with_cross_model)
        assert "CrossModelAttack" in stats
        assert "accuracy_cross_model_mean" in stats["CrossModelAttack"]
        assert stats["CrossModelAttack"]["accuracy_cross_model_mean"] == pytest.approx(95.0)

    def test_empty_results(self):
        stats = self.bench.compute_mean_accuracy({})
        assert stats == {}

    def test_single_file_single_attack(self):
        results = {
            "f.wav": {"attacks": {"A": {"accuracy": 75.0}}}
        }
        stats = self.bench.compute_mean_accuracy(results)
        assert stats["A"]["accuracy_mean"] == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# get_available_args()
# ---------------------------------------------------------------------------
class TestGetAvailableArgs:
    def test_returns_three_lists(self):
        bench = Benchmark()
        models, attacks, args = bench.get_available_args()
        assert isinstance(models, list)
        assert isinstance(attacks, list)
        assert isinstance(args, dict)

    def test_models_not_empty(self):
        bench = Benchmark()
        models, _, _ = bench.get_available_args()
        assert len(models) > 0

    def test_attacks_not_empty(self):
        bench = Benchmark()
        _, attacks, _ = bench.get_available_args()
        assert len(attacks) > 0

    def test_args_contains_known_keys(self):
        bench = Benchmark()
        _, _, args = bench.get_available_args()
        assert "gaussian_noise" in args
        assert "snr_db" in args["gaussian_noise"]
