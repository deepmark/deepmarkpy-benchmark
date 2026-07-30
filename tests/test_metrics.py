"""Tests for src/utils/metrics.py — audio quality metrics."""

import numpy as np
import pytest

from deepmarkpy.utils.metrics import trim_audio_to_match, psnr, si_sdr


# ---------------------------------------------------------------------------
# trim_audio_to_match()
# ---------------------------------------------------------------------------
class TestTrimAudioToMatch:
    def test_same_length_unchanged(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        ra, rb = trim_audio_to_match(a, b)
        np.testing.assert_array_equal(ra, a)
        np.testing.assert_array_equal(rb, b)

    def test_first_longer(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0])
        ra, rb = trim_audio_to_match(a, b)
        assert len(ra) == len(rb) == 2
        np.testing.assert_array_equal(ra, [1.0, 2.0])

    def test_second_longer(self):
        a = np.array([1.0])
        b = np.array([2.0, 3.0, 4.0])
        ra, rb = trim_audio_to_match(a, b)
        assert len(ra) == len(rb) == 1
        np.testing.assert_array_equal(rb, [2.0])


# ---------------------------------------------------------------------------
# psnr()
# ---------------------------------------------------------------------------
class TestPSNR:
    def test_identical_signals_returns_inf(self):
        a = np.array([0.5, -0.3, 0.8])
        assert psnr(a, a) == float("inf")

    def test_known_value(self):
        a = np.array([1.0, 0.0, -1.0])
        b = np.array([1.0, 0.1, -1.0])
        result = psnr(a, b, max_value=1.0)
        # MSE = (0.1^2)/3 ≈ 0.00333, PSNR = 10*log10(1/0.00333) ≈ 24.77
        assert 24 < result < 26

    def test_different_lengths_trimmed(self):
        a = np.ones(10)
        b = np.ones(8)
        assert psnr(a, b) == float("inf")


# ---------------------------------------------------------------------------
# si_sdr()
# ---------------------------------------------------------------------------
class TestSISDR:
    def test_identical_signals_high_sdr(self):
        a = np.random.RandomState(0).randn(1000).astype(np.float32)
        result = si_sdr(a, a)
        assert result > 100  # near-infinite SDR

    def test_uncorrelated_low_sdr(self):
        np.random.seed(0)
        a = np.random.randn(10000)
        b = np.random.randn(10000)
        result = si_sdr(a, b)
        # Uncorrelated signals should have very low (negative) SI-SDR
        assert result < 10

    def test_scaled_copy_high_sdr(self):
        a = np.random.RandomState(1).randn(1000)
        b = a * 2.0  # perfectly scaled
        result = si_sdr(a, b)
        assert result > 100  # scale-invariant, so should be very high
