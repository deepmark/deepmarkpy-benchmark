"""Tests for native (non-Docker) attack plugins.

Only tests attacks that run locally without Docker containers.
"""

import numpy as np
import pytest

from plugin_manager import PluginManager


@pytest.fixture(scope="module")
def pm():
    return PluginManager()


@pytest.fixture(scope="module")
def attacks(pm):
    return pm.get_attacks()


def _make_attack(attacks, name):
    """Instantiate an attack by class name."""
    assert name in attacks, f"{name} not found in discovered attacks"
    return attacks[name]["class"]()


# ---------------------------------------------------------------------------
# SignInversionAttack
# ---------------------------------------------------------------------------
class TestSignInversionAttack:
    def test_inverts_signal(self, attacks, sample_audio):
        atk = _make_attack(attacks, "SignInversionAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr)
        np.testing.assert_array_almost_equal(result, -audio)

    def test_double_inversion_identity(self, attacks, sample_audio):
        atk = _make_attack(attacks, "SignInversionAttack")
        audio, sr = sample_audio
        result = atk.apply(atk.apply(audio, sampling_rate=sr), sampling_rate=sr)
        np.testing.assert_array_almost_equal(result, audio)

    def test_preserves_shape(self, attacks, sample_audio):
        atk = _make_attack(attacks, "SignInversionAttack")
        audio, sr = sample_audio
        assert atk.apply(audio, sampling_rate=sr).shape == audio.shape


# ---------------------------------------------------------------------------
# GaussianNoiseAttack
# ---------------------------------------------------------------------------
class TestGaussianNoiseAttack:
    def test_output_differs_from_input(self, attacks, sample_audio):
        atk = _make_attack(attacks, "GaussianNoiseAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, snr_db_gaussian_noise=20, sampling_rate=sr)
        assert not np.array_equal(result, audio)

    def test_preserves_shape(self, attacks, sample_audio):
        atk = _make_attack(attacks, "GaussianNoiseAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, snr_db_gaussian_noise=20, sampling_rate=sr)
        assert result.shape == audio.shape

    def test_higher_snr_less_noise(self, attacks, sample_audio):
        atk = _make_attack(attacks, "GaussianNoiseAttack")
        audio, sr = sample_audio
        noisy_low = atk.apply(audio, snr_db_gaussian_noise=10, sampling_rate=sr)
        noisy_high = atk.apply(audio, snr_db_gaussian_noise=40, sampling_rate=sr)
        noise_low = np.mean((audio - noisy_low) ** 2)
        noise_high = np.mean((audio - noisy_high) ** 2)
        assert noise_low > noise_high

    def test_uses_config_default(self, attacks, sample_audio):
        atk = _make_attack(attacks, "GaussianNoiseAttack")
        audio, sr = sample_audio
        # Should not raise — uses config default snr_db
        result = atk.apply(audio, sampling_rate=sr)
        assert result.shape == audio.shape


# ---------------------------------------------------------------------------
# CropBeginningAttack
# ---------------------------------------------------------------------------
class TestCropBeginningAttack:
    def test_crops_correct_percentage(self, attacks, sample_audio):
        atk = _make_attack(attacks, "CropBeginningAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr, crop_percentage_beginning=10)
        expected_len = len(audio) - int(len(audio) * 0.10)
        assert len(result) == expected_len

    def test_zero_crop_unchanged(self, attacks, sample_audio):
        atk = _make_attack(attacks, "CropBeginningAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr, crop_percentage_beginning=0)
        np.testing.assert_array_equal(result, audio)

    def test_requires_sampling_rate(self, attacks, sample_audio):
        atk = _make_attack(attacks, "CropBeginningAttack")
        audio, _ = sample_audio
        with pytest.raises(ValueError, match="sampling_rate"):
            atk.apply(audio)


# ---------------------------------------------------------------------------
# CropRandomAttack
# ---------------------------------------------------------------------------
class TestCropRandomAttack:
    def test_crops_correct_length(self, attacks, sample_audio):
        atk = _make_attack(attacks, "CropRandomAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr, crop_percentage_random=10)
        expected_len = len(audio) - int(len(audio) * 0.10)
        assert len(result) == expected_len

    def test_preserves_values(self, attacks, sample_audio):
        """Cropped audio should only contain values from the original."""
        atk = _make_attack(attacks, "CropRandomAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr, crop_percentage_random=10)
        # Every value in result should exist in the original
        for val in result[:10]:
            assert val in audio


# ---------------------------------------------------------------------------
# SmoothingAttack
# ---------------------------------------------------------------------------
class TestSmoothingAttack:
    def test_output_shape(self, attacks, sample_audio):
        atk = _make_attack(attacks, "SmoothingAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr, window_size=15)
        assert result.shape == audio.shape

    def test_smoothing_reduces_variation(self, attacks, sample_audio):
        atk = _make_attack(attacks, "SmoothingAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr, window_size=15)
        assert np.std(result) <= np.std(audio)


# ---------------------------------------------------------------------------
# QuantizationAttack
# ---------------------------------------------------------------------------
class TestQuantizationAttack:
    def test_output_shape(self, attacks, sample_audio):
        atk = _make_attack(attacks, "QuantizationAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr, bit_quantization=256)
        assert result.shape == audio.shape

    def test_fewer_levels_more_distortion(self, attacks):
        # Use a signal with wide dynamic range to ensure quantization is noticeable
        np.random.seed(42)
        audio = np.random.randn(16000).astype(np.float32)
        atk = _make_attack(attacks, "QuantizationAttack")
        q_fine = atk.apply(audio, sampling_rate=16000, bit_quantization=1024)
        q_coarse = atk.apply(audio, sampling_rate=16000, bit_quantization=4)
        err_fine = np.mean((audio - q_fine) ** 2)
        err_coarse = np.mean((audio - q_coarse) ** 2)
        assert err_coarse > err_fine


# ---------------------------------------------------------------------------
# WaveletAttack (requires pywt — skip if not installed)
# ---------------------------------------------------------------------------
pywt = pytest.importorskip("pywt", reason="pywt not installed")


class TestWaveletAttack:
    def test_output_shape(self, attacks, sample_audio):
        if "WaveletDenoiseAttack" not in attacks:
            pytest.skip("WaveletDenoiseAttack not loaded (pywt missing)")
        atk = _make_attack(attacks, "WaveletDenoiseAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr)
        assert result.shape == audio.shape

    def test_modifies_signal(self, attacks, sample_audio):
        if "WaveletDenoiseAttack" not in attacks:
            pytest.skip("WaveletDenoiseAttack not loaded (pywt missing)")
        atk = _make_attack(attacks, "WaveletDenoiseAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr)
        assert not np.array_equal(result, audio)


# ---------------------------------------------------------------------------
# FlipSamplesAttack
# ---------------------------------------------------------------------------
class TestFlipSamplesAttack:
    def test_output_shape(self, attacks, sample_audio):
        atk = _make_attack(attacks, "FlipSamplesAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr)
        assert result.shape == audio.shape


# ---------------------------------------------------------------------------
# ZeroCrossInsertsAttack
# ---------------------------------------------------------------------------
class TestZeroCrossInsertsAttack:
    def test_output_longer_or_equal(self, attacks, sample_audio):
        atk = _make_attack(attacks, "ZeroCrossInsertsAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr)
        assert len(result) >= len(audio)


# ---------------------------------------------------------------------------
# LPCAttack
# ---------------------------------------------------------------------------
class TestLPCAttack:
    def test_output_shape(self, attacks, sample_audio):
        atk = _make_attack(attacks, "LPCAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr, order_lpc=12)
        assert result.shape == audio.shape

    def test_modifies_signal(self, attacks, sample_audio):
        atk = _make_attack(attacks, "LPCAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr, order_lpc=12)
        assert not np.array_equal(result, audio)
