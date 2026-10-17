"""Tests for native (non-Docker) attack plugins.

Only tests attacks that run locally without Docker containers.
"""

import shutil

import numpy as np
import pytest

from deepmarkpy.plugin_manager import PluginManager


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
        if "WaveletAttack" not in attacks:
            pytest.skip("WaveletAttack not loaded (pywt missing)")
        atk = _make_attack(attacks, "WaveletAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr)
        assert result.shape == audio.shape

    def test_modifies_signal(self, attacks, sample_audio):
        if "WaveletAttack" not in attacks:
            pytest.skip("WaveletAttack not loaded (pywt missing)")
        atk = _make_attack(attacks, "WaveletAttack")
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


# ---------------------------------------------------------------------------
# AacCompressionAttack
# ---------------------------------------------------------------------------
pytestmark_aac = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def _tone_plus_noise(sr=16000, dur=1.0, seed=0):
    """A signal with real spectral content (unlike a pure sine, which AAC's
    psychoacoustic model can code near-losslessly at almost any bitrate,
    masking the bitrate/distortion relationship being tested)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = (
        0.3 * np.sin(2 * np.pi * 300 * t)
        + 0.2 * np.sin(2 * np.pi * 1200 * t)
        + 0.05 * rng.standard_normal(t.size)
    )
    return sig.astype(np.float32), sr


@pytestmark_aac
class TestAacCompressionAttack:
    def test_preserves_shape(self, attacks, sample_audio):
        atk = _make_attack(attacks, "AacCompressionAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr)
        assert result.shape == audio.shape
        assert result.dtype == np.float32

    def test_modifies_signal(self, attacks, sample_audio):
        atk = _make_attack(attacks, "AacCompressionAttack")
        audio, sr = sample_audio
        result = atk.apply(audio, sampling_rate=sr, bitrate_aac=16)
        assert not np.array_equal(result, audio)

    def test_uses_config_default(self, attacks, sample_audio):
        atk = _make_attack(attacks, "AacCompressionAttack")
        audio, sr = sample_audio
        # Should not raise -- uses config default bitrate_aac
        result = atk.apply(audio, sampling_rate=sr)
        assert result.shape == audio.shape

    def test_requires_sampling_rate(self, attacks, sample_audio):
        atk = _make_attack(attacks, "AacCompressionAttack")
        audio, _ = sample_audio
        with pytest.raises(ValueError, match="sampling_rate"):
            atk.apply(audio)

    def test_lower_bitrate_more_distortion(self, attacks):
        atk = _make_attack(attacks, "AacCompressionAttack")
        audio, sr = _tone_plus_noise()
        high = atk.apply(audio, sampling_rate=sr, bitrate_aac=128)
        low = atk.apply(audio, sampling_rate=sr, bitrate_aac=16)
        err_high = np.mean((audio - high) ** 2)
        err_low = np.mean((audio - low) ** 2)
        assert err_low > err_high

    def test_round_trip_is_sample_aligned(self, attacks):
        """The .m4a container's edit list must keep AAC's ~1024-sample
        encoder priming delay from leaking out as a desync artifact --
        the reason this attack transcodes through .m4a rather than raw
        ADTS (see the class docstring)."""
        atk = _make_attack(attacks, "AacCompressionAttack")
        audio, sr = _tone_plus_noise(seed=1)
        result = atk.apply(audio, sampling_rate=sr)
        window = 8000
        ref = audio[:window] - audio[:window].mean()
        out = result[:window] - result[:window].mean()
        corr = np.correlate(out, ref, mode="full")
        lag = corr.argmax() - (window - 1)
        assert lag == 0
