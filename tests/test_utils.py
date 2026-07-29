"""Tests for src/utils/utils.py — SNR, resampling, renormalization, config loading."""

import json
import os
import tempfile

import numpy as np
import pytest

from deepmarkpy.utils.utils import snr, resample_audio, renormalize_audio, load_config


# ---------------------------------------------------------------------------
# snr()
# ---------------------------------------------------------------------------
class TestSNR:
    def test_identical_signals_returns_inf(self, sample_audio):
        audio, _ = sample_audio
        assert snr(audio, audio) == np.inf

    def test_known_snr_value(self):
        signal = np.ones(1000)
        noise = np.random.RandomState(0).randn(1000) * 0.1
        noisy = signal + noise
        result = snr(signal, noisy)
        assert 15 < result < 25  # ~20 dB for noise_std=0.1 on unit signal

    def test_mismatched_lengths_truncates(self):
        short = np.ones(100)
        long = np.ones(200)
        result = snr(short, long)
        assert result == np.inf  # both are identical after truncation

    def test_pure_noise(self):
        signal = np.zeros(1000)
        noisy = np.ones(1000)
        result = snr(signal, noisy)
        # signal_power = 0, so SNR = 10*log10(0/noise) → -inf or very negative
        assert result <= 0 or np.isneginf(result)


# ---------------------------------------------------------------------------
# resample_audio()
# ---------------------------------------------------------------------------
class TestResampleAudio:
    def test_same_rate_returns_unchanged(self, sample_audio):
        audio, sr = sample_audio
        result = resample_audio(audio, sr, sr)
        np.testing.assert_array_equal(result, audio)

    def test_downsample_changes_length(self, sample_audio):
        audio, sr = sample_audio
        result = resample_audio(audio, sr, sr // 2)
        assert len(result) < len(audio)
        assert abs(len(result) - len(audio) // 2) <= 2

    def test_upsample_changes_length(self, sample_audio):
        audio, sr = sample_audio
        result = resample_audio(audio, sr, sr * 2)
        assert len(result) > len(audio)
        assert abs(len(result) - len(audio) * 2) <= 2

    def test_roundtrip_preserves_shape(self, sample_audio):
        audio, sr = sample_audio
        up = resample_audio(audio, sr, sr * 2)
        down = resample_audio(up, sr * 2, sr)
        assert abs(len(down) - len(audio)) <= 2


# ---------------------------------------------------------------------------
# renormalize_audio()
# ---------------------------------------------------------------------------
class TestRenormalizeAudio:
    def test_matches_original_range(self):
        orig = np.array([-0.8, 0.0, 0.8])
        proc = np.array([0.1, 0.2, 0.3])
        result = renormalize_audio(orig, proc)
        assert pytest.approx(result.min(), abs=1e-6) == orig.min()
        assert pytest.approx(result.max(), abs=1e-6) == orig.max()

    def test_silent_processed_returns_as_is(self):
        orig = np.array([-1.0, 1.0])
        silent = np.array([0.0, 0.0])
        result = renormalize_audio(orig, silent)
        np.testing.assert_array_equal(result, silent)

    def test_single_value_processed(self):
        orig = np.array([-1.0, 1.0])
        constant = np.array([5.0, 5.0])
        result = renormalize_audio(orig, constant)
        np.testing.assert_array_equal(result, constant)


# ---------------------------------------------------------------------------
# load_config()
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def test_valid_config(self, tmp_path):
        cfg = {"sampling_rate": 16000, "watermark_size": 16}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg))
        result = load_config(str(path))
        assert result == cfg

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.json")

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{invalid json")
        with pytest.raises(ValueError):
            load_config(str(path))

    def test_empty_json_object(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{}")
        result = load_config(str(path))
        assert result == {}
