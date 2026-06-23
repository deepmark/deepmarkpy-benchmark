"""Tests for the detection_reliability module."""

import numpy as np
import pytest

from utils.detection_reliability import (
    _detect,
    _detect_is_positive_zero_bit,
    run_detection_reliability,
)
from utils.detection_reliability_report_generator import (
    _format_count,
    _format_metric,
    _format_pct,
    _short_model_name,
    _metric_label,
    generate_detection_reliability_report,
)


class TestDetectIsPositiveZeroBit:
    def test_scalar_true(self):
        assert _detect_is_positive_zero_bit(1) is True

    def test_scalar_false(self):
        assert _detect_is_positive_zero_bit(0) is False

    def test_numpy_array_positive(self):
        assert _detect_is_positive_zero_bit(np.array(1)) is True

    def test_numpy_array_zero(self):
        assert _detect_is_positive_zero_bit(np.array(0)) is False

    def test_list_positive(self):
        assert _detect_is_positive_zero_bit([1]) is True

    def test_list_zero(self):
        assert _detect_is_positive_zero_bit([0]) is False

    def test_empty_list(self):
        assert _detect_is_positive_zero_bit([]) is False

    def test_numpy_array_1d(self):
        assert _detect_is_positive_zero_bit(np.array([1, 0, 1])) is True

    def test_numpy_array_all_zeros(self):
        assert _detect_is_positive_zero_bit(np.array([0, 0, 0])) is False


class TestDetect:
    """Tests for _detect with mocked models."""

    class _ZeroBitModel:
        def __init__(self, returns):
            self._returns = returns

        def detect(self, audio, sr):
            return self._returns

    class _ConfidenceModel:
        def __init__(self, watermark, confidence):
            self._watermark = watermark
            self._confidence = confidence

        def detect(self, audio, sr):
            return self._watermark, self._confidence

    def test_zero_bit_positive(self):
        model = self._ZeroBitModel(np.array(1))
        assert _detect(model, np.zeros(100), 16000, False) is True

    def test_zero_bit_negative(self):
        model = self._ZeroBitModel(np.array(0))
        assert _detect(model, np.zeros(100), 16000, False) is False

    def test_confidence_above_threshold(self):
        model = self._ConfidenceModel(np.array([1, 0, 1]), 0.8)
        assert _detect(model, np.zeros(100), 16000, True, 0.5) is True

    def test_confidence_below_threshold(self):
        model = self._ConfidenceModel(np.array([1, 0, 1]), 0.3)
        assert _detect(model, np.zeros(100), 16000, True, 0.5) is False

    def test_confidence_at_threshold(self):
        model = self._ConfidenceModel(np.array([1, 0, 1]), 0.5)
        assert _detect(model, np.zeros(100), 16000, True, 0.5) is True

    def test_confidence_no_threshold_falls_back_to_zero_bit(self):
        model = self._ConfidenceModel(np.array([0]), 0.9)
        assert _detect(model, np.zeros(100), 16000, True, None) is False


class TestFormatHelpers:
    def test_format_count_normal(self):
        assert _format_count(3, 10) == "3/10"

    def test_format_count_zero_total(self):
        assert _format_count(0, 0) == "N/A"

    def test_format_pct_normal(self):
        assert _format_pct(1, 4) == "25.0\\%"

    def test_format_pct_zero_total(self):
        assert _format_pct(0, 0) == "N/A"

    def test_format_metric_none(self):
        assert _format_metric(None) == "N/A"

    def test_format_metric_na_string(self):
        assert _format_metric("N/A") == "N/A"

    def test_format_metric_float(self):
        assert _format_metric(3.14159) == "3.14"

    def test_short_model_name_strips_model(self):
        assert _short_model_name("AudioSealModel") == "AudioSeal"

    def test_short_model_name_strips_watermark(self):
        assert _short_model_name("TestWatermark") == "Test"

    def test_short_model_name_no_suffix(self):
        assert _short_model_name("Perth") == "Perth"

    def test_short_model_name_escapes_underscore(self):
        assert _short_model_name("My_Model") == "My\\_"

    def test_metric_label_known(self):
        assert _metric_label("pesq") == "PESQ (1--4.66)"

    def test_metric_label_unknown(self):
        assert _metric_label("some_metric") == "SOME METRIC"


class TestRunDetectionReliability:
    """Integration tests with a mocked benchmark."""

    class _MockModel:
        def __init__(self, is_zero_bit=True):
            self._is_zero_bit = is_zero_bit

        def generate_watermark(self):
            return np.array([1, 0, 1, 0])

        def embed(self, audio, watermark_data, sampling_rate):
            return audio + 0.001

        def detect(self, audio, sampling_rate):
            if self._is_zero_bit:
                return np.array(1)
            return np.array([1, 0, 1, 0]), 0.8

    class _MockBenchmark:
        ALWAYS_ON_METRICS = ("pesq", "visqol", "stoi")

        def __init__(self, is_zero_bit=True, returns_confidence=False,
                     detection_threshold=None):
            self.models = {
                "TestModel": {
                    "class": lambda: TestRunDetectionReliability._MockModel(is_zero_bit),
                    "config": {
                        "is_zero_bit": is_zero_bit,
                        "returns_confidence": returns_confidence,
                        "detection_threshold": detection_threshold,
                        "sampling_rate": 16000,
                    },
                }
            }
            self.attacks = {}

    def test_zero_bit_no_attacks(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        import soundfile as sf
        sr = 16000
        audio = np.sin(np.linspace(0, 1, sr)).astype(np.float32)
        sf.write(str(audio_file), audio, sr)

        benchmark = self._MockBenchmark(is_zero_bit=True)
        result = run_detection_reliability(
            benchmark, [str(audio_file)], "TestModel",
        )

        assert result["model_name"] == "TestModel"
        assert result["is_zero_bit"] is True
        assert result["n_files"] == 1
        assert result["no_attack"]["false_positive_count"] == 1
        assert result["no_attack"]["false_negative_count"] == 0

    def test_confidence_model_no_attacks(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        import soundfile as sf
        sr = 16000
        audio = np.sin(np.linspace(0, 1, sr)).astype(np.float32)
        sf.write(str(audio_file), audio, sr)

        benchmark = self._MockBenchmark(
            is_zero_bit=False, returns_confidence=True,
            detection_threshold=0.5,
        )
        result = run_detection_reliability(
            benchmark, [str(audio_file)], "TestModel",
        )

        assert result["is_zero_bit"] is False
        assert result["detection_threshold"] == 0.5
        assert result["no_attack"]["false_positive_count"] == 1
        assert result["no_attack"]["false_negative_count"] == 0

    def test_rejects_unsupported_model(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        import soundfile as sf
        sf.write(str(audio_file), np.zeros(16000), 16000)

        benchmark = self._MockBenchmark(
            is_zero_bit=False, returns_confidence=False,
        )
        with pytest.raises(ValueError, match="requires either"):
            run_detection_reliability(
                benchmark, [str(audio_file)], "TestModel",
            )

    def test_rejects_confidence_without_threshold(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        import soundfile as sf
        sf.write(str(audio_file), np.zeros(16000), 16000)

        benchmark = self._MockBenchmark(
            is_zero_bit=False, returns_confidence=True,
            detection_threshold=None,
        )
        with pytest.raises(ValueError, match="detection_threshold"):
            run_detection_reliability(
                benchmark, [str(audio_file)], "TestModel",
            )

    def test_model_not_found(self):
        benchmark = self._MockBenchmark()
        with pytest.raises(ValueError, match="not found"):
            run_detection_reliability(
                benchmark, ["fake.wav"], "NonExistentModel",
            )


class TestReportGeneration:
    def test_generates_tex_file(self, tmp_path):
        result = {
            "model_name": "PerthModel",
            "is_zero_bit": True,
            "detection_threshold": None,
            "n_files": 5,
            "no_attack": {
                "false_positive_count": 1,
                "false_negative_count": 0,
            },
            "attacks": {},
        }
        tex_path = generate_detection_reliability_report(
            result, report_dir=str(tmp_path),
        )
        assert tex_path.endswith(".tex")
        with open(tex_path) as f:
            content = f.read()
        assert "Perth" in content
        assert "False Positive" in content

    def test_generates_with_attacks(self, tmp_path):
        result = {
            "model_name": "AudioSealModel",
            "is_zero_bit": False,
            "detection_threshold": 0.5,
            "n_files": 10,
            "no_attack": {
                "false_positive_count": 2,
                "false_negative_count": 1,
            },
            "attacks": {
                "GaussianNoiseAttack": {
                    "accuracy_mean": 85.0,
                    "metrics": {"pesq": 3.2, "visqol": 4.0, "stoi": 0.91},
                    "false_positive_count": 3,
                    "false_positive_attempts": 10,
                    "false_negative_count": 4,
                    "false_negative_attempts": 10,
                },
            },
        }
        tex_path = generate_detection_reliability_report(
            result, report_dir=str(tmp_path),
        )
        with open(tex_path) as f:
            content = f.read()
        assert "AudioSeal" in content
        assert "GaussianNoise" in content
        assert "85.00" in content
        assert "Audio Distortion" in content

    def test_generates_with_quality_metrics(self, tmp_path):
        result = {
            "model_name": "PerthModel",
            "is_zero_bit": True,
            "detection_threshold": None,
            "n_files": 3,
            "no_attack": {
                "false_positive_count": 0,
                "false_negative_count": 0,
                "metrics": {"pesq": 4.1, "stoi": 0.98, "visqol": 4.5},
            },
            "attacks": {},
        }
        tex_path = generate_detection_reliability_report(
            result, report_dir=str(tmp_path),
        )
        with open(tex_path) as f:
            content = f.read()
        assert "Watermarked Audio Quality" in content
        assert "4.10" in content
