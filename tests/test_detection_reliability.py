"""Tests for the detection_reliability module."""

import json
import time

import numpy as np
import pytest

from deepmarkpy.config import init_template, load_config_data
from deepmarkpy.utils.detection_reliability import (
    _compute_metric_stats,
    _detect,
    run_detection_reliability,
)
from deepmarkpy.utils.detection_reliability_report_generator import (
    _format_count,
    _format_pct,
    _short_model_name,
    generate_detection_reliability_report,
)
# Metric formatting and labelling are shared by every generator, so a
# caption or a unit reads the same wherever it appears.
from deepmarkpy.utils.latex_helpers import (
    format_metric_cell as _format_metric,
    metric_label as _metric_label,
)


class TestDetect:
    """Tests for _detect with mocked models using is_watermarked()."""

    class _ZeroBitModel:
        def __init__(self, returns):
            self._returns = returns

        def detect(self, audio, sr):
            return self._returns

        def is_watermarked(self, detect_output):
            return bool(np.any(detect_output)) if detect_output is not None else False

    class _ConfidenceModel:
        def __init__(self, watermark, confidence, threshold=0.5):
            self._watermark = watermark
            self._confidence = confidence
            self._threshold = threshold

        def detect(self, audio, sr):
            return self._watermark, self._confidence

        def is_watermarked(self, detect_output):
            _wm, conf = detect_output
            return float(conf) >= self._threshold

    class _NumpyConfidenceModel(_ConfidenceModel):
        """Compares a numpy confidence, so its decision is a numpy.bool_."""

        def is_watermarked(self, detect_output):
            _wm, conf = detect_output
            return np.float32(conf) >= self._threshold

    def test_zero_bit_positive(self):
        model = self._ZeroBitModel(np.array(1))
        assert _detect(model, np.zeros(100), 16000) is True

    def test_zero_bit_negative(self):
        model = self._ZeroBitModel(np.array(0))
        assert _detect(model, np.zeros(100), 16000) is False

    def test_confidence_above_threshold(self):
        model = self._ConfidenceModel(np.array([1, 0, 1]), 0.8, threshold=0.5)
        assert _detect(model, np.zeros(100), 16000) is True

    def test_confidence_below_threshold(self):
        model = self._ConfidenceModel(np.array([1, 0, 1]), 0.3, threshold=0.5)
        assert _detect(model, np.zeros(100), 16000) is False

    def test_confidence_at_threshold(self):
        model = self._ConfidenceModel(np.array([1, 0, 1]), 0.5, threshold=0.5)
        assert _detect(model, np.zeros(100), 16000) is True

    def test_numpy_confidence_above_threshold(self):
        """A numpy.bool_ decision comes back as a Python bool, which the
        result's JSON file can hold."""
        model = self._NumpyConfidenceModel(np.array([1, 0, 1]), 0.8, threshold=0.5)
        assert _detect(model, np.zeros(100), 16000) is True

    def test_numpy_confidence_below_threshold(self):
        model = self._NumpyConfidenceModel(np.array([1, 0, 1]), 0.3, threshold=0.5)
        assert _detect(model, np.zeros(100), 16000) is False


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
        assert _format_metric("pesq", None) == "N/A"

    def test_format_metric_float(self):
        assert _format_metric("pesq", 3.14159) == "3.14"

    def test_format_metric_uses_the_unit_the_metric_is_read_in(self):
        # A percentage, a 0-1 fraction shown as a percentage, and a score
        # whose useful resolution is below 0.01 all format differently.
        assert _format_metric("accuracy", 91.5) == "91.50\\%"
        assert _format_metric("ber", 0.085) == "8.50\\%"
        assert _format_metric("stoi", 0.9412) == "0.9412"

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

    class _MockZeroBitModel:
        def generate_watermark(self):
            return np.array([1, 0, 1, 0])

        def embed(self, audio, watermark_data, sampling_rate):
            return audio + 0.001

        def detect(self, audio, sampling_rate):
            return np.array(1)

        def is_watermarked(self, detect_output):
            return bool(np.any(detect_output)) if detect_output is not None else False

    class _MockConfidenceModel:
        def generate_watermark(self):
            return np.array([1, 0, 1, 0])

        def embed(self, audio, watermark_data, sampling_rate):
            return audio + 0.001

        def detect(self, audio, sampling_rate):
            return np.array([1, 0, 1, 0]), 0.8

        def is_watermarked(self, detect_output):
            _wm, confidence = detect_output
            return float(confidence) >= 0.5

    class _MockNumpyConfidenceModel(_MockConfidenceModel):
        """Compares a numpy confidence, so its decision is a numpy.bool_."""

        def is_watermarked(self, detect_output):
            _wm, confidence = detect_output
            return np.float32(confidence) >= 0.5

    class _MockAttack:
        def __init__(self, version=None):
            self.config = {}

        def apply(self, audio, **kwargs):
            return np.array(audio, copy=True)

    class _MockUnsupportedModel:
        def generate_watermark(self):
            return np.array([1, 0, 1, 0])

        def embed(self, audio, watermark_data, sampling_rate):
            return audio + 0.001

        def detect(self, audio, sampling_rate):
            return np.array([1, 0, 1, 0])

    class _MockBenchmark:
        ALWAYS_ON_METRICS = ("pesq", "visqol", "stoi")

        def __init__(self, model_cls, is_zero_bit=True, detection_threshold=None):
            self.models = {
                "TestModel": {
                    "class": model_cls,
                    "config": {
                        "is_zero_bit": is_zero_bit,
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

        benchmark = self._MockBenchmark(
            model_cls=self._MockZeroBitModel, is_zero_bit=True,
        )
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
            model_cls=self._MockConfidenceModel,
            is_zero_bit=False, detection_threshold=0.5,
        )
        result = run_detection_reliability(
            benchmark, [str(audio_file)], "TestModel",
        )

        assert result["is_zero_bit"] is False
        assert result["detection_threshold"] == 0.5
        assert result["no_attack"]["false_positive_count"] == 1
        assert result["no_attack"]["false_negative_count"] == 0

    def test_numpy_decisions_are_stored_as_python_bools(self, tmp_path):
        """The mode writes its result with json.dump, which cannot hold the
        numpy.bool_ a model deciding by a numpy comparison returns."""
        import soundfile as sf
        from deepmarkpy.run import to_json_safe
        from deepmarkpy.utils.metric_resolver import MetricResolver

        audio_file = tmp_path / "test.wav"
        sr = 16000
        audio = np.sin(np.linspace(0, 1, sr)).astype(np.float32)
        sf.write(str(audio_file), audio, sr)

        benchmark = self._MockBenchmark(
            model_cls=self._MockNumpyConfidenceModel,
            is_zero_bit=False, detection_threshold=0.5,
        )
        benchmark.attacks = {
            "GaussianNoiseAttack": {"class": self._MockAttack, "config": {}},
        }
        result = run_detection_reliability(
            benchmark, [str(audio_file)], "TestModel",
            attack_types=["GaussianNoiseAttack"], metric_resolver=MetricResolver(),
        )

        json.dumps(to_json_safe(dict(result)))
        record = result["per_file"][str(audio_file)]
        assert record["no_attack_fp"] is True
        assert record["attacks"]["GaussianNoiseAttack"]["fp"] is True

    def test_rejects_unsupported_model(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        import soundfile as sf
        sf.write(str(audio_file), np.zeros(16000), 16000)

        benchmark = self._MockBenchmark(
            model_cls=self._MockUnsupportedModel,
            is_zero_bit=False,
        )
        with pytest.raises(ValueError, match="does not implement is_watermarked"):
            run_detection_reliability(
                benchmark, [str(audio_file)], "TestModel",
            )

    def test_model_not_found(self):
        benchmark = self._MockBenchmark(model_cls=self._MockZeroBitModel)
        with pytest.raises(ValueError, match="not found"):
            run_detection_reliability(
                benchmark, ["fake.wav"], "NonExistentModel",
            )


class TestReportGeneration:
    BASELINE_QUALITY_CAPTION = (
        "Audio quality of the watermarked audio compared to the original"
    )

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
                "metrics": {
                    "pesq": {"mean": 4.1},
                    "stoi": {"mean": 0.98},
                    "visqol": {"mean": 4.5},
                },
            },
            "attacks": {},
        }
        tex_path = generate_detection_reliability_report(
            result, report_dir=str(tmp_path),
        )
        with open(tex_path) as f:
            content = f.read()
        assert "Audio quality" in content
        assert "4.10" in content

    @staticmethod
    def _shipped_template_resolver():
        raw = json.loads(init_template("detection_reliability"))
        return load_config_data(raw, quiet=True).resolver

    @staticmethod
    def _baseline_report(tmp_path, resolver, **kwargs):
        """The .tex for three files whose no-attack baseline was measured
        over every signal metric any group enables, as the run measures it."""
        measured = resolver.all_signal_metrics()
        per_file = {
            f"clip{index}.wav": {
                "no_attack_fp": False,
                "no_attack_fn": False,
                "no_attack_metrics": {m: 4.0 + 0.1 * index for m in measured},
                "attacks": {},
            }
            for index in range(3)
        }
        result = {
            "model_name": "PerthModel",
            "is_zero_bit": True,
            "detection_threshold": None,
            "n_files": 3,
            "no_attack": {
                "false_positive_count": 0,
                "false_negative_count": 0,
                "metrics": {
                    m: _compute_metric_stats(
                        [record["no_attack_metrics"][m]
                         for record in per_file.values()],
                        resolver.statistics_for(None, m), m,
                    )
                    for m in measured
                },
            },
            "attacks": {},
            "per_file": per_file,
        }
        tex_path = generate_detection_reliability_report(
            result, report_dir=str(tmp_path), resolver=resolver, **kwargs,
        )
        with open(tex_path) as f:
            return f.read()

    def test_the_shipped_template_reports_the_baseline_quality(self, tmp_path):
        """The template enables its signal metrics per group and none under
        metrics.defaults; the baseline tables print what the groups enable."""
        resolver = self._shipped_template_resolver()
        assert resolver.signal_metrics_for_group(None) == []
        content = self._baseline_report(tmp_path, resolver)
        assert self.BASELINE_QUALITY_CAPTION in content
        assert "Speech intelligibility of the watermarked audio" in content
        assert _metric_label("pesq") in content
        assert _metric_label("stoi") in content

    def test_every_duration_part_reports_the_baseline_quality(self, tmp_path):
        content = self._baseline_report(
            tmp_path, self._shipped_template_resolver(),
            duration_partitions=[
                ("< 5s", ["clip0.wav"]),
                ("> 5s", ["clip1.wav", "clip2.wav"]),
                ("Overall", ["clip0.wav", "clip1.wav", "clip2.wav"]),
            ],
        )
        assert content.count("\\section{No-Attack Baseline}") == 3
        assert content.count(self.BASELINE_QUALITY_CAPTION) == 3


class TestTimingsMeanTheSameThingHere:
    """This mode detects twice per attack, so which call is timed matters.

    It runs detect on the attacked clean signal for the false-positive
    rate and on the attacked watermarked signal for the false negative.
    Only the second matches what ``Benchmark.run`` times, and the
    un-attacked detect belongs to the baseline, not to any attack.
    """

    MARK = 7.0
    EMBED, ATTACK = 0.01, 0.03
    CLEAN_DETECT, ATTACKED_DETECT = 0.05, 0.20

    class _Model:
        def __init__(self, outer):
            self.outer = outer
            self.config = {"is_zero_bit": True, "sampling_rate": 16000}

        def generate_watermark(self):
            return np.array([1])

        def embed(self, audio, watermark_data, sampling_rate):
            time.sleep(self.outer.EMBED)
            return audio

        def detect(self, audio, sampling_rate):
            attacked = audio[0] == self.outer.MARK
            time.sleep(self.outer.ATTACKED_DETECT if attacked
                       else self.outer.CLEAN_DETECT)
            return 1

        def is_watermarked(self, detect_output):
            return bool(detect_output)

    class _Attack:
        MARK = 7.0
        PAUSE = 0.03

        def __init__(self, version=None):
            self.config = {}

        def apply(self, audio, **kwargs):
            time.sleep(self.PAUSE)
            marked = np.array(audio, copy=True)
            marked[0] = self.MARK
            return marked

    @pytest.fixture
    def result(self, tmp_path):
        import soundfile as sf
        from deepmarkpy.utils.metric_resolver import MetricResolver

        path = tmp_path / "a.wav"
        sf.write(str(path),
                 np.sin(np.linspace(0, 1, 16000)).astype(np.float32), 16000)

        outer = self
        model_cls = lambda: TestTimingsMeanTheSameThingHere._Model(outer)

        class _Benchmark:
            models = {"TestModel": {
                "class": model_cls,
                "config": {"is_zero_bit": True, "sampling_rate": 16000},
            }}
            attacks = {"GaussianNoiseAttack": {
                "class": TestTimingsMeanTheSameThingHere._Attack,
                "config": {},
            }}

        resolver = MetricResolver(efficiency={
            "enabled": True,
            "metrics": {m: {"enabled": True, "statistics": ["mean"]}
                        for m in ("embed_latency", "detect_latency",
                                  "attack_latency")},
        })
        return run_detection_reliability(
            _Benchmark(), [str(path)], "TestModel",
            attack_types=["GaussianNoiseAttack"], metric_resolver=resolver,
        )

    def test_detect_latency_reaches_the_per_attack_table(self, result):
        """It was copied before the call that measures it, so it was absent."""
        timings = result["attacks"]["GaussianNoiseAttack"]["timings"]
        assert "detect_latency" in timings

    def test_the_attacks_detect_is_the_one_reported(self, result):
        timings = result["attacks"]["GaussianNoiseAttack"]["timings"]
        assert timings["detect_latency"]["mean"] == pytest.approx(
            self.ATTACKED_DETECT, abs=0.05,
        )

    def test_the_baseline_keeps_its_own_detect(self, result):
        """The un-attacked detect must not be overwritten by an attack's."""
        timings = result["no_attack"]["timings"]
        assert timings["detect_latency"]["mean"] == pytest.approx(
            self.CLEAN_DETECT, abs=0.05,
        )

    def test_the_per_file_record_does_not_mix_the_two(self, result):
        record = next(iter(result["per_file"].values()))
        assert record["detect_latency"] == pytest.approx(
            self.CLEAN_DETECT, abs=0.05,
        )
        assert record["attacks"]["GaussianNoiseAttack"]["detect_latency"] == \
            pytest.approx(self.ATTACKED_DETECT, abs=0.05)

    def test_embedding_is_carried_onto_the_attack_once_per_file(self, result):
        timings = result["attacks"]["GaussianNoiseAttack"]["timings"]
        assert timings["embed_latency"]["mean"] == pytest.approx(
            self.EMBED, abs=0.05,
        )


class TestCrossModelGetsItsSecondModelHereToo:
    """The plugin reads the second model's name from its kwargs only.

    ``Benchmark.run`` falls back to the plugin's config.json default; this
    pass did not, so with no ``attack_parameters`` override every file
    raised inside the per-attack ``try`` and the row came out with zero
    attempts instead of a measurement.
    """

    class _Model:
        def generate_watermark(self):
            return np.array([1])

        def embed(self, audio, watermark_data, sampling_rate):
            return audio + 0.001

        def detect(self, audio, sampling_rate):
            return 1

        def is_watermarked(self, detect_output):
            return bool(detect_output)

    def _run(self, tmp_path, default_second_model):
        import soundfile as sf
        from deepmarkpy.plugins.attacks.cross_model.attack import CrossModelAttack
        from deepmarkpy.utils.metric_resolver import MetricResolver

        path = tmp_path / "a.wav"
        sf.write(str(path),
                 np.sin(np.linspace(0, 1, 16000)).astype(np.float32), 16000)

        config = {"is_zero_bit": True, "sampling_rate": 16000}

        class _Benchmark:
            models = {
                "TestModel": {"class": self._Model, "config": config},
                "OtherModel": {"class": self._Model, "config": config},
            }
            attacks = {"CrossModelAttack": {
                "class": CrossModelAttack,
                "config": {"different_model_name_cross_model":
                           default_second_model},
            }}

        return run_detection_reliability(
            _Benchmark(), [str(path)], "TestModel",
            attack_types=["CrossModelAttack"], metric_resolver=MetricResolver(),
        )

    def test_the_plugin_default_is_used_without_an_override(self, tmp_path):
        row = self._run(tmp_path, "OtherModel")["attacks"]["CrossModelAttack"]
        assert row["false_positive_attempts"] == 1
        assert row["false_negative_attempts"] == 1

    def test_an_unknown_second_model_stops_the_run(self, tmp_path):
        with pytest.raises(ValueError, match="different_model_name_cross_model"):
            self._run(tmp_path, "NotAModel")
