"""``is_watermarked()`` is fed the raw return of ``detect()``.

A model defines that method over whatever its own ``detect()`` returns.
AudioSeal's reads the confidence out of a ``(watermark, confidence)``
pair, so handing it only the watermark array raises inside the model and
the whole run reports nothing detected -- an accuracy of 100% beside a
count of 0/N, which is not a result anyone can act on.

These drive ``Benchmark.run_no_attacks`` against stub models rather than
the report generator, because that is where the two are wired together.
"""

import numpy as np
import pytest
import soundfile as sf

from deepmarkpy.benchmark import Benchmark
from deepmarkpy.core.base_model import BaseModel, implements_is_watermarked


class _StubModel(BaseModel):
    """A model with no config.json; the benchmark supplies the config."""

    returns_confidence = False

    def __init__(self):  # noqa: D107 - deliberately skips BaseModel.__init__
        self._config = {}
        self.base_url = None

    def embed(self, audio, watermark_data, sampling_rate):
        return audio

    def detect(self, audio, sampling_rate):
        raise NotImplementedError


class ConfidenceModel(_StubModel):
    """Returns a pair, and decides on the confidence half of it."""

    def detect(self, audio, sampling_rate):
        return np.ones(16, dtype=np.int32), 0.91

    def is_watermarked(self, detect_output):
        _watermark, confidence = detect_output
        return float(confidence) >= 0.5


class PlainModel(_StubModel):
    """Returns the watermark alone."""

    def detect(self, audio, sampling_rate):
        return np.ones(16, dtype=np.int32)

    def is_watermarked(self, detect_output):
        return bool(np.any(detect_output))


class SilentModel(_StubModel):
    """Implements no decision at all."""

    def detect(self, audio, sampling_rate):
        return np.ones(16, dtype=np.int32)


@pytest.fixture
def audio_files(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"clip{index}.wav"
        sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
        paths.append(str(path))
    return paths


def run(model_cls, audio_files, **config):
    benchmark = Benchmark.__new__(Benchmark)
    benchmark.models = {
        "StubModel": {
            "class": model_cls,
            "config": {"sampling_rate": 16000, "watermark_size": 16, **config},
        }
    }
    return benchmark.run_no_attacks(
        filepaths=audio_files, wm_model="StubModel",
        watermark_data=np.ones(16, dtype=np.int32),
        calculate_quality_metrics=False,
    )


class TestDetectOutputReachesTheModelUnsplit:
    def test_a_confidence_model_decides_on_its_own_pair(self, audio_files):
        results = run(ConfidenceModel, audio_files, returns_confidence=True)

        assert results["supports_detection"] is True
        detected = [f.get("detected") for f in results["files"]]
        assert detected == [True, True, True], (
            "is_watermarked() was handed the watermark instead of the pair"
        )
        # The pair is still split for the columns that need it.
        assert all(f["confidence"] == pytest.approx(0.91)
                   for f in results["files"])

    def test_a_plain_model_decides_on_the_watermark(self, audio_files):
        results = run(PlainModel, audio_files)
        assert [f.get("detected") for f in results["files"]] == [True] * 3

    def test_a_model_without_the_method_records_nothing(self, audio_files):
        results = run(SilentModel, audio_files)
        assert results["supports_detection"] is False
        assert all("detected" not in f for f in results["files"])


class TestSupportPredicate:
    def test_inheriting_the_base_is_not_support(self):
        assert implements_is_watermarked(SilentModel()) is False

    def test_overriding_it_is(self):
        assert implements_is_watermarked(ConfidenceModel()) is True

    def test_a_class_is_judged_as_its_instances_are(self):
        """Config validation asks before any model is constructed."""
        assert implements_is_watermarked(SilentModel) is False
        assert implements_is_watermarked(ConfidenceModel) is True


class BrokenModel(_StubModel):
    """Its decision raises -- a contract mismatch, not a measurement."""

    def detect(self, audio, sampling_rate):
        return np.ones(16, dtype=np.int32)

    def is_watermarked(self, detect_output):
        _watermark, confidence = detect_output  # wrong shape on purpose
        return confidence > 0.5


class TestAFailedDecisionIsNotAZeroCount:
    def test_the_column_is_withdrawn_rather_than_reported_as_none_found(
        self, audio_files, caplog,
    ):
        results = run(BrokenModel, audio_files)

        assert results["supports_detection"] is False, (
            "0/N would read as 'never found' when nothing was measured"
        )
        assert all("detected" not in f for f in results["files"])
        assert any("failed on every file" in r.message for r in caplog.records)


class FlakyModel(_StubModel):
    """Decides on most files, raises on the second one."""

    def __init__(self):
        super().__init__()
        self._calls = 0

    def detect(self, audio, sampling_rate):
        return np.ones(16, dtype=np.int32)

    def is_watermarked(self, detect_output):
        self._calls += 1
        if self._calls == 2:
            raise ValueError("contract mismatch on this file")
        return True


class TestAPartialFailureIsNotANegative:
    """Some files decided, some raised: the count is over the decided ones.

    Counting every file in the denominator reported each failure as "not
    detected", so a model that found its watermark everywhere it answered
    showed 2/3 instead of 2/2.
    """

    def test_the_failed_file_is_left_out_of_the_count(self, audio_files):
        from deepmarkpy.utils.metric_resolver import MetricResolver
        from deepmarkpy.utils.no_attacks_report_generator import _summarize_model

        results = run(FlakyModel, audio_files)
        assert results["supports_detection"] is True
        assert [("detected" in f) for f in results["files"]] == [True, False, True]

        summary = _summarize_model(results, MetricResolver())
        assert summary["positive_detections"] == 2
        assert summary["detection_n"] == 2
