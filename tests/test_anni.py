"""ANNI adapter, opt-in configuration, and real CPU smoke coverage."""

import sys
import types
import warnings
from unittest.mock import Mock

import numpy as np
import pytest

from deepmarkpy.utils import metrics as m
from deepmarkpy.utils.metric_resolver import MetricResolver, worst_case_of


def test_both_scores_share_one_inference_and_keep_original_inputs(monkeypatch):
    reference = np.arange(22050, dtype=float)
    degraded = reference[:-7] * 0.9
    model = Mock(return_value={"D_peak": 2.0, "D_med": 0.5})
    monkeypatch.setattr(m, "_get_anni", lambda: model)
    result = m.compute_metrics(reference, degraded, 22050, m.ANNI_METRICS)
    assert result == {"anni_peak": 2.0, "anni_med": 0.5}
    model.assert_called_once()
    args, kwargs = model.call_args
    assert args[0] is reference and args[1] is degraded
    assert kwargs == {"sr": 22050}


def test_only_requested_score_returned(monkeypatch, sample_audio):
    monkeypatch.setattr(m, "compute_anni", lambda *args: {"anni_peak": 2, "anni_med": 1})
    audio, sr = sample_audio
    assert m.compute_metrics(audio, audio, sr, ["anni_med"]) == {"anni_med": 1}


def test_not_requested_does_not_load_anni(monkeypatch, sample_audio):
    load = Mock(side_effect=AssertionError("unexpected ANNI load"))
    monkeypatch.setattr(m, "_get_anni", load)
    audio, sr = sample_audio
    m.compute_metrics(audio, audio, sr, ["psnr"])
    load.assert_not_called()
    resolver = MetricResolver.from_attack_groups()
    assert not set(m.ANNI_METRICS) & set(resolver.all_signal_metrics())


def test_cached_cpu_model_and_missing_package(monkeypatch, caplog):
    factory = Mock(return_value=object())
    m._get_anni.cache_clear()
    try:
        monkeypatch.setitem(sys.modules, "anni", types.SimpleNamespace(ANNI=factory))
        assert m._get_anni() is m._get_anni()
        factory.assert_called_once_with(level=65.0, device="cpu")
        m._get_anni.cache_clear()
        monkeypatch.setitem(sys.modules, "anni", None)
        assert m.compute_anni([], [], 16000) == dict.fromkeys(m.ANNI_METRICS)
        assert "ANNI is unavailable" in caplog.text
    finally:
        m._get_anni.cache_clear()


@pytest.mark.parametrize("failure", [ValueError("short clip"), RuntimeError("inference failed")])
def test_failures_are_missing_not_zero(monkeypatch, failure, caplog):
    monkeypatch.setattr(m, "_get_anni", lambda: Mock(side_effect=failure))
    assert m.compute_anni([], [], 16000) == dict.fromkeys(m.ANNI_METRICS)
    assert str(failure) in caplog.text


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_invalid_scores_are_missing(monkeypatch, bad):
    monkeypatch.setattr(m, "_get_anni", lambda: Mock(return_value={"D_peak": bad, "D_med": 0}))
    assert m.compute_anni([], [], 16000) == dict.fromkeys(m.ANNI_METRICS)


def test_library_warnings_reach_logger(monkeypatch, caplog):
    def model(*args, **kwargs):
        warnings.warn("reference and test correlate weakly")
        return {"D_peak": 2, "D_med": 1}
    monkeypatch.setattr(m, "_get_anni", lambda: model)
    m.compute_anni([], [], 16000)
    assert "ANNI: reference and test correlate weakly" in caplog.text


def test_config_opt_in_and_lower_is_better():
    from deepmarkpy.config import load_config_data
    config = load_config_data({
        "mode": "benchmark", "models": ["AudioSealModel"],
        "calculate_quality_metrics": True,
        "metrics": {"defaults": {
            "anni_peak": {"enabled": True, "statistics": ["worst_case"]},
            "anni_med": {"enabled": True},
        }},
    })
    resolver = config.resolver
    assert resolver.all_signal_metrics() == list(m.ANNI_METRICS)
    for name in m.ANNI_METRICS:
        assert worst_case_of([0.2, 1.5, 0.7], name) == 1.5
    resolver.calculate_quality_metrics = False
    assert not set(m.ANNI_METRICS) & set(resolver.all_signal_metrics())


def test_real_anni_cpu_identity_and_changed_signal():
    pytest.importorskip("anni")
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from deepmarkpy.execution.pipelined import _blas_limits, _init_worker, _timed_metrics

    sr = 16000
    t = np.arange(sr // 2) / sr
    reference = 0.1 * np.sin(2 * np.pi * 1000 * t)
    m._get_anni.cache_clear()
    try:
        same = m.compute_anni(reference, reference, sr)
        changed = m.compute_anni(reference, reference * 1.2, sr)
        for name in m.ANNI_METRICS:
            assert same[name] == pytest.approx(0, abs=1e-6)
            assert changed[name] > same[name]
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker, initargs=(_blas_limits(),),
        ) as pool:
            worker_scores, _ = pool.submit(
                _timed_metrics, reference, reference * 1.2, sr, m.ANNI_METRICS,
            ).result(timeout=60)
        assert worker_scores == pytest.approx(changed)
        assert m.compute_anni(reference[:100], reference[:100], sr) == dict.fromkeys(m.ANNI_METRICS)
    finally:
        m._get_anni.cache_clear()
