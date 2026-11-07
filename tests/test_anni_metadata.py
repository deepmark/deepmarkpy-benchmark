"""Persist optional ANNI availability without loading it when disabled."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deepmarkpy import run
from deepmarkpy.utils import metrics
from deepmarkpy.utils.metric_resolver import MetricResolver


def write_metadata(tmp_path, monkeypatch, enabled, calculate=True):
    monkeypatch.setattr(run, "nisqa_status", lambda: {"available": False, "reason": "test"})
    monkeypatch.setattr(run, "_git_revision", lambda: "test")
    resolver = MetricResolver(
        defaults={"anni_peak": {"enabled": enabled}}, calculate_quality_metrics=calculate,
    )
    benchmark = SimpleNamespace(attacks={}, models={}, plugin_manager=SimpleNamespace(failed=[]))
    path = run.write_run_metadata(
        tmp_path, SimpleNamespace(), benchmark, [], resolver=resolver,
    )
    with open(path) as handle:
        return json.load(handle)


def test_missing_anni_reason_is_saved_and_warned(tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "anni", None)
    expected = metrics.anni_status()
    assert expected["available"] is False and expected["reason"]
    metadata = write_metadata(tmp_path, monkeypatch, enabled=True)
    assert metadata["metrics"]["anni"] == expected
    resolver = MetricResolver(defaults={"anni_peak": {"enabled": True}})
    run._warn_about_metric_services([SimpleNamespace(resolver=resolver)])
    assert "[W008]" in caplog.text and expected["reason"] in caplog.text


def test_rejected_input_does_not_mark_package_unavailable(tmp_path, monkeypatch):
    factory = Mock()
    monkeypatch.setitem(sys.modules, "anni", SimpleNamespace(ANNI=factory))
    monkeypatch.setattr(metrics, "_get_anni", lambda: Mock(side_effect=ValueError("short clip")))
    assert metrics.compute_anni([], [], 16000) == dict.fromkeys(metrics.ANNI_METRICS)
    metadata = write_metadata(tmp_path, monkeypatch, enabled=True)
    assert metadata["metrics"]["anni"] == {"available": True, "reason": "N/A"}
    factory.assert_not_called()


@pytest.mark.parametrize("enabled,calculate", [(False, True), (True, False)])
def test_disabled_anni_is_not_probed(tmp_path, monkeypatch, enabled, calculate):
    probe = Mock(side_effect=AssertionError("ANNI should not be imported"))
    monkeypatch.setattr(run, "anni_status", probe)
    metadata = write_metadata(tmp_path, monkeypatch, enabled=enabled, calculate=calculate)
    resolver = MetricResolver(
        defaults={"anni_peak": {"enabled": enabled}}, calculate_quality_metrics=calculate,
    )
    run._warn_about_metric_services([SimpleNamespace(resolver=resolver)])
    assert "anni" not in metadata["metrics"]
    probe.assert_not_called()
