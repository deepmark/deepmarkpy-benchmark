"""A run on services from deepmark-services checks they are still the images it started."""

import json

import pytest

from deepmarkpy import deployment as dep

DEPLOYMENT = {
    "engine": "docker",
    "gpu": {"uuid": "GPU-1", "name": "NVIDIA L4"},
    "services": {
        "audioseal": {"device": "cuda", "image": "deepmark-audioseal:gpu", "image_id": "sha256:aaaa1111",
                      "precision": "float32", "verdicts": ["audioseal.embed.x.json"], "reason": "PASS"},
        "encodec": {"device": "cpu", "image": "deepmark-encodec:cpu", "image_id": "sha256:eeee2222",
                    "precision": "float32", "verdicts": [], "reason": "no PASS"},
    },
}
PORTS = {"audioseal": "AUDIOSEAL_PORT", "encodec": "ENCODEC_PORT"}
ENV = {"AUDIOSEAL_PORT": "5001", "ENCODEC_PORT": "10007"}


def test_matching_containers_pass():
    running = {"5001": "sha256:aaaa1111", "10007": "eeee2222"}
    assert dep.check_deployment(DEPLOYMENT, PORTS, running, ENV) == []


def test_a_swapped_image_is_reported():
    running = {"5001": "sha256:ffff9999", "10007": "sha256:eeee2222"}
    problems = dep.check_deployment(DEPLOYMENT, PORTS, running, ENV)
    assert len(problems) == 1 and problems[0].startswith("audioseal: runs image")


def test_a_stopped_service_is_reported():
    problems = dep.check_deployment(DEPLOYMENT, PORTS, {"5001": "sha256:aaaa1111"}, ENV)
    assert problems == ["encodec: not running on port 10007"]


def test_no_file_means_no_deployment_and_no_check(tmp_path, monkeypatch):
    monkeypatch.setenv(dep.DEPLOYMENT_ENV, str(tmp_path / "missing.json"))
    assert dep.load_deployment() is None
    assert dep.verify_running_deployment() == []
    assert dep.deployment_summary() is None


def test_none_switches_it_off(monkeypatch):
    monkeypatch.setenv(dep.DEPLOYMENT_ENV, "none")
    assert dep.deployment_path() is None


def test_the_summary_says_where_each_service_ran_and_why(tmp_path, monkeypatch):
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(DEPLOYMENT))
    monkeypatch.setenv(dep.DEPLOYMENT_ENV, str(path))
    summary = dep.deployment_summary()
    assert summary["gpu"]["name"] == "NVIDIA L4"
    assert summary["services"]["audioseal"]["device"] == "cuda"
    assert summary["services"]["audioseal"]["verdicts"] == ["audioseal.embed.x.json"]
