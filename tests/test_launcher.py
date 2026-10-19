"""deepmark-services must start what a run calls, and nothing unverified on the GPU."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepmarkpy import launcher
from deepmarkpy.parity.verdicts import write_verdict

REPO = Path(__file__).resolve().parent.parent
GPU = SimpleNamespace(name="NVIDIA H200 NVL MIG 2g.35gb", uuid="MIG-50bf93fa")
IMAGES = {
    "audioseal": {"cpu": {"tag": "deepmark-audioseal:cpu", "image_id": "sha256:aaaacpu00000"},
                  "gpu": {"tag": "deepmark-audioseal:gpu", "image_id": "sha256:aaaagpu00000"}},
    "neural_vocoder": {"cpu": {"tag": "deepmark-neural_vocoder:cpu", "image_id": "sha256:bbbbcpu00000"},
                       "gpu": {"tag": "deepmark-neural_vocoder:gpu", "image_id": "sha256:bbbbgpu00000"}},
    "vae": {"cpu": {"tag": "deepmark-vae:cpu", "image_id": "sha256:cccccpu00000"},
            "gpu": {"tag": "deepmark-vae:gpu", "image_id": "sha256:ccccgpu00000"}},
    "network_transmission": {"cpu": {"tag": "deepmark-network_transmission:cpu", "image_id": "sha256:dddd"}},
}
GATES = {
    "audioseal": {"ops": {"embed": {"tier": "NUMERIC"}, "detect": {"tier": "NUMERIC"}}},
    "neural_vocoder": {"gpu_precision": "float64", "ops": {"apply": {"tier": "NUMERIC"}}},
    "vae": {"ops": {"apply": {"tier": "STOCHASTIC"}}},
}


def verdict(tmp_path, service, op, image_id, *, passed=True, device=GPU.name, override=None,
            strict=True, precision="float32"):
    env = {"engine": "plugin_inference.X", "numerics": {"strict": strict}, "device_name": device,
           "precision": precision}
    write_verdict({"service": service, "op": op, "tier": "NUMERIC", "passed": passed,
                   "candidate": {"image_id": image_id, "env": env, "engine_override": override},
                   "reference": {"image_id": "sha256:cpu"}, "corpus": {"sha256": "x"}}, tmp_path)


def placed(tmp_path, services, gpu=GPU):
    return {p.service: p for p in launcher.place(services, IMAGES, GATES, gpu, tmp_path)}


class TestPlacement:
    def test_without_gpu_everything_is_cpu(self, tmp_path):
        p = placed(tmp_path, ["audioseal"], gpu=None)["audioseal"]
        assert (p.device, p.image) == ("cpu", "deepmark-audioseal:cpu")

    def test_a_pass_for_every_op_puts_it_on_the_gpu(self, tmp_path):
        verdict(tmp_path, "audioseal", "embed", "sha256:aaaagpu00000")
        verdict(tmp_path, "audioseal", "detect", "sha256:aaaagpu00000")
        p = placed(tmp_path, ["audioseal"])["audioseal"]
        assert (p.device, p.image, len(p.verdicts)) == ("cuda", "deepmark-audioseal:gpu", 2)

    def test_one_missing_op_keeps_it_on_the_cpu_and_says_which(self, tmp_path):
        verdict(tmp_path, "audioseal", "embed", "sha256:aaaagpu00000")
        p = placed(tmp_path, ["audioseal"])["audioseal"]
        assert p.device == "cpu" and "no verdict for detect" in p.reason

    def test_a_measured_fail_is_cited_not_sent_back_to_the_gate(self, tmp_path):
        verdict(tmp_path, "audioseal", "embed", "sha256:aaaagpu00000")
        verdict(tmp_path, "audioseal", "detect", "sha256:aaaagpu00000", passed=False)
        p = placed(tmp_path, ["audioseal"])["audioseal"]
        assert p.device == "cpu" and p.reason.startswith("parity FAIL") and "audioseal.detect." in p.reason

    def test_a_later_pass_for_the_same_deployment_wins(self, tmp_path):
        for op in ("embed", "detect"):
            verdict(tmp_path, "audioseal", op, "sha256:aaaagpu00000", passed=False)
            verdict(tmp_path, "audioseal", op, "sha256:aaaagpu00000")
        assert placed(tmp_path, ["audioseal"])["audioseal"].device == "cuda"

    @pytest.mark.parametrize("bad", [
        {"passed": False},
        {"device": "NVIDIA L4"},                                # measured on another GPU
        {"override": "/override/gpu_engines.py:AudioSealEngineGPU"},  # a study engine, not the shipped one
        {"strict": False},                                      # TF32 on
    ])
    def test_a_pass_that_does_not_vouch_for_this_deployment_is_ignored(self, tmp_path, bad):
        for op in ("embed", "detect"):
            verdict(tmp_path, "audioseal", op, "sha256:aaaagpu00000", **bad)
        assert placed(tmp_path, ["audioseal"])["audioseal"].device == "cpu"

    def test_a_pass_for_an_older_image_is_ignored(self, tmp_path):
        for op in ("embed", "detect"):
            verdict(tmp_path, "audioseal", op, "sha256:0ldgpu000000")
        assert placed(tmp_path, ["audioseal"])["audioseal"].device == "cpu"

    def test_precision_must_match_the_gate(self, tmp_path):
        verdict(tmp_path, "neural_vocoder", "apply", "sha256:bbbbgpu00000", precision="float32")
        assert placed(tmp_path, ["neural_vocoder"])["neural_vocoder"].device == "cpu"
        verdict(tmp_path, "neural_vocoder", "apply", "sha256:bbbbgpu00000", precision="float64")
        p = placed(tmp_path, ["neural_vocoder"])["neural_vocoder"]
        assert (p.device, p.precision) == ("cuda", "float64")

    def test_stochastic_and_gateless_services_stay_on_the_cpu(self, tmp_path):
        p = placed(tmp_path, ["vae", "network_transmission"])
        assert p["vae"].device == "cpu" and "stochastic" in p["vae"].reason
        assert p["network_transmission"].device == "cpu"

    def test_an_unbuilt_service_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="build_images"):
            launcher.place(["encodec"], IMAGES, GATES, None, tmp_path)


class TestComposeOverride:
    P = [launcher.Placement("audioseal", "cuda", "deepmark-audioseal:gpu", "sha256:a"),
         launcher.Placement("encodec", "cpu", "deepmark-encodec:cpu", "sha256:e")]

    def test_docker_reserves_exactly_one_device(self):
        text = launcher.compose_override(self.P, "docker", "GPU-123")
        assert 'device_ids: ["GPU-123"]' in text and "nvidia.com/gpu" not in text
        assert "DEEPMARK_DEVICE=cuda" in text and "NVIDIA_TF32_OVERRIDE=0" in text

    def test_podman_names_the_device_by_cdi(self):
        text = launcher.compose_override(self.P, "/usr/bin/podman", "MIG-1")
        assert "nvidia.com/gpu=MIG-1" in text and "deploy:" not in text

    def test_cpu_services_are_pinned_to_their_image_and_the_cpu(self):
        text = launcher.compose_override(self.P, "docker", "GPU-123")
        block = text.split("  encodec:")[1]
        assert "image: deepmark-encodec:cpu" in block and "DEEPMARK_DEVICE=cpu" in block
        assert "devices" not in block

    def test_the_command_starts_only_these_services_without_building(self):
        cmd = launcher.compose_command("docker", Path("/b/o.yml"), ["audioseal"])
        assert cmd[:2] == ["docker", "compose"] and cmd[-3:] == ["-d", "--no-build", "audioseal"]
        assert launcher.compose_command("podman", Path("/b/o.yml"), ["x"])[0] == "podman-compose"


class TestWhichServices:
    def test_every_compose_service_has_a_port_variable(self):
        ports = launcher.compose_service_ports((REPO / "docker-compose.yml").read_text())
        assert ports["audioseal"] == "AUDIOSEAL_PORT"
        assert ports["speech_enhancement1"] == "SPEECH_ENHANCEMENT_PORT1"
        assert ports["nisqa"] == "NISQA_PORT"
        assert len(ports) == 17

    def test_urls_map_to_services_by_port(self):
        ports = {"audioseal": "AUDIOSEAL_PORT", "encodec": "ENCODEC_PORT"}
        env = {"AUDIOSEAL_PORT": "35001", "ENCODEC_PORT": "40007"}
        urls = ["http://localhost:35001", "http://localhost:40007/attack", "http://localhost:35001"]
        assert launcher.services_for_urls(urls, ports, env) == ["audioseal", "encodec"]
        with pytest.raises(ValueError, match="no compose service"):
            launcher.services_for_urls(["http://localhost:1"], ports, env)

    def test_a_config_needs_its_model_its_http_attacks_and_crossmodels_second_model(self, tmp_path, monkeypatch):
        for var, port in (("AUDIOSEAL_PORT", "5001"), ("ENCODEC_PORT", "10007"), ("WAVMARK_PORT", "8001")):
            monkeypatch.setenv(var, port)
        config = tmp_path / "benchmark.json"
        config.write_text(json.dumps({
            "mode": "benchmark", "general": {"wav_files_dir": str(tmp_path)}, "models": ["AudioSealModel"],
            "attacks": {"groups": [], "list": ["EncodecAttack", "GaussianNoiseAttack", "CrossModelAttack"]},
            "attack_parameters": {"CrossModelAttack": {"different_model_name_cross_model": "WavMarkModel"}},
        }))
        assert launcher.services_for_configs([str(config)]) == ["audioseal", "encodec", "wavmark"]


def test_waiting_returns_only_the_services_that_never_answered():
    ports = {"a": "A_PORT", "b": "B_PORT"}
    env = {"A_PORT": "1", "B_PORT": "2"}
    calls = []

    def get(url):
        calls.append(url)
        if url.startswith("http://localhost:2"):
            raise ConnectionError
    assert launcher.wait_until_serving(["a", "b"], ports, timeout_s=0, environ=env, get=get,
                                       sleep=lambda s: None) == ["b"]
    assert "http://localhost:1/docs" in calls
