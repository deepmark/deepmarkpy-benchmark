from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from deepmarkpy.core.base_attack import BaseAttack
from deepmarkpy.plugins.attacks.diffusion.attack import DiffusionAttack
from deepmarkpy.plugins.attacks.vae.attack import VAEAttack
from deepmarkpy.server.attack_service import run_attack_payload


class IdentityAttack(BaseAttack):
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        assert kwargs["sampling_rate"] == 16000
        assert kwargs["strength"] == 0.5
        return audio * kwargs["strength"]


def test_remote_attack_posts_canonical_payload(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"audio": [0.25, -0.25], "sampling_rate": 16000}

    def post(url, *, json, timeout):
        calls.append((url, json, timeout))
        return Response()

    monkeypatch.setenv("VAE_PORT", "12345")
    monkeypatch.setattr("deepmarkpy.core.remote_attack.requests.post", post)

    result = VAEAttack().apply(
        np.asarray([0.5, -0.5], dtype=np.float32),
        sampling_rate=np.int64(16000),
        model_name="test-model",
    )

    assert calls == [
        (
            "http://localhost:12345/attack",
            {
                "audio": [0.5, -0.5],
                "sampling_rate": 16000,
                "model_name": "test-model",
            },
            600.0,
        )
    ]
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [0.25, -0.25])


def test_remote_attack_forwards_legacy_alias_as_canonical(monkeypatch):
    payloads = []

    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"audio": [1.0]},
    )

    def post(_url, *, json, timeout):
        payloads.append(json)
        return response

    monkeypatch.setattr("deepmarkpy.core.remote_attack.requests.post", post)

    DiffusionAttack().apply(
        np.ones(1, dtype=np.float32),
        sampling_rate=16000,
        diffusion_steps=9,
    )

    assert payloads[0]["steps"] == 9
    assert "diffusion_steps" not in payloads[0]


def test_remote_attack_requires_sampling_rate():
    with pytest.raises(ValueError, match="sampling_rate"):
        VAEAttack().apply(np.ones(4, dtype=np.float32))


def test_remote_attack_supports_shared_and_attack_specific_hosts(monkeypatch):
    monkeypatch.setenv("DEEPMARK_ATTACK_HOST", "shared.example")
    assert VAEAttack().endpoint == "http://shared.example:10001"

    monkeypatch.setenv("VAE_PORT_HOST", "vae.example")
    monkeypatch.setenv("VAE_PORT", "12001")
    assert VAEAttack().endpoint == "http://vae.example:12001"


def test_service_adapter_executes_packaged_implementation():
    attack = IdentityAttack()
    response = run_attack_payload(
        attack,
        {
            "audio": [1.0, -1.0],
            "sampling_rate": 16000,
            "strength": 0.5,
        },
    )

    assert response == {
        "audio": [0.5, -0.5],
        "sampling_rate": 16000,
    }
