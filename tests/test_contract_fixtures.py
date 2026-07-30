"""Consistency checks for the P0.4 HTTP-contract fixtures (no Docker needed).

Live contract verification happens through ``scripts/contract_check.py``
(which needs the compose services); this suite only locks the committed
fixture set itself: every compose service has a contract, classifications are
recorded and well-formed, and recorded fixtures carry the artifacts the
verify mode depends on. This keeps the repo's no-Docker test invariant while
making silent fixture loss or hand-editing visible.
"""

import json
import os

import numpy as np
import pytest

CONTRACTS_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "contracts")

# The 14 compose services (docker-compose.yml service names).
ALL_SERVICES = frozenset({
    "audioseal", "aware", "perth", "silentcipher", "timbrewm", "wavmark",
    "vae", "diffusion", "neural_vocoder", "opus_codec",
    "network_transmission", "speech_enhancement1", "speech_enhancement2",
    "speech_tokenization",
})

# Classification lock (see tests/fixtures/contracts/README.md).
EXPECTED_CLASSIFICATION = {
    "audioseal": "deterministic",
    "aware": "deterministic",
    "perth": "deterministic",
    "silentcipher": "deterministic",
    "timbrewm": "deterministic",
    "wavmark": "deterministic",
    "neural_vocoder": "deterministic",
    "opus_codec": "deterministic",
    "speech_tokenization": "deterministic",
    "speech_enhancement1": "deterministic",
    "vae": "stochastic",
    "speech_enhancement2": "stochastic",
    "network_transmission": "stochastic",
    "diffusion": "stochastic",
}


def _contract(service):
    path = os.path.join(CONTRACTS_DIR, service, "contract.json")
    assert os.path.exists(path), f"missing contract fixture for {service}"
    with open(path) as f:
        return json.load(f)


def test_every_service_has_a_contract():
    on_disk = {d for d in os.listdir(CONTRACTS_DIR)
               if os.path.isdir(os.path.join(CONTRACTS_DIR, d))}
    assert on_disk == ALL_SERVICES, (
        f"contract dirs and compose services disagree — "
        f"missing: {sorted(ALL_SERVICES - on_disk)}; "
        f"unexpected: {sorted(on_disk - ALL_SERVICES)}"
    )


@pytest.mark.parametrize("service", sorted(ALL_SERVICES))
def test_classification_locked(service):
    contract = _contract(service)
    assert contract["classification"] == EXPECTED_CLASSIFICATION[service], (
        f"{service}: classification changed from the recorded "
        f"'{EXPECTED_CLASSIFICATION[service]}' — re-classification requires "
        "owner sign-off (REORG_PLAN.md §4.3)"
    )


@pytest.mark.parametrize("service", sorted(ALL_SERVICES))
def test_recorded_fixtures_are_complete(service):
    contract = _contract(service)
    assert not contract.get("recording_blocked"), (
        f"{service} is marked recording_blocked — every service has recorded "
        "fixtures (see tests/fixtures/contracts/README.md)"
    )
    assert contract["endpoints"], f"{service}: no endpoints recorded"
    for path, endpoint in contract["endpoints"].items():
        assert endpoint["response_sha256"], f"{service} {path}: missing hash"
        assert len(endpoint["double_call_stats"]) == 2, (
            f"{service} {path}: double-call evidence missing"
        )
    npz_path = os.path.join(CONTRACTS_DIR, service, "arrays.npz")
    assert os.path.exists(npz_path), f"{service}: arrays.npz missing"
    arrays = np.load(npz_path)
    assert "request_audio" in arrays, f"{service}: request_audio not stored"
