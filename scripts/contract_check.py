"""Record / verify HTTP-contract fixtures for the compose services.

Usage (repo root; base image + services built per README; Docker running):

    python scripts/contract_check.py record            # all 14 services
    python scripts/contract_check.py record vae aware  # subset
    python scripts/contract_check.py verify            # re-record & compare
    python scripts/contract_check.py verify opus_codec

For each service this script starts the compose service, waits for readiness,
sends the canonical request(s) **twice**, classifies the service by the
double-call result (byte-identical raw response bodies → deterministic), and
writes fixtures under ``tests/fixtures/contracts/<service>/``:

- ``contract.json`` — endpoint, non-audio request fields, response metadata
  (raw-body SHA-256, keys, classification, double-call evidence, image id);
- ``arrays.npz`` — the audio/watermark arrays of requests and responses.

Requests mirror the host-side clients byte-for-byte (same JSON field names
and construction; see each client's ``model.py``/``attack.py``). Canonical
audio is the §4.3 input — ``0.5 * np.random.default_rng(42).standard_normal(SR)``
at the model's config ``sampling_rate`` (attacks: 16000 Hz). Model fixtures
chain ``/embed`` → ``/detect`` (detect consumes the embed response audio).
Watermarks are ``np.random.default_rng(42).integers(0, 2, watermark_size)``.
``speech_enhancement_1`` is recorded with ``noise_strength=0.0`` in
the request (a legal kwargs-path value), making it byte-comparable.

Verification criteria: deterministic services must return the exact
recorded raw body (SHA-256); stochastic services (``diffusion``,
``network_transmission``, ``speech_enhancement_2``) are checked structurally —
same JSON keys, finite floats, response length within the recorded range, RMS
within [0.25x, 4x] of the recorded mean. Byte-identity claims are
same-machine, same-image-lineage only.
"""

import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO / "tests" / "fixtures" / "contracts"
ATTACK_SR = 16000
READY_TIMEOUT_S = 1200
STOCHASTIC_EXPECTED = {"diffusion", "network_transmission", "speech_enhancement2"}

# Services classified stochastic even when a same-container double call is
# byte-identical. network_transmission's RTP path is timing/instance
# dependent: calls within one container lifetime match, calls from a fresh
# container differ, so byte-identity cannot gate it. (`tc netem` does not
# engage under Docker Desktop's VM — `tc qdisc show` stays `noqueue`.)
STOCHASTIC_OVERRIDES = {
    "network_transmission":
        "instance-dependent RTP path: same-container double call identical, "
        "fresh-container call differs; netem does not engage under Docker "
        "Desktop",
}

# service -> (plugin dir, host port .env var default, kind)
SERVICES = {
    "audioseal": ("src/deepmarkpy/plugins/models/audio_seal", 5001, "model"),
    "aware": ("src/deepmarkpy/plugins/models/aware", 9004, "model"),
    "perth": ("src/deepmarkpy/plugins/models/perth", 7010, "model"),
    "silentcipher": ("src/deepmarkpy/plugins/models/silent_cipher", 7001, "model"),
    "timbrewm": ("src/deepmarkpy/plugins/models/timbrewm", 9001, "model"),
    "wavmark": ("src/deepmarkpy/plugins/models/wavmark", 8001, "model"),
    "vae": ("src/deepmarkpy/plugins/attacks/vae", 10001, "attack"),
    "encodec": ("src/deepmarkpy/plugins/attacks/encodec", 10007, "attack"),
    "descript_audio_codec": ("src/deepmarkpy/plugins/attacks/descript_audio_codec", 10008, "attack"),
    "diffusion": ("src/deepmarkpy/plugins/attacks/diffusion", 10002, "attack"),
    "neural_vocoder": ("src/deepmarkpy/plugins/attacks/neural_vocoder", 10004, "attack"),
    "opus_codec": ("src/deepmarkpy/plugins/attacks/opus_codec", 10023, "attack"),
    "network_transmission": ("src/deepmarkpy/plugins/attacks/network_transmission", 10020, "attack"),
    "speech_enhancement1": ("src/deepmarkpy/plugins/attacks/speech_enhancement_1", 10005, "attack"),
    "speech_enhancement2": ("src/deepmarkpy/plugins/attacks/speech_enhancement_2", 10006, "attack"),
    "speech_tokenization": ("src/deepmarkpy/plugins/attacks/speech_tokenization", 10003, "attack"),
}

# Per-request timeout: speech_enhancement2 instantiates ClearVoice per request
# (downloads weights on the container's first request), so it gets longer.
REQUEST_TIMEOUT_S = {"speech_enhancement2": 2400}
DEFAULT_REQUEST_TIMEOUT_S = 900

# The canonical input is ≈1 s unless a plugin needs otherwise. wavmark's
# encode_watermark yields zero usable chunks for exactly 1 s of audio
# (wm_add_util.py:50 assert) — it needs strictly more; 2 s is the smallest
# round override and matches real benchmark usage (multi-second files).
INPUT_SECONDS = {"wavmark": 2}


def load_config(plugin_dir: str) -> dict:
    with open(REPO / plugin_dir / "config.json") as f:
        return json.load(f)


def canonical_audio(sr: int, seconds: int = 1) -> np.ndarray:
    """The §4.3 canonical input at the given sampling rate."""
    return 0.5 * np.random.default_rng(42).standard_normal(sr * seconds)


def canonical_watermark(size: int) -> np.ndarray:
    return np.random.default_rng(42).integers(0, 2, size=size)


def build_attack_extra_fields(service: str, cfg: dict) -> dict:
    """Non-audio request fields, mirroring each attack client's payload."""
    if service == "diffusion":
        return {"diffusion_steps": cfg["steps_diffusion"]}
    if service == "opus_codec":
        return {"bitrate": cfg["bitrate_opus_codec"],
                "framesize": cfg["framesize_opus_codec"]}
    if service == "speech_enhancement1":
        # §4.3: recorded with noise_strength=0.0 in the request.
        return {"noise_strength": 0.0}
    if service == "speech_enhancement2":
        return {"model_name": cfg["model_name_se2"]}
    if service == "descript_audio_codec":
        return {"n_codebooks_dac": cfg["n_codebooks_dac"]}
    if service == "network_transmission":
        keys = [
            "bitrate_bps_netem", "frame_duration_ms_netem", "delay_ms_netem",
            "jitter_ms_netem", "packet_loss_netem", "duplication_netem",
            "reorder_netem", "corruption_netem", "fec_enabled_netem",
            "expected_loss_netem", "ns_enabled_netem", "vad_enabled_netem",
            "agc_enabled_netem", "agc_target_lufs_netem",
            "playout_delay_ms_netem",
        ]
        return {k: cfg[k] for k in keys}
    return {}


def compose(*args: str) -> None:
    subprocess.run(["docker-compose", *args], cwd=REPO, check=True,
                   capture_output=True, text=True, timeout=1800)


def wait_ready(port: int, service: str) -> None:
    deadline = time.time() + READY_TIMEOUT_S
    url = f"http://localhost:{port}/docs"
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(f"{service} not ready on port {port} "
                       f"after {READY_TIMEOUT_S}s")


def post_twice(url: str, payload: dict, timeout: int) -> tuple:
    r1 = requests.post(url, json=payload, timeout=timeout)
    r2 = requests.post(url, json=payload, timeout=timeout)
    r1.raise_for_status()
    r2.raise_for_status()
    return r1, r2


def response_stats(body: dict) -> dict:
    audio = body.get("audio") or body.get("watermarked_audio")
    if audio is None:
        return {"audio_len": None, "audio_rms": None}
    arr = np.asarray(audio, dtype=np.float64)
    return {"audio_len": int(arr.size),
            "audio_rms": float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0}


def image_id(service: str) -> str:
    out = subprocess.run(
        ["docker-compose", "images", "-q", service],
        cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def record_endpoint(url: str, payload: dict, timeout: int) -> dict:
    """Double-call an endpoint; return the fixture record for it."""
    r1, r2 = post_twice(url, payload, timeout)
    identical = r1.content == r2.content
    return {
        "request_non_audio": {k: v for k, v in payload.items()
                              if k not in ("audio", "watermark_data")},
        "response_status": r1.status_code,
        "response_keys": sorted(json.loads(r1.content).keys()),
        "response_sha256": hashlib.sha256(r1.content).hexdigest(),
        "double_call_identical": identical,
        "double_call_stats": [response_stats(json.loads(r.content))
                              for r in (r1, r2)],
        "_body": json.loads(r1.content),
    }


def run_service(service: str, mode: str) -> bool:
    plugin_dir, port, kind = SERVICES[service]
    cfg = load_config(plugin_dir)
    timeout = REQUEST_TIMEOUT_S.get(service, DEFAULT_REQUEST_TIMEOUT_S)
    base = f"http://localhost:{port}"
    out_dir = CONTRACTS_DIR / service

    logger.info("[%s] starting service ...", service)
    compose("up", "-d", service)
    try:
        wait_ready(port, service)
        arrays = {}
        endpoints = {}
        if kind == "model":
            sr = cfg["sampling_rate"]
            audio = canonical_audio(sr, INPUT_SECONDS.get(service, 1))
            wm = canonical_watermark(cfg["watermark_size"])
            arrays["request_audio"] = audio
            arrays["request_watermark"] = wm
            embed_payload = {"audio": audio.tolist(),
                             "watermark_data": wm.tolist(),
                             "sampling_rate": sr}
            embed = record_endpoint(f"{base}/embed", embed_payload, timeout)
            wm_audio = embed["_body"]["watermarked_audio"]
            arrays["embed_response_audio"] = np.asarray(wm_audio, dtype=np.float64)
            detect_payload = {"audio": wm_audio, "sampling_rate": sr}
            detect = record_endpoint(f"{base}/detect", detect_payload, timeout)
            det_wm = detect["_body"].get("watermark")
            arrays["detect_response_watermark"] = np.asarray(
                det_wm if det_wm is not None else [], dtype=np.float64)
            endpoints["/embed"] = embed
            endpoints["/detect"] = detect
        else:
            audio = canonical_audio(ATTACK_SR)
            arrays["request_audio"] = audio
            payload = {"audio": audio.tolist(), "sampling_rate": ATTACK_SR,
                       **build_attack_extra_fields(service, cfg)}
            attack = record_endpoint(f"{base}/attack", payload, timeout)
            arrays["attack_response_audio"] = np.asarray(
                attack["_body"]["audio"], dtype=np.float64)
            endpoints["/attack"] = attack

        deterministic = all(e["double_call_identical"]
                            for e in endpoints.values())
        classification = "deterministic" if deterministic else "stochastic"
        note = None
        if service in STOCHASTIC_OVERRIDES:
            classification = "stochastic"
            note = STOCHASTIC_OVERRIDES[service]

        if mode == "record":
            out_dir.mkdir(parents=True, exist_ok=True)
            contract = {
                "service": service,
                "kind": kind,
                "recorded": date.today().isoformat(),
                "image_id": image_id(service),
                "classification": classification,
                "classification_note": note,
                "endpoints": {p: {k: v for k, v in e.items() if k != "_body"}
                              for p, e in endpoints.items()},
            }
            with open(out_dir / "contract.json", "w") as f:
                json.dump(contract, f, indent=2)
            np.savez_compressed(out_dir / "arrays.npz", **arrays)
            logger.info("[%s] recorded: %s", service, classification)
            return True

        # verify mode
        with open(out_dir / "contract.json") as f:
            stored = json.load(f)
        ok = True
        if stored["classification"] == "deterministic":
            for path, e in endpoints.items():
                if e["response_sha256"] != stored["endpoints"][path]["response_sha256"]:
                    logger.error("[%s] %s: raw response differs from fixture",
                                 service, path)
                    ok = False
        else:
            for path, e in endpoints.items():
                se = stored["endpoints"][path]
                if e["response_keys"] != se["response_keys"]:
                    logger.error("[%s] %s: response keys changed", service, path)
                    ok = False
                    continue
                lens = [s["audio_len"] for s in se["double_call_stats"]]
                rmss = [s["audio_rms"] for s in se["double_call_stats"]]
                for s in e["double_call_stats"]:
                    if None in lens or s["audio_len"] is None:
                        continue
                    if not (min(lens) * 0.9 <= s["audio_len"] <= max(lens) * 1.1):
                        logger.error("[%s] %s: length %s outside recorded range %s",
                                     service, path, s["audio_len"], lens)
                        ok = False
                    mean_rms = sum(rmss) / len(rmss)
                    if mean_rms and not (0.25 * mean_rms <= s["audio_rms"]
                                         <= 4.0 * mean_rms):
                        logger.error("[%s] %s: RMS %s outside tolerance of %s",
                                     service, path, s["audio_rms"], mean_rms)
                        ok = False
                for s in e["double_call_stats"]:
                    if s["audio_rms"] is not None and not math.isfinite(s["audio_rms"]):
                        logger.error("[%s] %s: non-finite audio", service, path)
                        ok = False
        if ok and classification != stored["classification"]:
            logger.error("[%s] classification changed: %s -> %s",
                         service, stored["classification"], classification)
            ok = False
        logger.info("[%s] verify: %s", service, "PASS" if ok else "FAIL")
        return ok
    finally:
        subprocess.run(["docker-compose", "rm", "-sf", service],
                       cwd=REPO, capture_output=True, timeout=300)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("record", "verify"):
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    selected = sys.argv[2:] or list(SERVICES)
    unknown = set(selected) - set(SERVICES)
    if unknown:
        raise SystemExit(f"unknown service(s): {sorted(unknown)}. "
                         f"Valid: {sorted(SERVICES)}")
    results = {}
    for service in selected:
        try:
            results[service] = run_service(service, mode)
        except Exception as exc:  # keep going; report at the end
            logger.error("[%s] ERROR: %s", service, exc)
            results[service] = False
    logger.info("\n=== %s summary ===", mode)
    for service, ok in results.items():
        logger.info("%-22s %s", service, "OK" if ok else "FAILED")
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
