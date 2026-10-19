"""Parity verdicts on disk, and the question placement asks: may this run on GPU?

A verdict is one JSON file under ``parity/verdicts/``, named
``<service>.<op>.<image id prefix>.json``. It records:
- the judgement and every case's evidence;
- both runs' environments;
- the corpus hash and the git revision.

It is keyed by the **candidate image's ID** and its **engine configuration**:
the engine class that ran, the numerics settings, and the GPU model it ran on
(MIG profile included). A rebuilt image, or the
same image run with another engine or precision, has no verdict until it is
measured. Nothing carries over silently. BigVGAN showed why: its float32
engine fails on the GPU and its float64 engine passes, in one image.
"""

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

# The repository's parity/verdicts/ when run from a checkout; set
# DEEPMARK_VERDICT_DIR when the package is installed elsewhere.
VERDICT_DIR = Path(
    os.environ.get("DEEPMARK_VERDICT_DIR")
    or Path(__file__).resolve().parents[3] / "parity" / "verdicts"
)


def engine_config(env: Dict) -> str:
    """Fingerprint of what ran: engine class, numerics, GPU model and precision.

    The GPU model is part of it because numerics differ across architectures
    and libraries pick kernels by device. A verdict measured on one GPU (or
    MIG profile) says nothing about another; that one runs the gate itself.
    """
    key = {"engine": env.get("engine"), "numerics": env.get("numerics"), "device": env.get("device_name")}
    if env.get("precision", "float32") != "float32":
        key["precision"] = env["precision"]  # float32 keys stay as they were
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]


def verdict_path(service: str, op: str, image_id: str, config: str, directory: Path = VERDICT_DIR) -> Path:
    short = image_id.split(":")[-1][:12]
    return Path(directory) / f"{service}.{op}.{short}.{config}.json"


def write_verdict(record: Dict, directory: Path = VERDICT_DIR) -> Path:
    for key in ("service", "op", "tier", "passed", "candidate", "reference", "corpus"):
        if key not in record:
            raise ValueError(f"verdict lacks {key!r}")
    record = dict(record)
    record.setdefault("created_utc", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    record["candidate"] = dict(record["candidate"])
    record["candidate"]["engine_config"] = engine_config(record["candidate"].get("env", {}))
    path = verdict_path(record["service"], record["op"], record["candidate"]["image_id"],
                        record["candidate"]["engine_config"], directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)
    return path


def read_verdicts(directory: Path = VERDICT_DIR) -> Iterable[Dict]:
    for path in sorted(Path(directory).glob("*.json")):
        yield json.loads(path.read_text())


def shipped_engine_verdict(service: str, op: str, image_id: str, device_name: str,
                           precision: str = "float32", directory: Path = VERDICT_DIR) -> Optional[Dict]:
    """The verdict, PASS or FAIL, on whether ``service``'s ``op`` may run on this GPU.

    It must be:
    - for this exact image;
    - on this GPU model (MIG profile included);
    - from the engine the image ships, not a study override;
    - under strict numerics;
    - at the precision the service will run at.

    A PASS wins over a FAIL for the same deployment.
    """
    found = None
    for record in read_verdicts(directory):
        candidate = record.get("candidate", {})
        env = candidate.get("env", {})
        if (record.get("service"), record.get("op")) != (service, op):
            continue
        if candidate.get("image_id") != image_id or env.get("device_name") != device_name:
            continue
        if candidate.get("engine_override") or not (env.get("numerics") or {}).get("strict"):
            continue
        if env.get("precision", "float32") != precision:
            continue
        if record.get("passed"):
            return record
        found = record
    return found


def shipped_engine_pass(service: str, op: str, image_id: str, device_name: str,
                        precision: str = "float32", directory: Path = VERDICT_DIR) -> Optional[Dict]:
    """The PASS that lets a deployment run ``service``'s ``op`` on this GPU, or None."""
    record = shipped_engine_verdict(service, op, image_id, device_name, precision, directory)
    return record if record and record.get("passed") else None


def passing_verdict(service: str, op: str, image_id: str, config: str,
                    directory: Path = VERDICT_DIR) -> Optional[Dict]:
    """The PASS verdict for exactly this image and engine configuration, or None."""
    path = verdict_path(service, op, image_id, config, directory)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    candidate = record.get("candidate", {})
    if candidate.get("image_id") != image_id or candidate.get("engine_config") != config or not record.get("passed"):
        return None
    return record
