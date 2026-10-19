"""Parity verdicts on disk, and the question placement asks: may this run on GPU?

A verdict is one JSON file under ``parity/verdicts/``, named
``<service>.<op>.<image id prefix>.json``. It records:
- the judgement and every case's evidence;
- both runs' environments;
- the corpus hash and the git revision.

It is keyed by the **candidate image's ID**. A rebuilt image has a new ID and
so has no verdict until it is measured again; nothing carries over silently.
"""

import datetime
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


def verdict_path(service: str, op: str, image_id: str, directory: Path = VERDICT_DIR) -> Path:
    short = image_id.split(":")[-1][:12]
    return Path(directory) / f"{service}.{op}.{short}.json"


def write_verdict(record: Dict, directory: Path = VERDICT_DIR) -> Path:
    for key in ("service", "op", "tier", "passed", "candidate", "reference", "corpus"):
        if key not in record:
            raise ValueError(f"verdict lacks {key!r}")
    record = dict(record)
    record.setdefault("created_utc", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    path = verdict_path(record["service"], record["op"], record["candidate"]["image_id"], directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)
    return path


def read_verdicts(directory: Path = VERDICT_DIR) -> Iterable[Dict]:
    for path in sorted(Path(directory).glob("*.json")):
        yield json.loads(path.read_text())


def passing_verdict(service: str, op: str, image_id: str, directory: Path = VERDICT_DIR) -> Optional[Dict]:
    """The PASS verdict for exactly this image, or None."""
    path = verdict_path(service, op, image_id, directory)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("candidate", {}).get("image_id") != image_id or not record.get("passed"):
        return None
    return record
