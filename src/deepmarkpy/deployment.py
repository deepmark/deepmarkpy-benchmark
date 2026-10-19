"""What deepmark-services started, and whether that is still what is running.

deepmark-services writes build/deployment.json: per service, the image it
started, the device (cpu or cuda), the precision, and the parity verdicts
that put it on the GPU. A run with that file present checks that each
service's running container is that image before it measures anything, and
copies the record into run_metadata.json. A report then states which
services ran where, and on what evidence.

Without the file, nothing changes: services started by hand with plain
``docker compose`` run as before, and the metadata records no deployment.
``DEEPMARK_DEPLOYMENT`` points elsewhere, or is ``none`` to ignore the file.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DEPLOYMENT_ENV = "DEEPMARK_DEPLOYMENT"


def deployment_path() -> Optional[Path]:
    value = os.environ.get(DEPLOYMENT_ENV)
    if value is not None and value.strip().lower() in ("", "none"):
        return None
    return Path(value) if value else REPO / "build" / "deployment.json"


def load_deployment() -> Optional[Dict]:
    path = deployment_path()
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _engine(deployment: Dict) -> Optional[str]:
    engine = deployment.get("engine") or "docker"
    return engine if shutil.which(engine) else None


def running_images(engine: str) -> Dict[str, str]:
    """``{published port: image ID}`` for the containers running now."""
    ps = subprocess.run([engine, "ps", "--format", "{{.Names}}|{{.Ports}}"],
                        capture_output=True, text=True, timeout=30)
    images = {}
    for line in ps.stdout.splitlines():
        name, _, ports = line.partition("|")
        inspect = subprocess.run([engine, "inspect", "--format", "{{.Image}}", name.strip()],
                                 capture_output=True, text=True, timeout=30)
        image = inspect.stdout.strip()
        for port in ports.replace(",", " ").split():
            if "->" in port:
                images[port.split("->")[0].rsplit(":", 1)[-1]] = image
    return images


def _same_image(a: str, b: str) -> bool:
    return bool(a) and bool(b) and a.split(":")[-1] == b.split(":")[-1]


def check_deployment(deployment: Dict, service_ports: Dict[str, str], running: Dict[str, str],
                     environ=os.environ) -> List[str]:
    """Problems with the running services against the recorded deployment; empty when they match."""
    problems = []
    for service, record in deployment.get("services", {}).items():
        port = environ.get(service_ports.get(service, ""), "")
        image = running.get(port)
        if image is None:
            problems.append(f"{service}: not running on port {port or '?'}")
        elif not _same_image(image, record["image_id"]):
            problems.append(f"{service}: runs image {image[:19]}, but deepmark-services started "
                            f"{record['image']} ({record['image_id'][:19]})")
    return problems


def verify_running_deployment() -> List[str]:
    """Check the recorded deployment against what is running. Empty when there is none."""
    deployment = load_deployment()
    if deployment is None:
        return []
    engine = _engine(deployment)
    if engine is None:
        return [f"{deployment.get('engine')} is not on PATH, so the deployment in "
                f"{deployment_path()} cannot be checked"]
    from deepmarkpy.launcher import compose_service_ports

    ports = compose_service_ports((REPO / "docker-compose.yml").read_text())
    return check_deployment(deployment, ports, running_images(engine))


def deployment_summary() -> Optional[Dict]:
    """The recorded deployment, for run_metadata.json."""
    deployment = load_deployment()
    if deployment is None:
        return None
    return {
        "created_utc": deployment.get("created_utc"),
        "gpu": deployment.get("gpu"),
        "services": {
            name: {k: rec.get(k) for k in ("device", "image", "image_id", "precision", "verdicts")}
            for name, rec in deployment.get("services", {}).items()
        },
    }
