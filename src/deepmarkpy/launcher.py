"""deepmark-services: start the services a benchmark needs, each on the CPU or on a verified GPU.

    deepmark-services plan --config benchmark.json --gpu 0       # what would run where, and why
    deepmark-services up   --config benchmark.json --gpu 0       # start them
    deepmark-services up   audioseal encodec --gpu MIG-...       # name services directly
    deepmark-services up   --config benchmark.json               # all on the CPU
    deepmark-services up   ... --dry-run                         # print the compose file and command only

The services a config needs are the ones the benchmark itself would call:
- its models;
- its dockerized attacks;
- CrossModelAttack's second model;
- NISQA, when a NISQA metric is on.

With ``--gpu`` (an index such as 0, or a GPU-/MIG- UUID), a service runs on
that GPU only when ``parity/verdicts/`` holds a PASS for every operation in
its gate. The PASS must be for the exact image recorded in build/images.json,
on this GPU model, from the engine the image ships, at the precision the gate
sets. Anything else runs on the CPU, and ``plan`` says why. Nothing
unverified reaches the GPU.

The command writes build/compose.deploy.yml, an override pinning each
service to its recorded image and, for GPU services, its device and
numerics. It runs ``docker compose`` (or ``podman-compose`` with
--engine podman) with that file, and records the result in
build/deployment.json. Plain ``docker compose`` with the same files does the
same thing, and the printed command shows how.
"""

import argparse
import datetime
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger("deepmark-services")

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build"


@dataclass
class Placement:
    service: str
    device: str                      # "cpu" or "cuda"
    image: str
    image_id: str
    precision: str = "float32"
    reason: str = ""
    verdicts: List[str] = field(default_factory=list)


def compose_service_ports(compose_text: str) -> Dict[str, str]:
    """``{compose service: port variable}`` from the root compose file's port lines."""
    ports, service, in_services = {}, None, False
    for line in compose_text.splitlines():
        if re.match(r"^services:\s*$", line):
            in_services = True
            continue
        if in_services and re.match(r"^\S", line):
            in_services = False
        if not in_services:
            continue
        match = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
        if match:
            service = match.group(1)
            continue
        variable = re.search(r"\$\{(\w+)\}", line) if service and "127.0.0.1:" in line else None
        if variable and service not in ports:
            ports[service] = variable.group(1)
    return ports


def services_for_urls(urls: List[str], ports: Dict[str, str], environ=os.environ) -> List[str]:
    """Map the URLs a run calls to the compose services that publish their ports."""
    by_port = {environ.get(var): service for service, var in ports.items() if environ.get(var)}
    found, unknown = [], []
    for url in urls:
        port = str(urlparse(url).port)
        if port in by_port:
            if by_port[port] not in found:
                found.append(by_port[port])
        else:
            unknown.append(url)
    if unknown:
        raise ValueError(f"no compose service publishes {unknown}; check the *_PORT values in .env")
    return found


def services_for_configs(config_paths: List[str]) -> List[str]:
    """The compose services a run of these config files calls."""
    from deepmarkpy.benchmark import Benchmark
    from deepmarkpy.config import load_configs
    from deepmarkpy.run import service_endpoints

    benchmark = Benchmark()
    urls = []
    for config in load_configs(config_paths, benchmark.attacks, benchmark.models, quiet=True):
        urls += [url for _, _, url in service_endpoints(benchmark, config, config.models, cross_model=True)]
    ports = compose_service_ports((REPO / "docker-compose.yml").read_text())
    return services_for_urls(urls, ports)


def place(services: List[str], images: Dict, gates: Dict, gpu=None, verdict_dir: Optional[Path] = None) -> List[Placement]:
    """Decide CPU or GPU per service. ``gpu`` is a gpu_slice.SliceStatus, or None."""
    from deepmarkpy.parity import verdicts

    verdict_dir = verdict_dir or verdicts.VERDICT_DIR
    placements = []
    for service in services:
        built = images.get(service, {})
        if "cpu" not in built:
            raise ValueError(f"{service}: no image recorded in build/images.json; run scripts/build_images.py")
        cpu = Placement(service, "cpu", built["cpu"]["tag"], built["cpu"]["image_id"])
        gate = gates.get(service)
        if gpu is None:
            cpu.reason = "no --gpu given"
        elif not gate:
            cpu.reason = "no GPU path for this service"
        elif any(spec["tier"] == "STOCHASTIC" for spec in gate["ops"].values()):
            cpu.reason = "stochastic: judged on distributions, not yet verified"
        elif "gpu" not in built:
            cpu.reason = "no GPU image built (scripts/build_images.py --variant gpu)"
        else:
            precision = gate.get("gpu_precision", "float32")
            found, missing = [], []
            for op in gate["ops"]:
                record = verdicts.shipped_engine_pass(
                    service, op, built["gpu"]["image_id"], gpu.name, precision, verdict_dir)
                if record:
                    found.append(verdicts.verdict_path(
                        service, op, built["gpu"]["image_id"],
                        record["candidate"]["engine_config"], verdict_dir).name)
                else:
                    missing.append(op)
            if not missing:
                placements.append(Placement(service, "cuda", built["gpu"]["tag"], built["gpu"]["image_id"],
                                            precision, f"PASS on {gpu.name}", found))
                continue
            cpu.reason = (f"no PASS for {', '.join(missing)} on {gpu.name}: "
                          f"run deepmark-parity for {service} on this GPU")
        placements.append(cpu)
    return placements


def compose_override(placements: List[Placement], engine: str, gpu_uuid: Optional[str]) -> str:
    """The override file: every service pinned to its image; GPU ones given their device."""
    lines = ["# Generated by deepmark-services; regenerated on every run, do not edit.", "services:"]
    for p in placements:
        lines += [f"  {p.service}:", f"    image: {p.image}", "    environment:",
                  f"      - DEEPMARK_DEVICE={p.device}"]
        if p.device == "cuda":
            lines += ["      - NVIDIA_TF32_OVERRIDE=0", f"      - DEEPMARK_PRECISION={p.precision}"]
            if os.path.basename(engine) == "podman":
                # podman-compose ignores deploy.resources; CDI names the device instead.
                lines += ["    devices:", f"      - nvidia.com/gpu={gpu_uuid}"]
            else:
                lines += ["    deploy:", "      resources:", "        reservations:", "          devices:",
                          "            - driver: nvidia", f'              device_ids: ["{gpu_uuid}"]',
                          "              capabilities: [gpu]"]
    return "\n".join(lines) + "\n"


def compose_command(engine: str, override: Path, services: List[str]) -> List[str]:
    base = ["podman-compose"] if os.path.basename(engine) == "podman" else ["docker", "compose"]
    return [*base, "-f", str(REPO / "docker-compose.yml"), "-f", str(override), "up", "-d", "--no-build", *services]


def _print_plan(placements: List[Placement], gpu) -> None:
    where = f"{gpu.name} ({gpu.uuid})" if gpu else "no GPU"
    print(f"GPU: {where}")
    for p in placements:
        print(f"  {p.service:22s} {p.device:5s} {p.image:40s} {p.reason}")


def _check_tags(engine: str, placements: List[Placement]) -> None:
    """Refuse when a tag no longer points at the image the verdicts and records are for."""
    for p in placements:
        out = subprocess.run([engine, "image", "inspect", "--format", "{{.Id}}", p.image],
                             capture_output=True, text=True)
        actual = out.stdout.strip()
        if out.returncode != 0:
            raise SystemExit(f"{p.image} is not in {engine}'s image store; run scripts/build_images.py")
        if actual.split(":")[-1] != p.image_id.split(":")[-1]:
            raise SystemExit(f"{p.image} is {actual[:19]}, not the recorded {p.image_id[:19]}; it was "
                             "rebuilt since scripts/build_images.py recorded it. Re-run that, and the gate.")


def main(argv=None) -> int:
    from deepmarkpy.utils.gpu_slice import SliceUnavailable, check_slice
    from deepmarkpy.utils.utils import load_env_file

    parser = argparse.ArgumentParser(prog="deepmark-services", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("plan", "up"))
    parser.add_argument("services", nargs="*", help="compose service names (or use --config)")
    parser.add_argument("--config", nargs="+", default=[], help="benchmark config file(s) to start services for")
    parser.add_argument("--gpu", default=None,
                        help="GPU for verified services: an index (0) or a GPU-/MIG- UUID; omit for CPU only")
    parser.add_argument("--engine", default=os.environ.get("CONTAINER_ENGINE", "docker"))
    parser.add_argument("--dry-run", action="store_true", help="print the compose file and command; start nothing")
    a = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env_file(str(REPO / ".env"))

    services = list(dict.fromkeys(a.services + (services_for_configs(a.config) if a.config else [])))
    if not services:
        parser.error("name services or pass --config")
    images = json.loads((BUILD / "images.json").read_text())
    gates = {k: v for k, v in json.loads((REPO / "parity" / "gates.json").read_text()).items()
             if not k.startswith("_")}
    try:
        gpu = check_slice(a.gpu, min_free_mib=2048) if a.gpu else None
    except SliceUnavailable as exc:
        logger.error("%s", exc)
        return 2

    placements = place(services, images, gates, gpu)
    _print_plan(placements, gpu)
    if a.command == "plan":
        return 0

    override = BUILD / "compose.deploy.yml"
    text = compose_override(placements, a.engine, gpu.uuid if gpu else None)
    cmd = compose_command(a.engine, override, [p.service for p in placements])
    if a.dry_run:
        print(f"\n# {override}\n{text}\n$ {' '.join(cmd)}")
        return 0
    _check_tags(a.engine, placements)
    BUILD.mkdir(exist_ok=True)
    override.write_text(text)
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        return result.returncode
    (BUILD / "deployment.json").write_text(json.dumps({
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "engine": a.engine,
        "gpu": {"uuid": gpu.uuid, "name": gpu.name} if gpu else None,
        "services": {p.service: asdict(p) for p in placements},
    }, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
