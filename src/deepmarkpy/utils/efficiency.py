"""The efficiency metric family: what the run cost in time and memory.

These describe the machine and the deployment, not the watermarking
method, so they do not reproduce across runs.

Every line this module logs carries ``TERMINAL_TAG`` so a run's timing
output can be read, or filtered out, on its own:

    deepmark-benchmark --config c.json 2>&1 | grep '\\[efficiency\\]'
"""

import logging
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Prefixes every line this family prints, whatever the log format.
TERMINAL_TAG = "[efficiency]"


def log(message, *args, level=logging.INFO):
    """Log one tagged efficiency line."""
    logger.log(level, f"{TERMINAL_TAG} {message}", *args)


@contextmanager
def measure(record, key, enabled=True):
    """Time the block and store the elapsed seconds under ``key``.

    Args:
        record: the dict to write into -- the per-file or per-attack
            result entry, so the number travels with what it describes.
        key: the metric name, e.g. ``"embed_latency"``.
        enabled: whether the config asked for this metric. When it did
            not, the block runs untimed and nothing is written.

    Wall-clock time, so it includes the HTTP round trip of a dockerized
    plugin.
    """
    if not enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        record[key] = time.perf_counter() - start


# How long to wait on the docker CLI before giving up on a footprint.
_DOCKER_TIMEOUT_S = 15


def _run_docker(args):
    """Run a docker command, or return None when docker is unusable."""
    if not shutil.which("docker"):
        return None
    try:
        result = subprocess.run(
            ["docker", *args], capture_output=True, text=True,
            timeout=_DOCKER_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - never fatal to a run
        log("docker unavailable (%s); footprint not measured", exc,
            level=logging.WARNING)
        return None
    if result.returncode != 0:
        log("docker exited %s; footprint not measured", result.returncode,
            level=logging.WARNING)
        return None
    return result.stdout


def _running_containers():
    """``{published port: container name}`` for what is up right now."""
    output = _run_docker(["ps", "--format", "{{.Names}}|{{.Ports}}"])
    if not output:
        return {}
    by_port = {}
    for line in output.splitlines():
        name, _, ports = line.partition("|")
        # Published ports read like "127.0.0.1:5001->5001/tcp".
        for port in re.findall(r"[:\s](\d+)->", ports):
            by_port[port] = name.strip()
    return by_port


def _memory_usage(names):
    """``{container: (used MiB, limit MiB)}`` for the given containers."""
    if not names:
        return {}
    output = _run_docker(
        ["stats", "--no-stream", "--format", "{{.Name}}|{{.MemUsage}}", *names]
    )
    if not output:
        return {}

    scale = {"B": 1 / 1024 ** 2, "KiB": 1 / 1024, "MiB": 1.0,
             "GiB": 1024.0, "TiB": 1024.0 ** 2}

    def mib(text):
        match = re.match(r"\s*([\d.]+)\s*([KMGT]?i?B)", text)
        if not match:
            return None
        factor = scale.get(match.group(2))
        return float(match.group(1)) * factor if factor else None

    usage = {}
    for line in output.splitlines():
        name, _, figure = line.partition("|")
        used, _, limit = figure.partition("/")
        usage[name.strip()] = (mib(used), mib(limit))
    return usage


def container_snapshot(entries):
    """Memory held by the services this run used, for those still running.

    Args:
        entries: ``[(kind, label, url)]`` -- the models, the dockerized
            attacks and the metric services the run actually touched.

    Returns:
        ``[(kind, label, container, used MiB, limit MiB)]``, one row per
        entry whose port a running container publishes. Anything native,
        stopped, or unreachable is left out rather than reported as zero.
        The figure is the whole container's memory at one moment.
    """
    by_port = _running_containers()
    if not by_port:
        return []

    matched = []
    for kind, label, url in entries:
        try:
            port = urlparse(url).port
        except Exception:  # noqa: BLE001 - a malformed url simply has none
            port = None
        container = by_port.get(str(port)) if port else None
        if container:
            matched.append((kind, label, container))

    usage = _memory_usage(sorted({c for _, _, c in matched}))
    rows = []
    for kind, label, container in matched:
        used, limit = usage.get(container, (None, None))
        if used is not None:
            rows.append((kind, label, container, used, limit))
    return rows
