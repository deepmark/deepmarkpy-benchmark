"""Name the GPU slice a run may use, and refuse it when it is not ours to use.

GPU hosts for this benchmark are often MIG-partitioned and shared, so GPU
services are addressed by exactly one UUID, ``DEEPMARK_GPU_DEVICE`` (a
``MIG-...`` or ``GPU-...`` UUID as printed by ``nvidia-smi -L``), and never by
index or ``all``. Indices silently move to another slice when an administrator
repartitions the GPU; a UUID just stops existing, which this module reports.

:func:`check_slice` refuses a slice that is missing, that names a whole GPU in
MIG mode (only its MIG devices can run work), that has less free memory than
asked, or that runs a compute process owned by another user. A process whose
owner cannot be read counts as another user's: refusing a free slice costs a
retry, while running on someone else's costs them their job.

NVML is queried directly because ``nvidia-smi -i MIG-<uuid>`` reports "No
devices were found" on drivers where the same UUID works for CUDA. pynvml
(``nvidia-ml-py``) is imported only when a check runs, so CPU-only installs
never need it.

    python -m deepmarkpy.utils.gpu_slice                    # vet $DEEPMARK_GPU_DEVICE
    python -m deepmarkpy.utils.gpu_slice --min-free-gib 20
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

GPU_DEVICE_ENV = "DEEPMARK_GPU_DEVICE"


class SliceUnavailable(RuntimeError):
    """The named GPU slice cannot be used; the message says why."""


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    uid: Optional[int]  # None when /proc no longer (or never) showed the pid
    used_mib: int


@dataclass(frozen=True)
class SliceStatus:
    uuid: str
    name: str
    total_mib: int
    used_mib: int
    processes: Tuple[GpuProcess, ...]

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib

    def foreign_processes(self, uid: Optional[int] = None) -> Tuple[GpuProcess, ...]:
        """Processes not owned by ``uid`` (default: this user), unknown owners included."""
        own = os.getuid() if uid is None else uid
        return tuple(p for p in self.processes if p.uid != own)


def _load_nvml():
    try:
        import pynvml
    except ImportError as exc:
        raise SliceUnavailable(
            "pynvml is not installed; `pip install nvidia-ml-py` to vet a GPU slice"
        ) from exc
    return pynvml


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _owner(pid: int) -> Optional[int]:
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None


def _find_handle(nvml, uuid: str):
    """Return ``(handle, is_parent_in_mig_mode)`` for ``uuid``, or ``(None, False)``."""
    for index in range(nvml.nvmlDeviceGetCount()):
        gpu = nvml.nvmlDeviceGetHandleByIndex(index)
        try:
            mig_enabled = nvml.nvmlDeviceGetMigMode(gpu)[0] == 1
        except nvml.NVMLError:
            mig_enabled = False  # GPUs without MIG support raise here
        if _text(nvml.nvmlDeviceGetUUID(gpu)) == uuid:
            return gpu, mig_enabled
        if not mig_enabled:
            continue
        for slot in range(nvml.nvmlDeviceGetMaxMigDeviceCount(gpu)):
            try:
                mig = nvml.nvmlDeviceGetMigDeviceHandleByIndex(gpu, slot)
            except nvml.NVMLError:
                continue  # unpopulated slot
            if _text(nvml.nvmlDeviceGetUUID(mig)) == uuid:
                return mig, False
    return None, False


def slice_status(uuid: str, nvml=None) -> SliceStatus:
    """Read the name, memory and compute processes of the device ``uuid``."""
    nvml = nvml or _load_nvml()
    nvml.nvmlInit()
    try:
        handle, parent_in_mig_mode = _find_handle(nvml, uuid)
        if handle is None:
            raise SliceUnavailable(
                f"{uuid} is not on this host; `nvidia-smi -L` lists the devices "
                "that are. The GPU may have been repartitioned."
            )
        if parent_in_mig_mode:
            raise SliceUnavailable(
                f"{uuid} is a whole GPU in MIG mode; name one of its MIG "
                "devices instead (`nvidia-smi -L`)."
            )
        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
        processes = tuple(
            GpuProcess(p.pid, _owner(p.pid), (p.usedGpuMemory or 0) >> 20)
            for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle)
        )
        return SliceStatus(
            uuid=uuid,
            name=_text(nvml.nvmlDeviceGetName(handle)),
            total_mib=memory.total >> 20,
            used_mib=memory.used >> 20,
            processes=processes,
        )
    finally:
        nvml.nvmlShutdown()


def check_slice(
    uuid: Optional[str] = None,
    min_free_mib: int = 0,
    allow_foreign: bool = False,
    nvml=None,
) -> SliceStatus:
    """Return the status of ``uuid`` (default ``$DEEPMARK_GPU_DEVICE``) if it is usable.

    Raises :class:`SliceUnavailable` naming every reason it is not.
    """
    uuid = (uuid or os.environ.get(GPU_DEVICE_ENV, "")).strip()
    if not uuid:
        raise SliceUnavailable(
            f"{GPU_DEVICE_ENV} is not set; set it to the MIG or GPU UUID this "
            "run may use (`nvidia-smi -L`)."
        )
    if not uuid.startswith(("MIG-", "GPU-")):
        raise SliceUnavailable(
            f"{GPU_DEVICE_ENV}={uuid!r} is not a device UUID. Indices move "
            "when a GPU is repartitioned, so only MIG-/GPU- UUIDs are accepted."
        )

    status = slice_status(uuid, nvml)

    problems = []
    foreign = status.foreign_processes()
    if foreign and not allow_foreign:
        listed = ", ".join(
            f"pid {p.pid} (uid {'unknown' if p.uid is None else p.uid}, {p.used_mib} MiB)"
            for p in foreign
        )
        problems.append(f"{uuid} is in use by another user: {listed}")
    if status.free_mib < min_free_mib:
        problems.append(
            f"{uuid} has {status.free_mib} MiB free, {min_free_mib} MiB required"
        )
    if problems:
        raise SliceUnavailable("; ".join(problems))
    return status


def main(argv=None) -> int:
    from deepmarkpy.utils.utils import load_env_file

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default=None,
                        help=f"UUID to vet (default: ${GPU_DEVICE_ENV}, read from .env too)")
    parser.add_argument("--min-free-gib", type=float, default=0.0)
    parser.add_argument("--allow-foreign", action="store_true",
                        help="report other users' processes without refusing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env_file()
    try:
        status = check_slice(
            args.device,
            min_free_mib=int(args.min_free_gib * 1024),
            allow_foreign=args.allow_foreign,
        )
    except SliceUnavailable as exc:
        logger.error("%s", exc)
        return 1

    summary = asdict(status)
    summary["free_mib"] = status.free_mib
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
