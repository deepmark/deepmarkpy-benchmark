"""A GPU run must use only the device it names, and only while nobody else is on it.

The device is a whole GPU (by index or UUID) or one MIG slice (by UUID). No GPU
is needed: NVML is replaced by a fake device tree, a single GPU that is either
whole or in MIG mode with several MIG devices.
"""

import os
from types import SimpleNamespace

import pytest

from deepmarkpy.utils.gpu_slice import (
    GPU_DEVICE_ENV,
    SliceUnavailable,
    check_slice,
    main,
)

OURS = "MIG-50bf93fa-b44c-530a-acf3-c79763c56eec"
NEIGHBOUR = "MIG-2978f192-782d-5459-9e4d-136c05d90bbc"
PARENT = "GPU-69e9f719-dbd7-1bc8-b0b6-fb7f878affd8"


class FakeNVML:
    class NVMLError(Exception):
        pass

    def __init__(self, procs=None, used_mib=28, mig_mode=True):
        self.procs = procs or {}
        self.used_mib = used_mib
        self.mig_mode = mig_mode
        self.shutdowns = 0
        self.migs = [OURS, NEIGHBOUR, None]  # None: unpopulated slot

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        self.shutdowns += 1

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return PARENT

    def nvmlDeviceGetMigMode(self, handle):
        return (1 if self.mig_mode else 0, 1 if self.mig_mode else 0)

    def nvmlDeviceGetUUID(self, handle):
        return handle.encode()  # older pynvml returns bytes

    def nvmlDeviceGetMaxMigDeviceCount(self, handle):
        return len(self.migs)

    def nvmlDeviceGetMigDeviceHandleByIndex(self, handle, slot):
        if self.migs[slot] is None:
            raise self.NVMLError("not found")
        return self.migs[slot]

    def nvmlDeviceGetName(self, handle):
        return "NVIDIA H200 NVL MIG 2g.35gb"

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(total=33280 << 20, used=self.used_mib << 20)

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return [SimpleNamespace(pid=pid, usedGpuMemory=mib << 20)
                for pid, mib in self.procs.get(handle, [])]


@pytest.fixture
def owners(monkeypatch):
    """Map pid -> uid as /proc would; pids not listed have vanished."""
    table = {}

    def fake_owner(pid):
        return table.get(pid)

    monkeypatch.setattr("deepmarkpy.utils.gpu_slice._owner", fake_owner)
    return table


class TestTheSliceMustBeNamed:
    def test_unset_is_refused(self, monkeypatch):
        monkeypatch.delenv(GPU_DEVICE_ENV, raising=False)
        with pytest.raises(SliceUnavailable, match=GPU_DEVICE_ENV):
            check_slice(nvml=FakeNVML())

    @pytest.mark.parametrize("name", ["all", "cuda:0", "gpu0"])
    def test_anything_but_an_index_or_uuid_is_refused(self, name):
        with pytest.raises(SliceUnavailable, match="not a GPU index"):
            check_slice(name, nvml=FakeNVML())

    def test_a_whole_gpu_by_index_resolves_to_its_uuid(self, owners):
        assert check_slice("0", nvml=FakeNVML(mig_mode=False)).uuid == PARENT

    def test_an_index_of_a_gpu_in_mig_mode_asks_for_a_slice(self, owners):
        with pytest.raises(SliceUnavailable, match="MIG mode"):
            check_slice("0", nvml=FakeNVML())

    def test_an_index_past_the_last_gpu_is_refused(self, owners):
        with pytest.raises(SliceUnavailable, match="does not exist"):
            check_slice("3", nvml=FakeNVML(mig_mode=False))

    def test_the_environment_names_it(self, monkeypatch, owners):
        monkeypatch.setenv(GPU_DEVICE_ENV, OURS)
        assert check_slice(nvml=FakeNVML()).uuid == OURS

    def test_a_vanished_uuid_says_the_gpu_may_be_repartitioned(self, owners):
        gone = "MIG-ff1e51ba-09d9-5256-8a89-8fb1261d5e75"
        with pytest.raises(SliceUnavailable, match="repartitioned"):
            check_slice(gone, nvml=FakeNVML())

    def test_the_whole_gpu_in_mig_mode_is_refused(self, owners):
        with pytest.raises(SliceUnavailable, match="MIG mode"):
            check_slice(PARENT, nvml=FakeNVML())

    def test_a_whole_gpu_without_mig_is_usable(self, owners):
        assert check_slice(PARENT, nvml=FakeNVML(mig_mode=False)).uuid == PARENT


class TestOnlyWhileNobodyElseIsOnIt:
    def test_an_idle_slice_passes(self, owners):
        status = check_slice(OURS, nvml=FakeNVML())
        assert status.processes == ()
        assert status.free_mib == 33280 - 28

    def test_our_own_processes_do_not_block(self, owners):
        owners[100] = os.getuid()
        nvml = FakeNVML(procs={OURS: [(100, 402)]})
        assert len(check_slice(OURS, nvml=nvml).processes) == 1

    def test_another_users_process_blocks_and_is_named(self, owners):
        owners[200] = os.getuid() + 1
        nvml = FakeNVML(procs={OURS: [(200, 2034)]})
        with pytest.raises(SliceUnavailable, match="pid 200"):
            check_slice(OURS, nvml=nvml)

    def test_an_unreadable_owner_counts_as_another_user(self, owners):
        nvml = FakeNVML(procs={OURS: [(300, 10)]})  # 300 absent from /proc
        with pytest.raises(SliceUnavailable, match="uid unknown"):
            check_slice(OURS, nvml=nvml)

    def test_a_neighbours_processes_are_not_ours_to_worry_about(self, owners):
        owners[400] = os.getuid() + 1
        nvml = FakeNVML(procs={NEIGHBOUR: [(400, 2034)]})
        assert check_slice(OURS, nvml=nvml).processes == ()

    def test_allow_foreign_reports_without_refusing(self, owners):
        owners[200] = os.getuid() + 1
        nvml = FakeNVML(procs={OURS: [(200, 2034)]})
        status = check_slice(OURS, allow_foreign=True, nvml=nvml)
        assert [p.pid for p in status.foreign_processes()] == [200]

    def test_too_little_free_memory_is_refused(self, owners):
        nvml = FakeNVML(used_mib=30000)
        with pytest.raises(SliceUnavailable, match="MiB free"):
            check_slice(OURS, min_free_mib=4096, nvml=nvml)

    def test_every_reason_is_reported_together(self, owners):
        owners[200] = os.getuid() + 1
        nvml = FakeNVML(procs={OURS: [(200, 2034)]}, used_mib=30000)
        with pytest.raises(SliceUnavailable) as err:
            check_slice(OURS, min_free_mib=4096, nvml=nvml)
        assert "another user" in str(err.value) and "MiB free" in str(err.value)

    def test_nvml_is_shut_down_even_on_refusal(self, owners):
        nvml = FakeNVML()
        with pytest.raises(SliceUnavailable):
            check_slice("MIG-00000000-0000-0000-0000-000000000000", nvml=nvml)
        assert nvml.shutdowns == 1


class TestCommandLine:
    def test_exit_status_reflects_the_verdict(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)  # no .env here
        monkeypatch.delenv(GPU_DEVICE_ENV, raising=False)
        assert main([]) == 1
