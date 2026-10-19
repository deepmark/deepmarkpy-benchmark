"""A service runs on the device its container names, with the numerics the parity gate measured.

torch is not installed in the test environment, so a small fake stands in for
it and records what was set.
"""

import re
import sys
import types
from pathlib import Path

import pytest

from deepmarkpy.serving import device as dev

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: torch._has_gpu, get_device_name=lambda i: "Fake GPU")
    torch._has_gpu = True
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
        cudnn=types.SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False, flags=None),
    )
    calls = {}
    torch._C = types.SimpleNamespace(
        _jit_set_profiling_executor=lambda v: calls.__setitem__("profiling_executor", v),
        _jit_override_can_fuse_on_gpu=lambda v: calls.__setitem__("fuse_on_gpu", v),
        _jit_set_texpr_fuser_enabled=lambda v: calls.__setitem__("texpr_fuser", v),
    )
    torch._calls = calls
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    return torch


class TestServiceDevice:
    def test_the_default_is_the_cpu_and_touches_nothing(self, monkeypatch, fake_torch):
        monkeypatch.delenv(dev.DEVICE_ENV, raising=False)
        assert dev.service_device() == "cpu"
        assert fake_torch.backends.cuda.matmul.allow_tf32 is True  # CPU path left exactly as it was
        assert fake_torch._calls == {}

    def test_cuda_applies_the_measured_numerics(self, monkeypatch, fake_torch):
        monkeypatch.setenv(dev.DEVICE_ENV, "cuda")
        assert dev.service_device() == "cuda"
        b = fake_torch.backends
        assert (b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32, b.cudnn.benchmark, b.cudnn.deterministic) == (
            False, False, False, True)
        assert fake_torch._calls == {"profiling_executor": False, "fuse_on_gpu": False, "texpr_fuser": False}
        import os
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    def test_cuda_without_a_gpu_is_an_error_not_a_quiet_cpu_run(self, monkeypatch, fake_torch):
        fake_torch._has_gpu = False
        monkeypatch.setenv(dev.DEVICE_ENV, "cuda")
        with pytest.raises(RuntimeError, match="no GPU is visible"):
            dev.service_device()

    @pytest.mark.parametrize("value", ["gpu", "cuda:1", "mps"])
    def test_anything_else_is_refused(self, monkeypatch, value):
        monkeypatch.setenv(dev.DEVICE_ENV, value)
        with pytest.raises(ValueError, match=dev.DEVICE_ENV):
            dev.service_device()


class TestPrecision:
    def test_default_and_double(self, monkeypatch):
        monkeypatch.delenv(dev.PRECISION_ENV, raising=False)
        assert dev.service_precision() == "float32"
        monkeypatch.setenv(dev.PRECISION_ENV, "float64")
        assert dev.service_precision() == "float64"

    def test_half_precision_is_refused(self, monkeypatch):
        monkeypatch.setenv(dev.PRECISION_ENV, "float16")
        with pytest.raises(ValueError):
            dev.service_precision()


class TestMigFallback:
    def test_only_the_mig_allocator_assert_is_retried(self):
        assert dev.is_mig_allocator_assert(RuntimeError(
            'NVML_SUCCESS == r INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/CUDACachingAllocator.cpp":1016'))
        assert not dev.is_mig_allocator_assert(RuntimeError("CUDA out of memory"))
        assert not dev.is_mig_allocator_assert(ValueError("NVML_SUCCESS"))

    def test_other_errors_pass_through_untouched(self):
        def boom():
            raise RuntimeError("shape mismatch")
        with pytest.raises(RuntimeError, match="shape mismatch"):
            dev.with_cudnn_fallback(boom)

    def test_the_assert_is_retried_once_without_cudnn(self, fake_torch):
        state = {"calls": 0, "cudnn_off_on_retry": None}

        class Flags:
            def __init__(self, enabled):
                self.enabled = enabled

            def __enter__(self):
                state["cudnn_off_on_retry"] = not self.enabled

            def __exit__(self, *exc):
                return False

        fake_torch.backends.cudnn.flags = Flags

        def fn():
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("NVML_SUCCESS == r INTERNAL ASSERT FAILED")
            return "ok"

        assert dev.with_cudnn_fallback(fn) == "ok"
        assert state == {"calls": 2, "cudnn_off_on_retry": True}


def test_services_and_the_parity_gate_set_the_same_numerics():
    """The gate must measure what production runs; a flag set in one place only is drift."""
    flags = re.compile(r"(torch\.backends\.[\w.]+\s*=\s*\w+|torch\._C\._jit_\w+\(\w+\))")
    serving = set(flags.findall((REPO / "src/deepmarkpy/serving/device.py").read_text()))
    runner = set(flags.findall((REPO / "src/deepmarkpy/parity/runner.py").read_text()))
    assert serving and serving == runner
