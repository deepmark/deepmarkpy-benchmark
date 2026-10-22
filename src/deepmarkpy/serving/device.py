"""Where a service's engine runs, and the numerics it runs with.

A service runs on the CPU unless its container sets ``DEEPMARK_DEVICE=cuda``
(docker-compose.gpu.yml). A requested GPU that is not visible is an error, not
a fallback to the CPU. On the GPU, TF32, cuDNN autotuning and TorchScript
fusion are off, so results match the CPU and repeat. The CPU path is untouched.
``DEEPMARK_PRECISION=float64`` is read by engines that support it (BigVGAN).
"""

import logging
import os

logger = logging.getLogger(__name__)

DEVICE_ENV = "DEEPMARK_DEVICE"
PRECISION_ENV = "DEEPMARK_PRECISION"
DEVICES = ("cpu", "cuda")
PRECISIONS = ("float32", "float64")


def service_device() -> str:
    """The device this service's engine is placed on: ``cpu`` (default) or ``cuda``."""
    requested = (os.environ.get(DEVICE_ENV) or "cpu").strip().lower()
    if requested not in DEVICES:
        raise ValueError(f"{DEVICE_ENV}={requested!r}: expected one of {DEVICES}")
    if requested == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{DEVICE_ENV}=cuda but no GPU is visible to this container; "
                "start it with the GPU compose override (or set it to cpu)"
            )
        configure_numerics()
        logger.info("Engine device: cuda (%s)", torch.cuda.get_device_name(0))
    else:
        logger.info("Engine device: cpu")
    return requested


def service_precision() -> str:
    """``float32`` (default) or ``float64``, for engines that support the latter."""
    precision = (os.environ.get(PRECISION_ENV) or "float32").strip().lower()
    if precision not in PRECISIONS:
        raise ValueError(f"{PRECISION_ENV}={precision!r}: expected one of {PRECISIONS}")
    return precision


def configure_numerics() -> None:
    """Apply the GPU numerics the services were verified with. Call before the first CUDA work."""
    # Read when cuBLAS creates its handle, so it must be set before any CUDA work.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)


def is_mig_allocator_assert(exc: BaseException) -> bool:
    """PyTorch's allocator assert under MIG (it reports through NVML, which MIG forbids)."""
    return isinstance(exc, RuntimeError) and "NVML_SUCCESS" in str(exc)


def with_cudnn_fallback(fn):
    """Run ``fn()``; on the MIG allocator assert, run it once more without cuDNN.

    Under MIG the AudioSeal detector hits it on short inputs; a whole GPU never does.
    """
    try:
        return fn()
    except RuntimeError as exc:
        if not is_mig_allocator_assert(exc):
            raise
        import torch

        logger.warning("cuDNN hit the MIG allocator assert; retrying this call without cuDNN")
        with torch.backends.cudnn.flags(enabled=False):
            return fn()
