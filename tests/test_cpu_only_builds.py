"""The benchmark runs on CPU, so no image may carry a GPU runtime.

Nothing in the stack requests a GPU: docker-compose reserves no devices and
every service selects its device with ``cuda.is_available()``, which is False
in these containers. A CUDA build therefore adds gigabytes of libraries that
cannot execute. speech_enhancement_2 acquired one because ``clearvoice`` asks
only for ``torch>=2.0.1`` and the default wheel now bundles CUDA even on arm64.
"""

import glob
import os
import re

import pytest

PLUGINS = os.path.join(os.path.dirname(__file__), "..", "src", "deepmarkpy", "plugins")

# Distribution-name prefixes that only exist to drive an NVIDIA GPU.
GPU_ONLY = re.compile(r"^(nvidia-[a-z0-9_.-]*|triton|cupy-cuda\w*)\s*==", re.I)


def _pin_files():
    files = []
    for pattern in ("*/*/constraints*.txt", "*/*/requirements.txt",
                    "*/*/xcodec_requirements.txt"):
        files += glob.glob(os.path.join(PLUGINS, pattern))
    base = os.path.join(PLUGINS, "..", "..", "..")
    files += glob.glob(os.path.join(base, "constraints*.txt"))
    files += glob.glob(os.path.join(base, "requirements.txt"))
    return sorted(f for f in files if os.path.isfile(f))


def test_pin_files_were_found():
    assert len(_pin_files()) > 10, "glob found too few pin files to be meaningful"


@pytest.mark.parametrize("path", _pin_files(), ids=lambda p: os.path.relpath(p, PLUGINS))
def test_no_active_gpu_package_pins(path):
    offenders = [
        f"{n}: {line.strip()}"
        for n, line in enumerate(open(path), 1)
        if GPU_ONLY.match(line.strip())
    ]
    assert not offenders, (
        f"{os.path.relpath(path, PLUGINS)} pins GPU-only packages:\n  "
        + "\n  ".join(offenders)
        + "\nThe benchmark is CPU-only; install the +cpu wheel instead."
    )


def test_torch_pins_do_not_request_a_cuda_build():
    """An explicit +cuXXX local version would be unambiguously wrong here."""
    bad = []
    for path in _pin_files():
        for n, line in enumerate(open(path), 1):
            s = line.strip()
            if s.startswith("#") or not s.lower().startswith(("torch", "torchvision", "torchaudio")):
                continue
            if re.search(r"\+cu\d+", s):
                bad.append(f"{os.path.relpath(path, PLUGINS)}:{n}: {s}")
    assert not bad, "CUDA-tagged torch pins found:\n  " + "\n  ".join(bad)
