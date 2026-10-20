"""A GPU image variant must really be one, and every service must have a stated recipe.

scripts/build_images.py builds the GPU variant of most services by passing
the CUDA base through BASE_IMAGE. A Dockerfile that hard-codes its base would
build a "GPU" image with CPU torch, and the service would quietly run on the
CPU. The build itself refuses such an image; these tests catch the cause
before anything is built.
"""

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_build_images():
    spec = importlib.util.spec_from_file_location("build_images", REPO / "scripts" / "build_images.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_images = _load_build_images()
SERVICES = build_images.compose_services()


def test_every_compose_service_has_a_gpu_recipe():
    assert set(SERVICES) == set(build_images.GPU_RECIPES)


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_a_service_on_the_shared_base_takes_it_as_an_argument(service):
    dockerfile = (REPO / SERVICES[service]).read_text()
    froms = re.findall(r"^FROM\s+(\S+)", dockerfile, flags=re.MULTILINE)
    on_shared_base = any(f == "${BASE_IMAGE}" or f.startswith("ml-services-base") for f in froms)
    if not on_shared_base:
        assert build_images.GPU_RECIPES[service][0] == "none", (
            f"{service} has its own base, so it cannot get the CUDA base"
        )
        return
    assert "ARG BASE_IMAGE=ml-services-base:latest" in dockerfile
    assert froms and all(f == "${BASE_IMAGE}" for f in froms), (
        f"{service}: every stage must build FROM ${{BASE_IMAGE}}, found {froms}"
    )


def test_the_base_selects_its_torch_build_by_argument():
    dockerfile = (REPO / "Dockerfile.base").read_text()
    assert "ARG TORCH_VARIANT=cpu" in dockerfile
    assert "torch==2.7.1+${TORCH_VARIANT}" in dockerfile
    assert "whl/${TORCH_VARIANT}" in dockerfile


@pytest.mark.parametrize("service", sorted(s for s, r in build_images.GPU_RECIPES.items() if r[0] == "swap"))
def test_a_swap_recipe_names_cuda_builds_only(service):
    _, packages, index = build_images.GPU_RECIPES[service]
    assert re.fullmatch(r"cu\d+", index)
    for package in packages.split():
        assert package.endswith("+" + index), f"{service}: {package} is not a {index} build"


class TestGpuComposeOverride:
    """docker-compose.gpu.yml runs exactly the verified services on the GPU, as verified."""

    VERIFIED = {"audioseal", "descript_audio_codec", "neural_vocoder", "speech_enhancement1", "speech_enhancement2"}

    @pytest.fixture(scope="class")
    def override(self):
        import yaml

        return yaml.safe_load((REPO / "docker-compose.gpu.yml").read_text())["services"]

    def test_it_lists_exactly_the_verified_services(self, override):
        assert set(override) == self.VERIFIED

    @pytest.mark.parametrize("service", sorted(VERIFIED))
    def test_each_runs_its_gpu_image_on_the_named_gpu_with_strict_numerics(self, override, service):
        spec = override[service]
        assert spec["image"] == f"deepmark-{service}:gpu"
        assert build_images.GPU_RECIPES[service][0] in ("base", "swap", "same")
        assert {"DEEPMARK_DEVICE=cuda", "NVIDIA_TF32_OVERRIDE=0"} <= set(spec["environment"])
        device = spec["deploy"]["resources"]["reservations"]["devices"][0]
        assert device["device_ids"] == ["${DEEPMARK_GPU_DEVICE:-0}"] and device["capabilities"] == ["gpu"]

    def test_bigvgan_runs_in_float64(self, override):
        assert "DEEPMARK_PRECISION=float64" in override["neural_vocoder"]["environment"]
