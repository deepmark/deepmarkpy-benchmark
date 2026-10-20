"""docker-compose.gpu.yml builds and runs exactly the verified services on the GPU, as verified.

A service the override lists but builds without CUDA torch would refuse to
start (serving.device); one built from a hard-coded CPU base would never get
the CUDA one. These tests catch both before anything is built.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
VERIFIED = {"audioseal", "descript_audio_codec", "neural_vocoder", "speech_enhancement1", "speech_enhancement2"}
CUDA_BASE = {"audioseal", "descript_audio_codec", "neural_vocoder", "speech_enhancement1"}  # torch from the base image


@pytest.fixture(scope="module")
def override():
    return yaml.safe_load((REPO / "docker-compose.gpu.yml").read_text())["services"]


@pytest.fixture(scope="module")
def dockerfiles():
    services = yaml.safe_load((REPO / "docker-compose.yml").read_text())["services"]
    return {name: (REPO / spec["build"]["dockerfile"]).read_text() for name, spec in services.items()}


def test_it_lists_exactly_the_verified_services(override):
    assert set(override) == VERIFIED


@pytest.mark.parametrize("service", sorted(VERIFIED))
def test_each_runs_on_the_named_gpu_with_strict_numerics(override, service):
    spec = override[service]
    assert spec["image"] == f"deepmark-{service}:gpu"
    assert {"DEEPMARK_DEVICE=cuda", "NVIDIA_TF32_OVERRIDE=0"} <= set(spec["environment"])
    device = spec["deploy"]["resources"]["reservations"]["devices"][0]
    assert device["device_ids"] == ["${DEEPMARK_GPU_DEVICE:-0}"] and device["capabilities"] == ["gpu"]


@pytest.mark.parametrize("service", sorted(CUDA_BASE))
def test_services_on_the_shared_base_build_from_its_cuda_variant(override, dockerfiles, service):
    assert override[service]["build"]["args"]["BASE_IMAGE"] == "ml-services-base:cu126"
    dockerfile = dockerfiles[service]
    assert "ARG BASE_IMAGE=ml-services-base:latest" in dockerfile
    froms = re.findall(r"^FROM\s+(\S+)", dockerfile, flags=re.MULTILINE)
    assert froms and all(f == "${BASE_IMAGE}" for f in froms), f"{service}: every stage must build FROM ${{BASE_IMAGE}}"


def test_se2_swaps_in_the_cuda_build_of_its_own_torch(override, dockerfiles):
    assert override["speech_enhancement2"]["build"]["args"] == {"TORCH_INDEX": "cu130"}
    dockerfile = dockerfiles["speech_enhancement2"]
    assert "ARG TORCH_INDEX=cpu" in dockerfile
    assert 'torch==2.13.0+${TORCH_INDEX}' in dockerfile and 'whl/${TORCH_INDEX}' in dockerfile


def test_the_base_selects_its_torch_build_by_argument():
    dockerfile = (REPO / "Dockerfile.base").read_text()
    assert "ARG TORCH_VARIANT=cpu" in dockerfile
    assert "torch==2.7.1+${TORCH_VARIANT}" in dockerfile and "whl/${TORCH_VARIANT}" in dockerfile


def test_bigvgan_runs_in_float64(override):
    assert "DEEPMARK_PRECISION=float64" in override["neural_vocoder"]["environment"]


def test_services_not_on_the_gpu_keep_their_dockerfiles_unparameterised(dockerfiles):
    # Only GPU services take a base argument; the rest build exactly as before.
    for service, dockerfile in dockerfiles.items():
        if service not in CUDA_BASE:
            assert "BASE_IMAGE" not in dockerfile, service
