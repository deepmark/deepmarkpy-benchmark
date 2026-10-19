"""Build the service images, CPU or GPU variant, and record what was built.

    python scripts/build_images.py --variant cpu                 # every service
    python scripts/build_images.py --variant gpu audioseal vae   # a subset
    CONTAINER_ENGINE=podman python scripts/build_images.py ...

Services and their Dockerfiles come from docker-compose.yml. Images are
tagged ``deepmark-<service>:<variant>``. The base is tagged
``ml-services-base:latest`` (CPU) and ``ml-services-base:<cuda>`` (GPU).

How each service gets its GPU variant (``GPU_RECIPES``):

- ``base``: built FROM the CUDA build of ml-services-base, because the
  service inherits torch from the base.
- ``swap``: the CPU image with only its own torch swapped for a CUDA build
  (docker/gpu/Dockerfile.torch-swap), for services that install a torch
  version different from the base's.
- ``same``: the CPU image already carries a CUDA-capable torch, so one image
  serves both.
- ``none``: no GPU variant. NetworkTransmission and Opus run no torch.

Every built image is recorded in build/images.json:
- tag and image ID;
- the git revision;
- the torch version and its CUDA/cuDNN build, read from inside the image.

A GPU variant whose torch has no CUDA build fails the build rather than
quietly running on CPU. GPU parity verdicts are keyed by the image ID
recorded here, so a rebuilt image needs a new verdict.
"""

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

logger = logging.getLogger("build_images")

REPO = Path(__file__).resolve().parent.parent
BASE_CUDA = "cu126"

GPU_RECIPES = {
    "audioseal": ("base",),
    "wavmark": ("base",),
    "aware": ("base",),
    "perth": ("base",),
    "vae": ("base",),
    "encodec": ("base",),
    "descript_audio_codec": ("base",),
    "diffusion": ("base",),
    "neural_vocoder": ("base",),
    "speech_enhancement1": ("base",),
    "speech_tokenization": ("same",),  # installs torch 2.4.1 from PyPI: the cu121 build
    "nisqa": ("same",),  # installs torch 2.2.1 from PyPI: the cu121 build
    "speech_enhancement2": ("swap", "torch==2.13.0+cu130", "cu130"),
    "silentcipher": ("swap", "torch==2.0.0+cu118 torchaudio==2.0.1+cu118", "cu118"),
    "timbrewm": ("swap", "torch==2.0.0+cu118 torchaudio==2.0.1+cu118", "cu118"),
    "network_transmission": ("none",),
    "opus_codec": ("none",),
}

TORCH_PROBE = (
    "import json, torch; "
    "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, "
    "'cudnn': torch.backends.cudnn.version()}))"
)


def compose_services():
    """Return {service: dockerfile} from the root docker-compose.yml."""
    with open(REPO / "docker-compose.yml") as handle:
        compose = yaml.safe_load(handle)
    return {name: spec["build"]["dockerfile"] for name, spec in compose["services"].items()}


def run(cmd, log_path, dry_run):
    logger.info("$ %s", " ".join(cmd))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"build failed ({result.returncode}); see {log_path}")


def inspect(engine, tag):
    image_id = subprocess.run(
        [engine, "image", "inspect", "--format", "{{.Id}}", tag],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    probe = subprocess.run(
        [engine, "run", "--rm", "--entrypoint", "python", tag, "-c", TORCH_PROBE],
        capture_output=True, text=True,
    )
    torch_info = json.loads(probe.stdout.strip().splitlines()[-1]) if probe.returncode == 0 else None
    return image_id, torch_info


def git_rev():
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=REPO, capture_output=True, text=True).stdout.strip()
    return rev + ("-dirty" if dirty else "")


def build_base(engine, variant, args):
    torch_variant = "cpu" if variant == "cpu" else BASE_CUDA
    tag = "ml-services-base:latest" if variant == "cpu" else f"ml-services-base:{BASE_CUDA}"
    run([engine, "build", "-f", "Dockerfile.base", "--build-arg", f"TORCH_VARIANT={torch_variant}",
         "-t", tag, "."], args.log_dir / f"base-{variant}.log", args.dry_run)
    return tag


def build_service(engine, service, dockerfile, variant, args):
    tag = f"deepmark-{service}:{variant}"
    log = args.log_dir / f"{service}-{variant}.log"
    if variant == "cpu":
        run([engine, "build", "-f", dockerfile, "-t", tag, "."], log, args.dry_run)
        return tag

    recipe = GPU_RECIPES[service]
    kind = recipe[0]
    if kind == "none":
        logger.info("%s has no GPU variant; skipped", service)
        return None
    if kind == "base":
        run([engine, "build", "-f", dockerfile,
             "--build-arg", f"BASE_IMAGE=ml-services-base:{BASE_CUDA}", "-t", tag, "."], log, args.dry_run)
    elif kind == "same":
        run([engine, "tag", f"deepmark-{service}:cpu", tag], log, args.dry_run)
    elif kind == "swap":
        _, packages, index = recipe
        run([engine, "build", "-f", "docker/gpu/Dockerfile.torch-swap",
             "--build-arg", f"BASE=deepmark-{service}:cpu",
             "--build-arg", f"TORCH_PACKAGES={packages}",
             "--build-arg", f"TORCH_INDEX={index}", "-t", tag, "."], log, args.dry_run)
    return tag


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("services", nargs="*", help="compose service names (default: all)")
    parser.add_argument("--variant", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--engine", default=os.environ.get("CONTAINER_ENGINE", "docker"))
    parser.add_argument("--skip-base", action="store_true", help="reuse the existing base image")
    parser.add_argument("--manifest", type=Path, default=REPO / "build" / "images.json")
    parser.add_argument("--log-dir", type=Path, default=REPO / "build" / "logs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    services = compose_services()
    unknown = set(args.services) - set(services)
    if unknown:
        parser.error(f"not in docker-compose.yml: {sorted(unknown)}")
    missing_recipes = set(services) - set(GPU_RECIPES)
    if missing_recipes:
        parser.error(f"no GPU recipe for {sorted(missing_recipes)}; add one to GPU_RECIPES")
    selected = args.services or list(services)

    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
    rev = git_rev()

    failures = []

    def record(service, tag):
        # Written after every image, so a later failure keeps what was built.
        if args.dry_run or not tag:
            return
        image_id, torch_info = inspect(args.engine, tag)
        manifest.setdefault(service, {})[args.variant] = {
            "tag": tag,
            "image_id": image_id,
            "git_rev": rev,
            "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "torch": torch_info,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        logger.info("%-22s %-5s %s %s", service, args.variant, image_id[:12], torch_info)
        if args.variant == "gpu" and torch_info and not torch_info.get("cuda"):
            failures.append(f"{service}: GPU variant has torch {torch_info['torch']} without CUDA")

    if not args.skip_base:
        record("ml-services-base", build_base(args.engine, args.variant, args))
    for service in selected:
        record(service, build_service(args.engine, service, services[service], args.variant, args))

    if failures:
        raise SystemExit("\n".join(failures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
