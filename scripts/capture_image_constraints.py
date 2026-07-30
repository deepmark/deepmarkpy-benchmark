"""Capture per-image pip freezes into per-plugin constraint files (REORG_PLAN P0.2).

Run from the repo root, with the base image and all compose services built:

    python scripts/capture_image_constraints.py            # all images
    python scripts/capture_image_constraints.py aware vae  # subset

Valid selectors: any compose service name, plus the special selectors
``base`` (Dockerfile.base's image) and ``nv-builder`` (neural_vocoder's
builder stage). Unknown selectors are an error.

For each image this runs ``pip freeze`` as the container command (no app
import, no weights download), filters the output to plain ``name==version``
lines (editable and direct-URL entries cannot appear in a pip constraints
file — the aware editable install is pinned by its git-checkout commit
instead), and writes:

    <plugin_dir>/constraints.txt              per service image
    constraints.base.txt                      for Dockerfile.base (repo root)
    <neural_vocoder>/constraints.builder.txt  for its builder stage

The files record what the images resolved at capture time; re-capturing
changes the frozen dependency set.

Note: aware's ``constraints.stage1.txt`` (the interim state after its
editable install, before requirements.txt upgrades some pins) is NOT
regenerated here — it must be captured from a build stopped after the
editable-install step, because the final image's freeze is unsatisfiable
for that step.
"""

import logging
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent

SPECIAL_SELECTORS = {"base", "nv-builder"}

SERVICE_TO_PLUGIN = {
    "audioseal": "src/plugins/models/audio_seal",
    "aware": "src/plugins/models/aware",
    "perth": "src/plugins/models/perth",
    "silentcipher": "src/plugins/models/silent_cipher",
    "timbrewm": "src/plugins/models/timbrewm",
    "wavmark": "src/plugins/models/wavmark",
    "vae": "src/plugins/attacks/vae",
    "encodec": "src/plugins/attacks/encodec",
    "descript_audio_codec": "src/plugins/attacks/descript_audio_codec",
    "diffusion": "src/plugins/attacks/diffusion",
    "neural_vocoder": "src/plugins/attacks/neural_vocoder",
    "opus_codec": "src/plugins/attacks/opus_codec",
    "network_transmission": "src/plugins/attacks/network_transmission",
    "speech_enhancement1": "src/plugins/attacks/speech_enhancement_1",
    "speech_enhancement2": "src/plugins/attacks/speech_enhancement_2",
    "speech_tokenization": "src/plugins/attacks/speech_tokenization",
}

PIN_RE = re.compile(r"^[A-Za-z0-9._-]+==\S+$")


def freeze_via(cmd: list) -> tuple:
    """Run a pip-freeze command; return (kept_lines, dropped_lines)."""
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{out.stderr[-2000:]}")
    kept, dropped = [], []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        (kept if PIN_RE.match(line) else dropped).append(line)
    return kept, dropped


def write_constraints(path: Path, source: str, kept: list, dropped: list) -> None:
    header = [
        f"# pip constraints captured from {source}",
        "# by scripts/capture_image_constraints.py. Pins the image to its",
        "# resolved dependency set.",
    ]
    if dropped:
        header.append("# Excluded non-constraint entries (pinned elsewhere): "
                      + "; ".join(dropped))
    path.write_text("\n".join(header + sorted(kept, key=str.lower)) + "\n")
    logger.info("wrote %s (%d pins, %d excluded)",
                path.relative_to(REPO), len(kept), len(dropped))


def main() -> None:
    only = set(sys.argv[1:]) or None
    if only is not None:
        unknown = only - set(SERVICE_TO_PLUGIN) - SPECIAL_SELECTORS
        if unknown:
            raise SystemExit(
                f"unknown selector(s): {sorted(unknown)}. Valid: "
                f"{sorted(SERVICE_TO_PLUGIN)} plus {sorted(SPECIAL_SELECTORS)}"
            )
    if only is None or "base" in only:
        kept, dropped = freeze_via(
            ["docker", "run", "--rm", "ml-services-base:latest", "pip", "freeze"])
        write_constraints(REPO / "constraints.base.txt",
                          "ml-services-base:latest", kept, dropped)
    for svc, plugin in SERVICE_TO_PLUGIN.items():
        if only is not None and svc not in only:
            continue
        kept, dropped = freeze_via(
            ["docker-compose", "run", "--rm", "--no-deps", "-T", svc,
             "pip", "freeze"])
        write_constraints(REPO / plugin / "constraints.txt",
                          f"compose service '{svc}' image", kept, dropped)
    if only is None or "nv-builder" in only:
        build = subprocess.run(
            ["docker", "build", "--target", "builder", "-t", "nv-builder-freeze",
             "-f", "src/plugins/attacks/neural_vocoder/Dockerfile", "."],
            cwd=REPO, capture_output=True, text=True, timeout=3600)
        if build.returncode != 0:
            raise SystemExit(
                "neural_vocoder builder-stage build failed:\n"
                + build.stderr[-3000:]
            )
        kept, dropped = freeze_via(
            ["docker", "run", "--rm", "nv-builder-freeze", "pip", "freeze"])
        write_constraints(
            REPO / "src/plugins/attacks/neural_vocoder/constraints.builder.txt",
            "the neural_vocoder builder stage", kept, dropped)


if __name__ == "__main__":
    main()
