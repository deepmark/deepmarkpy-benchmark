"""Golden replay tests for native attacks.

Each golden in ``tests/fixtures/goldens/`` was recorded by
``scripts/generate_native_goldens.py`` in the canonical environment on
the designated machine. This suite replays every goldened attack under the
same protocol — canonical input ``0.5 *
np.random.default_rng(42).standard_normal(16000)``, ``np.random.seed(42)``
immediately before the invocation — and requires the output to be
byte-identical (dtype, shape, values) to the recorded fixture.

Byte-identity is a same-machine, same-environment claim: torch CPU
numerics differ across architectures, and mp3_compression shells out to the
machine's ffmpeg. In other environments a mismatch (as opposed to an error)
still indicates a real behavioral difference worth investigating.

Attacks not goldened — the machine-readable copy lives in the fixture
manifest (``tests/fixtures/goldens/manifest.json``, the source of truth this
suite parametrizes from); for reference the list is also recorded here:

- containerized, verified by the HTTP-contract fixtures instead: Encodec,
  DescriptAudioCodec, VAE, Diffusion, NeuralVocoder, SpeechEnhancement1/2,
  SpeechTokenization, OpusCodec, NetworkTransmission;
- model-callback (need live models/the registry; collusion_2 additionally
  unseedable — fresh ``default_rng()`` at attack.py:129): SameModel,
  CrossModel, Collusion, Collusion2, ZeroBitCollusion;
- corpus-dependent (need ``AIR_wav_files/`` and ``music/`` at CWD): Replay,
  Mixing;
- absent from the canonical environment (deps outside requirements.txt):
  Wavelet,
  TimeStretch, PitchShift, InvertedTimeStretch;
- not goldenable: Codec2Vocoder — pycodec2 carries C encoder state across in-process instantiations, so its output is
  invocation-history-dependent.

``test_goldens_and_exclusions_cover_the_attack_universe`` enforces that the
goldens plus these exclusions exactly cover the discovery-lock attack
universe, so the golden set cannot silently shrink and new attacks must be
explicitly goldened or excluded.
"""

import importlib.metadata
import json
import platform
import os
import random
import shutil

import numpy as np
import pytest

from deepmarkpy.plugin_manager import PluginManager
from tests.test_discovery_lock import CANONICAL_ATTACKS, OPTIONAL_DEP_ATTACKS

GOLDENS_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "goldens")

_MANIFEST_PATH = os.path.join(GOLDENS_DIR, "manifest.json")
if not os.path.exists(_MANIFEST_PATH):
    raise FileNotFoundError(
        f"{_MANIFEST_PATH} is missing — the golden fixtures are gone or were "
        "never generated. Regenerate them in the canonical environment with "
        "scripts/generate_native_goldens.py (a deliberate, reviewed decision)."
    )
with open(_MANIFEST_PATH) as _f:
    MANIFEST = json.load(_f)

SAMPLING_RATE = MANIFEST["sampling_rate"]
GLOBAL_SEED = MANIFEST["global_seed_before_each_invocation"]

# Ceiling on how far a golden may drift when the environment does not match
# the recording one. Chosen from measurement, not taste: macOS arm64 against
# Linux aarch64 on identical pins produced at most 4.3e-07 relative (ffmpeg's
# MP3 encoder; everything else was 5.8e-14 or better), and the smallest real
# behavior change made in this repo moved output by 3.1e-05.
CROSS_ENV_RTOL = 1e-6


def _numeric_env_mismatches():
    """What differs between here and where the goldens were recorded.

    The platform counts as much as the package versions: identical numpy on
    Linux and on macOS does not have to produce identical floats, and in
    practice does not. Anything listed here means byte-identity is not a
    claim that can be made, so the caller compares within tolerance instead.
    """
    mismatches = []
    recorded_machine = MANIFEST.get("machine")
    current_machine = f"{platform.system()} {platform.machine()}"
    if recorded_machine and recorded_machine != current_machine:
        mismatches.append(
            f"machine: recorded {recorded_machine}, running {current_machine}"
        )
    for pkg, recorded in MANIFEST["numeric_env"].items():
        try:
            installed = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            installed = "not installed"
        if installed != recorded:
            mismatches.append(f"{pkg}: recorded {recorded}, installed {installed}")
    return mismatches


@pytest.fixture(scope="module")
def pm():
    """One shared PluginManager — discovery imports ~50 plugin modules."""
    return PluginManager()


def _canonical_input():
    """Must match the manifest's input_expression exactly."""
    return 0.5 * np.random.default_rng(42).standard_normal(SAMPLING_RATE)


@pytest.mark.parametrize("cls_name", sorted(MANIFEST["attacks"]))
def test_native_attack_matches_golden(cls_name, pm):
    mismatches = _numeric_env_mismatches()
    if cls_name == "Mp3CompressionAttack" and shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH — required by Mp3CompressionAttack")
    attacks = pm.get_attacks()
    if cls_name not in attacks:
        pytest.skip(
            f"{cls_name} not discovered in this environment — goldens are "
            "recorded and verified in the canonical environment; "
            "discovery gaps are judged by test_discovery_lock.py"
        )
    meta = MANIFEST["attacks"][cls_name]
    golden = np.load(os.path.join(GOLDENS_DIR, f"{cls_name}.npz"))["output"]

    audio = _canonical_input()
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    out = attacks[cls_name]["class"]().apply(
        audio, sampling_rate=SAMPLING_RATE, **meta["params"]
    )

    assert str(out.dtype) == meta["output_dtype"], (
        f"{cls_name}: output dtype {out.dtype} != recorded {meta['output_dtype']}"
    )
    assert list(out.shape) == meta["output_shape"], (
        f"{cls_name}: output shape {out.shape} != recorded {meta['output_shape']}"
    )
    if not mismatches:
        assert np.array_equal(out, golden), (
            f"{cls_name}: output differs from its golden fixture — behavior "
            "changed relative to the recorded state"
        )
        return

    # Off the recording environment, byte-identity is not a claim anyone can
    # make: the same numpy on Linux and macOS need not produce identical
    # floats. Measured across those two, every attack agreed to 4.3e-07 or
    # better, while the smallest deliberate behavior change in this repo's
    # history moved output by 3.1e-05. The tolerance sits between them, so the
    # goldens still catch a real change on any platform.
    scale = max(float(np.abs(golden).max()), 1e-12)
    diff = float(np.abs(out - golden).max())
    assert diff / scale <= CROSS_ENV_RTOL, (
        f"{cls_name}: differs from its golden by {diff / scale:.2e} relative, "
        f"above the {CROSS_ENV_RTOL:.0e} cross-environment tolerance — too "
        f"large to be platform arithmetic, so behavior changed. Environment "
        f"differences: {'; '.join(mismatches)}"
    )


def test_goldens_and_exclusions_cover_the_attack_universe():
    """Goldens ∪ exclusions == the discovery-lock attack universe, disjointly.

    Prevents the golden set from silently shrinking (a removal from the
    generation script would otherwise produce a self-consistent smaller
    manifest) and forces an explicit golden-or-exclude decision for every
    newly added attack.
    """
    goldened = set(MANIFEST["attacks"])
    excluded = {name for names in MANIFEST["exclusions"].values() for name in names}
    universe = CANONICAL_ATTACKS | set(OPTIONAL_DEP_ATTACKS)
    overlap = goldened & excluded
    assert not overlap, f"attacks both goldened and excluded: {sorted(overlap)}"
    assert goldened | excluded == universe, (
        f"golden/exclusion accounting no longer covers the attack universe — "
        f"unaccounted: {sorted(universe - goldened - excluded)}; "
        f"unknown: {sorted((goldened | excluded) - universe)}"
    )


def test_every_golden_file_is_in_manifest():
    on_disk = {
        f[:-len(".npz")] for f in os.listdir(GOLDENS_DIR) if f.endswith(".npz")
    }
    assert on_disk == set(MANIFEST["attacks"]), (
        f"goldens on disk and manifest disagree — "
        f"only on disk: {sorted(on_disk - set(MANIFEST['attacks']))}; "
        f"only in manifest: {sorted(set(MANIFEST['attacks']) - on_disk)}"
    )
