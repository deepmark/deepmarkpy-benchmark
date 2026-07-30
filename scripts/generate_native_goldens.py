"""Generate golden fixtures for native attacks.

Run from the repo root, inside the canonical environment
(clean venv + pip install -r requirements.txt):

    python scripts/generate_native_goldens.py

For every goldenable native attack this script:

1. builds the canonical input — exactly
   ``0.5 * np.random.default_rng(42).standard_normal(SAMPLING_RATE)``
   (≈1 s of noise scaled to ±0.5) at ``SAMPLING_RATE`` = 16000 Hz (attacks do
   not declare a sampling rate in config; 16 kHz is the most common model
   rate and is fixed here — this script is the authoritative source of the
   exact expression);
2. seeds the legacy global RNG with ``np.random.seed(GLOBAL_SEED)`` and the
   stdlib RNG with ``random.seed(GLOBAL_SEED)`` immediately before the attack
   invocation (pins attacks drawing from ``np.random`` and ``crop_random``'s
   stdlib draws);
3. calls ``AttackCls().apply(input, sampling_rate=SAMPLING_RATE, **params)``
   with config-default parameters (``params`` is currently empty for every
   goldened attack; the hook exists for attacks whose config value is a list
   the benchmark expands per-value, should one ever be goldened);
4. repeats the invocation from scratch and requires the two outputs to be
   byte-identical (dtype + shape + values); nondeterministic attacks are NOT
   goldened — they are reported for escalation;
5. writes ``tests/fixtures/goldens/<ClassName>.npz`` (compressed: input is
   re-derivable, only the output array is stored) and a ``manifest.json``
   with environment lineage and per-attack metadata.

Byte-identity holds same-machine, same-environment only: fixtures are
recorded on the designated machine in the canonical venv. mp3_compression
shells out to ffmpeg, so its golden is additionally tied to the machine's
ffmpeg build (recorded in the manifest).

Excluded (with rationale, also recorded in the manifest):
- containerized attacks — verified by the HTTP-contract fixtures instead;
- model-callback attacks (need live model instances / the registry;
  collusion_2 additionally draws from a fresh ``np.random.default_rng()`` at
  attack.py:129, beyond ``np.random.seed``'s reach);
- corpus-dependent attacks (replay, mixing — need on-disk corpora);
- attacks absent from the canonical environment (wavelet, time_stretch, pitch_shift,
  inverted_time_stretch — import-time deps outside requirements.txt, absent
  from the canonical environment; see docs/KNOWN_DEFECTS.md D14).
"""

import datetime
import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deepmarkpy.plugin_manager import PluginManager  # noqa: E402

SAMPLING_RATE = 16000
INPUT_RNG_SEED = 42
GLOBAL_SEED = 42
GOLDENS_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "goldens")

# Packages on the goldens' numeric path. Their exact versions are recorded in
# the manifest; the replay test enforces byte-identity only when the running
# environment matches (byte-identity is same-machine, same-environment).
NUMERIC_ENV_PACKAGES = [
    "numpy", "scipy", "librosa", "soundfile", "audiocomplib",
]

# Attacks goldened here. Values: extra kwargs beyond sampling_rate (all
# empty today; non-empty only for attacks whose config default is a list
# the benchmark expands per-value).
GOLDEN_ATTACKS = {
    "AdditiveNoiseAttack": {},
    "BandstopFilterAttack": {},
    "ChorusAttack": {},
    "CropBeginningAttack": {},
    "CropRandomAttack": {},
    "CutSamplesAttack": {},
    "EchoAttack": {},
    "EqualizerAttack": {},
    "FlangerAttack": {},
    "FlipSamplesAttack": {},
    "GaussianNoiseAttack": {},
    "HighpassFilterAttack": {},
    "LPCAttack": {},
    "LowpassFilterAttack": {},
    "Mp3CompressionAttack": {},
    "PCMQuantizationAttack": {},
    "PinkNoiseAttack": {},
    "QuantizationAttack": {},
    "ReplacementAttack": {},
    "Replacement2Attack": {},
    "ResamplingPolyAttack": {},
    "SignInversionAttack": {},
    "SmoothingAttack": {},
    "STFTQuantizationAttack": {},
    "ZeroCrossInsertsAttack": {},
}

EXCLUSIONS = {
    "containerized (verified by the HTTP-contract fixtures)": [
        "EncodecAttack", "DescriptAudioCodecAttack",
        "VAEAttack", "DiffusionAttack", "NeuralVocoderAttack",
        "SpeechEnhancement1Attack", "SpeechEnhancement2Attack",
        "SpeechTokenizationAttack", "OpusCodecAttack",
        "NetworkTransmissionAttack",
    ],
    "model-callback (need live models/registry; protected by discovery locks"
    " + unit suite; collusion_2 additionally unseedable — fresh default_rng"
    " at attack.py:129)": [
        "SameModelAttack", "CrossModelAttack", "CollusionAttack",
        "Collusion2Attack", "ZeroBitCollusionAttack",
    ],
    "corpus-dependent (need AIR_wav_files/ and music/ at CWD)": [
        "ReplayAttack", "MixingAttack",
    ],
    "absent from the canonical environment (KNOWN_DEFECTS D14)": [
        "WaveletAttack", "TimeStretchAttack", "PitchShiftAttack",
        "InvertedTimeStretchAttack",
    ],
    "not goldenable (KNOWN_DEFECTS D16): pycodec2 carries C encoder state"
    " across in-process instantiations, so output is"
    " invocation-history-dependent": [
        "Codec2VocoderAttack",
    ],
}


def canonical_input() -> np.ndarray:
    """The canonical golden input."""
    return 0.5 * np.random.default_rng(INPUT_RNG_SEED).standard_normal(SAMPLING_RATE)


def run_attack(attack_cls, params: dict) -> np.ndarray:
    """One seeded attack invocation on a fresh instance."""
    audio = canonical_input()
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    return attack_cls().apply(audio, sampling_rate=SAMPLING_RATE, **params)


def ffmpeg_version() -> str:
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True)
    except FileNotFoundError:
        return "unavailable"
    return out.stdout.splitlines()[0] if out.returncode == 0 else "unavailable"


def main() -> None:
    os.makedirs(GOLDENS_DIR, exist_ok=True)
    pm = PluginManager()
    attacks = pm.get_attacks()

    manifest = {
        "recorded": datetime.date.today().isoformat(),
        "environment": "canonical: clean venv, "
                       "pip install -r requirements.txt, CPython "
                       + platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "ffmpeg": ffmpeg_version(),
        "numeric_env": {
            pkg: importlib.metadata.version(pkg) for pkg in NUMERIC_ENV_PACKAGES
        },
        "sampling_rate": SAMPLING_RATE,
        "input_expression":
            "0.5 * np.random.default_rng(42).standard_normal(16000)",
        "global_seed_before_each_invocation": GLOBAL_SEED,
        "exclusions": EXCLUSIONS,
        "attacks": {},
        "nondeterministic": {},
    }

    for cls_name, params in GOLDEN_ATTACKS.items():
        if cls_name not in attacks:
            raise SystemExit(
                f"{cls_name} not discovered — generation must run in the "
                "canonical environment."
            )
        attack_cls = attacks[cls_name]["class"]
        first = run_attack(attack_cls, params)
        second = run_attack(attack_cls, params)
        identical = (
            first.dtype == second.dtype
            and first.shape == second.shape
            and np.array_equal(first, second)
        )
        if not identical:
            manifest["nondeterministic"][cls_name] = (
                "double-run outputs differ under the seed protocol — "
                "NOT goldened; escalate per §4.3"
            )
            logger.info("NONDETERMINISTIC (no golden written): %s", cls_name)
            continue
        np.savez_compressed(
            os.path.join(GOLDENS_DIR, f"{cls_name}.npz"), output=first
        )
        manifest["attacks"][cls_name] = {
            "params": params,
            "output_dtype": str(first.dtype),
            "output_shape": list(first.shape),
        }
        logger.info("golden written: %s (%s, shape %s)",
                    cls_name, first.dtype, first.shape)

    with open(os.path.join(GOLDENS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("manifest written; %d goldens, %d nondeterministic",
                len(manifest["attacks"]), len(manifest["nondeterministic"]))


if __name__ == "__main__":
    main()
