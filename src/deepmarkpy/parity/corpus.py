"""The fixed inputs every parity run uses: real speech plus edge cases, hashed.

A parity verdict means "on these inputs". The corpus is therefore built
deterministically and recorded with a hash per case, so a verdict can name
exactly what it was measured on.

Edge cases (each from the first real clip unless noted):
- ``len_<n>``: the clip tiled or cut to n samples. 57,143 is TimeStretch's
  output for 5 s. 63,999 / 64,000 / 64,001 bracket the length at which the
  AudioSeal detector hit a MIG-only cuDNN assert in the GPU study. 16,000
  and 480,000 are short and long.
- ``silence``, ``quiet`` (peak −60 dBFS), ``clipped`` (driven 8× into ±1).
- ``noise`` (white, σ 0.3, seeded) and ``sweep`` (log sine, 50 Hz to 0.45·sr).
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np

from deepmarkpy.utils.utils import load_audio

EDGE_LENGTHS = (16000, 57143, 63999, 64000, 64001, 480000)
SYNTHETIC_SAMPLES = 80000
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3")


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype="<f8").tobytes()).hexdigest()


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    reps = -(-n // x.size)
    return np.tile(x, reps)[:n].copy()


def _sweep(n: int, sr: int) -> np.ndarray:
    t = np.arange(n) / sr
    f0, f1, duration = 50.0, 0.45 * sr, n / sr
    k = np.log(f1 / f0)
    return 0.5 * np.sin(2 * np.pi * f0 * duration / k * (np.exp(t / duration * k) - 1.0))


def build_corpus(sources: Path, root: Path, sampling_rate: int = 16000) -> Dict:
    """Write ``root/<sr>/clean/<case>.npy`` and ``root/<sr>/manifest.json``; return the manifest."""
    files = sorted(f for f in os.listdir(sources) if f.lower().endswith(AUDIO_EXTENSIONS))
    if not files:
        raise ValueError(f"no audio files in {sources}")

    cases: Dict[str, np.ndarray] = {}
    kinds: Dict[str, Dict] = {}
    for name in files:
        audio, _ = load_audio(str(Path(sources) / name), target_sr=sampling_rate)
        stem = Path(name).stem
        cases[stem] = np.asarray(audio, dtype=np.float64).ravel()
        kinds[stem] = {"kind": "speech", "source": name}

    first_stem = Path(files[0]).stem
    speech = cases[first_stem]
    for n in EDGE_LENGTHS:
        cases[f"len_{n}"] = _fit(speech, n)
        kinds[f"len_{n}"] = {"kind": "length", "source": files[0]}
    base = _fit(speech, SYNTHETIC_SAMPLES)
    peak = float(np.max(np.abs(base))) or 1.0
    rng = np.random.default_rng(20261019)
    synthetic = {
        "silence": np.zeros(SYNTHETIC_SAMPLES),
        "quiet": base / peak * 1e-3,
        "clipped": np.clip(base / peak * 8.0, -1.0, 1.0),
        "noise": np.clip(0.3 * rng.standard_normal(SYNTHETIC_SAMPLES), -1.0, 1.0),
        "sweep": _sweep(SYNTHETIC_SAMPLES, sampling_rate),
    }
    for stem, arr in synthetic.items():
        cases[stem] = arr
        kinds[stem] = {"kind": "synthetic", "source": files[0] if stem in ("quiet", "clipped") else None}

    out = Path(root) / str(sampling_rate) / "clean"
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for stem in sorted(cases):
        arr = cases[stem]
        np.save(out / f"{stem}.npy", arr)
        entries[stem] = {**kinds[stem], "samples": int(arr.size), "sha256": _sha256(arr)}

    manifest = {
        "sampling_rate": sampling_rate,
        "sources": str(sources),
        "cases": entries,
        "corpus_sha256": corpus_hash(entries),
    }
    (Path(root) / str(sampling_rate) / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def corpus_hash(entries: Dict[str, Dict]) -> str:
    joined = "\n".join(f"{stem}:{entries[stem]['sha256']}" for stem in sorted(entries))
    return hashlib.sha256(joined.encode()).hexdigest()


def load_manifest(root: Path, sampling_rate: int = 16000) -> Dict:
    return json.loads((Path(root) / str(sampling_rate) / "manifest.json").read_text())


def verify_inputs(directory: Path, entries: Dict[str, Dict]) -> None:
    """Raise unless ``directory`` holds exactly the manifest's cases, bit for bit."""
    present = {p.stem for p in Path(directory).glob("*.npy")}
    if present != set(entries):
        raise ValueError(f"{directory}: cases differ from the manifest: "
                         f"missing {sorted(set(entries) - present)}, extra {sorted(present - set(entries))}")
    for stem, entry in entries.items():
        if _sha256(np.load(Path(directory) / f"{stem}.npy")) != entry["sha256"]:
            raise ValueError(f"{directory}/{stem}.npy does not match its manifest hash")
