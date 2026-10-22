"""ReplayAttack: the FFT convolution must give what direct convolution gave.

The impulse responses ship separately (AIR_wav_files/), so this builds its
own: a decaying noise tail like a room's, 48,000 taps at 48 kHz, stored as
float32 the way librosa loads every response.
"""

import random

import librosa
import numpy as np
import pytest
import soundfile as sf

from deepmarkpy.plugins.attacks.replay.attack import ReplayAttack

SR = 16000


@pytest.fixture
def air_folder(tmp_path):
    rng = np.random.default_rng(7)
    for i in range(2):
        t = np.arange(48000) / 48000
        ir = (rng.standard_normal(48000) * np.exp(-t * (6 + i))).astype(np.float32)
        ir[0] = 1.0
        sf.write(tmp_path / f"air_test_{i}.wav", ir, 48000, subtype="FLOAT")
    return tmp_path


def _direct(audio, air_path):
    """The pre-FFT algorithm, for reference."""
    ir, ir_sr = librosa.load(air_path, sr=None, mono=True)
    ir = librosa.resample(ir, orig_sr=ir_sr, target_sr=SR)
    out = np.convolve(audio, ir, mode="full")[: len(audio)]
    return out / np.max(np.abs(out)) * 0.95


def test_matches_direct_convolution(air_folder):
    audio = 0.1 * np.random.default_rng(0).standard_normal(5 * SR)
    attack = ReplayAttack()
    for path in sorted(air_folder.glob("*.wav")):
        got = attack._convolve_with_air(audio, SR, str(path))
        want = _direct(audio, str(path))
        assert got.dtype == np.float64 and got.shape == audio.shape
        assert np.max(np.abs(got - want)) < 1e-12


def test_apply_end_to_end_is_seeded_and_keeps_length(air_folder):
    audio = 0.1 * np.random.default_rng(1).standard_normal(5 * SR)
    kwargs = dict(sampling_rate=SR, air_folder_replay=str(air_folder))
    outs = []
    for _ in range(2):
        random.seed(3)
        np.random.seed(3)
        outs.append(ReplayAttack().apply(audio, **kwargs))
    assert outs[0].shape == audio.shape
    assert np.array_equal(outs[0], outs[1])

