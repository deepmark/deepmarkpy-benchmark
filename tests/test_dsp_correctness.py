"""DSP-level guarantees for native attacks.

Each test here pins a property that was once wrong, so a regression shows up
as a named failure rather than as a plausible number in a results table.
"""

import numpy as np
import pytest

from deepmarkpy.benchmark import Benchmark
from deepmarkpy.plugin_manager import PluginManager

SR = 16000


@pytest.fixture(scope="module")
def attacks():
    return PluginManager().get_attacks()


def _noise(n=SR, seed=0, scale=0.5):
    return scale * np.random.default_rng(seed).standard_normal(n)


class TestBandstopStability:
    """The filter must stay bounded at every sampling rate it can be handed."""

    @pytest.mark.parametrize("sampling_rate", [8000, 16000, 22050, 32000, 44100, 48000, 96000])
    def test_output_is_bounded(self, attacks, sampling_rate):
        out = attacks["BandstopFilterAttack"]["class"]().apply(
            _noise(), sampling_rate=sampling_rate
        )
        assert np.isfinite(out).all(), "filter produced non-finite samples"
        # Transfer-function form reached ~1e147 here; a stable bandstop cannot
        # amplify broadband noise by more than a small factor.
        assert np.abs(out).max() < 10.0, (
            f"output blew up to {np.abs(out).max():.3e} at {sampling_rate} Hz"
        )


class TestZeroCrossInsertsKeepsHead:
    """Inserting pauses must not consume the start of the file."""

    @pytest.mark.parametrize("seconds", [1, 2, 4, 10])
    def test_head_is_preserved(self, attacks, seconds):
        # A DC-offset head delays the first zero crossing, so a dropped head is
        # unambiguous rather than hidden behind a crossing at sample 0.
        head = np.full(4000, 0.4)
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(SR * seconds - 4000) / SR)
        audio = np.concatenate([head, tone])
        first_crossing = int(np.where(np.diff(np.sign(audio)))[0][0])

        out = attacks["ZeroCrossInsertsAttack"]["class"]().apply(
            audio.copy(), sampling_rate=SR
        )

        assert len(out) >= len(audio), "attack shortened the signal"
        assert np.allclose(out[:first_crossing], audio[:first_crossing]), (
            f"{seconds}s file: samples before the first zero crossing were dropped"
        )


class TestIntegerCastsRound:
    """Truncation costs ~6 dB against rounding and biases toward zero."""

    @pytest.mark.parametrize("pcm", [8, 16])
    def test_no_sign_correlated_bias(self, attacks, pcm):
        audio = np.clip(_noise(64000, seed=7, scale=0.3), -0.99, 0.99)
        out = attacks["PCMQuantizationAttack"]["class"]().apply(
            audio.copy(), sampling_rate=SR, pcm_quantization=pcm
        )
        lsb = 1.0 / (127.0 if pcm == 8 else 32767.0)
        # Truncation toward zero yields exactly -0.5 LSB here; rounding ~0.
        bias = np.mean((out - audio) * np.sign(audio)) / lsb
        assert abs(bias) < 0.05, f"pcm={pcm}: sign-correlated bias {bias:+.3f} LSB"

    def test_quantization_noise_matches_rounding_theory(self, attacks):
        audio = np.clip(_noise(200000, seed=7, scale=0.3), -0.99, 0.99)
        out = attacks["PCMQuantizationAttack"]["class"]().apply(
            audio.copy(), sampling_rate=SR, pcm_quantization=16
        )
        lsb = 1.0 / 32767.0
        mse = np.mean(np.square(out - audio))
        # Rounding gives lsb^2/12; truncation gives lsb^2/3 (4x, +6 dB).
        assert mse < (lsb ** 2 / 12) * 1.5, "noise power consistent with truncation"


class TestInvertedTimeStretchDoesNotFakeSuccess:
    """A failed attack must not be reported as an attack that did nothing."""

    def test_failure_propagates(self, attacks):
        attack = attacks["InvertedTimeStretchAttack"]["class"]()
        # A zero rate makes the inverted second pass undefined, so the
        # underlying stretch raises.
        with pytest.raises(Exception):
            attack.apply(_noise(), sampling_rate=SR, rate_inverted_time_stretch=0.0)

    def test_does_not_return_input_unchanged(self, attacks):
        attack = attacks["InvertedTimeStretchAttack"]["class"]()
        audio = _noise()
        try:
            out = attack.apply(audio.copy(), sampling_rate=SR)
        except Exception:
            return  # raising is the corrected behavior
        assert not np.array_equal(out, audio), (
            "attack returned its input verbatim; it would score the detector's "
            "unattacked ceiling"
        )


class TestAttackSnrIsRecorded:
    """The effective SNR of an attack must be visible, not inferred."""

    def test_absolute_noise_snr_tracks_input_level(self, attacks):
        cls = attacks["AdditiveNoiseAttack"]["class"]
        measured = {}
        for rms in (0.02, 0.1, 0.4):
            audio = _noise(seed=3)
            audio = audio / np.sqrt(np.mean(audio ** 2)) * rms
            out = cls().apply(audio.copy(), sampling_rate=SR)
            measured[rms] = Benchmark._attack_snr_db(audio, out)

        spread = max(measured.values()) - min(measured.values())
        # Absolute-amplitude noise moves dB for dB with input level; this is
        # the property the recorded field exists to expose.
        assert spread > 20, f"expected a wide SNR spread, saw {spread:.1f} dB"
        assert measured[0.4] > measured[0.02], "louder input must yield higher SNR"

    def test_snr_parameterized_sibling_is_level_invariant(self, attacks):
        cls = attacks["GaussianNoiseAttack"]["class"]
        measured = []
        for rms in (0.02, 0.1, 0.4):
            audio = _noise(seed=3)
            audio = audio / np.sqrt(np.mean(audio ** 2)) * rms
            out = cls().apply(audio.copy(), sampling_rate=SR)
            measured.append(Benchmark._attack_snr_db(audio, out))
        assert max(measured) - min(measured) < 1.0, (
            f"SNR-parameterized attack drifted with level: {measured}"
        )

    def test_returns_none_when_lengths_differ(self):
        assert Benchmark._attack_snr_db(np.zeros(100), np.zeros(90)) is None

    def test_returns_none_for_identical_signals(self):
        audio = _noise(1000)
        assert Benchmark._attack_snr_db(audio, audio.copy()) is None
