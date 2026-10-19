"""The parity gate must pass what the study showed is equivalent and fail everything else.

The numbers come from the GPU study's TF32-off measurements
(scripts/gpu_study/results/raw/parity/). A tier that rejected those, or
accepted TF32's damage, would be the wrong gate.
"""

import math

import numpy as np
import pytest

from deepmarkpy.parity.compare import bit_flips, masked_diff, waveform_diff
from deepmarkpy.parity.tiers import (
    BITS, CONFIDENCE, EXACT, METRICS, NUMERIC, QUANTIZED, REPEAT, SIGNAL,
    CaseEvidence, judge,
)

SR = 16000
ALL = (SIGNAL, REPEAT, BITS, METRICS)


def speech_like(n=80000, seed=0):
    rng = np.random.default_rng(seed)
    return 0.1 * rng.standard_normal(n)


def evidence(case, ref, cand, tier=NUMERIC, **kw):
    diff = waveform_diff(ref, cand, SR, frame_atol=tier.flip_frame_atol)
    kw.setdefault("repeat_identical", True)
    kw.setdefault("bit_flips", 0)
    kw.setdefault("metric_deltas", {"pesq": 0.0, "visqol": 0.0, "stoi": 0.0})
    return CaseEvidence(case=case, signal=diff, **kw)


class TestCompare:
    def test_identical_signals(self):
        x = speech_like()
        d = waveform_diff(x, x.copy(), SR)
        assert d.identical and d.max_abs == 0.0 and d.rel_l2 == 0.0 and math.isinf(d.snr_db)

    def test_relative_l2_is_the_snr_of_the_difference(self):
        x = speech_like()
        d = waveform_diff(x, x + 1e-5 * speech_like(seed=1), SR)
        assert d.rel_l2 == pytest.approx(1e-5, rel=0.05)
        assert d.snr_db == pytest.approx(100.0, abs=0.5)

    def test_a_length_mismatch_is_a_result_not_a_trim(self):
        x = speech_like()
        d = waveform_diff(x, x[:-1], SR)
        assert not d.same_length and math.isinf(d.max_abs)

    def test_non_finite_output_is_caught(self):
        x = speech_like()
        y = x.copy()
        y[10] = np.nan
        assert not waveform_diff(x, y, SR).finite

    def test_a_silent_reference_gives_a_finite_relative_error(self):
        d = waveform_diff(np.zeros(16000), np.full(16000, 1e-9), SR)
        assert math.isfinite(d.rel_l2)

    def test_frames_over_locates_a_local_error(self):
        x = speech_like()
        y = x.copy()
        y[16000 + 5] += 0.01  # sample 16005 sits in frame 50 (320 samples per frame)
        d = waveform_diff(x, y, SR, frame_atol=1e-4)
        assert d.frames_over == (50,)

    def test_masking_removes_only_the_named_frames(self):
        x = speech_like()
        y = x.copy()
        y[16005] += 0.01
        y[100] += 2e-5
        m = masked_diff(x, y, SR, frames=[50])
        assert m.max_abs == pytest.approx(2e-5)

    def test_bit_flips(self):
        assert bit_flips([0, 1, 1, 0], [0, 1, 0, 0]) == 1
        assert bit_flips([0.0, 1.0], [0.4, 0.6]) == 0  # rounds like the benchmark does
        assert bit_flips([0, 1], [0, 1, 1]) is None
        assert bit_flips(None, [1]) is None


class TestNumericTier:
    def test_the_studys_tf32_off_audioseal_passes(self):
        # Worst clip at 104 dB, max |d| 3.1e-5, 0 bit flips.
        x = speech_like()
        cases = [evidence(f"c{i}", x, x + 6e-6 * speech_like(seed=i)) for i in range(5)]
        assert judge(cases, NUMERIC, ALL).passed

    def test_bigvgan_at_the_edge_passes(self):
        # Median 9.4e-6, worst 1.2e-5: inside 1e-5 / 3e-5.
        x = speech_like()
        noise = [9.4e-6, 9.4e-6, 1.2e-5, 9.0e-6, 9.4e-6]
        cases = [evidence(f"c{i}", x, x + s * speech_like(seed=i)) for i, s in enumerate(noise)]
        j = judge(cases, NUMERIC, ALL)
        assert j.passed, j.failures

    def test_tf32_damage_fails(self):
        # TF32 on: codecs at 20-40 dB.
        x = speech_like()
        j = judge([evidence("c", x, x + 1e-2 * speech_like(seed=3))], NUMERIC, ALL)
        assert not j.passed
        assert any("relative L2" in f for f in j.failures)

    def test_a_median_over_1e5_fails_even_when_each_case_is_under_3e5(self):
        x = speech_like()
        cases = [evidence(f"c{i}", x, x + 2e-5 * speech_like(seed=i)) for i in range(3)]
        j = judge(cases, NUMERIC, ALL)
        assert [f for f in j.failures if f.startswith("median")]

    def test_one_flipped_bit_fails(self):
        x = speech_like()
        j = judge([evidence("c", x, x, bit_flips=1)], NUMERIC, ALL)
        assert not j.passed and "bit" in j.failures[0]

    def test_a_candidate_that_does_not_repeat_fails(self):
        x = speech_like()
        assert not judge([evidence("c", x, x, repeat_identical=False)], NUMERIC, ALL).passed

    @pytest.mark.parametrize("missing", [
        {"repeat_identical": None}, {"bit_flips": None}, {"metric_deltas": {}},
    ])
    def test_missing_evidence_is_a_failure_not_a_pass(self, missing):
        x = speech_like()
        j = judge([evidence("c", x, x, **missing)], NUMERIC, ALL)
        assert not j.passed

    def test_an_empty_corpus_proves_nothing(self):
        assert not judge([], NUMERIC, ALL).passed

    def test_a_metric_delta_over_its_bound_fails(self):
        x = speech_like()
        ev = evidence("c", x, x, metric_deltas={"pesq": 2e-3, "visqol": 0.0, "stoi": 0.0})
        assert not judge([ev], NUMERIC, ALL).passed

    def test_confidence_is_bounded_when_required(self):
        ev = CaseEvidence(case="d", repeat_identical=True, bit_flips=0, conf_delta=6.3e-5)
        assert judge([ev], NUMERIC, (REPEAT, BITS, CONFIDENCE)).passed
        ev.conf_delta = 2e-4
        assert not judge([ev], NUMERIC, (REPEAT, BITS, CONFIDENCE)).passed


class TestExactTier:
    def test_identical_passes(self):
        x = speech_like()
        assert judge([evidence("c", x, x.copy(), tier=EXACT)], EXACT, ALL).passed

    def test_one_ulp_fails(self):
        x = speech_like()
        y = x.copy()
        y[0] = np.nextafter(y[0], 1.0)
        assert not judge([evidence("c", x, y, tier=EXACT)], EXACT, ALL).passed


class TestQuantizedTier:
    def _flip(self, x, frame=50, size=0.01):
        y = x + 5e-7 * speech_like(seed=9)
        y[frame * 320 + 3] += size
        return y

    def _ev(self, x, y, explained):
        ev = evidence("c", x, y, tier=QUANTIZED, explained_frames=explained)
        ev.signal_outside_flips = masked_diff(x, y, SR, ev.signal.frames_over)
        return ev

    def test_an_explained_flip_passes(self):
        x = speech_like()
        assert judge([self._ev(x, self._flip(x), explained=(50,))], QUANTIZED, ALL).passed

    def test_an_unexplained_flip_fails(self):
        x = speech_like()
        j = judge([self._ev(x, self._flip(x), explained=())], QUANTIZED, ALL)
        assert any("not near-ties" in f for f in j.failures)

    def test_a_flip_without_any_tie_evidence_fails(self):
        x = speech_like()
        j = judge([self._ev(x, self._flip(x), explained=None)], QUANTIZED, ALL)
        assert any("no near-tie evidence" in f for f in j.failures)

    def test_too_many_flip_frames_fail_even_if_explained(self):
        x = speech_like()
        y = x.copy()
        frames = list(range(0, 250, 50)) + list(range(1, 6))  # 10 of 250 frames = 4%
        for f in frames:
            y[f * 320] += 0.01
        j = judge([self._ev(x, y, explained=tuple(frames))], QUANTIZED, ALL)
        assert any("frames flipped" in f for f in j.failures)

    def test_outside_the_flips_the_numeric_bounds_still_apply(self):
        x = speech_like()
        y = self._flip(x)
        y += 1e-3 * speech_like(seed=4)  # broad error everywhere
        j = judge([self._ev(x, y, explained=tuple(range(250)))], QUANTIZED, ALL)
        assert not j.passed


class TestNearSilentReferences:
    def test_a_silent_reference_is_judged_on_absolute_error_alone(self):
        ref = np.zeros(80000)
        cand = 5e-8 * speech_like(seed=5)  # -146 dBFS: nothing, though "infinitely" relative
        j = judge([evidence("silence", ref, cand)], NUMERIC, ALL)
        assert j.passed, j.failures
        assert j.rel_l2_median is None  # no loud case, so no relative statistic

    def test_but_a_real_error_on_silence_still_fails(self):
        ref = np.zeros(80000)
        cand = 2e-3 * speech_like(seed=5)  # peaks near 1e-3: well over the 1e-4 bound
        assert not judge([evidence("silence", ref, cand)], NUMERIC, ALL).passed

    def test_speech_is_still_judged_relatively(self):
        x = speech_like()
        assert waveform_diff(x, x, SR).relative_applies
        j = judge([evidence("c", x, x + 5e-5 * speech_like(seed=6))], NUMERIC, ALL)
        assert any("relative L2" in f for f in j.failures)
