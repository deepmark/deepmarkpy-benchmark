"""Guards for metric selection, metric caveats, and metric behaviour.

Full-reference metrics that compare signals sample-by-sample report a
timing offset as if it were quality loss. The benchmark deliberately does
not resynchronize — a desynchronization attack is supposed to move the
time axis — so those values are still reported, but flagged so nobody
reads them as quality scores.
"""

import numpy as np
import pytest

from deepmarkpy.plugin_manager import PluginManager
from deepmarkpy.utils.attack_groups import (
    ATTACK_GROUPS,
    get_group_for_attack,
    get_metric_caveat,
    get_metrics_for_attack,
)
from deepmarkpy.utils.metrics import ALL_METRICS, mcd, psnr, si_sdr, trim_audio_to_match

# Attacks deliberately left out of every group, with the reason. Empty by
# policy: an ungrouped attack silently receives the full metric slate,
# including metrics its own family rejects as misleading.
UNGROUPED_BY_DESIGN = {}


class TestEveryAttackIsGrouped:
    def test_no_discovered_attack_is_ungrouped(self):
        """The reverse of the existing group->discovered check.

        Only this direction catches a new plugin that never got a group,
        which is how three attacks came to be scored on metrics their own
        families exclude.
        """
        orphans = sorted(
            name for name in PluginManager().get_attacks()
            if get_group_for_attack(name) is None and name not in UNGROUPED_BY_DESIGN
        )
        assert not orphans, (
            f"attacks belong to no group and would receive the full metric "
            f"slate: {orphans}. Add each to its group in attack_groups.py, or "
            f"to UNGROUPED_BY_DESIGN with a reason."
        )

    def test_no_attack_is_in_two_groups(self):
        seen = {}
        for key, group in ATTACK_GROUPS.items():
            for attack in group["attacks"]:
                assert attack not in seen, (
                    f"{attack} is in both {seen[attack]} and {key}"
                )
                seen[attack] = key

    @pytest.mark.parametrize("attack,group", [
        ("AdditiveNoiseAttack", "audio_distortion"),
        ("Replacement2Attack", "desynchronization"),
        ("VAEAttack", "ai_attacks"),
    ])
    def test_previously_orphaned_attacks_sit_with_their_family(self, attack, group):
        assert get_group_for_attack(attack) == group

    def test_grouped_attack_gets_fewer_metrics_than_the_fallback(self):
        """The fallback hands out every metric; a real group narrows it."""
        assert len(get_metrics_for_attack("Replacement2Attack")) < len(ALL_METRICS)
        assert set(get_metrics_for_attack("UnknownAttack")) == set(ALL_METRICS)


class TestMetricCaveats:
    def test_sample_aligned_metrics_are_flagged_for_desync_attacks(self):
        for metric in ("psnr", "si_sdr", "stoi", "mcd", "ncm"):
            assert get_metric_caveat("ZeroCrossInsertsAttack", metric), (
                f"{metric} is sample-aligned and must be flagged for a "
                f"time-shifting attack"
            )

    def test_internally_aligned_metrics_are_not_flagged(self):
        """PESQ and ViSQOL align internally, so a shift does not fool them."""
        for metric in ("pesq", "visqol"):
            assert get_metric_caveat("ZeroCrossInsertsAttack", metric) is None

    def test_reference_free_metrics_are_never_flagged(self):
        assert get_metric_caveat("ZeroCrossInsertsAttack", "nisqa_mos") is None

    def test_non_desync_attacks_carry_no_blanket_caveat(self):
        for metric in ("psnr", "stoi", "mcd"):
            assert get_metric_caveat("GaussianNoiseAttack", metric) is None

    def test_sign_inversion_flags_si_sdr_only(self):
        """SI-SDR is scale-invariant, so it cannot see a polarity flip."""
        assert get_metric_caveat("SignInversionAttack", "si_sdr")
        assert get_metric_caveat("SignInversionAttack", "psnr") is None


class TestMetricValues:
    """Known-good behaviour for the metrics implemented in this repo.

    Ranges and orderings rather than exact floats, so a dependency bump
    does not require re-recording while a scale or argument-order error
    still fails.
    """

    @staticmethod
    def _signal(n=4000):
        t = np.linspace(0, 1, n, endpoint=False)
        return 0.5 * np.sin(2 * np.pi * 220 * t)

    def test_psnr_of_identical_signals_is_infinite(self):
        x = self._signal()
        assert np.isinf(psnr(x, x.copy()))

    def test_psnr_decreases_as_noise_grows(self):
        x = self._signal()
        rng = np.random.default_rng(0)
        light = psnr(x, x + 0.001 * rng.standard_normal(x.size))
        heavy = psnr(x, x + 0.100 * rng.standard_normal(x.size))
        assert light > heavy

    def test_si_sdr_is_scale_invariant(self):
        x = self._signal()
        rng = np.random.default_rng(1)
        degraded = x + 0.01 * rng.standard_normal(x.size)
        assert si_sdr(x, degraded) == pytest.approx(si_sdr(x, degraded * 7.5), abs=1e-6)

    def test_si_sdr_cannot_see_polarity(self):
        """Documents the behaviour behind SignInversionAttack's caveat."""
        x = self._signal()
        assert si_sdr(x, -x) > 100

    def test_mcd_of_identical_signals_is_zero(self):
        x = self._signal(8000)
        assert mcd(x, x.copy()) == pytest.approx(0.0, abs=1e-6)

    def test_mcd_grows_with_timing_offset(self):
        """The reason MCD is flagged for desynchronization attacks.

        Uses noise rather than a tone: rolling a periodic signal by a whole
        number of periods is a no-op, which would hide the effect entirely.
        """
        x = np.random.default_rng(2).standard_normal(8000) * 0.5
        aligned = mcd(x, x.copy())
        shifted = mcd(x, np.roll(x, 400))
        assert aligned == pytest.approx(0.0, abs=1e-6)
        assert shifted > 1.0, "a timing shift must register as MCD distortion"

    def test_trim_matches_lengths_without_shifting(self):
        a, b = trim_audio_to_match(np.arange(10.0), np.arange(6.0))
        assert len(a) == len(b) == 6
        assert np.array_equal(a, np.arange(6.0))


class TestCaveatFootnotesMatchTheirReason:
    """Each caveat must print its own explanation, not a shared one.

    ``get_metric_caveat`` returns a different reason per case, and the report
    generators used to discard it and hardcode the time-shift wording, so
    SignInversion's SI-SDR cell was explained as a timing shift.
    """

    def test_distinct_reasons_get_distinct_markers(self):
        from deepmarkpy.utils.latex_helpers import MetricCaveats

        caveats = MetricCaveats()
        desync = caveats.mark("TimeStretchAttack", "psnr")
        polarity = caveats.mark("SignInversionAttack", "si_sdr")

        assert desync and polarity
        assert desync != polarity, "two unrelated caveats share one marker"

    def test_same_reason_reuses_its_marker(self):
        from deepmarkpy.utils.latex_helpers import MetricCaveats

        caveats = MetricCaveats()
        assert caveats.mark("TimeStretchAttack", "psnr") == caveats.mark(
            "TimeStretchAttack", "stoi"
        )

    def test_uncaveated_cell_is_unmarked(self):
        from deepmarkpy.utils.latex_helpers import MetricCaveats

        assert MetricCaveats().mark("GaussianNoiseAttack", "pesq") == ""

    def test_footnote_states_each_reason(self):
        from deepmarkpy.utils.latex_helpers import MetricCaveats

        caveats = MetricCaveats()
        caveats.mark("TimeStretchAttack", "psnr")
        caveats.mark("SignInversionAttack", "si_sdr")
        note = caveats.footnote()

        assert "timing shift" in note, "desynchronization reason missing"
        assert "polarity inversion" in note, "polarity reason missing"

    def test_footnote_is_empty_when_nothing_flagged(self):
        from deepmarkpy.utils.latex_helpers import MetricCaveats

        caveats = MetricCaveats()
        caveats.mark("GaussianNoiseAttack", "pesq")
        assert caveats.footnote() == ""

    def test_every_reason_completes_the_footnote_sentence(self):
        """Reasons are rendered as "This metric <reason>." and must read."""
        from deepmarkpy.utils.attack_groups import (
            ATTACK_GROUPS, get_metric_caveat,
        )
        from deepmarkpy.utils.metrics import ALL_METRICS

        seen = set()
        for group in ATTACK_GROUPS.values():
            for attack in group["attacks"]:
                for metric in ALL_METRICS:
                    reason = get_metric_caveat(attack, metric)
                    if reason:
                        seen.add(reason)
        assert seen, "no caveats defined at all"
        for reason in seen:
            first = reason.split()[0]
            assert not first[0].isupper(), (
                f"caveat should continue 'This metric ...', got {reason!r}"
            )
            assert not reason.endswith("."), f"caveat ends with a period: {reason!r}"


class TestEveryReportGeneratorAnnotatesCaveats:
    """A caveat added to attack_groups must reach every generator that prints
    per-attack quality metrics, not just the two that were updated first."""

    GENERATORS = [
        "report_generator",
        "detailed_report_generator",
        "detection_reliability_report_generator",
    ]

    def test_detection_reliability_table_marks_and_explains(self):
        """Built, not grepped: this generator was the one left unannotated."""
        from deepmarkpy.utils.detection_reliability_report_generator import (
            _always_on_table,
        )

        attacks = {
            "TimeStretchAttack": {"metrics": {"pesq": 3.1, "visqol": 4.2, "stoi": 0.9}},
            "GaussianNoiseAttack": {"metrics": {"pesq": 3.5, "visqol": 4.4, "stoi": 0.95}},
        }
        table = _always_on_table(
            attacks, list(attacks), "Quality metrics.", "tab:test"
        )

        assert "\\textsuperscript{\\dag}" in table, (
            "a sample-aligned metric on a desynchronization attack was printed "
            "with no caveat marker"
        )
        assert "timing shift" in table, "the marker has no explanation"

    def test_unaffected_table_carries_no_footnote(self):
        from deepmarkpy.utils.detection_reliability_report_generator import (
            _always_on_table,
        )

        attacks = {"GaussianNoiseAttack": {"metrics": {"pesq": 3.5, "visqol": 4.4, "stoi": 0.95}}}
        table = _always_on_table(attacks, list(attacks), "Quality metrics.", "tab:test")
        assert "textsuperscript" not in table

    @pytest.mark.parametrize("module", GENERATORS)
    def test_generator_does_not_hardcode_a_caveat_sentence(self, module):
        import importlib
        import inspect

        mod = importlib.import_module(f"deepmarkpy.utils.{module}")
        src = inspect.getsource(mod)
        assert "reports the shift" not in src and "reflects the shift" not in src, (
            f"{module} hardcodes the time-shift wording instead of using the "
            "reason get_metric_caveat returns"
        )
