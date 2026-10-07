"""Guards for metric selection, metric caveats, and metric behaviour.

Full-reference metrics that compare signals sample-by-sample report a
timing offset as if it were quality loss. The benchmark deliberately does
not resynchronize — a desynchronization attack is supposed to move the
time axis — so those values are still reported, but marked with a dagger
and a footnote so nobody reads them as quality scores.
"""

import numpy as np
import pytest

from deepmarkpy.plugin_manager import PluginManager
from deepmarkpy.utils.attack_groups import (
    ATTACK_GROUPS,
    get_group_for_attack,
    get_metric_caveat,
)
from deepmarkpy.utils.latex_helpers import MetricCaveats, format_metric_cell
from deepmarkpy.utils.metrics import mcd, psnr, si_sdr, trim_audio_to_match

# Attacks deliberately left out of every group, with the reason. Empty by
# policy: an ungrouped attack silently receives the full metric slate,
# including metrics its own family rejects as misleading.
UNGROUPED_BY_DESIGN = {}


class TestEveryAttackIsGrouped:
    def test_no_discovered_attack_is_ungrouped(self):
        """The reverse of the existing group->discovered check.

        Only this direction catches a new plugin that never got a group,
        which would be scored on metrics its own family excludes.
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
        ("VAEAttack", "ai_attacks"),
    ])
    def test_previously_orphaned_attacks_sit_with_their_family(self, attack, group):
        assert get_group_for_attack(attack) == group

    def test_grouped_attack_gets_fewer_metrics_than_the_fallback(self):
        """The fallback hands out every metric; a real group narrows it."""
        from deepmarkpy.utils.metric_resolver import (
            MetricResolver, SIGNAL_METRICS,
        )

        resolver = MetricResolver.from_attack_groups()
        assert len(resolver.metrics_for_attack("ZeroCrossInsertsAttack")) < len(SIGNAL_METRICS)
        assert set(resolver.metrics_for_attack("UnknownAttack")) == set(SIGNAL_METRICS)


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

    def test_a_version_name_carries_its_attack_caveat(self):
        """``SignInversionAttack (x)`` is SignInversion with other parameters."""
        polarity = get_metric_caveat("SignInversionAttack", "si_sdr")
        assert polarity
        assert get_metric_caveat("SignInversionAttack (x)", "si_sdr") == polarity
        assert get_metric_caveat("SignInversionAttack (x)", "psnr") is None
        assert get_metric_caveat("TimeStretchAttack (subtle)", "mcd")


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
    """Each caveat prints its own explanation, not a shared one."""

    def test_distinct_reasons_get_distinct_markers(self):
        caveats = MetricCaveats()
        desync = caveats.mark("TimeStretchAttack", "psnr")
        polarity = caveats.mark("SignInversionAttack", "si_sdr")

        assert desync and polarity
        assert desync != polarity, "two unrelated caveats share one marker"

    def test_same_reason_reuses_its_marker(self):
        caveats = MetricCaveats()
        assert caveats.mark("TimeStretchAttack", "psnr") == caveats.mark(
            "TimeStretchAttack", "stoi"
        )

    def test_uncaveated_cell_is_unmarked(self):
        assert MetricCaveats().mark("GaussianNoiseAttack", "pesq") == ""

    def test_footnote_states_each_reason(self):
        caveats = MetricCaveats()
        caveats.mark("TimeStretchAttack", "psnr")
        caveats.mark("SignInversionAttack", "si_sdr")
        note = caveats.footnote()

        assert "timing shift" in note, "desynchronization reason missing"
        assert "polarity inversion" in note, "polarity reason missing"

    def test_footnote_is_empty_when_nothing_flagged(self):
        caveats = MetricCaveats()
        caveats.mark("GaussianNoiseAttack", "pesq")
        assert caveats.footnote() == ""

    def test_every_reason_completes_the_footnote_sentence(self):
        """Reasons are rendered as "This metric <reason>." and must read."""
        from deepmarkpy.utils.metric_resolver import SIGNAL_METRICS

        seen = set()
        for group in ATTACK_GROUPS.values():
            for attack in group["attacks"]:
                for metric in SIGNAL_METRICS:
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


DAGGER = "\\textsuperscript{\\dag}"

# Three files' values per metric. MCD is worse high, PESQ and STOI low, so
# every metric's worst case differs from its mean.
PER_FILE = {
    "pesq": (2.0, 2.1, 2.2),
    "mcd": (9.0, 10.0, 11.0),
    "stoi": (0.5, 0.6, 0.7),
}

# Their mean and worst case, as the basic and detection-reliability reports
# receive them.
SUMMARY = {
    "pesq": {"mean": 2.1, "worst_case": 2.0},
    "mcd": {"mean": 10.0, "worst_case": 11.0},
    "stoi": {"mean": 0.6, "worst_case": 0.5},
}


def _caveat_resolver(statistics):
    """PESQ, MCD and STOI enabled for every group, each with ``statistics``."""
    from deepmarkpy.utils.metric_resolver import MetricResolver

    return MetricResolver(
        defaults={metric: {"enabled": True} for metric in SUMMARY},
        statistics=statistics,
    )


def _basic_tex(tmp_path, attack, statistics):
    from deepmarkpy.utils.report_generator import BenchmarkReportGenerator

    entry = {"accuracy_n": 3, "accuracy_mean": 70.0}
    for metric, values in SUMMARY.items():
        for statistic, value in values.items():
            entry[f"{metric}_{statistic}"] = value
    generator = BenchmarkReportGenerator(
        str(tmp_path), resolver=_caveat_resolver(statistics),
    )
    return generator.generate_latex_table(
        {attack: entry}, group_key=get_group_for_attack(attack),
    )


def _detailed_tex(tmp_path, attack, statistics):
    from deepmarkpy.utils.detailed_report_generator import (
        DetailedReportGenerator,
    )

    results = {
        f"f{index}.wav": {
            "watermarked_audio_quality": {
                "pesq": 4.41, "mcd": 0.37, "stoi": 0.9912,
            },
            "attacks": {attack: {
                "accuracy": 70.0,
                "attacked_audio_quality_wm": {
                    metric: values[index] for metric, values in PER_FILE.items()
                },
            }},
        }
        for index in range(3)
    }
    generator = DetailedReportGenerator(
        str(tmp_path), resolver=_caveat_resolver(statistics),
    )
    return generator.generate_latex_report(
        generator.aggregate_results(results), "TestModel",
    )


def _detection_reliability_tex(tmp_path, attack, statistics):
    from deepmarkpy.utils.detection_reliability_report_generator import (
        _metric_tables,
    )

    attacks = {attack: {"metrics": {
        metric: dict(values) for metric, values in SUMMARY.items()
    }}}
    return _metric_tables(
        attacks, [attack], get_group_for_attack(attack),
        _caveat_resolver(statistics), "Test.", "tab:test",
    )


class TestEveryReportGeneratorAnnotatesCaveats:
    """Every per-attack quality table marks an unreadable value and says why."""

    BUILDERS = [_basic_tex, _detailed_tex, _detection_reliability_tex]
    REPORTS = ["basic", "detailed", "detection_reliability"]
    SHAPES = [["mean", "worst_case"], ["mean"]]
    SHAPE_IDS = ["table_per_metric", "compact"]

    @pytest.mark.parametrize("statistics", SHAPES, ids=SHAPE_IDS)
    @pytest.mark.parametrize("build", BUILDERS, ids=REPORTS)
    def test_a_desynchronization_row_is_marked_and_explained(
            self, tmp_path, build, statistics):
        tex = build(tmp_path, "TimeStretchAttack", statistics)

        for metric in ("mcd", "stoi"):
            mean = format_metric_cell(metric, SUMMARY[metric]["mean"])
            worst = format_metric_cell(metric, SUMMARY[metric]["worst_case"])
            assert mean + DAGGER in tex, f"{metric} printed with no marker"
            # The first statistic of a row carries the marker, no other.
            assert worst + DAGGER not in tex
        pesq = format_metric_cell("pesq", SUMMARY["pesq"]["mean"])
        assert pesq in tex and pesq + DAGGER not in tex

        reason = get_metric_caveat("TimeStretchAttack", "mcd")
        assert f"This metric {reason}." in tex, "the marker has no explanation"

    @pytest.mark.parametrize("statistics", SHAPES, ids=SHAPE_IDS)
    def test_the_no_attack_row_is_never_marked(self, tmp_path, statistics):
        tex = _detailed_tex(tmp_path, "TimeStretchAttack", statistics)
        baseline = [line for line in tex.splitlines()
                    if line.strip().startswith("No Attack")]
        assert baseline, "no baseline row was printed"
        assert not any("textsuperscript" in line for line in baseline)

    @pytest.mark.parametrize("build", BUILDERS, ids=REPORTS)
    def test_an_unaffected_table_carries_no_marker_or_footnote(
            self, tmp_path, build):
        tex = build(tmp_path, "GaussianNoiseAttack", ["mean", "worst_case"])
        assert "textsuperscript" not in tex
        assert "for completeness" not in tex

    @pytest.mark.parametrize("attack,metric", [
        ("TimeStretchAttack", "mcd"),
        ("SignInversionAttack", "si_sdr"),
    ])
    def test_the_shipped_template_marks_what_it_enables(
            self, tmp_path, attack, metric):
        """An unedited ``--init`` file reports both, so both are marked."""
        import json

        from deepmarkpy.config import init_template, load_config_data
        from deepmarkpy.utils.report_generator import BenchmarkReportGenerator

        resolver = load_config_data(
            json.loads(init_template("benchmark")), quiet=True,
        ).resolver
        group_key = get_group_for_attack(attack)
        assert metric in resolver.signal_metrics_for_group(group_key)

        stats = {attack: {"accuracy_n": 3, "accuracy_mean": 70.0,
                          f"{metric}_mean": 10.0}}
        tex = BenchmarkReportGenerator(str(tmp_path), resolver=resolver) \
            .generate_latex_table(stats, group_key=group_key)

        assert "10.00" + DAGGER in tex
        assert f"This metric {get_metric_caveat(attack, metric)}." in tex


class TestBerAgreesBetweenReports:
    """Both reports reduce the same BER samples, not accuracy's summary."""

    SAMPLES = [98.2, 95.0, 88.0, 100.0, 51.8, 72.5, 99.0, 64.0, 100.0, 83.3]
    STATISTICS = ["mean", "std", "median", "p5", "p10", "p95", "p99",
                  "worst_case"]

    def _detailed(self):
        from deepmarkpy.utils.detailed_report_generator import (
            _ber_statistic, _statistics,
        )
        aggregated = _statistics(self.SAMPLES)
        return {s: _ber_statistic(aggregated, s) for s in self.STATISTICS}

    def _basic(self):
        from deepmarkpy.benchmark import Benchmark
        target = {}
        Benchmark._apply_statistics(
            target, "ber",
            1.0 - np.array(self.SAMPLES, dtype=float) / 100.0,
            self.STATISTICS,
        )
        return {s: target[f"ber_{s}"] for s in self.STATISTICS}

    @pytest.mark.parametrize("statistic", STATISTICS)
    def test_every_statistic_matches_the_basic_report(self, statistic):
        assert self._detailed()[statistic] == pytest.approx(
            self._basic()[statistic], abs=1e-12,
        )

    def test_the_tails_are_the_right_way_round(self):
        """A low BER percentile is the good tail, not the bad one."""
        ber = self._detailed()
        assert ber["p5"] <= ber["median"] <= ber["p95"] <= ber["worst_case"]
