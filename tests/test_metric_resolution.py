"""Per-group metric and statistic resolution.

These pin the inheritance rules the config file promises:

* a metric can be enabled or disabled per attack group, not only globally;
* inheritance is per metric, so a group section states only what it
  changes;
* subgroups inherit from their parent group, which inherits from defaults;
* deleting ``per_group`` leaves ``defaults`` applying uniformly;
* omitting ``statistics`` entirely yields all eight;
* the shipped defaults reproduce the ``ATTACK_GROUPS`` matrix exactly.
"""

import json

import pytest

from deepmarkpy.config import load_configs
from deepmarkpy.utils.attack_groups import (
    ATTACK_GROUPS,
    ATTACK_SUBGROUPS,
    CONFIG_GROUP_KEYS,
)
from deepmarkpy.utils.metric_resolver import (
    ALWAYS_ON_METRICS,
    ALL_STATISTICS,
    CANONICAL_METRIC_ORDER,
    EFFICIENCY_METRICS,
    MetricResolver,
)


# The metrics block takes signal and accuracy metrics only: an
# efficiency metric named here is E043, because its enable flag
# would be read from the efficiency section regardless.
CONFIGURABLE_METRICS = [m for m in CANONICAL_METRIC_ORDER
                        if m not in EFFICIENCY_METRICS]


def build(tmp_path, **overrides):
    """Write and load a benchmark config from the given keys."""
    data = {
        "mode": "benchmark",
        "models": ["AudioSealModel"],
        "calculate_quality_metrics": True,
    }
    data.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return load_configs([str(path)])[0].resolver


class TestRequirement1PerGroupSelection:
    """Each group may choose its own metrics and its own statistics."""

    def test_a_metric_can_be_on_for_one_group_and_off_for_another(self, tmp_path):
        resolver = build(tmp_path, metrics={
            "defaults": {"mcd": {"enabled": False}},
            "per_group": {"desynchronization": {"mcd": {"enabled": True}}},
        })
        assert resolver.is_enabled("desynchronization", "mcd") is True
        assert resolver.is_enabled("audio_distortion", "mcd") is False

    def test_each_group_can_ask_for_different_statistics(self, tmp_path):
        resolver = build(tmp_path, statistics=["mean"], metrics={
            "defaults": {"pesq": {"enabled": True}},
            "per_group": {
                "transmission": {"pesq": {"statistics": ["median", "p99"]}},
            },
        })
        assert resolver.statistics_for("transmission", "pesq") == ["median", "p99"]
        assert resolver.statistics_for("audio_distortion", "pesq") == ["mean"]

    def test_enabling_a_metric_a_group_never_showed_makes_it_available(self, tmp_path):
        """The taxonomy excludes ``mcd`` from ``audio_distortion``; the
        config can put it back."""
        assert "mcd" not in ATTACK_GROUPS["audio_distortion"]["quality_metrics"]
        resolver = build(tmp_path, metrics={
            "defaults": {"mcd": {"enabled": True}},
        })
        assert resolver.is_enabled("audio_distortion", "mcd") is True
        assert "mcd" in resolver.metrics_for_attack("GaussianNoiseAttack")


class TestRequirement1InheritanceIsPerMetric:
    def test_a_group_section_only_states_what_it_changes(self, tmp_path):
        resolver = build(tmp_path, metrics={
            "defaults": {
                "pesq": {"enabled": True},
                "stoi": {"enabled": True},
                "mcd": {"enabled": True},
            },
            "per_group": {"desynchronization": {"stoi": {"enabled": False}}},
        })
        assert resolver.is_enabled("desynchronization", "stoi") is False
        # Untouched by the group section, so inherited unchanged.
        assert resolver.is_enabled("desynchronization", "pesq") is True
        assert resolver.is_enabled("desynchronization", "mcd") is True

    def test_a_group_may_override_statistics_without_touching_enabled(self, tmp_path):
        resolver = build(tmp_path, statistics=["mean"], metrics={
            "defaults": {"pesq": {"enabled": True}},
            "per_group": {"ai_attacks": {"pesq": {"statistics": ["worst_case"]}}},
        })
        assert resolver.is_enabled("ai_attacks", "pesq") is True
        assert resolver.statistics_for("ai_attacks", "pesq") == ["worst_case"]

    def test_a_group_may_override_enabled_without_touching_statistics(self, tmp_path):
        resolver = build(tmp_path, statistics=["median"], metrics={
            "defaults": {"pesq": {"enabled": True}},
            "per_group": {"ai_attacks": {"pesq": {"enabled": False}}},
        })
        assert resolver.is_enabled("ai_attacks", "pesq") is False
        assert resolver.statistics_for("ai_attacks", "pesq") == ["median"]


class TestSubgroupInheritance:
    """A subgroup falls back to its parent group before the defaults."""

    def test_subgroup_inherits_from_its_parent_group(self, tmp_path):
        resolver = build(tmp_path, metrics={
            "defaults": {"psnr": {"enabled": True}},
            "per_group": {"audio_editing": {"psnr": {"enabled": False}}},
        })
        assert resolver.is_enabled("temporal_editing", "psnr") is False
        assert resolver.is_enabled("audio_effects", "psnr") is False

    def test_subgroup_overrides_its_parent(self, tmp_path):
        resolver = build(tmp_path, metrics={
            "defaults": {"psnr": {"enabled": True}},
            "per_group": {
                "audio_editing": {"psnr": {"enabled": False}},
                "audio_effects": {"psnr": {"enabled": True}},
            },
        })
        assert resolver.is_enabled("audio_effects", "psnr") is True
        assert resolver.is_enabled("temporal_editing", "psnr") is False

    def test_subgroup_statistics_fall_through_parent_then_defaults(self, tmp_path):
        resolver = build(tmp_path, statistics=["mean"], metrics={
            "defaults": {"mcd": {"enabled": True}},
            "per_group": {
                "audio_editing": {"mcd": {"statistics": ["median"]}},
                "audio_effects": {"mcd": {"statistics": ["p95", "worst_case"]}},
            },
        })
        assert resolver.statistics_for("audio_effects", "mcd") == ["p95", "worst_case"]
        assert resolver.statistics_for("temporal_editing", "mcd") == ["median"]
        assert resolver.statistics_for("audio_distortion", "mcd") == ["mean"]

    def test_an_attack_is_computed_for_both_its_group_and_its_subgroup(self, tmp_path):
        """The two sections can differ, and neither may end up with a hole."""
        resolver = build(tmp_path, metrics={
            "defaults": {m: {"enabled": False} for m in CONFIGURABLE_METRICS
                         if m not in ("accuracy", "ber", "emr")},
            "per_group": {
                "audio_editing": {"pesq": {"enabled": True}},
                "temporal_editing": {"stoi": {"enabled": True}},
            },
        })
        needed = resolver.metrics_for_attack("CropBeginningAttack")
        assert set(needed) == {"pesq", "stoi"}


class TestRequirement2PerGroupIsOptional:
    def test_deleting_per_group_applies_defaults_everywhere(self, tmp_path):
        resolver = build(tmp_path, metrics={
            "defaults": {"pesq": {"enabled": True}, "mcd": {"enabled": False}},
        })
        for group in CONFIG_GROUP_KEYS:
            assert resolver.is_enabled(group, "pesq") is True
            assert resolver.is_enabled(group, "mcd") is False

    def test_an_empty_per_group_block_changes_nothing(self, tmp_path):
        resolver = build(tmp_path, metrics={
            "defaults": {"pesq": {"enabled": True}}, "per_group": {},
        })
        assert resolver.is_enabled("desynchronization", "pesq") is True


class TestRequirement3UnusedGroupsAreNotErrors:
    def test_a_group_not_in_this_run_is_ignored_with_a_note(self, tmp_path):
        """Covered end to end in test_config_validation; asserted here too
        because it is a stated requirement of the resolution rules."""
        path = tmp_path / "config.json"
        path.write_text(json.dumps({
            "mode": "benchmark",
            "models": ["AudioSealModel"],
            "calculate_quality_metrics": True,
            "attacks": {"groups": ["audio_distortion"]},
            "metrics": {"per_group": {"ai_attacks": {"mcd": {"enabled": False}}}},
        }))
        config = load_configs([str(path)])[0]
        assert config.resolver.is_enabled("ai_attacks", "mcd") is False
        assert [w.code for w in config.warnings if w.code == "W001"]


class TestRequirement4DefaultsReproduceTheTaxonomy:
    def test_builtin_matrix_matches_attack_groups_exactly(self):
        resolver = MetricResolver.from_attack_groups()
        for group_key, definition in ATTACK_GROUPS.items():
            declared = set(
                definition["quality_metrics"]
                + definition["intelligibility_metrics"]
                + definition["nisqa_metrics"]
            )
            enabled = set(resolver.signal_metrics_for_group(group_key))
            assert enabled == declared | set(ALWAYS_ON_METRICS), \
                f"{group_key} drifted from ATTACK_GROUPS"

    def test_builtin_matrix_matches_attack_subgroups_exactly(self):
        resolver = MetricResolver.from_attack_groups()
        for key, definition in ATTACK_SUBGROUPS.items():
            declared = set(
                definition["quality_metrics"]
                + definition["intelligibility_metrics"]
                + definition["nisqa_metrics"]
            )
            assert set(resolver.signal_metrics_for_group(key)) == declared

    @pytest.mark.parametrize("mode", ["benchmark", "detection_reliability"])
    def test_shipped_template_reproduces_the_builtin_matrix(self, mode):
        """An unedited --init file must not change what a run displays.

        Reads the packaged template rather than configs/, which holds
        working files that are meant to be edited.
        """
        path = f"src/deepmarkpy/config_templates/{mode}.json"
        shipped = load_configs([path])[0].resolver
        builtin = MetricResolver.from_attack_groups()
        drift = [
            (group, metric)
            for group in CONFIG_GROUP_KEYS
            for metric in CANONICAL_METRIC_ORDER
            if metric not in ("ber",)
            and shipped.is_enabled(group, metric) != builtin.is_enabled(group, metric)
        ]
        assert drift == []


class TestRequirement5StatisticsFallback:
    def test_omitting_statistics_entirely_yields_all_eight(self, tmp_path):
        resolver = build(tmp_path, metrics={"defaults": {"pesq": {"enabled": True}}})
        assert resolver.statistics_for("audio_distortion", "pesq") == list(ALL_STATISTICS)

    def test_the_top_level_list_applies_when_a_metric_says_nothing(self, tmp_path):
        resolver = build(tmp_path, statistics=["mean", "p95"],
                         metrics={"defaults": {"pesq": {"enabled": True}}})
        assert resolver.statistics_for("audio_distortion", "pesq") == ["mean", "p95"]

    def test_resolution_order_is_group_then_default_then_toplevel_then_all(self, tmp_path):
        resolver = build(tmp_path, statistics=["mean", "p95"], metrics={
            "defaults": {
                "pesq": {"enabled": True, "statistics": ["median"]},
                "mcd": {"enabled": True},
                "visqol": {"enabled": True},
            },
            "per_group": {
                "transmission": {"pesq": {"statistics": ["worst_case"]}},
            },
        })
        # 1. group override
        assert resolver.statistics_for("transmission", "pesq") == ["worst_case"]
        # 2. per-metric default
        assert resolver.statistics_for("audio_distortion", "pesq") == ["median"]
        # 3. top-level list
        assert resolver.statistics_for("audio_distortion", "mcd") == ["mean", "p95"]

    def test_column_order_follows_the_list_that_applied(self, tmp_path):
        resolver = build(tmp_path, metrics={"defaults": {
            "pesq": {"enabled": True,
                     "statistics": ["worst_case", "mean", "median"]}}})
        assert resolver.statistics_for(None, "pesq") == [
            "worst_case", "mean", "median",
        ]

    def test_emr_has_no_statistics_at_all(self, tmp_path):
        resolver = build(tmp_path, statistics=["mean"])
        assert resolver.statistics_for("audio_distortion", "emr") == []


class TestMandatoryAndDerivedMetrics:
    def test_accuracy_is_always_enabled(self, tmp_path):
        resolver = build(tmp_path, metrics={"defaults": {}})
        for group in CONFIG_GROUP_KEYS:
            assert resolver.is_enabled(group, "accuracy") is True

    def test_an_unmentioned_metric_is_off(self, tmp_path):
        """No hidden fallback re-enables what the config did not ask for."""
        resolver = build(tmp_path, metrics={"defaults": {"pesq": {"enabled": True}}})
        assert resolver.is_enabled("audio_distortion", "mcd") is False
        assert resolver.is_enabled("audio_distortion", "nisqa_mos") is False

    def test_omitting_the_metrics_block_uses_the_builtin_matrix(self, tmp_path):
        resolver = build(tmp_path)
        builtin = MetricResolver.from_attack_groups()
        for group in CONFIG_GROUP_KEYS:
            assert (resolver.signal_metrics_for_group(group)
                    == builtin.signal_metrics_for_group(group))


class TestCalculateQualityMetricsSwitch:
    """The master switch, as specified: config wins when on, the always-on
    trio applies when it is off or absent."""

    def test_on_means_the_config_decides(self, tmp_path):
        resolver = build(tmp_path, calculate_quality_metrics=True, metrics={
            "defaults": {"mcd": {"enabled": True}, "pesq": {"enabled": False}},
        })
        assert resolver.is_enabled("audio_distortion", "mcd") is True
        assert resolver.is_enabled("audio_distortion", "pesq") is False

    def test_off_means_only_the_always_on_trio(self, tmp_path):
        resolver = build(tmp_path, calculate_quality_metrics=False, metrics={
            "defaults": {"mcd": {"enabled": True}, "pesq": {"enabled": False}},
        })
        assert resolver.signal_metrics_for_group("audio_distortion") == [
            "pesq", "visqol", "stoi",
        ]
        assert resolver.is_enabled("audio_distortion", "mcd") is False

    def test_absent_behaves_like_off(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({
            "mode": "benchmark", "models": ["AudioSealModel"],
            "metrics": {"defaults": {"mcd": {"enabled": True}}},
        }))
        resolver = load_configs([str(path)])[0].resolver
        assert resolver.is_enabled("audio_distortion", "mcd") is False
        assert resolver.is_enabled("audio_distortion", "pesq") is True

    def test_turning_it_on_never_drops_the_always_on_trio(self, tmp_path):
        """Without a ``metrics`` block, on only adds to what off computes.

        That is what the ``--calculate_quality_metrics`` flag produces: the
        built-in matrix decides, and it has to keep PESQ and STOI for the
        desynchronization attacks, whose taxonomy entry names neither.
        """
        on = build(tmp_path, calculate_quality_metrics=True)
        off = build(tmp_path, calculate_quality_metrics=False)
        dropped = {
            attack: sorted(set(off.metrics_for_attack(attack))
                           - set(on.metrics_for_attack(attack)))
            for definition in ATTACK_GROUPS.values()
            for attack in definition["attacks"]
        }
        assert {a: m for a, m in dropped.items() if m} == {}

    def test_derived_metrics_still_honour_their_flags_when_off(self, tmp_path):
        """ber and emr come from the accuracy array, so they cost nothing."""
        resolver = build(tmp_path, calculate_quality_metrics=False, metrics={
            "defaults": {"ber": {"enabled": True}, "emr": {"enabled": False}},
        })
        assert resolver.is_enabled("audio_distortion", "ber") is True
        assert resolver.is_enabled("audio_distortion", "emr") is False

    def test_statistics_still_resolve_from_config_when_off(self, tmp_path):
        resolver = build(tmp_path, calculate_quality_metrics=False,
                         statistics=["median"])
        assert resolver.statistics_for("audio_distortion", "pesq") == ["median"]


class TestMetricOrdering:
    def test_metrics_come_back_in_canonical_order(self, tmp_path):
        resolver = build(tmp_path, metrics={"defaults": {
            "stoi": {"enabled": True},
            "pesq": {"enabled": True},
            "nisqa_mos": {"enabled": True},
        }})
        assert resolver.metrics_for_group("audio_distortion") == [
            "accuracy", "pesq", "stoi", "nisqa_mos",
        ]

    def test_buckets_split_the_way_reports_lay_them_out(self, tmp_path):
        resolver = build(tmp_path)
        assert resolver.metrics_for_group("audio_distortion", "quality") == [
            "pesq", "psnr", "si_sdr", "visqol",
        ]
        assert resolver.metrics_for_group("audio_distortion", "intelligibility") == [
            "stoi", "sii", "ncm",
        ]
        assert resolver.metrics_for_group("audio_distortion", "robustness") == [
            "accuracy", "ber", "emr",
        ]


class TestBaselineCoverage:
    def test_the_no_attack_baseline_carries_every_groups_metrics(self, tmp_path):
        """The baseline row appears in every group's table, so it must hold
        any metric any group might show."""
        resolver = build(tmp_path, metrics={
            "defaults": {m: {"enabled": False} for m in ("pesq", "mcd", "stoi")},
            "per_group": {
                "audio_distortion": {"pesq": {"enabled": True}},
                "desynchronization": {"mcd": {"enabled": True}},
            },
        })
        assert set(resolver.all_signal_metrics()) == {"pesq", "mcd"}

    def test_no_signal_metric_enabled_is_detectable(self, tmp_path):
        resolver = build(tmp_path, metrics={"defaults": {
            m: {"enabled": False} for m in CONFIGURABLE_METRICS
            if m not in ("accuracy", "ber", "emr")
        }})
        assert resolver.any_signal_metric_enabled() is False


class TestMetricFamiliesAreConsistent:
    def test_metric_families_agree_across_the_two_modules(self):
        """metrics.py and metric_resolver.py both split the families.

        The resolver deliberately does not import metrics.py -- that
        module pulls in librosa/pesq/pystoi, and config validation runs
        before any of that. So the split is duplicated on purpose, and
        has to be checked rather than shared.
        """

        from deepmarkpy.utils import metrics as m
        from deepmarkpy.utils import metric_resolver as r

        assert tuple(m.NISQA_METRICS) == r.NISQA_METRICS
        assert tuple(m.INTELLIGIBILITY_METRICS) == r.INTELLIGIBILITY_METRICS
        # metrics.QUALITY_METRICS folds NISQA in; the resolver tables them
        # separately, so compare the part they both call "quality".
        assert tuple(
            x for x in m.QUALITY_METRICS if x not in m.NISQA_METRICS
        ) == r.QUALITY_METRICS
        assert set(m.ALL_METRICS) == set(r.SIGNAL_METRICS)
