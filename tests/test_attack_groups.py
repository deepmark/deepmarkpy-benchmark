"""Tests for src/utils/attack_groups.py."""

import pytest

from deepmarkpy.plugin_manager import PluginManager
from deepmarkpy.utils.attack_groups import (
    ATTACK_GROUPS,
    get_attacks_for_groups,
    get_group_for_attack,
    group_attacks,
)
from deepmarkpy.utils.metric_resolver import MetricResolver


class TestGroupedAttacksMatchPlugins:
    """All hardcoded attack names must correspond to real plugins."""

    @pytest.fixture(autouse=True)
    def _discover(self):
        self.available = set(PluginManager().get_attacks().keys())

    def test_all_grouped_attacks_exist(self):
        for group_key, group in ATTACK_GROUPS.items():
            for attack in group["attacks"]:
                assert attack in self.available, (
                    f"{attack} in group '{group_key}' is not a discovered plugin"
                )


class TestGetAttacksForGroups:
    def test_single_group(self):
        attacks = get_attacks_for_groups("audio_distortion")
        assert "GaussianNoiseAttack" in attacks

    def test_multiple_groups(self):
        attacks = get_attacks_for_groups(
            ["audio_distortion", "transmission"]
        )
        assert "GaussianNoiseAttack" in attacks
        assert "ReplayAttack" in attacks

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError):
            get_attacks_for_groups("nonexistent_group")


class TestGetGroupForAttack:
    def test_known_attack(self):
        assert get_group_for_attack("GaussianNoiseAttack") == "audio_distortion"

    def test_unknown_attack_returns_none(self):
        assert get_group_for_attack("FakeAttack") is None


class TestGroupAttacks:
    def test_organizes_by_group(self):
        grouped = group_attacks(["GaussianNoiseAttack", "ReplayAttack"])
        assert "audio_distortion" in grouped
        assert "transmission" in grouped

    def test_unknown_attacks_fall_into_other(self):
        grouped = group_attacks(["FakeAttack"])
        assert "other" in grouped
        assert grouped["other"]["attacks"] == ["FakeAttack"]


class TestMetricsForAttack:
    """The taxonomy decides the default metric set for each attack, via
    the matrix ``MetricResolver.from_attack_groups`` builds from it."""

    @staticmethod
    def _metrics(attack):
        return MetricResolver.from_attack_groups().metrics_for_attack(attack)

    def test_returns_group_metrics(self):
        metrics = self._metrics("GaussianNoiseAttack")
        assert "pesq" in metrics
        assert "stoi" in metrics

    def test_process_disruption_has_metrics(self):
        metrics = self._metrics("SameModelAttack")
        assert "pesq" in metrics
        assert "nisqa_mos" in metrics

    def test_unknown_attack_gets_the_full_set(self):
        """An ungrouped attack lands in "other", which enables everything."""
        from deepmarkpy.utils.metric_resolver import SIGNAL_METRICS
        assert set(self._metrics("FakeAttack")) == set(SIGNAL_METRICS)
