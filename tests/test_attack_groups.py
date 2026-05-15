"""Tests for src/utils/attack_groups.py."""

import pytest

from plugin_manager import PluginManager
from utils.attack_groups import (
    ATTACK_GROUPS,
    get_attacks_for_groups,
    get_group_for_attack,
    get_metrics_for_attack,
    group_attacks,
)


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


class TestGetMetricsForAttack:
    def test_returns_group_metrics(self):
        metrics = get_metrics_for_attack("GaussianNoiseAttack")
        assert "pesq" in metrics
        assert "stoi" in metrics

    def test_process_disruption_has_no_metrics(self):
        # Quality metrics are intentionally empty for this group
        assert get_metrics_for_attack("SameModelAttack") == []

    def test_unknown_attack_returns_all_metrics(self):
        from utils.metrics import ALL_METRICS
        assert set(get_metrics_for_attack("FakeAttack")) == set(ALL_METRICS)
