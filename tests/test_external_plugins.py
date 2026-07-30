"""Tests for external plugin-directory discovery and the failed registry."""

import textwrap

import pytest

from deepmarkpy.plugin_manager import PluginManager


@pytest.fixture
def external_dir(tmp_path):
    """A drop-in plugin directory: one working attack, one broken attack."""
    good = tmp_path / "my_attack"
    good.mkdir()
    (good / "attack.py").write_text(textwrap.dedent("""
        import numpy as np
        from deepmarkpy.core.base_attack import BaseAttack

        class ExternalDropInAttack(BaseAttack):
            def apply(self, audio, **kwargs):
                return np.asarray(audio)
    """))
    (good / "config.json").write_text('{"gain_external_drop_in": 1.0}')

    broken = tmp_path / "broken_attack"
    broken.mkdir()
    (broken / "attack.py").write_text("import module_that_does_not_exist\n")
    return tmp_path


def test_external_plugin_registers_with_config(external_dir):
    pm = PluginManager(external_plugins_dir=str(external_dir))
    attacks = pm.get_attacks()
    assert "ExternalDropInAttack" in attacks
    assert attacks["ExternalDropInAttack"]["config"] == {"gain_external_drop_in": 1.0}


def test_external_plugin_failure_is_recorded_not_raised(external_dir):
    pm = PluginManager(external_plugins_dir=str(external_dir))
    failed_paths = [p for p in pm.failed if p.endswith("broken_attack/attack.py")]
    assert failed_paths, f"broken plugin missing from failed registry: {pm.failed}"


def test_env_var_names_the_external_dir(external_dir, monkeypatch):
    monkeypatch.setenv("DEEPMARK_PLUGINS_DIR", str(external_dir))
    pm = PluginManager()
    assert "ExternalDropInAttack" in pm.get_attacks()


def test_no_external_dir_changes_nothing():
    pm = PluginManager()
    assert "ExternalDropInAttack" not in pm.get_attacks()
    assert pm.external_plugins_dir is None or "ExternalDropInAttack" not in pm.get_attacks()
