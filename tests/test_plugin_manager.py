"""Tests for src/plugin_manager.py — plugin discovery and config loading."""

import pytest

from deepmarkpy.plugin_manager import PluginManager
from deepmarkpy.core.base_attack import BaseAttack
from deepmarkpy.core.base_model import BaseModel


class TestPluginDiscovery:
    @pytest.fixture(autouse=True)
    def _create_pm(self):
        self.pm = PluginManager()

    def test_discovers_attacks(self):
        attacks = self.pm.get_attacks()
        assert isinstance(attacks, dict)
        assert len(attacks) > 0

    def test_discovers_models(self):
        models = self.pm.get_models()
        assert isinstance(models, dict)
        assert len(models) > 0

    def test_attack_classes_inherit_base(self):
        for name, entry in self.pm.get_attacks().items():
            assert issubclass(entry["class"], BaseAttack), f"{name} does not inherit BaseAttack"

    def test_model_classes_inherit_base(self):
        for name, entry in self.pm.get_models().items():
            assert issubclass(entry["class"], BaseModel), f"{name} does not inherit BaseModel"

    def test_attack_entries_have_config(self):
        for name, entry in self.pm.get_attacks().items():
            assert "config" in entry, f"{name} missing 'config' key"
            # config can be None (same_model) or a dict
            assert entry["config"] is None or isinstance(entry["config"], dict)

    def test_model_entries_have_config(self):
        for name, entry in self.pm.get_models().items():
            assert "config" in entry, f"{name} missing 'config' key"
            assert isinstance(entry["config"], dict), f"{name} has invalid config"

    def test_model_configs_have_required_keys(self):
        required = {"sampling_rate", "watermark_size"}
        for name, entry in self.pm.get_models().items():
            config = entry["config"]
            for key in required:
                assert key in config, f"{name} config missing '{key}'"

    def test_known_attacks_discovered(self):
        attacks = self.pm.get_attacks()
        expected = [
            "DescriptAudioCodecAttack",
            "DiffusionAttack",
            "EncodecAttack",
            "GaussianNoiseAttack",
            "NeuralVocoderAttack",
            "SignInversionAttack",
            "CropBeginningAttack",
            "SmoothingAttack",
            "SpeechEnhancement1Attack",
            "SpeechEnhancement2Attack",
            "SpeechTokenizationAttack",
            "VAEAttack",
        ]
        for name in expected:
            assert name in attacks, f"Expected attack '{name}' not discovered"

    def test_known_models_discovered(self):
        models = self.pm.get_models()
        expected = ["AudioSealModel", "WavMarkModel", "PerthModel"]
        for name in expected:
            assert name in models, f"Expected model '{name}' not discovered"
