"""Discovery lock tests — freeze the exact plugin registry.

These tests lock the exact sets of attack and model class names that
``PluginManager`` registers in the *canonical environment*, plus the provenance of each registration (the plugin directory whose
``attack.py``/``model.py`` defines the class). Their purpose is to turn the
silent-vanish failure mode into a hard test failure: plugin import errors are
logged and swallowed, so a plugin can disappear from the registry with no
error — most critically when a packaging change lets discovery return an empty
registry with exit code 0. Provenance is locked because discovery registers every subclass found
in a module's namespace, so a class imported across plugin modules (e.g.
``mixing`` imports ``EqualizerAttack``) can keep a same-named registry entry
alive even when its own directory has dropped out of the walked tree.

Canonical environment (the reference for all discovery assertions), recorded 2026-07-29 at commit
fec866b on the designated machine (macOS arm64, Darwin 25.3.0):

    python3 -m venv <venv>                      # CPython 3.11.5
    <venv>/bin/pip install -r requirements.txt  # pip 23.2.1
    <venv>/bin/pip install pytest               # pytest 9.1.1

``pytest`` is not in ``requirements.txt``; it was installed solely to execute
the suite and was verified (by re-running discovery before and after the
install) not to change the registered sets. ``pycodec2>=4.0.0`` is a floating
lower bound in ``requirements.txt``; it resolved to 4.1.1 at record time.

In the canonical environment four attack classes are absent because their
import-time dependencies are not in ``requirements.txt`` (a frozen,
intentional gap — the register notes ``wavelet``, ``pitch_shift``,
``time_stretch``; empirically ``inverted_time_stretch`` also vanishes because
it imports ``plugins.attacks.time_stretch.attack``). These four appear in
``OPTIONAL_DEP_ATTACKS`` below and may legitimately register in developer
environments that happen to have ``pywt``/``pyrubberband`` installed; the
exact-set assertion therefore binds only when neither optional dependency is
importable, while the subset, no-unknown-names, and provenance assertions
bind everywhere.

The asserted sets below are load-bearing: no removals ever, and additions
only as a deliberate, reviewed decision.
"""

import importlib.util
import inspect
from pathlib import Path

import pytest

from deepmarkpy.plugin_manager import PluginManager

# The exact model class-name set registered in the canonical environment,
# mapped to the plugin directory whose model.py defines each class.
CANONICAL_MODEL_DIRS = {
    "AudioSealModel": "audio_seal",
    "AwareModel": "aware",
    "PerthModel": "perth",
    "SilentCipherModel": "silent_cipher",
    "TimbreWMModel": "timbrewm",
    "WavMarkModel": "wavmark",
}
CANONICAL_MODELS = frozenset(CANONICAL_MODEL_DIRS)

# The exact attack class-name set registered in the canonical environment,
# mapped to the plugin directory whose attack.py defines each class.
CANONICAL_ATTACK_DIRS = {
    "AdditiveNoiseAttack": "additive_noise",
    "BandstopFilterAttack": "bandstop_filter",
    "ChorusAttack": "chorus",
    "Codec2VocoderAttack": "codec2_vocoder",
    "CollusionAttack": "collusion",
    "Collusion2Attack": "collusion_2",
    "CropBeginningAttack": "crop_beginning",
    "CropRandomAttack": "crop_random",
    "CrossModelAttack": "cross_model",
    "CutSamplesAttack": "cut_samples",
    "DescriptAudioCodecAttack": "descript_audio_codec",
    "DiffusionAttack": "diffusion",
    "EchoAttack": "echo",
    "EncodecAttack": "encodec",
    "EqualizerAttack": "equalizer",
    "FlangerAttack": "flanger",
    "FlipSamplesAttack": "flip_samples",
    "GaussianNoiseAttack": "gaussian_noise",
    "HighpassFilterAttack": "highpass_filter",
    "LPCAttack": "lpc",
    "LowpassFilterAttack": "lowpass_filter",
    "MixingAttack": "mixing",
    "Mp3CompressionAttack": "mp3_compression",
    "NetworkTransmissionAttack": "network_transmission",
    "NeuralVocoderAttack": "neural_vocoder",
    "OpusCodecAttack": "opus_codec",
    "PCMQuantizationAttack": "pcm_quantization",
    "PinkNoiseAttack": "pink_noise",
    "QuantizationAttack": "quantization",
    "ReplacementAttack": "replacement",
    "Replacement2Attack": "replacement_2",
    "ReplayAttack": "replay",
    "ResamplingPolyAttack": "resampling_poly",
    "SameModelAttack": "same_model",
    "SignInversionAttack": "sign_inversion",
    "SmoothingAttack": "smoothing",
    "SpeechEnhancement1Attack": "speech_enhancement_1",
    "SpeechEnhancement2Attack": "speech_enhancement_2",
    "SpeechTokenizationAttack": "speech_tokenization",
    "STFTQuantizationAttack": "stft_quantization",
    "VAEAttack": "vae",
    "ZeroBitCollusionAttack": "zero_bit_collusion",
    "ZeroCrossInsertsAttack": "zero_cross_inserts",
}
CANONICAL_ATTACKS = frozenset(CANONICAL_ATTACK_DIRS)

# Attack classes absent from the canonical environment because their
# import-time dependency is not in requirements.txt (D14, frozen). Maps
# class name -> (defining plugin directory, module whose absence removes
# the class from discovery).
OPTIONAL_DEP_ATTACKS = {
    "WaveletAttack": ("wavelet", "pywt"),
    "TimeStretchAttack": ("time_stretch", "pyrubberband"),
    "PitchShiftAttack": ("pitch_shift", "pyrubberband"),
    "InvertedTimeStretchAttack": ("inverted_time_stretch", "pyrubberband"),
}

# Attack sample for the config.json-content assertions. Each entry maps a
# class name to a key that exists only in that plugin's config.json (attack
# parameter names are globally unique), proving the directory's own config was
# attached. Classes imported into sibling plugin modules (EqualizerAttack,
# HighpassFilterAttack, TimeStretchAttack) are deliberately excluded: they can
# be re-registered under another directory's walk entry, so their attached
# config is walk-order-dependent.
ATTACK_CONFIG_SAMPLE = {
    "GaussianNoiseAttack": "snr_db_gaussian_noise",
    "VAEAttack": "model_name_vae",
    "OpusCodecAttack": "bitrate_opus_codec",
    "DiffusionAttack": "steps_diffusion",
    "EncodecAttack": "bandwidth_encodec",
}


def _installed_optional_deps():
    """Return the optional plugin dependencies importable in this environment."""
    optional_deps = sorted({dep for _, dep in OPTIONAL_DEP_ATTACKS.values()})
    return [
        dep for dep in optional_deps
        if importlib.util.find_spec(dep) is not None
    ]


@pytest.fixture(scope="module")
def pm():
    """One shared PluginManager — discovery imports ~50 plugin modules."""
    return PluginManager()


class TestModelDiscoveryLock:
    """The model registry must match the canonical set exactly, in any environment.

    No model has an optional import-time dependency: all six register from a
    plain ``pip install -r requirements.txt``, so exact equality binds
    unconditionally.
    """

    def test_exact_model_set(self, pm):
        registered = set(pm.get_models().keys())
        missing = CANONICAL_MODELS - registered
        extra = registered - CANONICAL_MODELS
        assert not missing, (
            f"Model plugins silently vanished from discovery: {sorted(missing)}. "
            "PluginManager swallows import errors — check the 'Failed to import' "
            "log lines. The canonical set is documented in this file's docstring."
        )
        assert not extra, (
            f"Unexpected model registrations: {sorted(extra)}. The lock set may "
            "only be extended deliberately, as a reviewed decision."
        )


class TestAttackDiscoveryLock:
    """The attack registry must cover the canonical set and add nothing unknown."""

    def test_all_canonical_attacks_registered(self, pm):
        missing = CANONICAL_ATTACKS - set(pm.get_attacks().keys())
        assert not missing, (
            f"Attack plugins silently vanished from discovery: {sorted(missing)}. "
            "Every dependency these plugins need is in requirements.txt, so in a "
            "canonical environment this means a real discovery regression; "
            "in a developer environment it can also mean the environment is stale "
            "relative to requirements.txt. PluginManager swallows import errors — "
            "check the 'Failed to import' log lines."
        )

    def test_no_unknown_attacks_registered(self, pm):
        allowed = CANONICAL_ATTACKS | set(OPTIONAL_DEP_ATTACKS)
        unknown = set(pm.get_attacks().keys()) - allowed
        assert not unknown, (
            f"Unexpected attack registrations: {sorted(unknown)}. The lock set may "
            "only be extended deliberately, as a reviewed decision."
        )

    def test_exact_attack_set_in_canonical_environment(self, pm):
        installed = _installed_optional_deps()
        if installed:
            pytest.skip(
                f"optional plugin dependencies installed: {installed} — the "
                "exact-set lock is defined against the canonical environment, "
                "which has neither"
            )
        registered = set(pm.get_attacks().keys())
        assert registered == CANONICAL_ATTACKS, (
            f"Canonical attack set mismatch. "
            f"Missing: {sorted(CANONICAL_ATTACKS - registered)}; "
            f"unexpected: {sorted(registered - CANONICAL_ATTACKS)}."
        )

    def test_optional_dep_attacks_register_when_dep_installed(self, pm):
        registered = set(pm.get_attacks().keys())
        for cls_name, (_, dep) in OPTIONAL_DEP_ATTACKS.items():
            if importlib.util.find_spec(dep) is not None:
                assert cls_name in registered, (
                    f"{cls_name} silently vanished from discovery even though "
                    f"its dependency '{dep}' is importable in this environment."
                )


class TestRegistrationProvenanceLock:
    """Every registration must come from its own plugin directory's walked file.

    Discovery registers every ``BaseAttack``/``BaseModel`` subclass found in a
    walked module's namespace, so a class imported across plugin modules (e.g.
    ``mixing`` imports ``EqualizerAttack``) keeps a same-named registry entry
    alive even if its own directory drops out of the walked tree — the
    walk/import divergence risk. Pinning each class's
    defining module and source location closes that hole, and also rejects a
    same-named impostor class defined elsewhere. ``__module__`` is fixed at
    class definition, so these assertions are walk-order-independent; the
    suffix comparison is prefix-agnostic so the packaged
    ``deepmarkpy.plugins.…`` prefix keeps the lock intact.
    """

    @staticmethod
    def _assert_provenance(entry, cls_name, expected_module_suffix, plugins_dir):
        cls = entry["class"]
        assert cls.__module__.endswith(expected_module_suffix), (
            f"{cls_name} is registered from module '{cls.__module__}', expected "
            f"a module ending in '{expected_module_suffix}' — the registry entry "
            "is not backed by the class's own plugin directory (walk/import "
            "divergence) or a same-named impostor class "
            "replaced the real plugin."
        )
        source = Path(inspect.getfile(cls)).resolve()
        walked = Path(plugins_dir).resolve()
        assert walked in source.parents, (
            f"{cls_name} is defined in '{source}', outside the walked plugins "
            f"directory '{walked}' — discovery is being fed by imports, not by "
            "the walked tree (walk/import divergence)."
        )

    def test_attack_registrations_defined_in_own_plugin_directory(self, pm):
        attacks = pm.get_attacks()
        provenance = {
            **CANONICAL_ATTACK_DIRS,
            **{name: dir_name for name, (dir_name, _) in OPTIONAL_DEP_ATTACKS.items()},
        }
        for cls_name, dir_name in provenance.items():
            if cls_name not in attacks:
                continue  # absence is judged by the set tests above
            self._assert_provenance(
                attacks[cls_name],
                cls_name,
                f"plugins.attacks.{dir_name}.attack",
                pm.plugins_dir,
            )

    def test_model_registrations_defined_in_own_plugin_directory(self, pm):
        models = pm.get_models()
        for cls_name, dir_name in CANONICAL_MODEL_DIRS.items():
            if cls_name not in models:
                continue  # absence is judged by the set tests above
            self._assert_provenance(
                models[cls_name],
                cls_name,
                f"plugins.models.{dir_name}.model",
                pm.plugins_dir,
            )


class TestConfigAttachmentLock:
    """Registered plugins must carry the config.json attached at discovery."""

    def test_every_registered_plugin_has_config_dict(self, pm):
        for kind, registry in (("attack", pm.get_attacks()), ("model", pm.get_models())):
            for cls_name, entry in registry.items():
                assert isinstance(entry["config"], dict), (
                    f"{kind} {cls_name} registered without a config dict (got "
                    f"{entry['config']!r}) — every plugin directory has a "
                    "config.json, so a None here means it went missing or "
                    "unparsable and discovery swallowed the error."
                )

    def test_sampled_attacks_have_their_own_config(self, pm):
        attacks = pm.get_attacks()
        for cls_name, distinctive_key in ATTACK_CONFIG_SAMPLE.items():
            assert cls_name in attacks, f"{cls_name} not discovered"
            config = attacks[cls_name]["config"]
            assert isinstance(config, dict), (
                f"{cls_name} registered without a config dict (got {config!r})"
            )
            assert distinctive_key in config, (
                f"{cls_name} config is missing its own key '{distinctive_key}' — "
                "the wrong directory's config.json may have been attached"
            )

    def test_same_model_attack_registers_empty_config(self, pm):
        attacks = pm.get_attacks()
        assert "SameModelAttack" in attacks, "SameModelAttack not discovered"
        assert attacks["SameModelAttack"]["config"] == {}, (
            "same_model's config.json is empty on disk and must attach as {} "
            f"(got {attacks['SameModelAttack']['config']!r})"
        )

    def test_all_models_have_config_with_required_keys(self, pm):
        models = pm.get_models()
        for cls_name in sorted(CANONICAL_MODELS):
            assert cls_name in models, f"{cls_name} not discovered"
            config = models[cls_name]["config"]
            assert isinstance(config, dict), (
                f"{cls_name} registered without a config dict (got {config!r})"
            )
            for key in ("sampling_rate", "watermark_size"):
                assert key in config, f"{cls_name} config missing '{key}'"
