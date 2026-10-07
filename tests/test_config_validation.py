"""Every validation error code, raised by the input that should raise it.

The point of the error catalog is that a user fixes a config file in one
pass: every problem is reported together, each with a stable code, the
exact JSON path, the offending value, and a suggestion when the value
looks like a typo. These tests pin one input per code, so a message can
be reworded but a code cannot silently stop firing.
"""

import json
import re

import pytest

from deepmarkpy.benchmark import expand_attacks
from deepmarkpy.config import (
    ConfigError,
    MODE_KEYS,
    VALID_MODES,
    init_template,
    load_config_data,
    load_configs,
)
from deepmarkpy.core.base_model import BaseModel
from deepmarkpy.plugin_manager import PluginManager
from deepmarkpy.run import _peek_plugins_dir
from deepmarkpy.utils.attack_groups import ATTACK_GROUPS


# A minimal file that validates, used as the base every case perturbs.
BASE = {
    "mode": "benchmark",
    "models": ["AudioSealModel"],
    # On, so the metrics block is what decides -- with it off the
    # always-on trio applies instead and the enable flags are ignored,
    # which is its own case below.
    "calculate_quality_metrics": True,
}

# Stand-ins for the plugin registries, so validation is testable without
# importing every plugin.
ATTACKS = {
    "GaussianNoiseAttack": {
        "config": {"snr_db_gaussian_noise": 35},
        "_raw_config": {
            "default": {"snr_db_gaussian_noise": 35},
            "mild": {"snr_db_gaussian_noise": 45},
        },
    },
    "EchoAttack": {"config": {"delay_echo": 0.1}, "_raw_config": {"delay_echo": 0.1}},
    "LowpassFilterAttack": {"config": {"cutoff_lowpass": 4000}},
}
MODELS = {"AudioSealModel": {}, "PerthModel": {}}
# A group is only selectable when every attack it declares was discovered.
DISTORTION = {
    **{name: {} for name in ATTACK_GROUPS["audio_distortion"]["attacks"]},
    **ATTACKS,
}


class _DecidingModel(BaseModel):
    """Overrides is_watermarked(). Validation inspects it, never builds it."""

    def is_watermarked(self, detect_output):
        return bool(detect_output)


class _UndecidedModel(BaseModel):
    """Inherits the base is_watermarked(), which raises."""


# Entries that carry the model class, as discovery registers them.
CLASS_MODELS = {
    "PerthModel": {"class": _DecidingModel, "config": {}},
    "WavMarkModel": {"class": _UndecidedModel, "config": {}},
}


def write(tmp_path, overrides, name="config.json", base=BASE):
    """Write ``base`` merged with ``overrides``; a None value drops the key."""
    data = dict(base)
    for key, value in overrides.items():
        if value is None and key in data:
            del data[key]
        else:
            data[key] = value
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def codes(paths, **kwargs):
    """Validate and return the set of error codes raised."""
    if isinstance(paths, str):
        paths = [paths]
    kwargs.setdefault("attacks_registry", ATTACKS)
    kwargs.setdefault("models_registry", MODELS)
    with pytest.raises(ConfigError) as excinfo:
        load_configs(paths, **kwargs)
    return {issue.code for issue in excinfo.value.issues}, excinfo.value


def issue_for(error, code):
    return next(i for i in error.issues if i.code == code)


class TestFileLevelCodes:
    def test_E001_missing_file(self, tmp_path):
        found, _ = codes(str(tmp_path / "nope.json"))
        assert "E001" in found

    def test_E002_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"mode": "benchmark",}')
        found, error = codes(str(path))
        assert "E002" in found
        # Names the actual mistake, not just the parser's state.
        assert "trailing comma" in issue_for(error, "E002").message

    def test_E002_a_utf16_file_is_a_config_error(self, tmp_path):
        """What Windows PowerShell 5.1's '>' writes when --init is redirected."""
        path = tmp_path / "utf16.json"
        path.write_bytes(json.dumps(BASE).encode("utf-16"))
        found, error = codes(str(path))
        assert "E002" in found
        assert "UTF-8" in issue_for(error, "E002").message

    def test_a_utf8_byte_order_mark_is_accepted(self, tmp_path):
        """What 'Out-File -Encoding utf8', the fix E002 suggests, writes in
        PowerShell 5.1. The plugins_dir peek before discovery reads it too."""
        path = tmp_path / "bom.json"
        path.write_bytes(json.dumps(
            {**BASE, "general": {"plugins_dir": "plugins"}}
        ).encode("utf-8-sig"))
        assert load_configs([str(path)], ATTACKS, MODELS)[0].mode == "benchmark"
        assert _peek_plugins_dir([str(path)]) == "plugins"

    def test_E003_root_not_an_object(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")
        found, _ = codes(str(path))
        assert "E003" in found


class TestModeCodes:
    def test_E004_mode_missing(self, tmp_path):
        found, _ = codes(write(tmp_path, {"mode": None}))
        assert "E004" in found

    def test_E005_mode_unknown_suggests_closest(self, tmp_path):
        found, error = codes(write(tmp_path, {"mode": "benchmrak"}))
        assert "E005" in found
        assert issue_for(error, "E005").suggestion == "benchmark"

    def test_E006_two_files_declare_the_same_mode(self, tmp_path):
        first = write(tmp_path, {}, name="a.json")
        second = write(tmp_path, {}, name="b.json")
        found, _ = codes([first, second])
        assert "E006" in found

    def test_two_files_with_different_modes_are_fine(self, tmp_path):
        first = write(tmp_path, {}, name="a.json")
        second = write(tmp_path, {"mode": "no_attacks"}, name="b.json")
        configs = load_configs([first, second], ATTACKS, MODELS)
        assert [c.mode for c in configs] == ["benchmark", "no_attacks"]


class TestKeyCodes:
    def test_E007_unknown_key_suggests_closest(self, tmp_path):
        found, error = codes(write(tmp_path, {"modles": ["X"]}))
        assert "E007" in found
        assert issue_for(error, "E007").suggestion == "models"

    def test_E008_key_belongs_to_a_different_mode(self, tmp_path):
        found, error = codes(write(
            tmp_path, {"mode": "no_attacks", "attacks": {"groups": []}}))
        assert "E008" in found
        assert "benchmark" in issue_for(error, "E008").message

    def test_E009_wrong_type(self, tmp_path):
        found, _ = codes(write(tmp_path, {"general": "report"}))
        assert "E009" in found

    def test_E039_unknown_general_key(self, tmp_path):
        found, error = codes(write(tmp_path, {"general": {"report_dirr": "x"}}))
        assert "E039" in found
        assert issue_for(error, "E039").suggestion == "report_dir"


class TestModelCodes:
    def test_empty_registry_rejects_model_names(self, tmp_path):
        found, _ = codes(write(tmp_path, {}), models_registry={})
        assert "E011" in found

    def test_E010_models_missing(self, tmp_path):
        found, _ = codes(write(tmp_path, {"models": None}))
        assert "E010" in found

    def test_E010_models_empty(self, tmp_path):
        found, _ = codes(write(tmp_path, {"models": []}))
        assert "E010" in found

    def test_E011_unknown_model_suggests_closest(self, tmp_path):
        found, error = codes(write(tmp_path, {"models": ["AudioSealMode"]}))
        assert "E011" in found
        assert issue_for(error, "E011").suggestion == "AudioSealModel"

    def test_E012_detection_reliability_takes_one_model(self, tmp_path):
        found, _ = codes(write(tmp_path, {
            "mode": "detection_reliability",
            "models": ["AudioSealModel", "PerthModel"],
        }))
        assert "E012" in found

    def test_E016_duplicate_model(self, tmp_path):
        found, _ = codes(write(
            tmp_path, {"models": ["AudioSealModel", "AudioSealModel"]}))
        assert "E016" in found

    def test_E044_detection_reliability_model_must_implement_is_watermarked(
            self, tmp_path):
        found, error = codes(write(tmp_path, {
            "mode": "detection_reliability", "models": ["WavMarkModel"],
        }), models_registry=CLASS_MODELS)
        assert "E044" in found
        issue = issue_for(error, "E044")
        assert issue.path == "models[0]"
        assert issue.value == "WavMarkModel"
        # Names the discovered models that do qualify.
        assert "PerthModel" in issue.message

    def test_E044_is_reported_alongside_E012(self, tmp_path):
        """The refused model still counts towards the one-model limit."""
        found, _ = codes(write(tmp_path, {
            "mode": "detection_reliability",
            "models": ["WavMarkModel", "PerthModel"],
        }), models_registry=CLASS_MODELS)
        assert {"E012", "E044"} <= found

    @pytest.mark.parametrize("mode", ["benchmark", "no_attacks"])
    def test_other_modes_do_not_need_is_watermarked(self, tmp_path, mode):
        config = load_configs([write(tmp_path, {
            "mode": mode, "models": ["WavMarkModel"],
        })], ATTACKS, CLASS_MODELS)[0]
        assert config.models == ["WavMarkModel"]

    def test_a_model_that_implements_is_watermarked_is_accepted(self, tmp_path):
        config = load_configs([write(tmp_path, {
            "mode": "detection_reliability", "models": ["PerthModel"],
        })], ATTACKS, CLASS_MODELS)[0]
        assert config.models == ["PerthModel"]

    def test_no_E044_without_a_model_class(self, tmp_path):
        """No registry (the pass before plugins load) or a name-only one
        has no class to ask, so the model is not refused there."""
        path = write(tmp_path, {
            "mode": "detection_reliability", "models": ["WavMarkModel"],
        })
        for registry in (None, {"WavMarkModel": {}}):
            config = load_configs([path], models_registry=registry,
                                  quiet=True)[0]
            assert config.models == ["WavMarkModel"]


class TestAttackCodes:
    @pytest.mark.parametrize("spec", ["EchoAttack", "EchoAttack:mild"])
    def test_empty_registry_rejects_attack_names(self, tmp_path, spec):
        found, _ = codes(
            write(tmp_path, {"attacks": {"list": [spec]}}), attacks_registry={},
        )
        assert "E014" in found

    def test_empty_registry_rejects_parameter_targets(self, tmp_path):
        found, _ = codes(write(tmp_path, {
            "attack_parameters": {"EchoAttack": {"delay_echo": 0.2}},
        }), attacks_registry={})
        assert "E017" in found

    def test_E013_unknown_group_suggests_closest(self, tmp_path):
        found, error = codes(write(
            tmp_path, {"attacks": {"groups": ["audio_distorsion"]}}))
        assert "E013" in found
        assert issue_for(error, "E013").suggestion == "audio_distortion"

    def test_E013_names_a_subgroup_as_not_selectable(self, tmp_path):
        found, error = codes(write(
            tmp_path, {"attacks": {"groups": ["temporal_editing"]}}))
        assert "E013" in found
        assert "report subsection" in issue_for(error, "E013").message

    def test_E014_unknown_attack_suggests_closest(self, tmp_path):
        found, error = codes(write(
            tmp_path, {"attacks": {"list": ["GausianNoiseAttack"]}}))
        assert "E014" in found
        assert issue_for(error, "E014").suggestion == "GaussianNoiseAttack"

    def test_E014_group_member_that_was_not_discovered(self, tmp_path):
        """A group runs every attack it declares, and the run refuses one
        whose plugin did not load."""
        found, error = codes(write(
            tmp_path, {"attacks": {"groups": ["audio_distortion"]}}))
        assert "E014" in found
        issue = issue_for(error, "E014")
        assert issue.path == "attacks.groups[0]"
        assert "PinkNoiseAttack" in issue.message
        assert "GaussianNoiseAttack" not in issue.message

    def test_E015_unknown_attack_version(self, tmp_path):
        found, error = codes(write(
            tmp_path, {"attacks": {"list": ["GaussianNoiseAttack:milde"]}}))
        assert "E015" in found
        assert issue_for(error, "E015").suggestion == "mild"

    def test_known_version_is_accepted(self, tmp_path):
        config = load_configs(
            [write(tmp_path, {"attacks": {"list": ["GaussianNoiseAttack:mild"]}})],
            ATTACKS, MODELS,
        )[0]
        assert config.attack_list == ["GaussianNoiseAttack:mild"]

    def test_E016_duplicate_attack(self, tmp_path):
        found, _ = codes(write(tmp_path, {"attacks": {
            "list": ["EchoAttack", "EchoAttack"]}}))
        assert "E016" in found


class TestAttackParameterCodes:
    # One Codec2 run per listed bitrate; an override replaces the list.
    CODEC2 = {**ATTACKS, "Codec2VocoderAttack": {
        "config": {"bitrate_codec2": [700, 1200, 2400]}}}
    # CrossModelAttack detects with a second, named model.
    CROSS_MODEL = {"CrossModelAttack": {
        "config": {"different_model_name_cross_model": "AudioSealModel"}}}

    def test_E017_unknown_attack_key(self, tmp_path):
        found, error = codes(write(
            tmp_path, {"attack_parameters": {"EchoAttackk": {"delay_echo": 1}}}))
        assert "E017" in found
        assert issue_for(error, "E017").suggestion == "EchoAttack"

    def test_E018_unknown_parameter_suggests_closest(self, tmp_path):
        found, error = codes(write(tmp_path, {
            "attack_parameters": {"EchoAttack": {"delay_eco": 0.2}}}))
        assert "E018" in found
        assert issue_for(error, "E018").suggestion == "delay_echo"

    def test_E019_wrong_parameter_type(self, tmp_path):
        found, _ = codes(write(tmp_path, {
            "attack_parameters": {"EchoAttack": {"delay_echo": "loud"}}}))
        assert "E019" in found

    def test_two_attacks_may_share_a_parameter_name(self, tmp_path):
        """Parameters are routed per attack entry, so both values apply,
        each to its own attack."""
        registry = dict(ATTACKS)
        registry["OtherEchoAttack"] = {"config": {"delay_echo": 0.3}}
        config = load_configs([write(tmp_path, {"attack_parameters": {
            "EchoAttack": {"delay_echo": 0.2},
            "OtherEchoAttack": {"delay_echo": 0.4},
        }})], registry, MODELS)[0]

        assert config.parameters_for("EchoAttack", None) == {"delay_echo": 0.2}
        assert config.parameters_for("OtherEchoAttack", None) == {"delay_echo": 0.4}

    def test_a_bare_name_targets_the_default_version_only(self, tmp_path):
        """The mislabelling trap: an override must not reach other versions."""
        config = load_configs([write(tmp_path, {"attack_parameters": {
            "GaussianNoiseAttack": {"snr_db_gaussian_noise": 25},
        }})], ATTACKS, MODELS)[0]

        assert config.parameters_for("GaussianNoiseAttack", "default") == {
            "snr_db_gaussian_noise": 25,
        }
        assert config.parameters_for("GaussianNoiseAttack", "mild") == {}

    def test_an_existing_version_can_be_overridden(self, tmp_path):
        config = load_configs([write(tmp_path, {"attack_parameters": {
            "GaussianNoiseAttack:mild": {"snr_db_gaussian_noise": 50},
        }})], ATTACKS, MODELS)[0]

        assert config.parameters_for("GaussianNoiseAttack", "mild") == {
            "snr_db_gaussian_noise": 50,
        }
        assert config.parameters_for("GaussianNoiseAttack", "default") == {}

    def test_default_and_a_named_version_can_differ(self, tmp_path):
        config = load_configs([write(tmp_path, {"attack_parameters": {
            "GaussianNoiseAttack": {"snr_db_gaussian_noise": 30},
            "GaussianNoiseAttack:mild": {"snr_db_gaussian_noise": 55},
        }})], ATTACKS, MODELS)[0]

        assert config.parameters_for("GaussianNoiseAttack", "default")[
            "snr_db_gaussian_noise"] == 30
        assert config.parameters_for("GaussianNoiseAttack", "mild")[
            "snr_db_gaussian_noise"] == 55

    def test_a_complete_new_version_is_defined(self, tmp_path):
        """All parameters given, so the version is added and selectable."""
        config = load_configs([write(tmp_path, {
            "attack_parameters": {
                "GaussianNoiseAttack:brutal": {"snr_db_gaussian_noise": 5},
            },
            "attacks": {"list": ["GaussianNoiseAttack:brutal"]},
        })], ATTACKS, MODELS)[0]

        assert config.synthetic_versions == {
            "GaussianNoiseAttack": {"brutal": {"snr_db_gaussian_noise": 5}},
        }
        assert config.parameters_for("GaussianNoiseAttack", "brutal") == {
            "snr_db_gaussian_noise": 5,
        }
        assert config.attack_list == ["GaussianNoiseAttack:brutal"]

    def test_a_partial_new_version_is_skipped_with_a_warning(self, tmp_path):
        """Only some parameters given, so it is not a version at all."""
        registry = dict(ATTACKS)
        registry["TwoParamAttack"] = {
            "config": {"alpha_two": 1.0, "beta_two": 2.0},
            "_raw_config": {"alpha_two": 1.0, "beta_two": 2.0},
        }
        config = load_configs([write(tmp_path, {"attack_parameters": {
            "TwoParamAttack:half": {"alpha_two": 9.0},
        }})], registry, MODELS)[0]

        assert config.synthetic_versions == {}
        assert config.parameters_for("TwoParamAttack", "half") == {}
        warned = next(w for w in config.warnings if w.code == "W010")
        assert "beta_two" in warned.message

    def test_E015_names_a_version_defined_only_partially(self, tmp_path):
        """A skipped version must not then be selectable."""
        registry = dict(ATTACKS)
        registry["TwoParamAttack"] = {
            "config": {"alpha_two": 1.0, "beta_two": 2.0},
            "_raw_config": {"alpha_two": 1.0, "beta_two": 2.0},
        }
        with pytest.raises(ConfigError) as excinfo:
            load_configs([write(tmp_path, {
                "attack_parameters": {"TwoParamAttack:half": {"alpha_two": 9.0}},
                "attacks": {"list": ["TwoParamAttack:half"]},
            })], registry, MODELS)
        issue = next(i for i in excinfo.value.issues if i.code == "E015")
        assert "give ALL of this attack's parameters" in issue.message

    def test_E015_message_lists_versions_the_config_defined(self, tmp_path):
        with pytest.raises(ConfigError) as excinfo:
            load_configs([write(tmp_path, {
                "attack_parameters": {
                    "GaussianNoiseAttack:brutal": {"snr_db_gaussian_noise": 5},
                },
                "attacks": {"list": ["GaussianNoiseAttack:savage"]},
            })], ATTACKS, MODELS)
        issue = next(i for i in excinfo.value.issues if i.code == "E015")
        assert "brutal" in issue.message

    def test_E017_is_raised_for_a_versioned_key_too(self, tmp_path):
        found, error = codes(write(tmp_path, {"attack_parameters": {
            "EchoAttackk:loud": {"delay_echo": 1}}}))
        assert "E017" in found
        assert issue_for(error, "E017").suggestion == "EchoAttack"

    def test_integer_is_accepted_where_the_default_is_a_float(self, tmp_path):
        """JSON writes 2 for 2.0, so an int must satisfy a float default."""
        config = load_configs([write(tmp_path, {"attack_parameters": {
            "EchoAttack": {"delay_echo": 1}}})], ATTACKS, MODELS)[0]
        assert config.parameters_for("EchoAttack", None)["delay_echo"] == 1

    def test_a_single_codec2_bitrate_is_accepted(self, tmp_path):
        """Stored as written, and run as a one-element list would be."""
        config = load_configs([write(tmp_path, {
            "attacks": {"list": ["Codec2VocoderAttack"]},
            "attack_parameters": {
                "Codec2VocoderAttack": {"bitrate_codec2": 1200}},
        })], self.CODEC2, MODELS)[0]

        assert config.parameters_for("Codec2VocoderAttack", None) == \
            {"bitrate_codec2": 1200}
        expanded = expand_attacks(["Codec2VocoderAttack"], self.CODEC2,
                                  parameters=config.parameters_for)
        assert [display for _, display, _, _ in expanded] == \
            ["Codec2VocoderAttack_1200"]

    @pytest.mark.parametrize("value", ["1200", True])
    def test_E019_a_single_codec2_bitrate_must_be_an_integer(self, tmp_path,
                                                            value):
        found, _ = codes(write(tmp_path, {"attack_parameters": {
            "Codec2VocoderAttack": {"bitrate_codec2": value}}}),
            attacks_registry=self.CODEC2)
        assert "E019" in found

    def test_E019_no_other_list_parameter_takes_a_single_value(self, tmp_path):
        """A range or per-band list given as one number would reach the
        attack unchanged and fail inside apply()."""
        registry = {"BandstopFilterAttack": {
            "config": {"freq_range_bandstop": [350, 500]}}}
        found, _ = codes(write(tmp_path, {"attack_parameters": {
            "BandstopFilterAttack": {"freq_range_bandstop": 400}}}),
            attacks_registry=registry)
        assert "E019" in found

    @pytest.mark.parametrize("value", [
        [1000], [], ["1300"], [700, 1000], [[700]], [True], 1000,
    ], ids=["unsupported", "empty", "string", "one_of_two_unsupported",
            "nested", "bool", "single_unsupported"])
    def test_E045_unsupported_codec2_bitrate(self, tmp_path, value):
        """At least one bitrate, and each an integer Codec2 supports."""
        found, error = codes(write(tmp_path, {
            "attacks": {"list": ["Codec2VocoderAttack"]},
            "attack_parameters": {
                "Codec2VocoderAttack": {"bitrate_codec2": value}},
        }), attacks_registry=self.CODEC2)
        assert "E045" in found
        issue = issue_for(error, "E045")
        assert issue.path == \
            "attack_parameters.Codec2VocoderAttack.bitrate_codec2"
        # Lists what Codec2 supports, not just the plugin's own default list.
        assert "1300" in issue.message

    def test_E045_applies_to_a_version_the_config_defines(self, tmp_path):
        found, error = codes(write(tmp_path, {
            "attacks": {"list": ["Codec2VocoderAttack:tiny"]},
            "attack_parameters": {
                "Codec2VocoderAttack:tiny": {"bitrate_codec2": [1000]}},
        }), attacks_registry=self.CODEC2)
        assert "E045" in found
        assert issue_for(error, "E045").path == \
            "attack_parameters.Codec2VocoderAttack:tiny.bitrate_codec2"

    def test_any_supported_codec2_bitrate_is_accepted(self, tmp_path):
        """Supported by Codec2, though absent from the plugin's own list."""
        config = load_configs([write(tmp_path, {
            "attacks": {"list": ["Codec2VocoderAttack"]},
            "attack_parameters": {
                "Codec2VocoderAttack": {"bitrate_codec2": [1300, 3200]}},
        })], self.CODEC2, MODELS)[0]
        assert config.parameters_for("Codec2VocoderAttack", None) == \
            {"bitrate_codec2": [1300, 3200]}

    def test_E011_unknown_cross_model_second_model(self, tmp_path):
        """The second model is checked against the discovered models, as
        each name under 'models' is."""
        found, error = codes(write(tmp_path, {
            "attacks": {"list": ["CrossModelAttack"]},
            "attack_parameters": {"CrossModelAttack": {
                "different_model_name_cross_model": "AudioSealMode"}},
        }), attacks_registry=self.CROSS_MODEL)
        assert "E011" in found
        issue = issue_for(error, "E011")
        assert issue.path == ("attack_parameters.CrossModelAttack."
                              "different_model_name_cross_model")
        assert issue.suggestion == "AudioSealModel"

    def test_a_discovered_cross_model_second_model_is_accepted(self, tmp_path):
        config = load_configs([write(tmp_path, {
            "attacks": {"list": ["CrossModelAttack"]},
            "attack_parameters": {"CrossModelAttack": {
                "different_model_name_cross_model": "PerthModel"}},
        })], self.CROSS_MODEL, MODELS)[0]
        assert config.parameters_for("CrossModelAttack", None) == \
            {"different_model_name_cross_model": "PerthModel"}


class TestStatisticCodes:
    def test_E021_unknown_statistic_suggests_closest(self, tmp_path):
        found, error = codes(write(tmp_path, {"statistics": ["mean", "medain"]}))
        assert "E021" in found
        assert issue_for(error, "E021").suggestion == "median"

    def test_E022_empty_statistics_list(self, tmp_path):
        found, error = codes(write(tmp_path, {"statistics": []}))
        assert "E022" in found
        assert "enabled" in issue_for(error, "E022").message

    def test_E022_empty_per_metric_statistics_list(self, tmp_path):
        found, _ = codes(write(tmp_path, {"metrics": {
            "defaults": {"pesq": {"statistics": []}}}}))
        assert "E022" in found

    def test_E023_duplicate_statistic(self, tmp_path):
        found, _ = codes(write(tmp_path, {"statistics": ["mean", "mean"]}))
        assert "E023" in found

    @pytest.mark.parametrize("overrides,path", [
        ({"statistics": ["std"]}, "statistics"),
        ({"metrics": {"defaults": {"accuracy": {"statistics": ["std"]}}}},
         "metrics.defaults.accuracy.statistics"),
        ({"metrics": {"defaults": {}, "per_group": {
            "audio_editing": {"accuracy": {"statistics": ["std"]}}}}},
         "metrics.per_group.audio_editing.accuracy.statistics"),
    ])
    def test_E046_accuracy_needs_more_than_std(self, tmp_path, overrides, path):
        """Given only the spread, the basic report prints it as the accuracy."""
        found, error = codes(write(tmp_path, overrides))
        assert "E046" in found
        issue = issue_for(error, "E046")
        assert issue.path == path
        assert "spread" in issue.message

    def test_no_E046_for_a_subsection_no_table_reads(self, tmp_path):
        """No table reads a subsection's accuracy statistics, so W016 says
        so and E046 does not apply."""
        config = load_configs([write(tmp_path, {"metrics": {
            "defaults": {"accuracy": {"enabled": True}},
            "per_group": {"temporal_editing": {
                "accuracy": {"statistics": ["std"]}}},
        }})], ATTACKS, MODELS)[0]
        assert [w.code for w in config.warnings if w.code == "W016"] == ["W016"]


class TestMetricCodes:
    def test_E024_unknown_metric_suggests_closest(self, tmp_path):
        found, error = codes(write(tmp_path, {"metrics": {
            "defaults": {"peqs": {"enabled": True}}}}))
        assert "E024" in found
        assert issue_for(error, "E024").suggestion == "pesq"

    def test_E025_ber_is_not_valid_for_detection_reliability(self, tmp_path):
        found, error = codes(write(tmp_path, {
            "mode": "detection_reliability",
            "metrics": {"defaults": {"ber": {"enabled": True}}},
        }))
        assert "E025" in found
        assert "binary" in issue_for(error, "E025").message

    def test_E026_accuracy_cannot_be_disabled(self, tmp_path):
        found, _ = codes(write(tmp_path, {"metrics": {
            "defaults": {"accuracy": {"enabled": False}}}}))
        assert "E026" in found

    def test_E029_emr_takes_no_statistics(self, tmp_path):
        found, _ = codes(write(tmp_path, {"metrics": {
            "defaults": {"emr": {"statistics": ["mean"]}}}}))
        assert "E029" in found

    def test_E027_unknown_group_suggests_closest(self, tmp_path):
        found, error = codes(write(tmp_path, {"metrics": {
            "per_group": {"desync": {"pesq": {"enabled": False}}}}}))
        assert "E027" in found
        assert issue_for(error, "E027").suggestion == "desynchronization"

    def test_E028_per_group_is_meaningless_without_attacks(self, tmp_path):
        found, _ = codes(write(tmp_path, {
            "mode": "no_attacks",
            "metrics": {"per_group": {"audio_distortion": {}}},
        }))
        assert "E028" in found

    def test_other_is_a_valid_group_key(self, tmp_path):
        config = load_configs([write(tmp_path, {"metrics": {
            "per_group": {"other": {"pesq": {"enabled": False}}}}})],
            ATTACKS, MODELS)[0]
        assert config.resolver.is_enabled("other", "pesq") is False


class TestCropAndDurationCodes:
    def test_E030_crop_wrong_type(self, tmp_path):
        found, _ = codes(write(tmp_path, {"crop_before_attack": "10%"}))
        assert "E030" in found

    @pytest.mark.parametrize("value", [0, 100, 150, -5])
    def test_E031_crop_out_of_range(self, tmp_path, value):
        found, _ = codes(write(tmp_path, {"crop_before_attack": value}))
        assert "E031" in found

    def test_crop_in_range_is_accepted(self, tmp_path):
        config = load_configs(
            [write(tmp_path, {"crop_before_attack": 12.5})], ATTACKS, MODELS,
        )[0]
        assert config.crop_before_attack == 12.5

    def test_E032_boundary_not_positive(self, tmp_path):
        found, _ = codes(write(tmp_path, {"duration_groups": {
            "boundaries": [5, -2]}}))
        assert "E032" in found

    def test_E033_boundaries_out_of_order(self, tmp_path):
        found, _ = codes(write(tmp_path, {"duration_groups": {
            "boundaries": [30, 10]}}))
        assert "E033" in found

    def test_E034_duplicate_boundary(self, tmp_path):
        found, _ = codes(write(tmp_path, {"duration_groups": {
            "boundaries": [5, 5]}}))
        assert "E034" in found

    def test_duration_labels_describe_the_bins(self, tmp_path):
        config = load_configs([write(tmp_path, {"duration_groups": {
            "boundaries": [5, 10], "include_overall": True}})],
            ATTACKS, MODELS)[0]
        assert config.duration_labels() == ["< 5.0s", "5.0–10.0s", "≥ 10.0s"]
        assert config.duration_include_overall is True


class TestComparisonCodes:
    # Two models compared over two selected groups, one of which computes
    # accuracy's median but not its mean.
    NARROWED = {
        "models": ["AudioSealModel", "PerthModel"],
        "attacks": {"list": ["GaussianNoiseAttack", "LowpassFilterAttack"]},
        "metrics": {
            "defaults": {"accuracy": {"statistics": ["mean", "median"]}},
            "per_group": {
                "audio_editing": {"accuracy": {"statistics": ["median"]}},
            },
        },
    }

    def test_E035_unknown_primary_statistic(self, tmp_path):
        found, error = codes(write(tmp_path, {"comparison": {
            "primary_statistic": "medain"}}))
        assert "E035" in found
        assert issue_for(error, "E035").suggestion == "median"

    def test_E035_std_cannot_be_the_primary_statistic(self, tmp_path):
        """A spread has no better end for the main table to rank toward."""
        found, error = codes(write(tmp_path, {
            "statistics": ["mean", "std"],
            "comparison": {"primary_statistic": "std"},
        }))
        assert "E035" in found
        assert "spread" in issue_for(error, "E035").message

    def test_E036_primary_statistic_is_never_computed(self, tmp_path):
        found, error = codes(write(tmp_path, {
            "statistics": ["mean"],
            "comparison": {"primary_statistic": "p99"},
        }))
        assert "E036" in found
        assert "never computed" in issue_for(error, "E036").message

    def test_primary_statistic_within_the_configured_set_is_accepted(self, tmp_path):
        config = load_configs([write(tmp_path, {
            "statistics": ["mean", "worst_case"],
            "comparison": {"primary_statistic": "worst_case"},
        })], ATTACKS, MODELS)[0]
        assert config.comparison_primary_statistic == "worst_case"

    def test_the_derived_primary_is_a_level_not_a_spread(self, tmp_path):
        """Unset, it is the first configured statistic other than std."""
        config = load_configs([write(tmp_path, {
            "models": ["AudioSealModel", "PerthModel"],
            "metrics": {"defaults": {
                "accuracy": {"statistics": ["std", "mean"]}}},
        })], ATTACKS, MODELS)[0]
        assert config.comparison_primary_statistic == "mean"

    def test_W015_a_selected_group_leaves_the_primary_out(self, tmp_path):
        """Its rows read N/A in the main table, but every number is still
        reported, so the run goes ahead."""
        config = load_configs([write(tmp_path, {
            **self.NARROWED, "comparison": {"primary_statistic": "mean"},
        })], ATTACKS, MODELS)[0]

        note = next(w for w in config.warnings if w.code == "W015")
        assert note.path == "metrics.per_group.audio_editing.accuracy.statistics"
        assert config.comparison_primary_statistic == "mean"

    def test_W015_covers_the_derived_primary_too(self, tmp_path):
        config = load_configs([write(tmp_path, self.NARROWED)],
                              ATTACKS, MODELS)[0]
        assert [w.path for w in config.warnings if w.code == "W015"] == \
            ["metrics.per_group.audio_editing.accuracy.statistics"]

    def test_W015_names_only_the_tables_that_rank(self, tmp_path):
        """A std table is shown uncoloured, so no row is ranked in it."""
        config = load_configs([write(tmp_path, {
            **self.NARROWED,
            "metrics": {
                "defaults": {"accuracy": {"statistics": ["mean", "median"]}},
                "per_group": {"audio_editing": {
                    "accuracy": {"statistics": ["median", "std"]}}},
            },
        })], ATTACKS, MODELS)[0]

        note = next(w for w in config.warnings if w.code == "W015")
        assert "ranked only in the median table(s)" in note.message

    def test_W015_covers_an_ungrouped_attack_when_every_attack_runs(
            self, tmp_path):
        """With no selection every discovered attack runs, and an ungrouped
        one's rows sit under 'other'."""
        path = write(tmp_path, {
            "models": ["AudioSealModel", "PerthModel"],
            "metrics": {
                "defaults": {"accuracy": {"statistics": ["mean", "median"]}},
                "per_group": {"other": {"accuracy": {"statistics": ["median"]}}},
            },
        })

        def noted(registry):
            config = load_configs([path], registry, MODELS)[0]
            return [w.path for w in config.warnings if w.code == "W015"]

        assert noted(ATTACKS) == []
        assert noted({**ATTACKS, "UngroupedAttack": {"config": {}}}) == \
            ["metrics.per_group.other.accuracy.statistics"]

    def test_no_W015_without_a_comparison(self, tmp_path):
        """A single model has no comparison table to leave rows out of."""
        config = load_configs([write(tmp_path, {
            **self.NARROWED, "models": ["AudioSealModel"],
            "comparison": {"primary_statistic": "mean"},
        })], ATTACKS, MODELS)[0]
        assert not [w for w in config.warnings if w.code == "W015"]

    def test_no_W015_for_unselected_groups_or_subsections(self, tmp_path):
        """A group this run does not reach has no rows, and a subsection's
        accuracy statistics never reach the comparison."""
        config = load_configs([write(tmp_path, {
            **self.NARROWED,
            "attacks": {"list": ["LowpassFilterAttack"]},
            "metrics": {
                "defaults": {"accuracy": {"statistics": ["mean", "median"]}},
                "per_group": {
                    "audio_distortion": {"accuracy": {"statistics": ["median"]}},
                    "frequency_filtering": {
                        "accuracy": {"statistics": ["median"]}},
                },
            },
            "comparison": {"primary_statistic": "mean"},
        })], ATTACKS, MODELS)[0]
        assert not [w for w in config.warnings if w.code == "W015"]


class TestSeedAndLegacyCodes:
    @pytest.mark.parametrize("seed", [-1, 2**32])
    def test_E037_seed_outside_numpy_range(self, tmp_path, seed):
        found, _ = codes(write(tmp_path, {"general": {"seed": seed}}))
        assert "E037" in found

    @pytest.mark.parametrize("seed", [None, 0, 2**32 - 1])
    def test_seed_range_endpoints_are_valid(self, tmp_path, seed):
        config = load_configs([write(tmp_path, {"general": {"seed": seed}})])[0]
        assert config.general["seed"] == seed

    def test_E037_seed_must_be_an_integer(self, tmp_path):
        found, _ = codes(write(tmp_path, {"general": {"seed": "42"}}))
        assert "E037" in found

    def test_E038_pre_2_1_modes_block(self, tmp_path):
        found, error = codes(write(tmp_path, {
            "modes": {"benchmark": True, "no_attacks": False}}))
        assert "E038" in found
        assert "--init benchmark" in issue_for(error, "E038").message

    def test_E038_positional_statistic_string(self, tmp_path):
        found, _ = codes(write(tmp_path, {"metrics": {
            "pesq": {"enabled": True, "statistics": "mean:T std:F"}}}))
        assert "E038" in found

    def test_E038_attacks_source_key(self, tmp_path):
        found, _ = codes(write(tmp_path, {"attacks": {"source": "all"}}))
        assert "E038" in found


class TestAllProblemsAreReportedTogether:
    def test_one_pass_reports_every_error(self, tmp_path):
        """The whole point of the catalog: no fix-one-rerun-repeat loop."""
        found, error = codes(write(tmp_path, {
            "models": ["AudioSealMode"],
            "statistics": ["mean", "medain"],
            "crop_before_attack": 150,
            "metrics": {"defaults": {"peqs": {"enabled": True}}},
            "duration_groups": {"boundaries": [30, 10]},
        }))
        assert {"E011", "E021", "E031", "E024", "E033"} <= found
        assert len(error.issues) >= 5

    def test_errors_from_several_files_are_reported_together(self, tmp_path):
        first = write(tmp_path, {"models": ["Nope"]}, name="a.json")
        second = write(tmp_path, {"mode": "no_attacks",
                                  "statistics": ["nope"]}, name="b.json")
        found, error = codes([first, second])
        sources = {issue.source for issue in error.issues}
        assert len(sources) == 2
        assert {"E011", "E021"} <= found

    def test_every_issue_carries_a_code_and_a_path(self, tmp_path):
        _, error = codes(write(tmp_path, {
            "models": [], "statistics": ["nope"], "crop_before_attack": -1,
        }))
        for issue in error.issues:
            assert issue.code and issue.path
            assert issue.source
            assert issue.render().startswith(f"[{issue.code}]")


class TestWarningsDoNotBlockTheRun:
    def test_per_group_for_an_unselected_group_is_only_noted(self, tmp_path):
        """Requirement: an unused group section must not stop the run."""
        config = load_configs([write(tmp_path, {
            "attacks": {"groups": ["audio_distortion"]},
            "metrics": {"per_group": {"transmission": {"mcd": {"enabled": False}}}},
        })], DISTORTION, MODELS)[0]

        warned = [w for w in config.warnings if w.code == "W001"]
        assert warned and warned[0].severity == "info"
        assert "transmission" in warned[0].path

    def test_metrics_block_ignored_without_the_master_switch(self, tmp_path):
        config = load_configs([write(tmp_path, {
            "calculate_quality_metrics": False,
            "metrics": {"defaults": {"mcd": {"enabled": True}}},
        })], ATTACKS, MODELS)[0]

        assert any(w.code == "W002" for w in config.warnings)

    def test_partial_nisqa_selection_says_it_saves_nothing(self, tmp_path):
        config = load_configs([write(tmp_path, {
            "metrics": {"defaults": {
                "nisqa_mos": {"enabled": True},
            }},
        })], ATTACKS, MODELS)[0]

        note = next(w for w in config.warnings if w.code == "W003")
        assert "single NISQA request" in note.message

    def test_include_overall_without_boundaries_is_only_a_warning(self, tmp_path):
        config = load_configs([write(tmp_path, {"duration_groups": {
            "boundaries": [], "include_overall": True}})], ATTACKS, MODELS)[0]
        assert any(w.code == "W005" for w in config.warnings)

    def test_W016_robustness_metric_under_a_subsection(self, tmp_path):
        """Accuracy, BER and EMR are tabled per top-level group; a
        subsection splits only the signal-metric tables."""
        config = load_configs([write(tmp_path, {"metrics": {
            "defaults": {"accuracy": {"enabled": True}},
            "per_group": {"temporal_editing": {
                "accuracy": {"statistics": ["worst_case"]},
                "ber": {"enabled": False},
                "pesq": {"enabled": False},
            }},
        }})], ATTACKS, MODELS)[0]

        warned = sorted(w.path for w in config.warnings if w.code == "W016")
        assert warned == ["metrics.per_group.temporal_editing.accuracy",
                          "metrics.per_group.temporal_editing.ber"]

    def test_no_W016_on_a_top_level_group(self, tmp_path):
        config = load_configs([write(tmp_path, {"metrics": {
            "defaults": {"accuracy": {"enabled": True}},
            "per_group": {"audio_editing": {
                "accuracy": {"statistics": ["worst_case"]}}},
        }})], ATTACKS, MODELS)[0]
        assert not [w for w in config.warnings if w.code == "W016"]


class TestDocumentationKeysAreIgnored:
    def test_underscore_keys_are_not_validated(self, tmp_path):
        """JSON has no comments; '_'-prefixed keys are the convention."""
        config = load_configs([write(tmp_path, {
            "_comment": "anything at all",
            "_another": ["a", "list", "of", "notes"],
            "metrics": {
                "_note": "explains the block",
                "defaults": {"_why": "explains pesq", "pesq": {"enabled": True}},
            },
        })], ATTACKS, MODELS)[0]
        assert config.resolver.is_enabled(None, "pesq") is True


class TestTemplates:
    @pytest.mark.parametrize("mode", VALID_MODES)
    def test_init_prints_a_file_that_validates(self, tmp_path, mode):
        path = tmp_path / f"{mode}.json"
        path.write_text(init_template(mode))
        config = load_configs([str(path)])[0]
        assert config.mode == mode

    @pytest.mark.parametrize("mode", VALID_MODES)
    def test_init_output_is_documented(self, mode):
        raw = json.loads(init_template(mode))
        documented = [k for k in raw if k.startswith("_")]
        assert len(documented) >= 5, "template carries no inline instructions"

    @pytest.mark.parametrize("mode", VALID_MODES)
    def test_packaged_template_is_what_init_prints(self, mode):
        """--init must print the packaged file, not a reconstruction of it."""
        path = f"src/deepmarkpy/config_templates/{mode}.json"
        with open(path, encoding="utf-8") as fh:
            assert fh.read() == init_template(mode)

    @pytest.mark.parametrize("mode", VALID_MODES)
    def test_the_repo_example_still_validates(self, mode):
        """configs/ holds working files, edited by whoever runs the benchmark.

        They are expected to diverge from the template -- that is the point
        of them -- but a broken example is worth catching, so this checks
        they load rather than that they match byte for byte.
        """
        config = load_configs([f"configs/{mode}.json"])[0]
        assert config.mode == mode

    @pytest.mark.parametrize("mode", ["benchmark", "detection_reliability"])
    def test_template_attack_list_is_the_discovered_set(self, mode):
        """The names and counts a template documents are what discovery finds."""
        text = init_template(mode)
        discovered = sorted(PluginManager().get_attacks())
        listed = json.loads(text)["_attacks_available"].split(": ", 1)[1]
        assert listed.rstrip(".").split(", ") == discovered
        for count in re.findall(r"all (\d+)", text):
            assert int(count) == len(discovered)

    def test_init_rejects_an_unknown_mode(self):
        with pytest.raises(ConfigError) as excinfo:
            init_template("benchmrak")
        assert excinfo.value.issues[0].suggestion == "benchmark"

    @pytest.mark.parametrize("mode", VALID_MODES)
    def test_template_only_uses_keys_its_mode_accepts(self, mode):
        raw = json.loads(init_template(mode))
        real = {k for k in raw if not k.startswith("_")}
        assert real <= MODE_KEYS[mode]

    def test_no_attacks_template_omits_attack_sections_entirely(self):
        raw = json.loads(init_template("no_attacks"))
        for key in ("attacks", "attack_parameters", "crop_before_attack",
                    "comparison"):
            assert key not in raw, (
                f"{key} is not valid in no_attacks mode, so the template must "
                f"not show it at all"
            )
        assert "per_group" not in json.dumps(raw.get("metrics", {}))

    def test_detection_reliability_template_omits_ber(self):
        raw = json.loads(init_template("detection_reliability"))
        assert "ber" not in raw["metrics"]["defaults"]


class TestSelectionCoverage:
    """An entry that silently does nothing must be named, not ignored."""

    def test_W011_parameters_for_an_unselected_attack(self, tmp_path):
        config = load_configs([write(tmp_path, {
            "attacks": {"list": ["GaussianNoiseAttack"]},
            "attack_parameters": {"EchoAttack": {"delay_echo": 0.2}},
        })], ATTACKS, MODELS)[0]

        note = next(w for w in config.warnings if w.code == "W011")
        assert "EchoAttack" in note.path
        assert note.severity == "info"

    def test_W011_a_defined_version_that_is_not_selected(self, tmp_path):
        """Defining a version is deliberate work; say if it will not run."""
        config = load_configs([write(tmp_path, {
            "attacks": {"list": ["GaussianNoiseAttack:mild"]},
            "attack_parameters": {
                "GaussianNoiseAttack:brutal": {"snr_db_gaussian_noise": 5},
            },
        })], ATTACKS, MODELS)[0]

        note = next(w for w in config.warnings if w.code == "W011")
        assert "brutal" in note.path

    def test_no_warning_when_the_attack_is_selected_bare(self, tmp_path):
        """A bare name runs every version, so a defined one is reachable."""
        config = load_configs([write(tmp_path, {
            "attacks": {"list": ["GaussianNoiseAttack"]},
            "attack_parameters": {
                "GaussianNoiseAttack:brutal": {"snr_db_gaussian_noise": 5},
            },
        })], ATTACKS, MODELS)[0]
        assert not [w for w in config.warnings if w.code == "W011"]

    def test_no_warning_when_reached_through_a_group(self, tmp_path):
        config = load_configs([write(tmp_path, {
            "attacks": {"groups": ["audio_distortion"]},
            "attack_parameters": {
                "GaussianNoiseAttack": {"snr_db_gaussian_noise": 20},
            },
        })], DISTORTION, MODELS)[0]
        assert not [w for w in config.warnings if w.code == "W011"]

    def test_no_warning_when_everything_runs(self, tmp_path):
        """An empty selection runs all attacks, so nothing is unreachable."""
        config = load_configs([write(tmp_path, {
            "attack_parameters": {"EchoAttack": {"delay_echo": 0.2}},
        })], ATTACKS, MODELS)[0]
        assert not [w for w in config.warnings if w.code == "W011"]

    def test_W011_detection_reliability_empty_selection_runs_no_attacks(
            self, tmp_path):
        """In this mode an empty selection measures the no-attack baseline
        alone, so no attack's parameters are applied."""
        config = load_configs([write(tmp_path, {
            "mode": "detection_reliability",
            "attacks": {"groups": [], "list": []},
            "attack_parameters": {"EchoAttack": {"delay_echo": 0.2}},
        })], ATTACKS, MODELS)[0]

        note = next(w for w in config.warnings if w.code == "W011")
        assert "EchoAttack" in note.path
        assert note.severity == "info"


class TestParkedEntries:
    """An underscore parks an entry without deleting it.

    JSON has no way to comment code out, so the _-prefix doubles as an
    on/off switch. What matters is that a parked entry is inert *and*
    that selecting it is then an error rather than a silent no-op.
    """

    def test_a_parked_version_is_not_defined(self, tmp_path):
        config = load_configs([write(tmp_path, {
            "attacks": {"list": ["GaussianNoiseAttack"]},
            "attack_parameters": {
                "_GaussianNoiseAttack:insane": {"snr_db_gaussian_noise": 1},
            },
        })], ATTACKS, MODELS)[0]

        assert config.synthetic_versions == {}
        assert config.parameters_for("GaussianNoiseAttack", "insane") == {}
        # Silent about the parked entry specifically: the parser never looks
        # inside a _ key, so there is nothing to report about it.
        assert not [
            w for w in config.warnings if w.code in ("W010", "W011")
        ], "a parked entry should raise no attack_parameters warning"

    def test_a_parked_override_does_not_apply(self, tmp_path):
        config = load_configs([write(tmp_path, {
            "attacks": {"list": ["GaussianNoiseAttack"]},
            "attack_parameters": {
                "_GaussianNoiseAttack": {"snr_db_gaussian_noise": 1},
            },
        })], ATTACKS, MODELS)[0]
        assert config.parameters_for("GaussianNoiseAttack", "default") == {}

    def test_selecting_a_parked_version_is_an_error(self, tmp_path):
        """The one thing that must not happen is a silent no-op."""
        found, error = codes(write(tmp_path, {
            "attacks": {"list": ["GaussianNoiseAttack:insane"]},
            "attack_parameters": {
                "_GaussianNoiseAttack:insane": {"snr_db_gaussian_noise": 1},
            },
        }))
        assert "E015" in found
        assert "insane" in issue_for(error, "E015").value

    def test_unparking_defines_the_version(self, tmp_path):
        """The same file with the underscore removed."""
        config = load_configs([write(tmp_path, {
            "attacks": {"list": ["GaussianNoiseAttack:insane"]},
            "attack_parameters": {
                "GaussianNoiseAttack:insane": {"snr_db_gaussian_noise": 1},
            },
        })], ATTACKS, MODELS)[0]

        assert config.synthetic_versions == {
            "GaussianNoiseAttack": {"insane": {"snr_db_gaussian_noise": 1}},
        }
        assert config.attack_list == ["GaussianNoiseAttack:insane"]


class TestJsonSyntaxErrorsAreLocatable:
    """A broken file must say where and why, not just that it broke.

    Python's own message describes the parser's state ("Expecting
    property name enclosed in double quotes"), which does not tell you a
    stray comma is the problem.
    """

    @staticmethod
    def _issue(tmp_path, text):
        path = tmp_path / "c.json"
        path.write_text(text)
        with pytest.raises(ConfigError) as excinfo:
            load_configs([str(path)], quiet=True)
        return excinfo.value.issues[0]

    def test_trailing_comma_in_an_object_is_named(self, tmp_path):
        issue = self._issue(tmp_path, '{\n  "mode": "benchmark",\n}\n')
        assert issue.code == "E002"
        assert "trailing comma" in issue.message

    def test_trailing_comma_in_an_array_is_named(self, tmp_path):
        issue = self._issue(tmp_path, '{\n  "models": ["A",]\n}\n')
        assert "trailing comma" in issue.message

    def test_the_offending_line_is_shown_with_a_caret(self, tmp_path):
        issue = self._issue(
            tmp_path, '{\n  "mode": "benchmark"\n  "models": ["A"]\n}\n')
        assert '"models": ["A"]' in issue.message
        assert "^" in issue.message

    def test_unbalanced_braces_are_counted(self, tmp_path):
        """The failure point is usually far from the unclosed brace."""
        issue = self._issue(
            tmp_path, '{\n  "a": {\n    "b": {}\n}\n')
        assert "do not balance" in issue.message

    def test_a_comment_says_to_use_underscore_keys(self, tmp_path):
        issue = self._issue(tmp_path, '{\n  // note\n  "mode": "benchmark"\n}\n')
        assert '"_"' in issue.message

    def test_single_quotes_are_named(self, tmp_path):
        issue = self._issue(tmp_path, "{\n  'mode': 'benchmark'\n}\n")
        assert "double quotes" in issue.message

    def test_an_empty_file_says_so(self, tmp_path):
        assert "empty" in self._issue(tmp_path, "").message

    def test_the_line_and_column_are_reported(self, tmp_path):
        issue = self._issue(tmp_path, '{\n  "mode": "benchmark",\n}\n')
        assert issue.path.startswith("line 3")


class TestSilentlyEmptyMetricsIsWarned:
    def test_W012_metrics_without_defaults(self, tmp_path):
        """per_group alone leaves every unnamed metric off."""
        config = load_configs([write(tmp_path, {"metrics": {
            "per_group": {"audio_distortion": {"pesq": {"enabled": True}}},
        }})], ATTACKS, MODELS)[0]

        note = next(w for w in config.warnings if w.code == "W012")
        assert "off" in note.message
        assert config.resolver.is_enabled("audio_distortion", "pesq") is True
        assert config.resolver.is_enabled("desynchronization", "pesq") is False

    def test_no_warning_when_defaults_is_present(self, tmp_path):
        config = load_configs([write(tmp_path, {"metrics": {
            "defaults": {"pesq": {"enabled": True}},
        }})], ATTACKS, MODELS)[0]
        assert not [w for w in config.warnings if w.code == "W012"]


class TestValidatingWithoutAFile:
    """An application that builds the configuration itself validates it here.

    A project embedding the benchmark assembles a config from its own UI
    and needs the same answer the CLI would give -- before writing
    anything to disk, so it can put the error in front of its own user.
    The two paths must not drift: the file loader and this one run the
    same checks over the same mapping.
    """

    def test_a_valid_mapping_returns_a_config(self):
        config = load_config_data(dict(BASE), quiet=True)
        assert config.mode == "benchmark"
        assert config.models == ["AudioSealModel"]

    def test_no_file_is_touched(self):
        """The source is a label, not a path, so it need not exist."""
        config = load_config_data(dict(BASE), source="wizard step 3", quiet=True)
        assert config.source == "wizard step 3"

    def test_it_raises_the_same_code_a_file_would(self, tmp_path):
        data = {**BASE, "attacks": {"groups": ["desync"], "list": []}}

        with pytest.raises(ConfigError) as from_memory:
            load_config_data(data, quiet=True)
        with pytest.raises(ConfigError) as from_file:
            load_configs([write(tmp_path, {"attacks": {"groups": ["desync"],
                                                       "list": []}})])

        assert [i.code for i in from_memory.value.issues] == \
               [i.code for i in from_file.value.issues] == ["E013"]
        assert from_memory.value.issues[0].suggestion == "desynchronization"

    def test_the_source_label_is_carried_into_the_issue(self):
        with pytest.raises(ConfigError) as error:
            load_config_data({**BASE, "mode": "nonsense"}, source="wizard")
        assert all(i.source == "wizard" for i in error.value.issues)

    def test_it_resolves_metrics_the_same_way_a_file_does(self, tmp_path):
        overrides = {
            "statistics": ["mean", "worst_case"],
            "metrics": {"defaults": {"pesq": {"enabled": True},
                                     "visqol": {"enabled": False}}},
        }
        from_memory = load_config_data({**BASE, **overrides}, quiet=True)
        from_file = load_configs([write(tmp_path, overrides)], quiet=True)[0]

        for group in (None, "audio_distortion", "desynchronization"):
            assert from_memory.resolver.metrics_for_group(group) == \
                   from_file.resolver.metrics_for_group(group)

    def test_a_non_mapping_is_rejected_not_crashed_on(self):
        with pytest.raises(ConfigError) as error:
            load_config_data(["not", "a", "mapping"])
        assert [i.code for i in error.value.issues] == ["E003"]

    def test_registry_checks_apply_from_memory_too(self):
        """The plugin registries are how an unknown attack is caught, and
        they reach this path exactly as they reach the file one."""
        with pytest.raises(ConfigError) as error:
            load_config_data(
                {**BASE, "attacks": {"groups": [], "list": ["NoSuchAttack"]}},
                attacks_registry={"GaussianNoiseAttack": {}},
                models_registry={"AudioSealModel": {}},
            )
        assert "E014" in [i.code for i in error.value.issues]


class TestEfficiencyMetricsBelongToTheirOwnSection:
    """Naming one under ``metrics`` is an error, not a flag that does nothing.

    They are in the canonical metric order, so the metrics block would take
    them without complaint -- but ``is_enabled`` reads them from the
    efficiency section, so the flag would have no effect.
    """

    def test_E043_efficiency_metric_in_the_metrics_block(self, tmp_path):
        path = write(tmp_path, {"metrics": {"defaults": {
            "embed_latency": {"enabled": True},
        }}})
        found, error = codes(path)
        assert "E043" in found
        assert "efficiency.metrics.embed_latency" in \
            issue_for(error, "E043").message

    def test_E043_applies_to_a_group_section_too(self, tmp_path):
        path = write(tmp_path, {"metrics": {"per_group": {
            "audio_distortion": {"attack_latency": {"enabled": False}},
        }}})
        found, _ = codes(path)
        assert "E043" in found

    def test_the_efficiency_section_itself_still_accepts_them(self, tmp_path):
        path = write(tmp_path, {"efficiency": {
            "enabled": True,
            "metrics": {"embed_latency": {"enabled": True}},
        }})
        config = load_configs([path], attacks_registry=ATTACKS,
                              models_registry=MODELS, quiet=True)[0]
        assert config.resolver.is_enabled(None, "embed_latency") is True


class TestABareAttackNameAndItsDefaultVersionAreOneTarget:
    """Which spelling each side uses is not a fact they share.

    The validator keys a bare name by how many versions the plugin
    declares; the run loop asks by how ``attacks.list`` spelled the
    attack. A mismatch dropped the override without a word.
    """

    SINGLE_VERSION = {"LowpassFilterAttack": {
        "config": {"cutoff_lowpass": 4000},
        "_raw_config": {"cutoff_lowpass": 4000},
    }}

    def _config(self, tmp_path, parameters):
        path = write(tmp_path, {
            "attacks": {"list": ["LowpassFilterAttack"]},
            "attack_parameters": parameters,
        })
        return load_configs([path], attacks_registry=self.SINGLE_VERSION,
                            models_registry=MODELS, quiet=True)[0]

    def test_a_default_suffixed_key_reaches_the_bare_lookup(self, tmp_path):
        config = self._config(tmp_path, {
            "LowpassFilterAttack:default": {"cutoff_lowpass": 3000},
        })
        assert config.parameters_for("LowpassFilterAttack", None) == \
            {"cutoff_lowpass": 3000}

    def test_a_bare_key_reaches_the_default_suffixed_lookup(self, tmp_path):
        config = self._config(tmp_path, {
            "LowpassFilterAttack": {"cutoff_lowpass": 2500},
        })
        assert config.parameters_for("LowpassFilterAttack", "default") == \
            {"cutoff_lowpass": 2500}

    def test_a_bare_key_still_does_not_reach_a_named_version(self, tmp_path):
        """The point of keying by version in the first place."""
        path = write(tmp_path, {
            "attacks": {"list": ["GaussianNoiseAttack"]},
            "attack_parameters": {
                "GaussianNoiseAttack": {"snr_db_gaussian_noise": 33},
            },
        })
        config = load_configs([path], attacks_registry=ATTACKS,
                              models_registry=MODELS, quiet=True)[0]
        assert config.parameters_for("GaussianNoiseAttack", "default") == \
            {"snr_db_gaussian_noise": 33}
        assert config.parameters_for("GaussianNoiseAttack", "mild") == {}

    @pytest.mark.parametrize("listed", ["LowpassFilterAttack",
                                        "LowpassFilterAttack:default"])
    def test_E016_both_spellings_set_for_a_single_version_attack(
            self, tmp_path, listed):
        """Two entries for one target: neither may quietly win, whichever
        spelling attacks.list uses."""
        found, error = codes(write(tmp_path, {
            "attacks": {"list": [listed]},
            "attack_parameters": {
                "LowpassFilterAttack": {"cutoff_lowpass": 2500},
                "LowpassFilterAttack:default": {"cutoff_lowpass": 3000},
            },
        }), attacks_registry=self.SINGLE_VERSION)
        assert "E016" in found
        assert issue_for(error, "E016").path == \
            "attack_parameters.LowpassFilterAttack:default"

    def test_E016_both_spellings_set_for_a_multi_version_attack(self, tmp_path):
        found, error = codes(write(tmp_path, {"attack_parameters": {
            "GaussianNoiseAttack": {"snr_db_gaussian_noise": 20},
            "GaussianNoiseAttack:default": {"snr_db_gaussian_noise": 5},
        }}))
        assert "E016" in found
        assert issue_for(error, "E016").path == \
            "attack_parameters.GaussianNoiseAttack:default"

    def test_E016_needs_no_plugin_registry(self, tmp_path):
        """The pass before plugins are imported reports it already."""
        found, _ = codes(write(tmp_path, {"attack_parameters": {
            "EchoAttack": {"delay_echo": 0.2},
            "EchoAttack:default": {"delay_echo": 0.3},
        }}), attacks_registry=None, models_registry=None)
        assert "E016" in found


class TestDefiningAVersionIgnoresDocumentationKeys:
    """W010 does not count ``_``-prefixed comment keys as unset parameters.

    Counted, a plugin whose config.json carries one could never have a
    version defined for it: every parameter given, and the entry still
    skipped as partial.
    """

    DOCUMENTED = {"Codec2VocoderAttack": {
        "config": {"_comment": "supported bitrates", "bitrate_codec2": [700]},
        "_raw_config": {"_comment": "supported bitrates",
                        "bitrate_codec2": [700]},
    }}

    def test_a_fully_specified_version_is_defined(self, tmp_path):
        path = write(tmp_path, {
            "attacks": {"list": ["Codec2VocoderAttack:tiny"]},
            "attack_parameters": {
                "Codec2VocoderAttack:tiny": {"bitrate_codec2": [700]},
            },
        })
        config = load_configs([path], attacks_registry=self.DOCUMENTED,
                              models_registry=MODELS, quiet=True)[0]
        assert config.synthetic_versions["Codec2VocoderAttack"] == \
            {"tiny": {"bitrate_codec2": [700]}}

    def test_a_partly_specified_version_is_still_skipped(self, tmp_path):
        registry = {"TwoParameterAttack": {
            "config": {"_comment": "docs", "a": 1, "b": 2},
            "_raw_config": {"_comment": "docs", "a": 1, "b": 2},
        }}
        path = write(tmp_path, {
            "attack_parameters": {"TwoParameterAttack:tiny": {"a": 5}},
        })
        config = load_configs([path], attacks_registry=registry,
                              models_registry=MODELS, quiet=True)[0]

        assert config.synthetic_versions == {}
        warned = next(w for w in config.warnings if w.code == "W010")
        # Names the parameter that is actually missing, not the comment.
        assert "b" in warned.message
        assert "_comment" not in warned.message
