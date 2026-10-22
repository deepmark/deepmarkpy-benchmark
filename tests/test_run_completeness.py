"""A run must not quietly do less than it was asked to: a port set in .env
reaches the host clients, and an attack, version or second model that cannot
run stops the run rather than being skipped."""

import os

import pytest

from deepmarkpy.benchmark import Benchmark
from deepmarkpy.utils.utils import load_env_file


class TestEnvFileReachesTheHost:
    def test_values_are_loaded(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("# ports\nVAE_PORT=19999\nAUDIOSEAL_PORT=18888\n")
        monkeypatch.delenv("VAE_PORT", raising=False)
        monkeypatch.delenv("AUDIOSEAL_PORT", raising=False)

        applied = load_env_file(str(env))

        assert applied == {"VAE_PORT": "19999", "AUDIOSEAL_PORT": "18888"}
        assert os.environ["VAE_PORT"] == "19999"

    def test_quotes_and_spaces_are_stripped(self, tmp_path, monkeypatch):
        # The shipped .env writes HOST with both.
        env = tmp_path / ".env"
        env.write_text('HOST = "0.0.0.0"\n')
        monkeypatch.delenv("HOST", raising=False)

        load_env_file(str(env))

        assert os.environ["HOST"] == "0.0.0.0"

    def test_real_environment_wins(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("VAE_PORT=19999\n")
        monkeypatch.setenv("VAE_PORT", "12345")

        applied = load_env_file(str(env))

        assert applied == {}, "the file overrode an explicit export"
        assert os.environ["VAE_PORT"] == "12345"

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_env_file(str(tmp_path / "nope.env")) == {}

    @pytest.mark.parametrize("line", ["", "   ", "# comment", "NOEQUALS"])
    def test_junk_lines_are_skipped(self, tmp_path, line):
        env = tmp_path / ".env"
        env.write_text(line + "\n")
        assert load_env_file(str(env)) == {}


class TestMissingAttacksAreFatal:
    """A requested attack that is not loadable must stop the run.

    These exercise the path a user actually takes. An earlier version of this
    class only called ``_require_attacks_available`` directly with an
    unfiltered list, which passed while ``--attack_groups`` was still dropping
    unavailable attacks before the guard could see them.
    """

    def _benchmark(self, failed=None):
        bench = Benchmark.__new__(Benchmark)
        bench.attacks = {"GaussianNoiseAttack": {}, "EchoAttack": {}}
        bench.models = {"FakeModel": {}}
        bench.plugin_manager = type("PM", (), {"failed": failed or {}})()
        return bench

    def test_requested_but_absent_attack_raises(self):
        bench = self._benchmark()
        with pytest.raises(ValueError, match="not available"):
            bench._require_attacks_available(["GaussianNoiseAttack", "WaveletAttack"])

    def test_import_failure_reason_is_surfaced(self):
        bench = self._benchmark(
            {"deepmarkpy.plugins.attacks.wavelet.attack": "No module named 'pywt'"}
        )
        with pytest.raises(ValueError, match="No module named 'pywt'"):
            bench._require_attacks_available(["WaveletAttack"])

    def test_all_present_is_silent(self):
        self._benchmark()._require_attacks_available(["GaussianNoiseAttack", "EchoAttack"])

    def test_run_rejects_an_unavailable_attack_before_touching_files(self):
        """run() must refuse the request, not quietly measure a smaller set."""
        bench = self._benchmark()
        with pytest.raises(ValueError, match="WaveletAttack"):
            bench.run(filepaths=["/nonexistent.wav"], wm_model="FakeModel",
                      attack_types=["GaussianNoiseAttack", "WaveletAttack"])

    def test_run_refuses_an_empty_explicit_request(self):
        """An explicitly empty set must not fall back to the whole registry."""
        bench = self._benchmark()
        with pytest.raises(ValueError, match="empty"):
            bench.run(filepaths=["/nonexistent.wav"], wm_model="FakeModel",
                      attack_types=[])

    def test_run_with_no_request_still_uses_every_attack(self):
        """The default path must keep working: None means all."""
        bench = self._benchmark()
        # Fails on the missing model, which proves it got past attack selection.
        with pytest.raises(ValueError, match="Model"):
            bench.run(filepaths=["/nonexistent.wav"], wm_model="NoSuchModel",
                      attack_types=None)


class TestAttackGroupsReachTheGuard:
    """--attack_groups must hand its resolved list over unfiltered."""

    def test_config_group_resolution_does_not_filter_against_the_registry(self):
        """Group expansion in the config layer leaves every member for the guard."""
        from deepmarkpy.config import ModeConfig
        from deepmarkpy.utils.attack_groups import ATTACK_GROUPS

        config = ModeConfig(mode="benchmark", source="c.json",
                            attack_groups=["audio_editing"])
        specs = config.selected_attack_specs()

        assert set(specs) == set(ATTACK_GROUPS["audio_editing"]["attacks"]), (
            "config group resolution filters against the plugin registry; "
            "unavailable attacks are dropped before Benchmark.run can object"
        )

    def test_empty_selection_means_every_attack_not_none_of_them(self):
        """No groups and no list is 'run everything', which run() expands."""
        from deepmarkpy.config import ModeConfig

        config = ModeConfig(mode="benchmark", source="c.json")
        assert config.selected_attack_specs() is None

    def test_explicit_list_and_groups_combine_without_duplicates(self):
        from deepmarkpy.config import ModeConfig

        config = ModeConfig(
            mode="benchmark", source="c.json",
            attack_groups=["audio_distortion"],
            attack_list=["GaussianNoiseAttack", "ReplayAttack"],
        )
        specs = config.selected_attack_specs()
        assert specs.count("GaussianNoiseAttack") == 1
        assert "ReplayAttack" in specs
        assert "PinkNoiseAttack" in specs

    def test_group_resolution_returns_declared_attacks_not_discovered_ones(self):
        from deepmarkpy.utils.attack_groups import ATTACK_GROUPS, get_attacks_for_groups

        group = "audio_editing"
        resolved = get_attacks_for_groups([group])
        assert set(resolved) == set(ATTACK_GROUPS[group]["attacks"]), (
            "group resolution must reflect what the group declares, so a "
            "missing plugin is visible rather than absent"
        )


class TestCrossModelReceivesItsSecondModel:
    """The run loop hands the cross-model attack its second model's name."""

    def test_the_resolved_name_is_handed_to_the_attack(self):
        import inspect

        from deepmarkpy import benchmark as benchmark_module

        # Both executors apply an attack through _apply_entry.
        source = inspect.getsource(benchmark_module.Benchmark._apply_entry)
        assert 'current_attack_kwargs[\n' \
               '                "different_model_name_cross_model"]' in source \
            or '"different_model_name_cross_model"] = different_model_name' in source, (
                "the resolved name is not passed to the attack"
            )

    def test_the_plugin_default_second_model_is_discovered(self):
        benchmark = Benchmark()
        entry = benchmark.attacks["CrossModelAttack"]
        default = (entry.get("config") or {}).get(
            "different_model_name_cross_model")
        assert default in benchmark.models, (
            f"the plugin default {default!r} is not a discovered model"
        )

    def test_an_unknown_second_model_stops_the_run_before_any_audio(
        self, tmp_path, monkeypatch,
    ):
        """The plugin's unknown default name stops the run before any audio."""
        import numpy as np
        import soundfile as sf

        from deepmarkpy.utils.metric_resolver import MetricResolver

        embedded = []

        class _Model:
            def generate_watermark(self):
                return np.ones(16, dtype=np.int32)

            def embed(self, audio, watermark_data, sampling_rate):
                embedded.append(sampling_rate)
                return audio

            def detect(self, audio, sampling_rate):
                return np.ones(16, dtype=np.int32)

        path = tmp_path / "a.wav"
        sf.write(str(path), np.zeros(16000, dtype=np.float32), 16000)

        benchmark = Benchmark()
        monkeypatch.setitem(
            benchmark.attacks["CrossModelAttack"], "config",
            {"different_model_name_cross_model": "NotAModel"},
        )
        monkeypatch.setitem(benchmark.models, "StubModel", {
            "class": _Model, "config": {"sampling_rate": 16000},
        })

        with pytest.raises(ValueError, match="different_model_name_cross_model"):
            benchmark.run(
                filepaths=[str(path)], wm_model="StubModel",
                attack_types=["GaussianNoiseAttack", "CrossModelAttack"],
                metric_resolver=MetricResolver(),
            )
        assert embedded == [], "audio was embedded before the name was checked"


class TestAVersionIsNeverSilentlyDropped:
    """An attack either takes the requested version or says it cannot."""

    class _TakesVersion:
        def __init__(self, version=None):
            self.version = version

    class _TakesNone:
        def __init__(self):
            self.version = "default-preset"

    class _RaisesInside:
        def __init__(self, version=None):
            raise TypeError("a bug in the plugin's own constructor")

    def test_a_versioned_attack_receives_its_version(self):
        from deepmarkpy.benchmark import instantiate_attack

        built = instantiate_attack(self._TakesVersion, "X", "aggressive")
        assert built.version == "aggressive"

    def test_a_versionless_attack_asked_for_a_version_is_refused(self):
        """Running its default preset would label the row as data it is not."""
        from deepmarkpy.benchmark import instantiate_attack

        with pytest.raises(ValueError, match="does not support versions"):
            instantiate_attack(self._TakesNone, "X", "aggressive")

    def test_a_versionless_attack_asked_for_the_default_is_silent(self, caplog):
        from deepmarkpy.benchmark import instantiate_attack

        with caplog.at_level("WARNING"):
            instantiate_attack(self._TakesNone, "X", "default")
        assert "does not support versions" not in caplog.text

    def test_a_constructor_bug_is_not_mistaken_for_a_missing_version(self):
        from deepmarkpy.benchmark import instantiate_attack

        with pytest.raises(TypeError, match="bug in the plugin"):
            instantiate_attack(self._RaisesInside, "X", "mild")


class TestAFailingAttackIsNamed:
    """A failure must say which attack and file, and keep its own type.

    run_multiple_models skips a model on OSError/ConnectionError/MemoryError
    (infrastructure) and stops on anything else (a bug), so the exception
    has to reach it unchanged; only the log gains the attack and file.
    """

    def _run(self, tmp_path, exc):
        import numpy as np
        import soundfile as sf

        from deepmarkpy.core.base_attack import BaseAttack

        class FakeModel:
            config = {"sampling_rate": 16000, "watermark_size": 16}

            def generate_watermark(self):
                return np.zeros(16, dtype=np.int32)

            def embed(self, audio, watermark_data, sampling_rate):
                return np.asarray(audio, dtype=np.float64)

            def detect(self, audio, sampling_rate):
                return np.zeros(16, dtype=np.int32)

        class BoomAttack(BaseAttack):
            def apply(self, audio, **kwargs):
                raise exc

        wav = tmp_path / "clip.wav"
        sf.write(wav, 0.1 * np.random.default_rng(0).standard_normal(16000), 16000)
        bench = Benchmark.__new__(Benchmark)
        bench.attacks = {"BoomAttack": {"class": BoomAttack, "config": {}}}
        bench.models = {"FakeModel": {"class": FakeModel, "config": FakeModel.config}}
        bench.plugin_manager = type("PM", (), {"failed": {}})()
        bench.run(filepaths=[str(wav)], wm_model="FakeModel", attack_types=["BoomAttack"],
                  calculate_quality_metrics=False)

    def test_an_infrastructure_failure_keeps_its_type(self, tmp_path, caplog):
        with caplog.at_level("ERROR"), pytest.raises(OSError, match="service gone"):
            self._run(tmp_path, OSError("service gone"))
        assert "Attack BoomAttack failed on" in caplog.text and "clip.wav" in caplog.text

    def test_a_bug_keeps_its_type(self, tmp_path, caplog):
        with caplog.at_level("ERROR"), pytest.raises(ValueError, match="buffer source array is read-only"):
            self._run(tmp_path, ValueError("buffer source array is read-only"))
        assert "Attack BoomAttack failed on" in caplog.text
