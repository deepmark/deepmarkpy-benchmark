"""A run must not quietly do less than it was asked to.

Covers the two ways that used to happen: a port set in .env never reaching
the host clients, and a requested attack whose plugin failed to import being
warned about and skipped while the run still exited 0.
"""

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
        """An explicitly empty set must not fall back to the whole registry.

        This is what a group whose plugins all failed to import used to
        produce: asking for one group and silently getting every attack.
        """
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
    """--attack_groups must hand its resolved list over unfiltered.

    Filtering unavailable attacks out in run.py left the guard with nothing to
    catch, so a group ran short silently; when every plugin in a group failed,
    the empty list fell through to "run everything".
    """

    def test_run_py_does_not_filter_group_results(self):
        import inspect
        from deepmarkpy import run as run_module

        src = inspect.getsource(run_module.main)
        start = src.index("if args.attack_groups:")
        block = src[start:start + 600]
        assert "available = set(attacks)" not in block, (
            "run.py filters group-resolved attacks against the registry again; "
            "unavailable ones are dropped before Benchmark.run can object"
        )

    def test_group_resolution_returns_declared_attacks_not_discovered_ones(self):
        from deepmarkpy.utils.attack_groups import ATTACK_GROUPS, get_attacks_for_groups

        group = "audio_editing"
        resolved = get_attacks_for_groups([group])
        assert set(resolved) == set(ATTACK_GROUPS[group]["attacks"]), (
            "group resolution must reflect what the group declares, so a "
            "missing plugin is visible rather than absent"
        )
