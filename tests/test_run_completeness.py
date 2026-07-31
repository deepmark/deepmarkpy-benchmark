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
    """--attack_groups resolves from a static table, so it can name an attack
    whose plugin never imported. That must stop the run, not shrink it."""

    def _benchmark(self):
        bench = Benchmark.__new__(Benchmark)
        bench.attacks = {"GaussianNoiseAttack": {}, "EchoAttack": {}}
        bench.plugin_manager = type("PM", (), {"failed": {}})()
        return bench

    def test_requested_but_absent_attack_raises(self):
        bench = self._benchmark()
        with pytest.raises(ValueError, match="not available"):
            bench._require_attacks_available(["GaussianNoiseAttack", "WaveletAttack"])

    def test_error_names_the_missing_attack(self):
        bench = self._benchmark()
        with pytest.raises(ValueError, match="WaveletAttack"):
            bench._require_attacks_available(["WaveletAttack"])

    def test_import_failure_reason_is_surfaced(self):
        bench = self._benchmark()
        bench.plugin_manager.failed = {
            "deepmarkpy.plugins.attacks.wavelet.attack": "No module named 'pywt'"
        }
        with pytest.raises(ValueError, match="No module named 'pywt'"):
            bench._require_attacks_available(["WaveletAttack"])

    def test_all_present_is_silent(self):
        bench = self._benchmark()
        bench._require_attacks_available(["GaussianNoiseAttack", "EchoAttack"])

    def test_empty_request_is_silent(self):
        self._benchmark()._require_attacks_available([])
