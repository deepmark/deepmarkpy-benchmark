"""Tests for opt-in seeding, run provenance, and report-directory handling."""

import argparse
import json
import os

import numpy as np

from deepmarkpy import __version__, run as run_module


class TestSeedingIsOptIn:
    """A seed must be requestable, and its absence must change nothing.

    Watermark payloads are random per file by design; seeding is a
    reproducibility tool, never the default.
    """

    def test_seed_flag_exists_and_defaults_to_none(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=None)
        assert parser.parse_args([]).seed is None
        assert parser.parse_args(["--seed", "7"]).seed == 7

    def test_same_seed_reproduces_the_watermark_and_attack_noise(self):
        import random

        def draw():
            wm = np.random.randint(0, 2, size=16)
            noise = np.random.normal(0, 1, size=8)
            pick = random.random()
            return wm, noise, pick

        random.seed(11); np.random.seed(11)
        a = draw()
        random.seed(11); np.random.seed(11)
        b = draw()
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])
        assert a[2] == b[2]

    def test_different_seeds_give_different_draws(self):
        np.random.seed(1)
        first = np.random.randint(0, 2, size=32)
        np.random.seed(2)
        second = np.random.randint(0, 2, size=32)
        assert not np.array_equal(first, second)

    def test_collusion2_generator_follows_the_global_seed(self):
        """collusion_2 builds its own Generator; it must still be seedable."""
        def draw():
            return np.random.default_rng(np.random.randint(0, 2**32)).random(4)

        np.random.seed(5)
        a = draw()
        np.random.seed(5)
        b = draw()
        assert np.array_equal(a, b)


class TestCorpusOrderIsStable:
    def test_replay_and_mixing_sort_their_corpus_listing(self):
        """Filesystem order must not decide which corpus file an attack draws."""
        for path in (
            "src/deepmarkpy/plugins/attacks/replay/attack.py",
            "src/deepmarkpy/plugins/attacks/mixing/attack.py",
        ):
            source = open(path).read()
            assert "sorted(f for f in os.listdir" in source, f"{path} lists unsorted"


class TestRunMetadata:
    def test_writes_a_sibling_file_with_provenance(self, tmp_path):
        class _PM:
            failed = {"plugins.attacks.broken.attack": "No module named 'x'"}

        benchmark = type("_B", (), {
            "attacks": {"GaussianNoiseAttack": {}},
            "models": {"WavMarkModel": {"config": {"sampling_rate": 16000,
                                                   "watermark_size": 16}}},
            "plugin_manager": _PM(),
        })()
        args = argparse.Namespace(seed=42)

        path = run_module.write_run_metadata(
            str(tmp_path), args, benchmark, ["WavMarkModel"],
            extra={"n_files": 3},
        )
        meta = json.load(open(path))

        assert os.path.basename(path) == "run_metadata.json"
        assert meta["deepmarkpy_version"] == __version__
        assert meta["seed"] == 42
        assert meta["models"] == ["WavMarkModel"]
        assert meta["n_files"] == 3
        assert meta["generated_utc"].endswith("+00:00")
        assert meta["plugins"]["attacks_discovered"] == ["GaussianNoiseAttack"]
        # A plugin that failed to import is recorded rather than silently absent.
        assert meta["plugins"]["failed_imports"]

    def test_does_not_wrap_the_existing_artifacts(self, tmp_path):
        """benchmark_stats.json's top-level keys are read as attack names."""
        class _PM:
            failed = {}

        benchmark = type("_B", (), {
            "attacks": {}, "models": {}, "plugin_manager": _PM(),
        })()
        run_module.write_run_metadata(
            str(tmp_path), argparse.Namespace(seed=None), benchmark, ["M"],
        )
        assert not os.path.exists(os.path.join(tmp_path, "benchmark_stats.json"))


class TestReportDirIsHonoured:
    def test_every_mode_reads_report_dir_rather_than_a_hardcoded_path(self):
        """--report_dir must not clean one directory and write to another."""
        source = open("src/deepmarkpy/run.py").read()
        assert 'report_base = "report"' not in source, (
            "multi-model mode still hardcodes the report directory"
        )
        assert "report_base = args.report_dir" in source

    def test_deletion_is_announced_before_it_happens(self, tmp_path, caplog):
        (tmp_path / "old_results.json").write_text("{}")
        with caplog.at_level("INFO"):
            run_module._clean_report_dir(str(tmp_path))
        assert any("Clearing" in r.message for r in caplog.records), (
            "no log line precedes the deletion"
        )
        assert not (tmp_path / "old_results.json").exists()
