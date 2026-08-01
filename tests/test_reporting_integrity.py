"""Tests for the reporting-integrity work: coverage counts, dispersion,
decode-failure accounting, the shared attack dispatcher, and the
comparative table's handling of non-comparable model families."""

import numpy as np
import pytest

from deepmarkpy.benchmark import Benchmark, apply_attack, expand_attacks
from deepmarkpy.utils.comparative_report_generator import (
    RANK_COLORS,
    ComparativeReportGenerator,
)
from deepmarkpy.utils.report_generator import BenchmarkReportGenerator


class _Recorder:
    """Attack stub recording the audio and kwargs it was called with."""

    def __init__(self, returns_tuple=False):
        self.returns_tuple = returns_tuple
        self.seen_audio = None
        self.seen_kwargs = None

    def apply(self, audio, **kwargs):
        self.seen_audio = audio
        self.seen_kwargs = kwargs
        out = np.asarray(audio) * 0.5
        return (out, "watermark") if self.returns_tuple else out


class TestSharedDispatcher:
    def test_collusion_receives_the_clean_original_not_the_target(self):
        """The spliced-in reference must differ from the array being attacked.

        Passing the target as its own 'original' turns the attack into a
        no-op, which previously fabricated a perfect reliability row.
        """
        attack = _Recorder()
        target = np.ones(16)
        clean = np.zeros(16)
        apply_attack(attack, "ZeroBitCollusionAttack", target, clean, {})
        assert attack.seen_kwargs["original_audio_collusion"] is clean
        assert not np.array_equal(
            attack.seen_kwargs["original_audio_collusion"], attack.seen_audio
        )

    def test_tuple_returning_attack_is_unpacked(self):
        attack = _Recorder(returns_tuple=True)
        audio, extra = apply_attack(attack, "CrossModelAttack", np.ones(8), np.zeros(8), {})
        assert isinstance(audio, np.ndarray)
        assert extra == "watermark"

    def test_plain_attack_returns_none_extra_and_is_squeezed(self):
        attack = _Recorder()
        audio, extra = apply_attack(
            attack, "GaussianNoiseAttack", np.ones((1, 8)), np.zeros((1, 8)), {}
        )
        assert extra is None
        assert audio.shape == (8,)

    def test_kwargs_are_not_mutated_for_the_caller(self):
        attack = _Recorder()
        kwargs = {"sampling_rate": 16000}
        apply_attack(attack, "ZeroBitCollusionAttack", np.ones(4), np.zeros(4), kwargs)
        assert "original_audio_collusion" not in kwargs


class TestExpandAttacks:
    def test_bitrate_list_expands_to_one_entry_per_value(self):
        registry = {"Codec2VocoderAttack": {"config": {"bitrate_codec2": [700, 2400]}}}
        expanded = expand_attacks(["Codec2VocoderAttack"], registry)
        assert [d for _, d, _ in expanded] == [
            "Codec2VocoderAttack_700", "Codec2VocoderAttack_2400",
        ]
        assert [o for _, _, o in expanded] == [
            {"bitrate_codec2": 700}, {"bitrate_codec2": 2400},
        ]

    def test_unsupported_bitrate_is_skipped(self):
        registry = {"Codec2VocoderAttack": {"config": {"bitrate_codec2": [700, 999]}}}
        expanded = expand_attacks(["Codec2VocoderAttack"], registry)
        assert [d for _, d, _ in expanded] == ["Codec2VocoderAttack_700"]

    def test_plain_attack_passes_through(self):
        registry = {"GaussianNoiseAttack": {"config": {"snr_db_gaussian_noise": 35}}}
        assert expand_attacks(["GaussianNoiseAttack"], registry) == [
            ("GaussianNoiseAttack", "GaussianNoiseAttack", {})
        ]

    def test_unknown_attack_passes_through_unchanged(self):
        assert expand_attacks(["NoSuchAttack"], {}) == [
            ("NoSuchAttack", "NoSuchAttack", {})
        ]


class TestAggregationTransparency:
    @staticmethod
    def _results(entries):
        return {
            f"f{i}.wav": {"attacks": {"A": entry}}
            for i, entry in enumerate(entries)
        }

    def test_reports_n_and_std_for_accuracy(self):
        stats = Benchmark.compute_mean_accuracy(
            Benchmark.__new__(Benchmark),
            self._results([
                {"accuracy": 90.0, "detection_valid": True},
                {"accuracy": 100.0, "detection_valid": True},
            ]),
        )["A"]
        assert stats["accuracy_n"] == 2
        assert stats["accuracy_std"] == pytest.approx(np.std([90.0, 100.0], ddof=1))

    def test_counts_decode_failures_separately_from_the_mean(self):
        """A 50.0 from a dead decoder must be distinguishable from a measured 50.0.

        Deliberately asymmetric (1 failure among 3 files) so that counting
        valid files instead of failed ones cannot produce the same number.
        """
        stats = Benchmark.compute_mean_accuracy(
            Benchmark.__new__(Benchmark),
            self._results([
                {"accuracy": 50.0, "detection_valid": False},
                {"accuracy": 100.0, "detection_valid": True},
                {"accuracy": 100.0, "detection_valid": True},
            ]),
        )["A"]
        assert stats["detection_failures"] == 1
        assert stats["accuracy_n"] == 3

    def test_no_failures_reports_zero_not_absent(self):
        stats = Benchmark.compute_mean_accuracy(
            Benchmark.__new__(Benchmark),
            self._results([
                {"accuracy": 100.0, "detection_valid": True},
                {"accuracy": 90.0, "detection_valid": True},
            ]),
        )["A"]
        assert stats["detection_failures"] == 0

    def test_all_failures_are_all_counted(self):
        stats = Benchmark.compute_mean_accuracy(
            Benchmark.__new__(Benchmark),
            self._results([
                {"accuracy": 50.0, "detection_valid": False},
                {"accuracy": 50.0, "detection_valid": False},
            ]),
        )["A"]
        assert stats["detection_failures"] == 2

    def test_metric_n_records_partial_coverage(self):
        stats = Benchmark.compute_mean_accuracy(
            Benchmark.__new__(Benchmark),
            self._results([
                {"accuracy": 100.0, "detection_valid": True,
                 "attacked_audio_quality_wm": {"pesq": 3.0, "stoi": 0.9}},
                {"accuracy": 100.0, "detection_valid": True,
                 "attacked_audio_quality_wm": {"pesq": None, "stoi": 0.8}},
            ]),
        )["A"]
        assert stats["pesq_n"] == 1
        assert stats["stoi_n"] == 2


class TestBasicReportSurfacesCoverage:
    def test_table_shows_n_dispersion_and_failure_marker(self):
        gen = BenchmarkReportGenerator.__new__(BenchmarkReportGenerator)
        table = gen.generate_latex_table({
            "GaussianNoiseAttack": {
                "accuracy_mean": 75.0, "accuracy_n": 4, "accuracy_std": 5.0,
                "detection_failures": 2, "pesq_mean": 3.1, "pesq_n": 2,
            }
        })
        assert "75.00" in table and "$\\pm$ 5.00" in table
        assert "(2)" in table, "decode-failure count not marked"
        assert "$n$=2" in table, "reduced metric coverage not marked"
        assert "random-guess floor" in table, "failure footnote missing"


class TestComparativeTableComparability:
    @staticmethod
    def _gen(meta):
        gen = ComparativeReportGenerator.__new__(ComparativeReportGenerator)
        gen.model_meta = meta
        return gen

    @staticmethod
    def _data_row(table, attack_display):
        """Return the table's data row for an attack, excluding header/footnote."""
        for line in table.splitlines():
            if line.strip().startswith(attack_display) and "&" in line:
                return line
        raise AssertionError(f"no data row for {attack_display} in:\n{table}")

    def test_zero_bit_column_is_marked_and_excluded_from_ranking(self):
        """Zero-bit scores must not be rank-coloured against bit accuracies.

        Uses three models so the multi-bit values genuinely rank (two
        distinct values), which is what makes the exclusion observable.
        """
        gen = self._gen({
            "PerthModel": {"is_zero_bit": True, "watermark_size": 10,
                           "sampling_rate": 16000, "n_files": 3},
            "WavMarkModel": {"is_zero_bit": False, "watermark_size": 16,
                             "sampling_rate": 16000, "n_files": 3},
            "AwareModel": {"is_zero_bit": False, "watermark_size": 20,
                           "sampling_rate": 16000, "n_files": 3},
        })
        table = gen.generate_accuracy_table({
            "PerthModel": {"GaussianNoiseAttack": 100.0},
            "WavMarkModel": {"GaussianNoiseAttack": 90.0},
            "AwareModel": {"GaussianNoiseAttack": 60.0},
        })

        # The header cell for the zero-bit model carries the marker.
        assert "\\textsuperscript{0} &" in table or "\\textsuperscript{0} \\\\" in table, \
            "zero-bit column header is not marked"

        row = self._data_row(table, "Gaussian")
        cells = [c.strip() for c in row.split("&")]
        perth_cell, wavmark_cell, aware_cell = cells[1], cells[2], cells[3]

        # Perth scores highest but must not be coloured as the winner of a
        # ranking it does not belong to.
        assert "textcolor" not in perth_cell, f"zero-bit cell was ranked: {perth_cell}"
        assert "100.00" in perth_cell

        # The multi-bit columns must rank *among themselves*: WavMark's 90 is
        # the best of {90, 60}, so it must carry rank-1 colour. If the zero-bit
        # 100 were included in the ranking, WavMark would drop to rank 2 and
        # take a different colour — that is exactly the regression to catch.
        best_color, second_color = RANK_COLORS[0][0], RANK_COLORS[1][0]
        assert best_color in wavmark_cell, (
            f"best multi-bit value not ranked first among multi-bit models: {wavmark_cell}"
        )
        assert second_color in aware_cell, f"second multi-bit rank lost: {aware_cell}"

    def test_zero_bit_models_are_ranked_among_themselves_never_against_multibit(self):
        """With only zero-bit models present, nothing is rank-coloured."""
        gen = self._gen({
            "PerthModel": {"is_zero_bit": True, "watermark_size": 10,
                           "sampling_rate": 16000, "n_files": 2},
            "OtherZeroBit": {"is_zero_bit": True, "watermark_size": 10,
                             "sampling_rate": 16000, "n_files": 2},
        })
        table = gen.generate_accuracy_table({
            "PerthModel": {"A": 100.0}, "OtherZeroBit": {"A": 0.0},
        })
        row = self._data_row(table, "A")
        assert "textcolor" not in row

    def test_note_states_payload_rate_and_n_per_model(self):
        gen = self._gen({
            "PerthModel": {"is_zero_bit": True, "watermark_size": 10,
                           "sampling_rate": 16000, "n_files": 6},
            "TimbreWMModel": {"is_zero_bit": False, "watermark_size": 10,
                              "sampling_rate": 22050, "n_files": 6},
        })
        table = gen.generate_accuracy_table({
            "PerthModel": {"A": 100.0}, "TimbreWMModel": {"A": 99.0},
        })
        assert "10-bit payload" in table
        assert "22050 Hz" in table
        assert "n=6" in table

    def test_without_metadata_the_table_still_renders(self):
        gen = self._gen({})
        table = gen.generate_accuracy_table({"M1": {"A": 1.0}, "M2": {"A": 2.0}})
        assert "A" in table


class TestMultiModelPath:
    """The multi-model path builds the comparative report's inputs.

    It is not covered by the rest of the suite, and a NameError introduced
    here aborts the whole run rather than skipping one model, because the
    tolerated-exception set deliberately excludes coding errors.
    """

    def test_collects_stats_and_metadata_for_every_model(self, tmp_path, monkeypatch):
        import argparse

        from deepmarkpy import run as run_module

        models = {
            "ZeroBitModel": {"config": {"is_zero_bit": True, "watermark_size": 10,
                                        "sampling_rate": 16000}},
            "MultiBitModel": {"config": {"is_zero_bit": False, "watermark_size": 40,
                                         "sampling_rate": 22050}},
        }
        benchmark = type("_B", (), {"models": models})()

        def fake_run_single_model(_benchmark, _filepaths, model_name, _args, output_dir=None):
            results = {"f.wav": {}}
            flattened = {"GaussianNoiseAttack": 90.0}
            stats = {"GaussianNoiseAttack": {"accuracy_mean": 90.0, "accuracy_n": 5}}
            return results, flattened, stats

        captured = {}

        class _FakeGenerator:
            def __init__(self, report_dir=None):
                pass

            def generate_full_report(self, all_results, all_stats, **kwargs):
                captured["stats"] = all_stats
                captured["meta"] = kwargs.get("model_meta")

        monkeypatch.setattr(run_module, "run_single_model", fake_run_single_model)
        monkeypatch.setattr(run_module, "_clean_report_dir", lambda *a, **k: None)
        monkeypatch.setattr(run_module, "_copy_deepmark_assets", lambda *a, **k: None)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setitem(
            __import__("sys").modules,
            "deepmarkpy.utils.comparative_report_generator",
            type("_M", (), {"ComparativeReportGenerator": _FakeGenerator}),
        )

        args = argparse.Namespace(
            calculate_quality_metrics=False,
            crop_before_attack=None,
            report_dir=str(tmp_path / "report"),
            seed=None,
        )
        run_module.run_multiple_models(
            benchmark, ["f.wav"], ["ZeroBitModel", "MultiBitModel"], args,
        )

        assert set(captured["stats"]) == {"ZeroBitModel", "MultiBitModel"}
        meta = captured["meta"]
        assert meta["ZeroBitModel"]["is_zero_bit"] is True
        assert meta["MultiBitModel"]["is_zero_bit"] is False
        assert meta["MultiBitModel"]["watermark_size"] == 40
        assert meta["MultiBitModel"]["sampling_rate"] == 22050
        assert meta["ZeroBitModel"]["n_files"] == 5
