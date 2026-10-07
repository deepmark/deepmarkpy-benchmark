"""Tests for the reporting-integrity work: coverage counts, dispersion,
decode-failure accounting, the shared attack dispatcher, and the
comparative table's handling of non-comparable model families."""

import numpy as np
import pytest

from deepmarkpy.benchmark import (
    Benchmark,
    apply_attack,
    audio_filename_label,
    expand_attacks,
)
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
        no-op and fabricates a perfect reliability row.
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
        assert [d for _, d, _, _ in expanded] == [
            "Codec2VocoderAttack_700", "Codec2VocoderAttack_2400",
        ]
        assert [o for _, _, o, _ in expanded] == [
            {"bitrate_codec2": 700}, {"bitrate_codec2": 2400},
        ]

    def test_unsupported_bitrate_is_skipped(self):
        registry = {"Codec2VocoderAttack": {"config": {"bitrate_codec2": [700, 999]}}}
        expanded = expand_attacks(["Codec2VocoderAttack"], registry)
        assert [d for _, d, _, _ in expanded] == ["Codec2VocoderAttack_700"]

    def test_a_bare_bitrate_attack_also_runs_a_config_defined_version(self):
        """A bare name runs every version at each of its bitrates. The one
        the config defines loads the plugin's default preset and takes its
        bitrates from the config."""
        registry = {"Codec2VocoderAttack": {"config": {"bitrate_codec2": [700, 2400]}}}
        expanded = expand_attacks(
            ["Codec2VocoderAttack"], registry,
            extra_versions={"Codec2VocoderAttack": {"hi": {"bitrate_codec2": [3200]}}},
        )
        assert [(d, o, v) for _, d, o, v in expanded] == [
            ("Codec2VocoderAttack_700 (default)", {"bitrate_codec2": 700}, "default"),
            ("Codec2VocoderAttack_2400 (default)", {"bitrate_codec2": 2400}, "default"),
            ("Codec2VocoderAttack_3200 (hi)", {"bitrate_codec2": 3200}, None),
        ]

    def test_each_declared_version_runs_at_its_own_bitrates(self):
        """A preset's own bitrate list decides its rows, not the default's."""
        registry = {"Codec2VocoderAttack": {
            "config": {"bitrate_codec2": [700]},
            "_raw_config": {
                "default": {"bitrate_codec2": [700]},
                "high": {"bitrate_codec2": [3200]},
            },
        }}
        assert [(d, o, v) for _, d, o, v in expand_attacks(
            ["Codec2VocoderAttack"], registry,
        )] == [
            ("Codec2VocoderAttack_700 (default)", {"bitrate_codec2": 700}, "default"),
            ("Codec2VocoderAttack_3200 (high)", {"bitrate_codec2": 3200}, "high"),
        ]
        assert [d for _, d, _, _ in expand_attacks(
            ["Codec2VocoderAttack:high"], registry,
        )] == ["Codec2VocoderAttack_3200 (high)"]

    def test_plain_attack_passes_through(self):
        registry = {"GaussianNoiseAttack": {"config": {"snr_db_gaussian_noise": 35}}}
        assert expand_attacks(["GaussianNoiseAttack"], registry) == [
            ("GaussianNoiseAttack", "GaussianNoiseAttack", {}, None)
        ]

    def test_unknown_attack_passes_through_unchanged(self):
        assert expand_attacks(["NoSuchAttack"], {}) == [
            ("NoSuchAttack", "NoSuchAttack", {}, None)
        ]

    def test_a_colon_inside_the_version_stays_in_the_version(self):
        """Validation splits at the first colon, and so does expansion: at
        the last, "GaussianNoiseAttack:v:2" would name a class
        "GaussianNoiseAttack:v" that benchmark mode skips."""
        registry = {"GaussianNoiseAttack": {
            "config": {"snr_db_gaussian_noise": 35},
            "_raw_config": {
                "default": {"snr_db_gaussian_noise": 35},
                "v:2": {"snr_db_gaussian_noise": 10},
            },
        }}
        (cls, display, _, version), = expand_attacks(
            ["GaussianNoiseAttack:v:2"], registry,
        )
        assert cls == "GaussianNoiseAttack"
        assert version == "v:2"
        assert display == "GaussianNoiseAttack (v:2)"

    def test_a_saved_audio_filename_replaces_what_a_path_cannot_hold(self):
        """A version name may hold a path separator or a character Windows
        reserves; in the saved-audio filename each becomes an underscore."""
        assert audio_filename_label("GaussianNoiseAttack (v:2)") == "GaussianNoiseAttack (v_2)"
        assert audio_filename_label("PresetNoiseAttack (lo/hi)") == "PresetNoiseAttack (lo_hi)"

    def test_a_safe_name_is_its_own_filename(self):
        """Spaces and parentheses are kept, so a safe name is unchanged."""
        for name in ("Codec2VocoderAttack_700", "GaussianNoiseAttack (mild)"):
            assert audio_filename_label(name) == name


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
    @staticmethod
    def _generator(tmp_path, statistics=("mean",)):
        """A generator whose columns are exactly ``statistics``."""
        from deepmarkpy.utils.metric_resolver import MetricResolver

        resolver = MetricResolver(
            defaults={
                "accuracy": {"enabled": True, "statistics": list(statistics)},
                "pesq": {"enabled": True, "statistics": list(statistics)},
            },
            calculate_quality_metrics=True,
        )
        return BenchmarkReportGenerator(str(tmp_path), resolver=resolver)

    def test_table_shows_n_dispersion_and_failure_marker(self, tmp_path):
        gen = self._generator(tmp_path, ("mean", "std"))
        table = gen.generate_latex_table({
            "GaussianNoiseAttack": {
                "accuracy_mean": 75.0, "accuracy_n": 4, "accuracy_std": 5.0,
                "detection_failures": 2, "pesq_mean": 3.1, "pesq_n": 2,
            }
        }, group_key="audio_distortion")
        assert "75.00" in table
        assert "5.00" in table, "configured dispersion column missing"
        assert "(2)" in table, "decode-failure count not marked"
        assert "$n$=2" in table, "reduced metric coverage not marked"
        assert "random-guess floor" in table, "failure footnote missing"

    def test_columns_are_exactly_what_the_config_asks_for(self, tmp_path):
        """No hardcoded column may survive the config, in either direction."""
        gen = self._generator(tmp_path, ("median", "worst_case"))
        table = gen.generate_latex_table({
            "GaussianNoiseAttack": {
                "accuracy_mean": 75.0, "accuracy_n": 4,
                "accuracy_median": 80.0, "accuracy_worst_case": 60.0,
            }
        }, group_key="audio_distortion")
        header = next(l for l in table.splitlines() if "Attack Type" in l)
        assert "Median" in header and "Worst Case" in header
        assert "Mean" not in header, "an unconfigured column was added anyway"


class TestComparativeTableComparability:
    # Steady keeps ~98% on every file; Erratic averages 82% and swings
    # widely, so the larger spread belongs to the worse model.
    STEADY_VS_ERRATIC = {
        "SteadyModel": {"GaussianNoiseAttack": {
            "accuracy_mean": 98.0, "accuracy_std": 2.74}},
        "ErraticModel": {"GaussianNoiseAttack": {
            "accuracy_mean": 82.0, "accuracy_std": 24.90}},
    }

    @staticmethod
    def _gen(meta, statistics=("mean",)):
        from deepmarkpy.utils.metric_resolver import MetricResolver

        gen = ComparativeReportGenerator.__new__(ComparativeReportGenerator)
        gen.model_meta = meta
        gen.resolver = MetricResolver(
            defaults={"accuracy": {"enabled": True,
                                   "statistics": list(statistics)}},
        )
        gen.primary_statistic = statistics[0]
        gen._has_deepmark_cls = False
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

    def test_a_std_table_ranks_no_model(self):
        """A spread has no better end, so its cells are shown uncoloured.

        The same row at the mean is ranked, so the data itself ranks.
        """
        gen = self._gen({}, ("mean", "std"))
        std_row = self._data_row(gen.generate_accuracy_table(
            self.STEADY_VS_ERRATIC, "std", with_note=False), "Gaussian")
        mean_row = self._data_row(gen.generate_accuracy_table(
            self.STEADY_VS_ERRATIC, "mean", with_note=False), "Gaussian")

        assert "textcolor" not in std_row, std_row
        assert "24.90" in std_row
        steady_mean_cell = mean_row.split("&")[1]
        assert RANK_COLORS[0][0] in steady_mean_cell, mean_row

    def test_the_further_statistics_say_std_is_left_unranked(self):
        """The sentence over the secondary tables names std as the exception."""
        gen = self._gen({}, ("mean", "std"))
        tex = gen.generate_latex_report(self.STEADY_VS_ERRATIC, include_radar=False)
        further = tex.split("Further Statistics", 1)[1]
        sentence = further[further.index("Every other statistic"):].split("\n\n", 1)[0]

        assert "standard deviation" in sentence, sentence
        assert "uncoloured" in sentence, sentence

    def test_the_further_statistics_name_std_only_when_it_is_among_them(self):
        """With std in the main table, every table below it is ranked."""
        gen = self._gen({}, ("std", "mean"))
        tex = gen.generate_latex_report(self.STEADY_VS_ERRATIC, include_radar=False)
        further = tex.split("Further Statistics", 1)[1]
        sentence = further[further.index("Every other statistic"):].split("\n\n", 1)[0]

        assert "standard deviation" not in sentence, sentence


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

        def fake_run_single_model(_benchmark, _filepaths, model_name, _config,
                                  _settings, output_dir=None):
            results = {"f.wav": {}}
            flattened = {"GaussianNoiseAttack": 90.0}
            stats = {"GaussianNoiseAttack": {"accuracy_mean": 90.0, "accuracy_n": 5}}
            return results, flattened, stats

        captured = {}

        class _FakeGenerator:
            def __init__(self, report_dir=None, resolver=None,
                         primary_statistic="mean"):
                pass

            def generate_full_report(self, all_stats, **kwargs):
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

        from deepmarkpy.config import ModeConfig

        config = ModeConfig(mode="benchmark", source="test.json",
                            models=["ZeroBitModel", "MultiBitModel"])
        settings = run_module.RunSettings(
            wav_files_dir=str(tmp_path), report_dir=str(tmp_path / "report"),
            seed=None, verbose=False, save_audio=False,
        )
        run_module.run_multiple_models(
            benchmark, ["f.wav"], ["ZeroBitModel", "MultiBitModel"],
            config, settings,
        )

        assert set(captured["stats"]) == {"ZeroBitModel", "MultiBitModel"}
        meta = captured["meta"]
        assert meta["ZeroBitModel"]["is_zero_bit"] is True
        assert meta["MultiBitModel"]["is_zero_bit"] is False
        assert meta["MultiBitModel"]["watermark_size"] == 40
        assert meta["MultiBitModel"]["sampling_rate"] == 22050
        assert meta["ZeroBitModel"]["n_files"] == 5


class TestExpansionIsDeduplicated:
    """A row must not be expanded twice.

    Naming a version in attacks.list while its attack also arrives from a
    group produced that version twice: attacked twice per file, then
    collapsed by the results dict, which keys on the display name. Pure
    wasted work, and invisible in the output.
    """

    @staticmethod
    def _registry():
        return {"GaussianNoiseAttack": {
            "config": {"snr_db_gaussian_noise": 35},
            "_raw_config": {
                "default": {"snr_db_gaussian_noise": 35},
                "mild": {"snr_db_gaussian_noise": 45},
            },
        }}

    def test_explicit_version_and_bare_name_yield_one_row_each(self):
        names = [
            display for _, display, _, _ in expand_attacks(
                ["GaussianNoiseAttack:mild", "GaussianNoiseAttack"],
                self._registry(),
            )
        ]
        assert names == [
            "GaussianNoiseAttack (mild)", "GaussianNoiseAttack (default)",
        ], names

    def test_the_explicit_spec_wins(self):
        """It comes first, so its parameters are the ones kept."""
        rows = expand_attacks(
            ["GaussianNoiseAttack:mild", "GaussianNoiseAttack"],
            self._registry(),
            parameters=lambda name, version: (
                {"snr_db_gaussian_noise": 99} if version == "mild" else {}
            ),
        )
        mild = next(kw for _, d, kw, _ in rows if d.endswith("(mild)"))
        assert mild == {"snr_db_gaussian_noise": 99}

    def test_the_same_spec_twice_is_one_row(self):
        names = [
            display for _, display, _, _ in expand_attacks(
                ["GaussianNoiseAttack:mild", "GaussianNoiseAttack:mild"],
                self._registry(),
            )
        ]
        assert names == ["GaussianNoiseAttack (mild)"]

    def test_default_and_bare_bitrate_attack_is_one_row_per_bitrate(self):
        """On a single-version plugin ':default' names the bare attack, so
        together they give each bitrate one row, not two."""
        names = [
            display for _, display, _, _ in expand_attacks(
                ["Codec2VocoderAttack:default", "Codec2VocoderAttack"],
                {"Codec2VocoderAttack": {"config": {"bitrate_codec2": [700, 2400]}}},
            )
        ]
        assert names == ["Codec2VocoderAttack_700", "Codec2VocoderAttack_2400"], names

    def test_a_bitrate_version_named_and_reached_bare_is_one_row(self):
        """The named config-defined version runs at its own bitrate, and
        the bare name adds only the rows it does not already have."""
        names = [
            display for _, display, _, _ in expand_attacks(
                ["Codec2VocoderAttack:hi", "Codec2VocoderAttack"],
                {"Codec2VocoderAttack": {"config": {"bitrate_codec2": [700]}}},
                extra_versions={"Codec2VocoderAttack": {"hi": {"bitrate_codec2": [3200]}}},
            )
        ]
        assert names == [
            "Codec2VocoderAttack_3200 (hi)", "Codec2VocoderAttack_700 (default)",
        ], names


class TestTheCropCaveatSurvivesDurationGrouping:
    """A cropped run says so whichever shape the report takes.

    Both reports stated it in their abstract, and the duration-grouped
    documents have no abstract -- so grouping a cropped run silently
    dropped the one sentence that says the numbers do not describe the
    whole signal.
    """

    CROP = 12.5

    @pytest.fixture(autouse=True)
    def _no_pdflatex(self, monkeypatch):
        monkeypatch.setattr(
            "deepmarkpy.utils.latex_helpers.compile_latex",
            lambda *a, **k: None,
        )

    def _grouped_stats(self):
        return {
            "< 5.0s": {"n_files": 3, "stats": {
                "GaussianNoiseAttack": {"accuracy_mean": 91.0, "accuracy_n": 3},
            }},
            "> 5.0s": {"n_files": 3, "stats": {
                "GaussianNoiseAttack": {"accuracy_mean": 95.0, "accuracy_n": 3},
            }},
        }

    def _basic_tex(self, tmp_path, crop):
        import json
        from deepmarkpy.utils.report_generator import BenchmarkReportGenerator

        stats_file = tmp_path / "stats.json"
        stats_file.write_text(json.dumps(self._grouped_stats()))
        generator = BenchmarkReportGenerator(str(tmp_path))
        generator.generate_full_report(
            str(stats_file), "TestModel", crop_before_attack=crop,
        )
        return (tmp_path / "benchmark_report.tex").read_text()

    def test_the_basic_grouped_report_carries_the_note(self, tmp_path):
        assert "A crop of 12.5\\%" in self._basic_tex(tmp_path, self.CROP)

    def test_an_uncropped_grouped_report_says_nothing(self, tmp_path):
        assert "A crop of" not in self._basic_tex(tmp_path, None)

    def test_the_note_appears_once_not_per_duration_part(self, tmp_path):
        tex = self._basic_tex(tmp_path, self.CROP)
        assert tex.count("A crop of") == 1

    def test_the_detailed_grouped_report_carries_the_note(self, tmp_path):
        from deepmarkpy.utils.detailed_report_generator import (
            DetailedReportGenerator,
        )

        results = {
            f"f{i}.wav": {
                "watermarked_audio_quality": {"pesq": 3.5},
                "attacks": {"GaussianNoiseAttack": {
                    "accuracy": 90.0,
                    "attacked_audio_quality_wm": {"pesq": 3.0},
                }},
            }
            for i in range(4)
        }
        generator = DetailedReportGenerator(str(tmp_path))
        generator.generate_full_report(
            results, model_name="TestModel",
            crop_before_attack=self.CROP,
            duration_partitions=[
                ("< 5.0s", ["f0.wav", "f1.wav"]),
                ("> 5.0s", ["f2.wav", "f3.wav"]),
            ],
        )
        tex = (tmp_path / "detailed_report.tex").read_text()
        assert "A crop of 12.5\\%" in tex
