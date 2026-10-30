"""Benchmark mode scores a model that decides presence on its own decision.

A model whose config.json declares ``returns_is_watermarked`` gets, per
attack, a true positive from ``is_watermarked()`` on the attacked watermarked
audio, a false positive from the same attack applied to the clean audio, and
accuracy over the files it detected. A zero-bit model gets the two counts
alone. A model that does not declare it is measured as before.
"""

import json
import pathlib

import numpy as np
import pytest
import soundfile as sf

from deepmarkpy.benchmark import Benchmark
from deepmarkpy.utils import metrics
from deepmarkpy.utils.metric_resolver import MetricResolver

# In process, so every attack runs on the ordered lane. Each bit nudges one
# block of samples; the block means give the bits by sign and the presence
# decision by size, which clean audio (zero-mean noise) does not reach.
MULTI_BIT_MODEL = '''
import numpy as np

from deepmarkpy.core.base_model import BaseModel


class DecidingModel(BaseModel):
    def embed(self, audio, watermark_data, sampling_rate, **kwargs):
        out = np.array(audio, dtype=np.float64)
        block = len(out) // len(watermark_data)
        for i, bit in enumerate(watermark_data):
            out[i * block:(i + 1) * block] += 5e-3 if bit else -5e-3
        return out

    def detect(self, audio, sampling_rate, **kwargs):
        audio = np.asarray(audio, dtype=np.float64)
        block = len(audio) // 16
        means = np.array([np.mean(audio[i * block:(i + 1) * block]) for i in range(16)])
        return (means > 0).astype(int), float(np.mean(np.abs(means)))

    def is_watermarked(self, detect_output):
        return detect_output[1] > 2.5e-3
'''

ZERO_BIT_MODEL = '''
import numpy as np

from deepmarkpy.core.base_model import BaseModel


class ZeroBitDecidingModel(BaseModel):
    def embed(self, audio, watermark_data, sampling_rate, **kwargs):
        return np.asarray(audio, dtype=np.float64) + 5e-3

    def detect(self, audio, sampling_rate, **kwargs):
        return int(np.mean(np.asarray(audio, dtype=np.float64)) > 2.5e-3)

    def is_watermarked(self, detect_output):
        return bool(detect_output)
'''

UNDECIDED_MODEL = MULTI_BIT_MODEL.replace("class DecidingModel", "class UndecidedModel").split(
    "    def is_watermarked")[0]

SILENCE_ATTACK = '''
import numpy as np

from deepmarkpy.core.base_attack import BaseAttack


class SilenceAttack(BaseAttack):
    """Leaves nothing for a detector to find."""

    def apply(self, audio, **kwargs):
        return np.zeros_like(np.asarray(audio, dtype=np.float64))
'''

MULTI_BIT_CONFIG = {
    "sampling_rate": 16000, "watermark_size": 16, "is_zero_bit": False,
    "returns_confidence": True, "returns_is_watermarked": True,
}
ATTACKS = ["NoAttack", "SilenceAttack"]
N_FILES = 3


def _plugin(root, name, filename, source, config):
    directory = pathlib.Path(root) / name
    directory.mkdir(parents=True)
    (directory / filename).write_text(source)
    (directory / "config.json").write_text(json.dumps(config))


@pytest.fixture(autouse=True)
def _no_nisqa(monkeypatch):
    monkeypatch.setattr(metrics, "_NISQA_ENDPOINT", None)
    monkeypatch.setattr(metrics, "_nisqa_unavailable", True)
    monkeypatch.setattr(metrics, "_nisqa_reason", "disabled in tests")


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    root = tmp_path_factory.mktemp("plugins")
    _plugin(root, "deciding", "model.py", MULTI_BIT_MODEL, MULTI_BIT_CONFIG)
    _plugin(root, "zero_bit", "model.py", ZERO_BIT_MODEL,
            {**MULTI_BIT_CONFIG, "is_zero_bit": True, "returns_confidence": False})
    _plugin(root, "undecided", "model.py", UNDECIDED_MODEL,
            {**MULTI_BIT_CONFIG, "returns_is_watermarked": False})
    _plugin(root, "silence", "attack.py", SILENCE_ATTACK, {})
    return Benchmark(external_plugins_dir=str(root))


@pytest.fixture(scope="module")
def filepaths(tmp_path_factory):
    directory = tmp_path_factory.mktemp("audio")
    rng = np.random.default_rng(3)
    paths = []
    for index in range(N_FILES):
        path = directory / f"clip{index}.wav"
        sf.write(str(path), 0.01 * rng.standard_normal(16000), 16000, subtype="FLOAT")
        paths.append(str(path))
    return paths


def _run(benchmark, filepaths, model):
    results = benchmark.run(
        filepaths=filepaths, wm_model=model, attack_types=ATTACKS,
        metric_resolver=MetricResolver.from_attack_groups(calculate_quality_metrics=False),
    )
    model_config = benchmark.models[model]["config"]
    stats = benchmark.compute_mean_accuracy(
        results, is_zero_bit=model_config.get("is_zero_bit", False),
    )
    return results, stats


class TestAModelThatDecidesPresence:
    def test_rows_carry_both_decisions(self, benchmark, filepaths):
        results, _ = _run(benchmark, filepaths, "DecidingModel")
        for file_results in results.values():
            kept = file_results["attacks"]["NoAttack"]
            assert kept["detected"] is True and kept["detected_clean"] is False
            removed = file_results["attacks"]["SilenceAttack"]
            assert removed["detected"] is False and removed["detected_clean"] is False

    def test_counts_true_and_false_positives_over_every_file(self, benchmark, filepaths):
        _, stats = _run(benchmark, filepaths, "DecidingModel")
        assert (stats["NoAttack"]["tp_count"], stats["NoAttack"]["tp_n"]) == (N_FILES, N_FILES)
        assert (stats["NoAttack"]["fp_count"], stats["NoAttack"]["fp_n"]) == (0, N_FILES)
        assert stats["SilenceAttack"]["tp_count"] == 0

    def test_accuracy_covers_the_detected_files_only(self, benchmark, filepaths):
        """An undetected file's message is not a reading, so it is no 50%."""
        _, stats = _run(benchmark, filepaths, "DecidingModel")
        assert stats["NoAttack"]["accuracy_mean"] == 100.0
        assert stats["NoAttack"]["accuracy_n"] == N_FILES
        assert (stats["NoAttack"]["emr_count"], stats["NoAttack"]["emr_rate"]) == (N_FILES, 1.0)
        silence = stats["SilenceAttack"]
        assert silence["accuracy_n"] == 0 and "accuracy_mean" not in silence
        # Exact recovery is over every file: none detected, none recovered.
        assert (silence["emr_count"], silence["emr_rate"]) == (0, 0.0)

    def test_the_undetected_rows_keep_their_raw_accuracy(self, benchmark, filepaths):
        results, _ = _run(benchmark, filepaths, "DecidingModel")
        for file_results in results.values():
            assert "accuracy" in file_results["attacks"]["SilenceAttack"]


class TestAZeroBitModel:
    def test_has_no_accuracy_ber_or_emr(self, benchmark, filepaths):
        results, stats = _run(benchmark, filepaths, "ZeroBitDecidingModel")
        for file_results in results.values():
            for row in file_results["attacks"].values():
                assert "accuracy" not in row and "detection_valid" not in row
        for entry in stats.values():
            assert not [key for key in entry if key.startswith(("accuracy", "ber", "emr"))]

    def test_its_whole_measurement_is_tp_and_fp(self, benchmark, filepaths):
        _, stats = _run(benchmark, filepaths, "ZeroBitDecidingModel")
        assert stats["NoAttack"]["tp_count"] == N_FILES
        assert stats["NoAttack"]["fp_count"] == 0
        assert stats["SilenceAttack"]["tp_count"] == 0


class TestAModelThatDoesNotDeclareIt:
    def test_is_measured_as_before(self, benchmark, filepaths):
        """No decisions, no clean-audio pass, and accuracy over every file."""
        results, stats = _run(benchmark, filepaths, "UndecidedModel")
        for file_results in results.values():
            for row in file_results["attacks"].values():
                assert "detected" not in row and "detected_clean" not in row
        assert not [key for entry in stats.values() for key in entry
                    if key.startswith(("tp_", "fp_"))]
        assert stats["SilenceAttack"]["accuracy_n"] == N_FILES


class TestADeclarationItCannotHonour:
    def test_is_refused_before_any_audio(self, benchmark, filepaths, monkeypatch):
        entry = benchmark.models["UndecidedModel"]
        monkeypatch.setitem(entry, "config",
                            {**entry["config"], "returns_is_watermarked": True})
        with pytest.raises(ValueError, match="does not implement is_watermarked"):
            benchmark.run(filepaths=filepaths, wm_model="UndecidedModel",
                          attack_types=ATTACKS)


class TestNoAttack:
    def test_returns_its_input_unchanged(self, benchmark):
        audio = np.random.default_rng(0).standard_normal(1000)
        out = benchmark.attacks["NoAttack"]["class"]().apply(audio, sampling_rate=16000)
        assert np.array_equal(out, audio)


class TestNoAttackLabel:
    def test_reports_name_it_no_attack(self):
        from deepmarkpy.utils.latex_helpers import display_attack_name
        assert display_attack_name("NoAttack") == "No attack"
        assert display_attack_name("NoAttack", split_camel_case=True) == "No attack"
        assert display_attack_name("GaussianNoiseAttack (a)") == "GaussianNoise (a)"


class TestComputeMeanAccuracyFromRows:
    """The aggregation alone, on hand-written rows."""

    @staticmethod
    def _results(rows):
        return {f"f{i}.wav": {"attacks": {"A": row}} for i, row in enumerate(rows)}

    def test_rows_without_decisions_aggregate_as_before(self):
        stats = Benchmark.__new__(Benchmark).compute_mean_accuracy(self._results([
            {"accuracy": 100.0}, {"accuracy": 50.0},
        ]))["A"]
        assert stats["accuracy_mean"] == 75.0 and stats["accuracy_n"] == 2
        assert "tp_count" not in stats and "fp_count" not in stats

    def test_accuracy_and_emr_follow_the_detected_rows(self):
        stats = Benchmark.__new__(Benchmark).compute_mean_accuracy(self._results([
            {"accuracy": 100.0, "detected": True, "detected_clean": True},
            {"accuracy": 80.0, "detected": True, "detected_clean": False},
            {"accuracy": 50.0, "detected": False, "detected_clean": False},
        ]))["A"]
        assert stats["accuracy_mean"] == 90.0 and stats["accuracy_n"] == 2
        # One file detected and exact, out of all three.
        assert (stats["emr_count"], stats["emr_rate"]) == (1, 1 / 3)
        assert (stats["tp_count"], stats["tp_n"]) == (2, 3)
        assert (stats["fp_count"], stats["fp_n"]) == (1, 3)


class TestReports:
    """The same reports, with TP and FP added; zero-bit tables are TP and FP alone."""

    @staticmethod
    def _rows(zero_bit, n=4):
        rows = []
        for i in range(n):
            row = {"detected": i < 3, "detected_clean": i == 0,
                   "attacked_audio_quality_wm": {"pesq": 3.0}}
            if not zero_bit:
                row.update(accuracy=100.0 if i < 2 else 75.0, detection_valid=True)
            rows.append(row)
        return {f"f{i}.wav": {"attacks": {"GaussianNoiseAttack": row,
                                          "NoAttack": dict(row, detected=True)}}
                for i, row in enumerate(rows)}

    def _basic_tex(self, tmp_path, zero_bit):
        from deepmarkpy.utils.report_generator import BenchmarkReportGenerator
        stats = Benchmark.__new__(Benchmark).compute_mean_accuracy(
            self._rows(zero_bit), is_zero_bit=zero_bit)
        path = tmp_path / "benchmark_stats.json"
        path.write_text(json.dumps(stats))
        BenchmarkReportGenerator(str(tmp_path), is_zero_bit=zero_bit).generate_full_report(
            str(path), model_name="M")
        return (tmp_path / "benchmark_report.tex").read_text()

    def test_a_deciding_model_gets_tp_and_fp_columns(self, tmp_path, no_pdflatex):
        tex = self._basic_tex(tmp_path, zero_bit=False)
        assert "3/4 (75.0\\%)" in tex and "1/4 (25.0\\%)" in tex
        assert "true positives only" in tex
        # In a table of their own, so the accuracy table keeps its width.
        def table(label):
            return tex.split(label, 1)[1].split("\\end{longtable}", 1)[0]
        assert "TP & FP" in table("tab:benchmark_detection_").replace("  ", " ")
        assert "TP" not in table("tab:benchmark_accuracy_")

    def test_a_zero_bit_model_is_headlined_by_its_tp_rate(self, tmp_path, no_pdflatex):
        tex = self._basic_tex(tmp_path, zero_bit=True)
        assert "TP rate across the" in tex
        assert "no accuracy, BER or EMR" in tex
        assert "Mean" not in tex.split("\\section{Audio Distortion")[1].split("\\end{longtable}")[0]

    def test_a_model_without_decisions_keeps_its_table(self, tmp_path, no_pdflatex):
        from deepmarkpy.utils.report_generator import BenchmarkReportGenerator
        results = {f"f{i}.wav": {"attacks": {"GaussianNoiseAttack": {"accuracy": 90.0}}}
                   for i in range(2)}
        stats = Benchmark.__new__(Benchmark).compute_mean_accuracy(results)
        path = tmp_path / "benchmark_stats.json"
        path.write_text(json.dumps(stats))
        BenchmarkReportGenerator(str(tmp_path)).generate_full_report(str(path), model_name="M")
        tex = (tmp_path / "benchmark_report.tex").read_text()
        assert "& TP" not in tex and "true positives only" not in tex

    def test_the_detailed_report_counts_from_the_rows(self, tmp_path, no_pdflatex):
        from deepmarkpy.utils.detailed_report_generator import DetailedReportGenerator
        tex_path = DetailedReportGenerator(str(tmp_path)).generate_full_report(
            self._rows(zero_bit=False), model_name="M")
        tex = pathlib.Path(tex_path).read_text()
        assert "3/4 (75.0\\%)" in tex and "1/4 (25.0\\%)" in tex
        # Accuracy over the three detected rows: (100 + 100 + 75) / 3.
        assert "91.67\\%" in tex

    def test_the_comparison_shows_a_zero_bit_tp_rate_and_detection_tables(
            self, tmp_path, no_pdflatex):
        from deepmarkpy.utils.comparative_report_generator import ComparativeReportGenerator
        bench = Benchmark.__new__(Benchmark)
        all_stats = {
            "DecidingModel": bench.compute_mean_accuracy(self._rows(zero_bit=False)),
            "ZeroBitModel": bench.compute_mean_accuracy(self._rows(zero_bit=True),
                                                        is_zero_bit=True),
        }
        generator = ComparativeReportGenerator(str(tmp_path))
        generator.model_meta = {"ZeroBitModel": {"is_zero_bit": True}}
        tex = generator.generate_latex_report(all_stats, include_radar=False)
        assert "\\section{Detection Comparison}" in tex
        accuracy = next(block for block in tex.split("\\begin{longtable}")
                        if "tab:comparison_accuracy_mean" in block)
        assert "75.00" in accuracy, "the zero-bit column shows its TP rate"
