"""The CLI, driven end to end from config files and from flags: ``main()``
runs real native attacks against an in-process model plugin with no
``base_url``. An autouse fixture makes NISQA unavailable and the docker CLI
list no container, so no test contacts a service or runs the docker CLI."""

import json
import pathlib
import re
import runpy
import sys

import numpy as np
import pytest
import soundfile as sf

from deepmarkpy import run as run_module
from deepmarkpy.utils import efficiency, metrics

pytestmark = pytest.mark.usefixtures("no_pdflatex")

# The docker CLI call, which _no_services replaces in every test.
RUN_DOCKER = efficiency._run_docker


@pytest.fixture(autouse=True)
def _no_services(monkeypatch):
    """NISQA reads as unavailable without a request; docker lists nothing."""
    monkeypatch.setattr(metrics, "_NISQA_ENDPOINT", None)
    monkeypatch.setattr(metrics, "_nisqa_unavailable", True)
    monkeypatch.setattr(metrics, "_nisqa_reason", "disabled in tests")
    monkeypatch.setattr(efficiency, "_run_docker", lambda args: "")


MODEL_SOURCE = '''
import numpy as np

from deepmarkpy.core.base_model import BaseModel


class DummyWatermarkModel(BaseModel):
    """Adds a tiny deterministic offset and reads it back."""

    WATERMARK_SIZE = 16

    # Detection reliability calls detect() on clean audio before anything
    # has been embedded, so this must be readable from the start.
    _last = None

    def generate_watermark(self):
        return np.random.randint(0, 2, size=self.WATERMARK_SIZE)

    def embed(self, audio, watermark_data, sampling_rate, **kwargs):
        self._last = np.asarray(watermark_data)
        return np.asarray(audio) + 1e-4

    def detect(self, audio, sampling_rate, **kwargs):
        if self._last is None:
            return np.zeros(self.WATERMARK_SIZE, dtype=int)
        # Flip one bit when the signal has drifted, so attacked audio
        # scores below a clean read rather than always perfect.
        detected = self._last.copy()
        if float(np.mean(np.abs(np.asarray(audio)))) > 0.30:
            detected[0] = 1 - detected[0]
        return detected

    def is_watermarked(self, detect_output):
        return detect_output is not None and len(detect_output) > 0
'''

# No shipped attack declares parameter presets -- every plugin config.json
# holds one flat parameter set -- so the plugin-declared-version path is
# exercised by a throwaway attack this file installs, exactly as the model
# side is. Naming it without a version expands into one run per preset;
# pin one, and the report labels it with the version it used.
ATTACK_SOURCE = '''
import numpy as np

from deepmarkpy.core.base_attack import BaseAttack


class PresetNoiseAttack(BaseAttack):
    """Additive noise at the configured SNR, in three presets."""

    def apply(self, audio, **kwargs):
        snr_db = kwargs.get(
            "snr_db_preset_noise", self.config.get("snr_db_preset_noise"),
        )
        audio = np.asarray(audio, dtype=float)
        signal_power = float(np.mean(audio ** 2))
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = np.random.randn(*audio.shape) * np.sqrt(noise_power)
        return audio + noise
'''

ATTACK_CONFIG = {
    "default": {"snr_db_preset_noise": 35},
    "mild": {"snr_db_preset_noise": 45},
    "aggressive": {"snr_db_preset_noise": 15},
}

ATTACK_CLASS = "PresetNoiseAttack"
ATTACK_PARAM = "snr_db_preset_noise"
ATTACK_SPEC = f"{ATTACK_CLASS}:aggressive"
ATTACK_NAME = f"{ATTACK_CLASS} (aggressive)"

MODEL_CONFIG = {
    "sampling_rate": 16000,
    "watermark_size": 16,
    "is_zero_bit": False,
    "returns_confidence": False,
}


@pytest.fixture
def plugins_dir(tmp_path):
    directory = tmp_path / "plugins" / "dummy"
    directory.mkdir(parents=True)
    (directory / "model.py").write_text(MODEL_SOURCE)
    (directory / "config.json").write_text(json.dumps(MODEL_CONFIG))

    attack = tmp_path / "plugins" / "presets"
    attack.mkdir(parents=True)
    (attack / "attack.py").write_text(ATTACK_SOURCE)
    (attack / "config.json").write_text(json.dumps(ATTACK_CONFIG))
    return str(tmp_path / "plugins")


@pytest.fixture
def audio_dir(tmp_path):
    directory = tmp_path / "audio"
    directory.mkdir()
    rng = np.random.default_rng(0)
    for index in range(3):
        # Long enough for PESQ/STOI, which refuse very short clips.
        samples = 16000 + index * 4000
        signal = 0.3 * np.sin(
            2 * np.pi * 440 * np.arange(samples) / 16000
        ) + 0.01 * rng.standard_normal(samples)
        sf.write(directory / f"clip{index}.wav", signal.astype(np.float32), 16000)
    return str(directory)


def write_config(tmp_path, name, **overrides):
    """Write a config file from ``overrides``; a None value drops the key."""
    data = {
        "mode": "benchmark",
        "models": ["DummyWatermarkModel"],
        "calculate_quality_metrics": True,
        "statistics": ["mean", "worst_case"],
        "attacks": {"list": [ATTACK_SPEC]},
        "metrics": {"defaults": {
            "accuracy": {"enabled": True},
            "ber": {"enabled": True},
            "emr": {"enabled": True},
            "pesq": {"enabled": True},
            "stoi": {"enabled": True},
            "psnr": {"enabled": False},
            "si_sdr": {"enabled": False},
            "mcd": {"enabled": False},
            "visqol": {"enabled": False},
            "sii": {"enabled": False},
            "ncm": {"enabled": False},
            "nisqa_mos": {"enabled": False},
            "nisqa_noi": {"enabled": False},
            "nisqa_dis": {"enabled": False},
            "nisqa_col": {"enabled": False},
            "nisqa_loud": {"enabled": False},
        }},
    }
    data.update(overrides)
    data = {key: value for key, value in data.items() if value is not None}
    if data["mode"] == "detection_reliability":
        # BER does not apply to this mode.
        data.get("metrics", {}).get("defaults", {}).pop("ber", None)
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def resolved_for(tmp_path, plugins_dir, **overrides):
    """The parameters each row of this config's run would apply."""
    from deepmarkpy.benchmark import Benchmark
    from deepmarkpy.config import load_configs

    benchmark = Benchmark(external_plugins_dir=plugins_dir)
    config = load_configs(
        [write_config(tmp_path, "resolved.json", **overrides)],
        benchmark.attacks, benchmark.models,
    )[0]
    return run_module.resolved_attack_parameters(benchmark, config)


def run_cli(*argv):
    return run_module.main(list(argv))


def write_model_plugin(plugins_dir, source, directory="dummy"):
    """Install ``source`` as a model plugin, replacing any already there."""
    target = pathlib.Path(plugins_dir) / directory
    target.mkdir(exist_ok=True)
    (target / "model.py").write_text(source)
    (target / "config.json").write_text(json.dumps(MODEL_CONFIG))


# Two config-defined versions whose names no filename can hold verbatim:
# one has a path separator, the other a character Windows reserves.
UNSAFE_VERSIONS = {
    "attacks": {"list": [f"{ATTACK_CLASS}:lo/hi", f"{ATTACK_CLASS}:v:2"]},
    "attack_parameters": {
        f"{ATTACK_CLASS}:lo/hi": {ATTACK_PARAM: 20},
        f"{ATTACK_CLASS}:v:2": {ATTACK_PARAM: 20},
    },
}
UNSAFE_VERSION_NAMES = {f"{ATTACK_CLASS} (lo/hi)", f"{ATTACK_CLASS} (v:2)"}
UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


def saved_audio(report_dir):
    """The files a single-mode run saved, checked to sit directly in audio/."""
    audio = report_dir / "audio"
    saved = [path for path in audio.rglob("*") if path.is_file()]
    assert {path.parent for path in saved} == {audio}, saved
    return saved


class TestBenchmarkMode:
    @pytest.mark.parametrize("grouped", [False, True])
    def test_zero_bit_stats_do_not_contain_ber(
        self, tmp_path, plugins_dir, audio_dir, grouped,
    ):
        directory = pathlib.Path(plugins_dir) / "dummy"
        source = MODEL_SOURCE.replace(
            "return detected", "return bool(np.array_equal(detected, self._last))",
        )
        (directory / "model.py").write_text(source)
        (directory / "config.json").write_text(json.dumps({
            **MODEL_CONFIG, "is_zero_bit": True,
        }))
        overrides = {}
        if grouped:
            overrides["duration_groups"] = {
                "boundaries": [1.1], "include_overall": False,
            }
        config = write_config(tmp_path, "zero_bit.json", **overrides)
        report_dir = tmp_path / "report"
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_OK

        stats = json.loads((report_dir / "benchmark_stats.json").read_text())
        groups = [group["stats"] for group in stats.values()] if grouped else [stats]
        assert groups
        for group in groups:
            entry = group[ATTACK_NAME]
            assert entry["accuracy_n"] > 0
            assert not any(key.startswith("ber_") for key in entry)
        for name in ("benchmark_report.tex", "detailed_report.tex"):
            tex = (report_dir / name).read_text()
            assert "BER" not in tex and "Bit error rate" not in tex

    def test_produces_results_stats_metadata_and_reports(
        self, tmp_path, plugins_dir, audio_dir, monkeypatch,
    ):
        """Only the configured metrics and statistics, no timings, no docker calls."""
        docker_calls = []
        monkeypatch.setattr(efficiency, "_run_docker",
                            lambda args: docker_calls.append(args))
        config = write_config(tmp_path, "benchmark.json")
        report_dir = tmp_path / "report"

        assert run_cli(
            "--config", config,
            "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir),
            "--plugins_dir", plugins_dir,
            "--seed", "7",
        ) == run_module.EXIT_OK

        for name in ("benchmark_results.json", "benchmark_stats.json",
                     "run_metadata.json", "benchmark_report.tex",
                     "detailed_report.tex", "benchmark_chart.png"):
            assert (report_dir / name).exists(), f"{name} was not written"

        metadata = json.loads((report_dir / "run_metadata.json").read_text())
        assert metadata["seed"] == 7
        assert metadata["models"] == ["DummyWatermarkModel"]
        assert metadata["config_file"] == config
        assert metadata["n_files"] == 3

        stats = json.loads((report_dir / "benchmark_stats.json").read_text())
        entry = stats[ATTACK_NAME]
        assert {"accuracy_mean", "accuracy_worst_case", "accuracy_n"} <= set(entry)
        assert "pesq_mean" in entry and "stoi_mean" in entry
        for metric in ("mcd", "psnr", "si_sdr", "sii", "ncm", "nisqa_mos"):
            assert not any(k.startswith(f"{metric}_") for k in entry), (
                f"{metric} was computed despite being disabled in the config"
            )
        assert "accuracy_median" not in entry

        # No timing is taken at all, rather than taken and filtered out.
        results = json.loads((report_dir / "benchmark_results.json").read_text())
        measured = next(iter(results.values()))["attacks"][ATTACK_NAME]
        assert not [k for k in measured if "latency" in k], measured
        assert not [k for k in entry if "latency" in k], entry
        tex = (report_dir / "benchmark_report.tex").read_text()
        assert "tab:benchmark_efficiency" not in tex
        assert "Attack time" not in tex
        detailed = (report_dir / "detailed_report.tex").read_text()
        assert "tab:efficiency_" not in detailed
        assert "Attack time" not in detailed
        assert docker_calls == [], docker_calls

    def test_a_bare_key_does_not_leak_onto_a_named_version(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """A bare class name targets the default version, not the named ones."""
        config = write_config(
            tmp_path, "bare.json",
            attacks={"list": ["PresetNoiseAttack:mild",
                              "PresetNoiseAttack:aggressive"]},
            attack_parameters={"PresetNoiseAttack": {"snr_db_preset_noise": 25}},
        )
        report_dir = tmp_path / "report"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
                "--seed", "3")

        attacks = next(iter(json.loads(
            (report_dir / "benchmark_results.json").read_text()).values()))["attacks"]
        mild = attacks["PresetNoiseAttack (mild)"]["attack_snr_db"]
        aggressive = attacks["PresetNoiseAttack (aggressive)"]["attack_snr_db"]

        assert mild > aggressive + 20, (
            f"the bare override reached both versions: {mild} vs {aggressive}"
        )
        # The presets, untouched: mild is 45 dB and aggressive is 15 dB.
        assert 40 < mild < 50 and 10 < aggressive < 20

    def test_naming_codec2_bare_runs_the_defined_version_too(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """One run per bitrate of each version, a config-defined one included."""
        pytest.importorskip("pycodec2")
        config = write_config(
            tmp_path, "codec2.json",
            attacks={"list": ["Codec2VocoderAttack"]},
            attack_parameters={
                "Codec2VocoderAttack": {"bitrate_codec2": [700]},
                "Codec2VocoderAttack:hi": {"bitrate_codec2": [3200]},
            },
        )
        report_dir = tmp_path / "report"
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_OK

        attacks = next(iter(json.loads(
            (report_dir / "benchmark_results.json").read_text()).values()))["attacks"]
        assert set(attacks) == {
            "Codec2VocoderAttack_700 (default)",
            "Codec2VocoderAttack_3200 (hi)",
        }

    def test_saved_audio_survives_any_version_name(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """Results keep a version name verbatim; its saved file gets a safe one."""
        config = write_config(tmp_path, "unsafe.json", **UNSAFE_VERSIONS)
        report_dir = tmp_path / "report"
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
            "--save_audio",
        ) == run_module.EXIT_OK

        attacks = next(iter(json.loads(
            (report_dir / "benchmark_results.json").read_text()).values()))["attacks"]
        assert set(attacks) == UNSAFE_VERSION_NAMES
        saved = saved_audio(report_dir)
        # Per clip: the watermarked audio, and the attacked audio of each
        # version.
        assert len(saved) == 3 * (1 + len(UNSAFE_VERSION_NAMES))
        assert not [p.name for p in saved if UNSAFE_FILENAME_CHARS.search(p.name)]

    def test_crop_before_attack_is_applied(self, tmp_path, plugins_dir, audio_dir):
        config = write_config(tmp_path, "crop.json", crop_before_attack=25)
        report_dir = tmp_path / "report"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)

        tex = (report_dir / "benchmark_report.tex").read_text()
        assert "crop of 25.0\\%" in tex

    def test_duration_groups_produce_one_part_per_bin(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """One part per bin, each stating the embedding cost."""
        config = write_config(
            tmp_path, "durations.json",
            duration_groups={"boundaries": [1.2], "include_overall": True},
            efficiency={"enabled": True, "metrics": {
                "embed_latency": {"statistics": ["mean"]},
                "attack_latency": {"statistics": ["mean"]},
            }},
        )
        report_dir = tmp_path / "report"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)

        stats = json.loads((report_dir / "benchmark_stats.json").read_text())
        assert "Overall" in stats
        assert any(label.startswith("<") for label in stats)

        tex = (report_dir / "benchmark_report.tex").read_text()
        parts = tex.count("\\part{")
        assert parts >= 2, tex
        assert tex.count("Embedding cost per file") == parts

    def test_overall_is_dropped_when_every_file_lands_in_one_bin(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """With one bin, "Overall" would repeat it verbatim, so it is dropped."""
        config = write_config(
            tmp_path, "one_bin.json",
            duration_groups={"boundaries": [30], "include_overall": True},
        )
        report_dir = tmp_path / "report"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)

        stats = json.loads((report_dir / "benchmark_stats.json").read_text())
        assert len(stats) == 1, stats
        assert "Overall" not in stats
        assert (report_dir / "benchmark_report.tex").read_text().count(
            "\\part{"
        ) == 1

    @pytest.mark.parametrize("include_overall", [True, False])
    def test_comparison_stats_cover_every_bin(
        self, tmp_path, plugins_dir, audio_dir, monkeypatch, include_overall,
    ):
        """The flat stats the comparative report ranks cover every bin's files."""
        returned = []
        original = run_module.run_single_model

        def capture(*args, **kwargs):
            out = original(*args, **kwargs)
            returned.append(out)
            return out

        monkeypatch.setattr(run_module, "run_single_model", capture)
        config = write_config(
            tmp_path, "bins.json",
            duration_groups={"boundaries": [1.2],
                             "include_overall": include_overall},
        )
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(tmp_path / "report"),
                "--plugins_dir", plugins_dir)

        (_, _, stats), = returned
        assert stats[ATTACK_NAME]["accuracy_n"] == 3


class TestNoAttacksMode:
    def test_writes_its_own_report_and_no_attack_results(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        config = write_config(
            tmp_path, "no_attacks.json", mode="no_attacks", attacks=None,
        )
        report_dir = tmp_path / "report"
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_OK

        assert (report_dir / "no_attacks_report.tex").exists()
        assert (report_dir / "no_attacks_DummyWatermarkModel.json").exists()

        tex = (report_dir / "no_attacks_report.tex").read_text()
        assert "PESQ" in tex
        assert "MCD" not in tex, "a disabled metric reached the report"


class TestDetectionReliabilityMode:
    def test_writes_reliability_results_and_report(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """Writes results and report; with no efficiency section, times nothing."""
        config = write_config(
            tmp_path, "dr.json", mode="detection_reliability",
        )
        report_dir = tmp_path / "report"
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_OK

        assert (report_dir / "detection_reliability_report.tex").exists()
        data = json.loads((report_dir / "detection_reliability.json").read_text())
        assert data["n_files"] == 3
        assert ATTACK_NAME in data["attacks"]

        entry = data["attacks"][ATTACK_NAME]
        assert {"accuracy_mean", "accuracy_worst_case"} <= set(entry)
        assert entry["false_positive_attempts"] == 3

        assert not entry.get("timings"), entry.get("timings")
        assert not data["no_attack"].get("timings")
        record = next(iter(data["per_file"].values()))
        assert not [k for k in record if "latency" in k], record
        for attack_record in record["attacks"].values():
            assert not [k for k in attack_record if "latency" in k], attack_record
        tex = (report_dir / "detection_reliability_report.tex").read_text()
        assert "tab:dr_efficiency" not in tex
        assert "Attack time" not in tex

    def test_saved_audio_survives_any_version_name(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """This mode's own run loop saves audio under the same naming rule."""
        config = write_config(
            tmp_path, "unsafe.json", mode="detection_reliability",
            **UNSAFE_VERSIONS,
        )
        report_dir = tmp_path / "report"
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
            "--save_audio",
        ) == run_module.EXIT_OK

        data = json.loads((report_dir / "detection_reliability.json").read_text())
        assert set(data["attacks"]) == UNSAFE_VERSION_NAMES
        saved = saved_audio(report_dir)
        # Per clip: the watermarked audio, and each version applied to the
        # clean and to the watermarked audio.
        assert len(saved) == 3 * (1 + 2 * len(UNSAFE_VERSION_NAMES))
        assert not [p.name for p in saved if UNSAFE_FILENAME_CHARS.search(p.name)]


class TestSeveralModesInOneInvocation:
    @pytest.mark.parametrize("repeated", [False, True])
    def test_every_mode_runs_and_none_erases_another(
        self, tmp_path, plugins_dir, audio_dir, repeated,
    ):
        """Each mode reports and saves its audio in audio/<mode>/."""
        benchmark = write_config(tmp_path, "benchmark.json")
        dr = write_config(tmp_path, "dr.json", mode="detection_reliability")
        no_attacks = write_config(tmp_path, "no_attacks.json",
                                  mode="no_attacks", attacks=None)

        configs = (
            ["--config", benchmark, "--config", dr, "--config", no_attacks]
            if repeated else ["--config", benchmark, dr, no_attacks]
        )
        report_dir = tmp_path / "report"
        assert run_cli(
            *configs,
            "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir),
            "--plugins_dir", plugins_dir,
            "--save_audio",
        ) == run_module.EXIT_OK

        assert (report_dir / "benchmark_report.tex").exists()
        assert (report_dir / "detection_reliability_report.tex").exists()
        assert (report_dir / "no_attacks_report.tex").exists()
        for mode in ("benchmark", "detection_reliability", "no_attacks"):
            assert (report_dir / "audio" / mode / "clip0_watermarked.wav").exists()


class TestFailureModes:
    # The deprecated launcher, which scripts run by path.
    LAUNCHER = pathlib.Path(__file__).resolve().parents[1] / "src" / "run.py"

    @staticmethod
    def _model_whose_service_drops(name="DummyWatermarkModel"):
        """The test model as ``name``, failing in embed like a stopped container."""
        return MODEL_SOURCE.replace("DummyWatermarkModel", name).replace(
            "self._last = np.asarray(watermark_data)",
            'raise ConnectionError("service dropped")',
        )

    def test_a_bad_config_exits_two_without_running(self, tmp_path, plugins_dir):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"mode": "benchmark", "models": ["Nope"]}))
        assert run_cli(
            "--config", str(path), "--wav_files_dir", str(tmp_path),
            "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_CONFIG_ERROR

    def test_a_missing_audio_directory_exits_one(
        self, tmp_path, plugins_dir,
    ):
        config = write_config(tmp_path, "benchmark.json")
        assert run_cli(
            "--config", config,
            "--wav_files_dir", str(tmp_path / "nope"),
            "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_RUNTIME_ERROR

    def test_no_audio_directory_anywhere_is_a_config_error(
        self, tmp_path, plugins_dir,
    ):
        config = write_config(tmp_path, "benchmark.json")
        assert run_cli(
            "--config", config, "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_CONFIG_ERROR

    def test_validate_only_needs_no_audio_directory(
        self, tmp_path, plugins_dir, caplog,
    ):
        """Validation needs no general.wav_files_dir, which --init leaves null."""
        config = write_config(tmp_path, "benchmark.json",
                              general={"wav_files_dir": None})
        with caplog.at_level("INFO"):
            assert run_cli(
                "--config", config, "--plugins_dir", plugins_dir,
                "--validate-only",
            ) == run_module.EXIT_OK
        # The summary says how to set it rather than printing None.
        assert "audio=not set" in caplog.text

    def test_validate_only_runs_nothing(self, tmp_path, plugins_dir, audio_dir):
        config = write_config(tmp_path, "benchmark.json")
        report_dir = tmp_path / "report"
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
            "--validate-only",
        ) == run_module.EXIT_OK
        assert not report_dir.exists(), "--validate-only wrote output"

    def test_an_unknown_cross_model_second_model_is_a_config_error(
        self, tmp_path, plugins_dir, audio_dir, capsys,
    ):
        """E011, before the report directory is cleared or any audio embedded."""
        embedded = tmp_path / "embedded"
        write_model_plugin(plugins_dir, MODEL_SOURCE.replace(
            "self._last = np.asarray(watermark_data)",
            f"open({str(embedded)!r}, 'w').close()\n"
            "        self._last = np.asarray(watermark_data)",
        ))
        config = write_config(
            tmp_path, "cross.json",
            attacks={"list": [ATTACK_SPEC, "CrossModelAttack"]},
            attack_parameters={"CrossModelAttack": {
                "different_model_name_cross_model": "DummyWatermarkModl",
            }},
        )
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        previous = report_dir / "benchmark_report.tex"
        previous.write_text("previous run")
        common = ("--config", config, "--wav_files_dir", audio_dir,
                  "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)

        assert run_cli(*common, "--validate-only") == run_module.EXIT_CONFIG_ERROR
        assert "[E011]" in capsys.readouterr().err
        assert run_cli(*common) == run_module.EXIT_CONFIG_ERROR
        assert "[E011]" in capsys.readouterr().err
        assert previous.read_text() == "previous run"
        assert not embedded.exists(), "audio was embedded before the check"

    @pytest.mark.parametrize("mode", ["benchmark", "detection_reliability"])
    def test_a_group_with_an_undiscovered_member_is_a_config_error(
        self, tmp_path, plugins_dir, audio_dir, monkeypatch, capsys, mode,
    ):
        """E014 for a member that failed to import, before anything is cleared."""
        from deepmarkpy.plugin_manager import PluginManager

        load = PluginManager._load_attacks

        def load_without_pink_noise(self):
            # What a missing optional dependency does: the class is absent
            # and the import error is recorded.
            load(self)
            self.attacks.pop("PinkNoiseAttack")
            self.failed["deepmarkpy.plugins.attacks.pink_noise.attack"] = (
                "No module named 'not_installed'"
            )

        monkeypatch.setattr(PluginManager, "_load_attacks", load_without_pink_noise)
        config = write_config(
            tmp_path, f"{mode}.json", mode=mode,
            attacks={"groups": ["audio_distortion"]},
        )
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        previous = report_dir / "previous_results.json"
        previous.write_text('{"completed": true}')
        common = ("--config", config, "--wav_files_dir", audio_dir,
                  "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)

        assert run_cli(*common, "--validate-only") == run_module.EXIT_CONFIG_ERROR
        assert "[E014]" in capsys.readouterr().err
        assert run_cli(*common) == run_module.EXIT_CONFIG_ERROR
        assert "[E014]" in capsys.readouterr().err
        assert previous.read_text() == '{"completed": true}'

    def test_a_reliability_model_without_is_watermarked_is_a_config_error(
        self, tmp_path, plugins_dir, audio_dir, capsys,
    ):
        """E044 for a model without is_watermarked(), from a file or the flags."""
        write_model_plugin(
            plugins_dir, MODEL_SOURCE.split("    def is_watermarked")[0],
        )
        config = tmp_path / "dr.json"
        config.write_text(json.dumps({
            "mode": "detection_reliability", "models": ["DummyWatermarkModel"],
        }))
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        previous = report_dir / "detection_reliability.json"
        previous.write_text('{"completed": true}')
        common = ("--wav_files_dir", audio_dir, "--report_dir", str(report_dir),
                  "--plugins_dir", plugins_dir)

        assert run_cli(
            "--config", str(config), "--validate-only", *common,
        ) == run_module.EXIT_CONFIG_ERROR
        assert "[E044]" in capsys.readouterr().err
        assert run_cli(
            "--config", str(config), *common,
        ) == run_module.EXIT_CONFIG_ERROR
        assert "[E044]" in capsys.readouterr().err
        assert run_cli(
            "--detection_reliability", "--wm_model", "DummyWatermarkModel",
            *common,
        ) == run_module.EXIT_CONFIG_ERROR
        assert "[E044]" in capsys.readouterr().err
        assert previous.read_text() == '{"completed": true}'

    @pytest.mark.parametrize("failure", ["config", "runtime"])
    def test_the_deprecated_launcher_exits_with_mains_code(
        self, tmp_path, plugins_dir, monkeypatch, failure,
    ):
        """python src/run.py exits with the code main() returned."""
        if failure == "config":
            argv = ["--config", str(tmp_path / "missing.json"),
                    "--wav_files_dir", str(tmp_path)]
            expected = run_module.EXIT_CONFIG_ERROR
        else:
            argv = ["--config", write_config(tmp_path, "benchmark.json"),
                    "--wav_files_dir", str(tmp_path / "nope"),
                    "--plugins_dir", plugins_dir]
            expected = run_module.EXIT_RUNTIME_ERROR

        monkeypatch.setattr(sys, "argv", [str(self.LAUNCHER), *argv])
        with pytest.warns(DeprecationWarning):
            with pytest.raises(SystemExit) as exit_info:
                runpy.run_path(str(self.LAUNCHER), run_name="__main__")
        assert exit_info.value.code == expected

    def test_a_reliability_run_that_fails_exits_one(
        self, tmp_path, plugins_dir, audio_dir, monkeypatch,
    ):
        """A mode that produced nothing fails the run."""
        def fail(*args, **kwargs):
            raise ValueError("the run could not start")

        monkeypatch.setattr(
            "deepmarkpy.utils.detection_reliability.run_detection_reliability",
            fail,
        )
        config = tmp_path / "dr.json"
        config.write_text(json.dumps({
            "mode": "detection_reliability", "models": ["DummyWatermarkModel"],
        }))
        report_dir = tmp_path / "report"
        assert run_cli(
            "--config", str(config), "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_RUNTIME_ERROR
        assert not (report_dir / "detection_reliability.json").exists()

    def test_a_baseline_whose_only_model_fails_exits_one(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        write_model_plugin(plugins_dir, self._model_whose_service_drops())
        config = tmp_path / "no_attacks.json"
        config.write_text(json.dumps({
            "mode": "no_attacks", "models": ["DummyWatermarkModel"],
        }))
        assert run_cli(
            "--config", str(config), "--wav_files_dir", audio_dir,
            "--report_dir", str(tmp_path / "report"),
            "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_RUNTIME_ERROR

    def test_a_comparison_where_every_model_fails_exits_one(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        write_model_plugin(plugins_dir, self._model_whose_service_drops())
        write_model_plugin(
            plugins_dir, self._model_whose_service_drops("OtherWatermarkModel"),
            "other",
        )
        config = write_config(
            tmp_path, "benchmark.json",
            models=["DummyWatermarkModel", "OtherWatermarkModel"],
        )
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(tmp_path / "report"),
            "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_RUNTIME_ERROR

    def test_a_comparison_one_model_survives_exits_zero(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """A model lost to an infrastructure failure is skipped; the run succeeds."""
        write_model_plugin(
            plugins_dir, self._model_whose_service_drops("OtherWatermarkModel"),
            "other",
        )
        config = write_config(
            tmp_path, "benchmark.json",
            models=["DummyWatermarkModel", "OtherWatermarkModel"],
        )
        report_dir = tmp_path / "report"
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_OK
        assert (report_dir / "DummyWatermarkModel" / "benchmark_results.json").exists()


class TestCliOverridesConfig:
    @pytest.mark.parametrize("extra", [
        [], ["--seed", "-1"], ["--seed", str(2**32), "--validate-only"],
    ])
    def test_invalid_seed_preserves_existing_reports(
        self, tmp_path, plugins_dir, audio_dir, extra,
    ):
        """An out-of-range seed, from the config or --seed, is refused."""
        config = write_config(
            tmp_path, "benchmark.json", general={"seed": 7 if extra else 2**32},
        )
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        existing = report_dir / "previous.json"
        existing.write_text('{"completed": true}')
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir), "--plugins_dir", plugins_dir, *extra,
        ) == run_module.EXIT_CONFIG_ERROR
        assert existing.read_text() == '{"completed": true}'

    def test_cli_wins_for_the_keys_that_exist_in_both(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        config = write_config(
            tmp_path, "benchmark.json",
            general={"report_dir": str(tmp_path / "from_config"), "seed": 1},
        )
        cli_dir = tmp_path / "from_cli"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(cli_dir), "--plugins_dir", plugins_dir,
                "--seed", "99")

        assert (cli_dir / "benchmark_report.tex").exists()
        assert not (tmp_path / "from_config").exists()
        metadata = json.loads((cli_dir / "run_metadata.json").read_text())
        assert metadata["seed"] == 99

    def test_config_applies_when_the_cli_is_silent(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        config_dir = tmp_path / "from_config"
        config = write_config(
            tmp_path, "benchmark.json",
            general={"wav_files_dir": audio_dir,
                     "report_dir": str(config_dir), "seed": 5},
        )
        run_cli("--config", config, "--plugins_dir", plugins_dir)

        metadata = json.loads((config_dir / "run_metadata.json").read_text())
        assert metadata["seed"] == 5


class TestServiceProbe:
    """A container that answers slowly is waited for, not reported as down."""

    @staticmethod
    def _benchmark():
        model_cls = type("Slow", (), {"__init__": lambda self: setattr(
            self, "base_url", "http://localhost:9/")})
        return type("_B", (), {"models": {"SlowModel": {"class": model_cls}}})()

    @staticmethod
    def _config():
        from deepmarkpy.config import ModeConfig
        return [ModeConfig(mode="benchmark", source="c.json",
                           models=["SlowModel"])]

    def test_a_service_that_answers_late_is_reachable(self, monkeypatch):
        import requests

        calls = {"n": 0}

        def flaky_get(url, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise requests.ConnectionError("still starting")
            return "ok"

        sleeps = []
        monkeypatch.setattr(requests, "get", flaky_get)
        monkeypatch.setattr("time.sleep", sleeps.append)
        messages = run_module._unreachable_model_services(
            self._benchmark(), self._config(),
        )
        assert messages == []
        assert calls["n"] == 3, "gave up before the service came up"
        # One pause between each pair of attempts, so a refused port is not
        # hit in a tight loop.
        assert sleeps == [run_module._SERVICE_PROBE_INTERVAL_S] * 2

    def test_a_service_that_never_answers_is_reported(self, monkeypatch):
        import requests

        monkeypatch.setattr(run_module, "_SERVICE_PROBE_BUDGET_S", 0)
        monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(
            requests.ConnectionError("down")))

        messages = run_module._unreachable_model_services(
            self._benchmark(), self._config(),
        )
        assert len(messages) == 1
        assert "docker compose up -d slow" in messages[0]

    def test_validate_only_does_not_wait(self, monkeypatch):
        """--validate-only is the flag you use BEFORE starting containers."""
        import requests

        calls = {"n": 0}

        def always_down(url, timeout=None):
            calls["n"] += 1
            raise requests.ConnectionError("down")

        monkeypatch.setattr(requests, "get", always_down)
        messages = run_module._unreachable_model_services(
            self._benchmark(), self._config(), wait=False,
        )
        assert len(messages) == 1
        assert calls["n"] == 1, "probed more than once with wait=False"


class TestParameterProvenance:
    """Which version ran with which values, before and after the run."""

    # Two presets overridden and two versions defined, all run through the
    # bare name.
    LADDER = {
        "PresetNoiseAttack:mild": {"snr_db_preset_noise": 45},
        "PresetNoiseAttack": {"snr_db_preset_noise": 30},
        "PresetNoiseAttack:aggressive": {"snr_db_preset_noise": 20},
        "PresetNoiseAttack:brutal": {"snr_db_preset_noise": 10},
        "PresetNoiseAttack:extreme": {"snr_db_preset_noise": 3},
    }
    EXPECTED = {
        "PresetNoiseAttack (mild)": 45,
        "PresetNoiseAttack (default)": 30,
        "PresetNoiseAttack (aggressive)": 20,
        "PresetNoiseAttack (brutal)": 10,
        "PresetNoiseAttack (extreme)": 3,
    }

    def _config(self, tmp_path):
        return write_config(
            tmp_path, "ladder.json",
            attacks={"list": ["PresetNoiseAttack"]},
            attack_parameters=self.LADDER,
        )

    def test_validate_only_reports_every_version(
        self, tmp_path, plugins_dir, audio_dir, caplog,
    ):
        with caplog.at_level("INFO"):
            run_cli("--config", self._config(tmp_path),
                    "--wav_files_dir", audio_dir,
                    "--plugins_dir", plugins_dir, "--validate-only")
        logged = "\n".join(r.message for r in caplog.records)
        for name, snr in self.EXPECTED.items():
            assert f"{name}: snr_db_preset_noise={snr}" in logged, (
                f"{name} not reported with its value"
            )

    def test_run_metadata_matches_what_each_version_measured(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """The values recorded are the ones that reached the signal."""
        report_dir = tmp_path / "report"
        run_cli("--config", self._config(tmp_path),
                "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
                "--seed", "1")

        resolved = json.loads(
            (report_dir / "run_metadata.json").read_text()
        )["attack_parameters_resolved"]
        assert {
            name: params["snr_db_preset_noise"]
            for name, params in resolved.items()
        } == self.EXPECTED

        attacks = next(iter(json.loads(
            (report_dir / "benchmark_results.json").read_text()).values()))["attacks"]
        assert set(attacks) == set(self.EXPECTED)
        for name, target in self.EXPECTED.items():
            measured = attacks[name]["attack_snr_db"]
            assert abs(measured - target) < 1.0, (
                f"{name} asked for {target} dB but measured {measured}"
            )

    def test_resolved_parameters_merge_preset_and_override(
        self, tmp_path, plugins_dir,
    ):
        """An override names only what it changes; the rest is the preset."""
        entry = resolved_for(
            tmp_path, plugins_dir,
            attacks={"list": ["FlipSamplesAttack"]},
            attack_parameters={"FlipSamplesAttack": {"num_flip_samples": 50}},
        )["FlipSamplesAttack"]
        assert entry["num_flip_samples"] == 50, "override not applied"
        assert "duration_flip_samples" in entry, (
            "untouched plugin parameters are missing, so the record is "
            "not what apply() will see"
        )


class TestParameterReportingPerMode:
    """The parameters reported are those of the attacks the mode runs."""

    def test_no_attacks_mode_reports_no_attack_parameters(self, tmp_path, plugins_dir):
        """This mode applies none, so listing every attack would be nonsense."""
        assert resolved_for(
            tmp_path, plugins_dir, mode="no_attacks", attacks=None,
        ) == {}

    def test_detection_reliability_empty_selection_is_no_attacks(
        self, tmp_path, plugins_dir,
    ):
        """Empty means baseline-only here, not 'every attack'."""
        assert resolved_for(
            tmp_path, plugins_dir, mode="detection_reliability",
            attacks={"groups": [], "list": []},
        ) == {}

    def test_benchmark_empty_selection_is_every_attack(self, tmp_path, plugins_dir):
        assert len(resolved_for(
            tmp_path, plugins_dir, attacks={"groups": [], "list": []},
        )) > 20

    def test_plugin_documentation_keys_are_not_reported_as_parameters(
        self, tmp_path, plugins_dir,
    ):
        """Codec2's config.json carries a "_comment" note of its own."""
        from deepmarkpy.benchmark import Benchmark
        from deepmarkpy.config import load_configs

        path = tmp_path / "c.json"
        path.write_text(json.dumps({
            "mode": "benchmark", "models": ["DummyWatermarkModel"],
            "calculate_quality_metrics": False,
            "attacks": {"list": ["Codec2VocoderAttack"]},
        }))
        benchmark = Benchmark(external_plugins_dir=plugins_dir)
        if "Codec2VocoderAttack" not in benchmark.attacks:
            pytest.skip("Codec2 plugin not installed")
        config = load_configs([str(path)], benchmark.attacks, benchmark.models)[0]

        for params in run_module.resolved_attack_parameters(benchmark, config).values():
            assert not [k for k in params if k.startswith("_")], params


class TestEfficiencySection:
    """Timings are measured only when the efficiency section asks for them."""

    EFFICIENCY = {
        "enabled": True,
        "metrics": {
            "embed_latency": {"statistics": ["mean"]},
            "detect_latency": {"statistics": ["mean"]},
            "attack_latency": {"statistics": ["mean", "worst_case"]},
        },
    }

    def _run(self, tmp_path, plugins_dir, audio_dir, name, **overrides):
        config = write_config(tmp_path, name, **overrides)
        report_dir = tmp_path / name.replace(".json", "")
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)
        return report_dir

    def test_timings_are_recorded_and_tabled_when_enabled(
        self, tmp_path, plugins_dir, audio_dir, caplog,
    ):
        from deepmarkpy.utils.efficiency import TERMINAL_TAG

        with caplog.at_level("INFO"):
            report_dir = self._run(tmp_path, plugins_dir, audio_dir,
                                   "eff_on.json", efficiency=self.EFFICIENCY)

        # Timing output carries its own tag, so it can be filtered on its own.
        tagged = [r.message for r in caplog.records if TERMINAL_TAG in r.message]
        assert any("Attack time" in line for line in tagged), tagged

        results = json.loads((report_dir / "benchmark_results.json").read_text())
        entry = next(iter(next(iter(results.values()))["attacks"].values()))
        for metric in ("embed_latency", "detect_latency", "attack_latency"):
            assert entry[metric] > 0, f"{metric} was not measured"

        stats = json.loads((report_dir / "benchmark_stats.json").read_text())
        row = next(iter(stats.values()))
        assert row["attack_latency_mean"] > 0
        assert "attack_latency_worst_case" in row

        tex = (report_dir / "benchmark_report.tex").read_text()
        assert "tab:benchmark_efficiency" in tex
        assert "Attack time (s)" in tex

        # The detailed report aggregates from the raw per-file results, so
        # it has to collect the timings itself rather than inherit them.
        detailed = (report_dir / "detailed_report.tex").read_text()
        assert "tab:efficiency_" in detailed
        assert "Attack time (s)" in detailed

        # Embedding is measured once per file and does not vary by attack,
        # so it is stated for the run rather than repeated down a column.
        for document in (tex, detailed):
            assert "Embedding cost per file" in document
            assert "Embed time (s)" not in document

    def test_a_disabled_metric_is_left_out_of_the_table(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        report_dir = self._run(
            tmp_path, plugins_dir, audio_dir, "eff_partial.json",
            efficiency={"enabled": True, "metrics": {
                "attack_latency": {"statistics": ["mean"]},
                "embed_latency": {"enabled": False},
                "detect_latency": {"enabled": False},
            }},
        )
        tex = (report_dir / "benchmark_report.tex").read_text()
        assert "Attack time (s)" in tex
        assert "Embed time" not in tex and "Detect time" not in tex


class TestReliabilityModeTimings:
    """The reliability mode times its own run loop, as the other modes do."""

    def _run(self, tmp_path, plugins_dir, audio_dir, **overrides):
        config = write_config(
            tmp_path, "reliability_eff.json", mode="detection_reliability",
            efficiency={"enabled": True, "metrics": {
                "embed_latency": {"statistics": ["mean"]},
                "detect_latency": {"statistics": ["mean"]},
                "attack_latency": {"statistics": ["mean", "worst_case"]},
            }},
            **overrides,
        )
        report_dir = tmp_path / "reliability"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)
        return report_dir

    def test_timings_are_measured_and_tabled(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        report_dir = self._run(tmp_path, plugins_dir, audio_dir)

        result = json.loads(
            (report_dir / "detection_reliability.json").read_text()
        )
        attack = next(iter(result["attacks"].values()))
        assert attack["timings"]["attack_latency"]["mean"] > 0
        assert attack["timings"]["detect_latency"]["mean"] > 0
        assert result["no_attack"]["timings"]["embed_latency"]["mean"] > 0

        tex = (report_dir / "detection_reliability_report.tex").read_text()
        assert "tab:dr_efficiency" in tex
        assert "Attack time (s)" in tex
        # Embedding does not vary by attack, so it is stated once.
        assert "Embedding cost per file" in tex
        assert "Embed time (s)" not in tex

    def test_every_duration_part_states_the_embedding_cost(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """Each part, rebuilt from per-file records, keeps the baseline timings."""
        report_dir = self._run(
            tmp_path, plugins_dir, audio_dir,
            duration_groups={"boundaries": [1.2], "include_overall": True},
        )
        tex = (report_dir / "detection_reliability_report.tex").read_text()
        parts = tex.count("\\part{")
        assert parts >= 2, tex
        assert tex.count("Embedding cost per file") == parts


class TestContainerMemorySection:
    """A snapshot of the running containers the run used, gated by the config."""

    ENABLED = {"enabled": True, "metrics": {
        "container_footprint": {"enabled": True},
        "embed_latency": {"enabled": False},
        "detect_latency": {"enabled": False},
        "attack_latency": {"enabled": False},
    }}

    # What a run that used the service on port 5001 reports for it.
    ROW = ("Model", "A", "deepmark-audioseal", 3348.0, 8192.0)

    @staticmethod
    def _one_running_container(monkeypatch):
        """docker lists deepmark-audioseal, publishing port 5001."""
        monkeypatch.setattr(efficiency, "_running_containers",
                            lambda: {"5001": "deepmark-audioseal"})
        monkeypatch.setattr(efficiency, "_memory_usage", lambda names: {
            "deepmark-audioseal": (3348.0, 8192.0)})

    def test_the_in_process_plugin_has_no_container_so_no_section(
        self, tmp_path, plugins_dir, audio_dir, monkeypatch,
    ):
        """A container is running, but none serves this in-process model."""
        self._one_running_container(monkeypatch)
        config = write_config(tmp_path, "containers.json",
                              efficiency=self.ENABLED)
        report_dir = tmp_path / "containers"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)

        tex = (report_dir / "benchmark_report.tex").read_text()
        assert "Container Memory" not in tex

    def test_the_section_renders_when_there_are_rows(self):
        from deepmarkpy.utils.latex_helpers import container_section

        section = container_section([
            ("Model", "AudioSealModel", "c1", 3348.0, 8192.0),
            ("Attack", "EncodecAttack", "c2", 900.0, 4096.0),
            ("Metric", "NISQA", "c3", 470.0, 4096.0),
        ])
        assert "\\section{Container Memory}" in section
        for name in ("AudioSealModel", "EncodecAttack", "NISQA"):
            assert name in section
        # The share of the limit is what says whether a service is at risk.
        assert "41\\%" in section

    def test_no_rows_means_no_section(self):
        from deepmarkpy.utils.latex_helpers import container_section

        assert container_section([]) == ""

    def test_a_missing_docker_cli_yields_no_rows(self, monkeypatch):
        monkeypatch.setattr(efficiency, "_run_docker", RUN_DOCKER)
        monkeypatch.setattr(efficiency.shutil, "which", lambda name: None)
        assert efficiency.container_snapshot(
            [("Model", "X", "http://localhost:5001")]
        ) == []

    def test_a_port_no_container_publishes_is_left_out(self, monkeypatch):
        self._one_running_container(monkeypatch)
        assert efficiency.container_snapshot([
            ("Model", "A", "http://localhost:5001"),
            ("Model", "X", "http://localhost:59999"),
        ]) == [self.ROW]

    def test_a_malformed_url_is_left_out(self, monkeypatch):
        self._one_running_container(monkeypatch)
        assert efficiency.container_snapshot([
            ("Model", "A", "http://localhost:5001"),
            ("Model", "Y", "not a url"),
            ("Model", "Z", None),
        ]) == [self.ROW]


class TestContainerSectionCoversTheWholeRun:
    """The section lists every service the report's own run used."""

    class _DockerAttack:
        endpoint = "http://localhost:9999/attack"

    def _rows_for(self, tmp_path, **overrides):
        from unittest.mock import patch

        from deepmarkpy.benchmark import Benchmark
        from deepmarkpy.config import load_config_data
        from deepmarkpy.run import _container_rows

        data = {
            "mode": "benchmark",
            "models": ["AudioSealModel", "PerthModel"],
            "calculate_quality_metrics": True,
            "efficiency": {"enabled": True, "metrics": {
                "container_footprint": {"enabled": True}}},
        }
        data.update(overrides)
        config = load_config_data(data, quiet=True)

        benchmark = Benchmark.__new__(Benchmark)
        benchmark.models = {}
        benchmark.attacks = {
            "DiffusionAttack": {"class": self._DockerAttack, "config": {}},
        }

        seen = {}

        def fake_snapshot(entries):
            seen["entries"] = entries
            return []

        with patch("deepmarkpy.utils.efficiency.container_snapshot",
                   fake_snapshot):
            _container_rows(benchmark, config, config.models)
        return seen.get("entries", [])

    def test_nisqa_counts_when_a_group_enables_it(self, tmp_path):
        entries = self._rows_for(tmp_path, metrics={
            "defaults": {m: {"enabled": False} for m in
                         ("pesq", "nisqa_mos", "nisqa_noi")},
            "per_group": {"audio_distortion": {"nisqa_mos": {"enabled": True}}},
        })
        assert any(label == "NISQA" for _, label, _ in entries), entries

    def test_nisqa_is_left_out_when_nothing_enables_it(self, tmp_path):
        entries = self._rows_for(tmp_path, metrics={
            "defaults": {m: {"enabled": False}
                         for m in ("nisqa_mos", "nisqa_noi", "nisqa_dis",
                                   "nisqa_col", "nisqa_loud")},
        })
        assert not any(label == "NISQA" for _, label, _ in entries), entries

    def test_benchmark_mode_without_a_selection_counts_every_attack(
        self, tmp_path,
    ):
        entries = self._rows_for(tmp_path)
        assert ("Attack", "DiffusionAttack",
                self._DockerAttack.endpoint) in entries, entries

    @pytest.mark.parametrize("mode", ["no_attacks", "detection_reliability"])
    def test_a_mode_that_runs_no_attacks_counts_none(self, tmp_path, mode):
        """No selection here means no attacks, so no attack service is listed."""
        # detection_reliability measures one model per config.
        entries = self._rows_for(tmp_path, mode=mode, models=["AudioSealModel"])
        assert not any(kind == "Attack" for kind, _, _ in entries), entries

    def test_each_model_report_asks_for_its_own_model_alone(
        self, tmp_path, plugins_dir, audio_dir, monkeypatch,
    ):
        write_model_plugin(
            plugins_dir,
            MODEL_SOURCE.replace("class DummyWatermarkModel",
                                 "class SecondWatermarkModel"),
            "second",
        )
        asked = []

        def record(_benchmark, _config, model_names):
            asked.append(list(model_names))
            return []

        monkeypatch.setattr(run_module, "_container_rows", record)
        config = write_config(
            tmp_path, "two_models.json",
            models=["DummyWatermarkModel", "SecondWatermarkModel"],
            efficiency=TestContainerMemorySection.ENABLED,
        )
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(tmp_path / "report"),
            "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_OK

        assert asked == [["DummyWatermarkModel"], ["SecondWatermarkModel"]]


class TestTheFlagInterface:
    """The measurement flags build a config that runs as the equivalent file does."""

    def _flag_run(self, tmp_path, plugins_dir, audio_dir, *extra):
        report_dir = tmp_path / "flags"
        code = run_cli(
            "--wm_model", "DummyWatermarkModel",
            "--attack_types", ATTACK_SPEC,
            "--calculate_quality_metrics",
            "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir),
            "--plugins_dir", plugins_dir,
            "--seed", "3", *extra,
        )
        return code, report_dir

    def test_the_quality_flag_keeps_pesq_and_stoi_for_desynchronization(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """The flag's per-group matrix keeps PESQ and STOI for desynchronization."""
        report_dir = tmp_path / "desync"
        assert run_cli(
            "--wm_model", "DummyWatermarkModel",
            "--attack_types", "FlipSamplesAttack",
            "--calculate_quality_metrics",
            "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir),
            "--plugins_dir", plugins_dir, "--seed", "3",
        ) == run_module.EXIT_OK

        entry = json.loads(
            (report_dir / "benchmark_stats.json").read_text()
        )["FlipSamplesAttack"]
        assert {"pesq_mean", "stoi_mean"} <= set(entry)
        tex = (report_dir / "benchmark_report.tex").read_text()
        assert "tab:benchmark_pesq_desynchronization" in tex
        assert "tab:benchmark_stoi_desynchronization" in tex

    def test_flags_and_the_equivalent_config_agree(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """The same run, expressed both ways, measures the same thing."""
        code, flag_dir = self._flag_run(tmp_path, plugins_dir, audio_dir)
        assert code == run_module.EXIT_OK
        # --calculate_quality_metrics also writes the detailed report.
        for name in ("benchmark_results.json", "benchmark_report.tex",
                     "detailed_report.tex"):
            assert (flag_dir / name).exists(), f"{name} was not written"

        # The config the flags build: no statistics and no metrics block, so
        # all eight statistics and the built-in matrix apply.
        config = write_config(tmp_path, "equivalent.json",
                              metrics=None, statistics=None)
        config_dir = tmp_path / "fromfile"
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(config_dir), "--plugins_dir", plugins_dir,
            "--seed", "3",
        ) == run_module.EXIT_OK

        def stats(report_dir):
            return json.loads((report_dir / "benchmark_stats.json").read_text())

        assert stats(flag_dir) == stats(config_dir)

    def test_an_attack_parameter_flag_reaches_the_attack(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """It lands on the default version only, as a bare config key does."""
        report_dir = tmp_path / "params"
        assert run_cli(
            "--wm_model", "DummyWatermarkModel",
            "--attack_types", ATTACK_CLASS,
            f"--{ATTACK_PARAM}", "25",
            "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir),
            "--plugins_dir", plugins_dir, "--seed", "3",
        ) == run_module.EXIT_OK

        attacks = next(iter(json.loads(
            (report_dir / "benchmark_results.json").read_text()
        ).values()))["attacks"]

        # 25 dB, away from every preset: default 35, mild 45, aggressive 15.
        assert 20 < attacks[f"{ATTACK_CLASS} (default)"]["attack_snr_db"] < 30
        # The presets it must not have touched.
        assert 10 < attacks[f"{ATTACK_CLASS} (aggressive)"]["attack_snr_db"] < 20
        assert 40 < attacks[f"{ATTACK_CLASS} (mild)"]["attack_snr_db"] < 50

    def test_attack_groups_select_a_family(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        report_dir = tmp_path / "grouped"
        assert run_cli(
            "--wm_model", "DummyWatermarkModel",
            "--attack_groups", "audio_distortion",
            "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir),
            "--plugins_dir", plugins_dir, "--seed", "3",
        ) == run_module.EXIT_OK
        results = json.loads(
            (report_dir / "benchmark_results.json").read_text()
        )
        assert "GaussianNoiseAttack" in next(iter(results.values()))["attacks"]

    def test_no_attacks_flag_selects_that_mode(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        report_dir = tmp_path / "baseline"
        assert run_cli(
            "--wm_model", "DummyWatermarkModel", "--no_attacks",
            "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir),
            "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_OK
        assert (report_dir / "no_attacks_report.tex").exists()

    def test_combined_mode_flags_with_several_models_are_refused(
        self, tmp_path, plugins_dir, audio_dir, capsys,
    ):
        """E012: reliability measures one model, with or without --no_attacks."""
        write_model_plugin(
            plugins_dir,
            MODEL_SOURCE.replace("class DummyWatermarkModel",
                                 "class SecondWatermarkModel"),
            "second",
        )
        report_dir = tmp_path / "combined"
        assert run_cli(
            "--wm_models", "DummyWatermarkModel", "SecondWatermarkModel",
            "--no_attacks", "--detection_reliability",
            "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir),
            "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_CONFIG_ERROR

        assert "[E012]" in capsys.readouterr().err
        assert not report_dir.exists()

    def test_the_config_file_wins_and_says_so(
        self, tmp_path, plugins_dir, audio_dir, caplog,
    ):
        """Both given: the file decides, and the ignored flag is named."""
        config = write_config(tmp_path, "wins.json",
                              models=["DummyWatermarkModel"])
        report_dir = tmp_path / "both"
        with caplog.at_level("WARNING"):
            assert run_cli(
                "--config", config, "--wm_model", "DummyWatermarkModel",
                "--no_attacks", "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir),
                "--plugins_dir", plugins_dir,
            ) == run_module.EXIT_OK

        assert "--no_attacks" in caplog.text and "Ignoring" in caplog.text
        # The config says benchmark mode, so that is what ran.
        assert (report_dir / "benchmark_report.tex").exists()

    def test_a_zero_crop_flag_is_named_when_the_config_wins(
        self, tmp_path, plugins_dir, audio_dir, caplog,
    ):
        """0 is a value the caller set, even though 0.0 == False."""
        config = write_config(tmp_path, "wins.json")
        with caplog.at_level("WARNING"):
            assert run_cli(
                "--config", config, "--crop_before_attack", "0",
                "--wav_files_dir", audio_dir, "--plugins_dir", plugins_dir,
                "--validate-only",
            ) == run_module.EXIT_OK

        assert "Ignoring --crop_before_attack" in caplog.text

    def test_a_zero_crop_from_the_flags_is_no_crop(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """--crop_before_attack 0 crops nothing, as a config file's null does."""
        code, report_dir = self._flag_run(
            tmp_path, plugins_dir, audio_dir, "--crop_before_attack", "0",
        )
        assert code == run_module.EXIT_OK
        assert (report_dir / "benchmark_results.json").exists()
        assert "crop of" not in (report_dir / "benchmark_report.tex").read_text()

    def test_an_unknown_parameter_flag_is_refused(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        with pytest.raises(SystemExit):
            run_cli("--wm_model", "DummyWatermarkModel",
                    "--not_a_parameter", "5",
                    "--wav_files_dir", audio_dir,
                    "--plugins_dir", plugins_dir, "--validate-only")

    def test_neither_a_config_nor_a_flag_asks_for_a_config(self):
        with pytest.raises(SystemExit):
            run_cli("--wav_files_dir", "whatever")

    def test_the_help_says_these_flags_run_without_a_config(self):
        """--help lists them, so it must not call every flag operational."""
        text = " ".join(run_module._build_parser().format_help().split())
        assert "operational only" not in text
        assert "Required unless --init or a compatibility flag is used." in text
