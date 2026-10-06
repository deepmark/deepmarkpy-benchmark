"""The CLI, driven end to end from config files.

Runs ``main()`` against a throwaway model plugin and real native attacks,
so the wiring the unit tests stub out -- argument parsing, config
loading, mode dispatch, the report calls -- is actually executed.

Docker is not involved: the model plugin here runs in-process and has no
``base_url``, so the service reachability check passes it over.
"""

import json
import pathlib

import numpy as np
import pytest
import soundfile as sf

from deepmarkpy import run as run_module

MODEL_SOURCE = '''
import numpy as np

from deepmarkpy.core.base_model import BaseModel


class DummyWatermarkModel(BaseModel):
    """Adds a tiny deterministic offset and reads it back.

    Real enough to exercise the pipeline: embedding perturbs the signal
    (so quality metrics have something to measure) and detection degrades
    once an attack has been applied.
    """

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
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture(autouse=True)
def _skip_pdflatex(monkeypatch):
    """These assert on the .tex and the JSON, not on PDF rendering."""
    monkeypatch.setattr(
        "deepmarkpy.utils.latex_helpers.compile_latex", lambda *a, **k: None,
    )
    for module in ("report_generator", "detailed_report_generator",
                   "no_attacks_report_generator",
                   "detection_reliability_report_generator"):
        monkeypatch.setattr(
            f"deepmarkpy.utils.{module}.compile_latex",
            lambda *a, **k: None, raising=False,
        )


def run_cli(*argv):
    return run_module.main(list(argv))


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
        self, tmp_path, plugins_dir, audio_dir,
    ):
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

    def test_stats_carry_only_the_configured_metrics(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        config = write_config(tmp_path, "benchmark.json")
        report_dir = tmp_path / "report"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)

        stats = json.loads((report_dir / "benchmark_stats.json").read_text())
        entry = stats[ATTACK_NAME]

        assert {"accuracy_mean", "accuracy_worst_case", "accuracy_n"} <= set(entry)
        assert "pesq_mean" in entry and "stoi_mean" in entry
        # Disabled metrics must not have been computed at all.
        for metric in ("mcd", "psnr", "si_sdr", "sii", "ncm", "nisqa_mos"):
            assert not any(k.startswith(f"{metric}_") for k in entry), (
                f"{metric} was computed despite being disabled in the config"
            )
        # And only the configured statistics.
        assert "accuracy_median" not in entry

    def test_attack_parameters_reach_the_attack(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """A per-version attack_parameters entry must change the measurement."""
        report_dir = tmp_path / "report"

        def snr_for(value):
            config = write_config(
                tmp_path, f"cfg{value}.json",
                attack_parameters={ATTACK_SPEC: {"snr_db_preset_noise": value}},
            )
            run_cli("--config", config, "--wav_files_dir", audio_dir,
                    "--report_dir", str(report_dir),
                    "--plugins_dir", plugins_dir, "--seed", "3")
            results = json.loads((report_dir / "benchmark_results.json").read_text())
            first = next(iter(results.values()))
            return first["attacks"][ATTACK_NAME]["attack_snr_db"]

        quiet, loud = snr_for(40), snr_for(10)
        assert quiet > loud + 15, (
            f"snr_db_preset_noise did not reach the attack: {quiet} vs {loud}"
        )

    def test_a_bare_key_does_not_leak_onto_a_named_version(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """Two versions in one run must keep measuring differently.

        A bare class name targets the default version only. If it reached
        the named ones too they would collapse to the same number while
        the report kept printing their distinct labels.
        """
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

    def test_a_complete_new_version_is_run_and_labelled(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """A version the plugin does not declare, defined in the config."""
        config = write_config(
            tmp_path, "newver.json",
            attacks={"list": ["PresetNoiseAttack:brutal"]},
            attack_parameters={
                "PresetNoiseAttack:brutal": {"snr_db_preset_noise": 5},
            },
        )
        report_dir = tmp_path / "report"
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
            "--seed", "3",
        ) == run_module.EXIT_OK

        attacks = next(iter(json.loads(
            (report_dir / "benchmark_results.json").read_text()).values()))["attacks"]
        assert "PresetNoiseAttack (brutal)" in attacks
        snr = attacks["PresetNoiseAttack (brutal)"]["attack_snr_db"]
        assert 3 < snr < 7, f"the defined version's parameter was not used: {snr}"

    def test_naming_the_attack_bare_runs_the_defined_version_too(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        config = write_config(
            tmp_path, "allver.json",
            attacks={"list": ["PresetNoiseAttack"]},
            attack_parameters={
                "PresetNoiseAttack:brutal": {"snr_db_preset_noise": 5},
            },
        )
        report_dir = tmp_path / "report"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)

        attacks = next(iter(json.loads(
            (report_dir / "benchmark_results.json").read_text()).values()))["attacks"]
        assert set(attacks) == {
            "PresetNoiseAttack (default)",
            "PresetNoiseAttack (mild)",
            "PresetNoiseAttack (aggressive)",
            "PresetNoiseAttack (brutal)",
        }

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
        config = write_config(
            tmp_path, "durations.json",
            duration_groups={"boundaries": [1.2], "include_overall": True},
        )
        report_dir = tmp_path / "report"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)

        stats = json.loads((report_dir / "benchmark_stats.json").read_text())
        assert "Overall" in stats
        assert any(label.startswith("<") for label in stats)

        tex = (report_dir / "benchmark_report.tex").read_text()
        assert "\\part{" in tex

    def test_overall_is_dropped_when_every_file_lands_in_one_bin(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """One bin makes "Overall" a verbatim copy of it, under a second
        heading, and the reader is left looking for a difference."""
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
        """The flat stats the comparative report ranks are over every file.

        Without "Overall" this used to take the first bin, so a multi-model
        comparison quietly left out every file in the later bins.
        """
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
            tmp_path, "no_attacks.json", mode="no_attacks",
            attacks=None,
        )
        # 'attacks' is not a valid key for this mode, so remove it.
        raw = json.loads(open(config).read())
        raw.pop("attacks", None)
        open(config, "w").write(json.dumps(raw))

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
        config = write_config(
            tmp_path, "dr.json", mode="detection_reliability",
        )
        raw = json.loads(open(config).read())
        raw["metrics"]["defaults"].pop("ber")
        open(config, "w").write(json.dumps(raw))

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


class TestSeveralModesInOneInvocation:
    def test_both_modes_run_and_neither_erases_the_other(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """The capability the per-mode config split had to preserve."""
        benchmark = write_config(tmp_path, "benchmark.json")

        dr_raw = json.loads(open(write_config(tmp_path, "dr.json")).read())
        dr_raw["mode"] = "detection_reliability"
        dr_raw["metrics"]["defaults"].pop("ber")
        dr_path = tmp_path / "dr.json"
        dr_path.write_text(json.dumps(dr_raw))

        report_dir = tmp_path / "report"
        assert run_cli(
            "--config", benchmark, str(dr_path),
            "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir),
            "--plugins_dir", plugins_dir,
        ) == run_module.EXIT_OK

        assert (report_dir / "benchmark_report.tex").exists()
        assert (report_dir / "detection_reliability_report.tex").exists()


class TestFailureModes:
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

    def test_validate_only_runs_nothing(self, tmp_path, plugins_dir, audio_dir):
        config = write_config(tmp_path, "benchmark.json")
        report_dir = tmp_path / "report"
        assert run_cli(
            "--config", config, "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
            "--validate-only",
        ) == run_module.EXIT_OK
        assert not report_dir.exists(), "--validate-only wrote output"


class TestCliOverridesConfig:
    @pytest.mark.parametrize("seed", [-1, 2**32])
    @pytest.mark.parametrize("from_cli", [False, True])
    @pytest.mark.parametrize("validate_only", [False, True])
    def test_invalid_seed_preserves_existing_reports(
        self, tmp_path, plugins_dir, audio_dir, seed, from_cli, validate_only,
    ):
        config = write_config(
            tmp_path, "benchmark.json", general={"seed": 7 if from_cli else seed},
        )
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        existing = report_dir / "previous.json"
        existing.write_text('{"completed": true}')
        extra = ["--seed", str(seed)] if from_cli else []
        if validate_only:
            extra.append("--validate-only")
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
    """A container that answers slowly must not read as one that is down.

    WavMark loads its checkpoint on the first request and takes over ten
    seconds to answer when cold, so a single short probe aborted a run
    that would have worked seconds later.
    """

    @staticmethod
    def _benchmark(base_url="http://localhost:9/"):
        model_cls = type("Slow", (), {"__init__": lambda self: setattr(
            self, "base_url", base_url)})
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

        monkeypatch.setattr(requests, "get", flaky_get)
        messages = run_module._unreachable_model_services(
            self._benchmark(), self._config(),
        )
        assert messages == []
        assert calls["n"] == 3, "gave up before the service came up"

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

    def test_a_model_without_a_base_url_is_skipped(self, monkeypatch):
        """An in-process plugin has no service to check."""
        messages = run_module._unreachable_model_services(
            self._benchmark(base_url=None), self._config(),
        )
        assert messages == []


class TestParameterProvenance:
    """Which version ran with which values must be answerable.

    Before the run, from --validate-only; after it, from
    run_metadata.json. Inferring it from the config by hand is exactly
    what silently mislabelled two identical rows before.
    """

    LADDER = {
        "PresetNoiseAttack:mild": {"snr_db_preset_noise": 45},
        "PresetNoiseAttack": {"snr_db_preset_noise": 35},
        "PresetNoiseAttack:aggressive": {"snr_db_preset_noise": 20},
        "PresetNoiseAttack:brutal": {"snr_db_preset_noise": 10},
        "PresetNoiseAttack:extreme": {"snr_db_preset_noise": 3},
    }
    EXPECTED = {
        "PresetNoiseAttack (mild)": 45,
        "PresetNoiseAttack (default)": 35,
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

    def test_run_metadata_records_every_version(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        report_dir = tmp_path / "report"
        run_cli("--config", self._config(tmp_path),
                "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)

        resolved = json.loads(
            (report_dir / "run_metadata.json").read_text()
        )["attack_parameters_resolved"]
        assert {
            name: params["snr_db_preset_noise"]
            for name, params in resolved.items()
        } == self.EXPECTED

    def test_the_five_versions_actually_measure_differently(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """The provenance must match what reached the signal."""
        report_dir = tmp_path / "report"
        run_cli("--config", self._config(tmp_path),
                "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir,
                "--seed", "1")

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
        from deepmarkpy.benchmark import Benchmark
        from deepmarkpy.config import load_configs

        benchmark = Benchmark(external_plugins_dir=plugins_dir)
        config = load_configs(
            [write_config(tmp_path, "partial.json",
                          attacks={"list": ["FlipSamplesAttack"]},
                          attack_parameters={
                              "FlipSamplesAttack": {"num_flip_samples": 50},
                          })],
            benchmark.attacks, benchmark.models,
        )[0]

        resolved = run_module.resolved_attack_parameters(benchmark, config)
        entry = resolved["FlipSamplesAttack"]
        assert entry["num_flip_samples"] == 50, "override not applied"
        assert "duration_flip_samples" in entry, (
            "untouched plugin parameters are missing, so the record is "
            "not what apply() will see"
        )


class TestParameterReportingPerMode:
    """An empty attack selection means something different per mode.

    The provenance table has to agree with the run loop, or it describes
    attacks that never ran.
    """

    def test_no_attacks_mode_reports_no_attack_parameters(self, tmp_path, plugins_dir):
        """This mode applies none, so listing every attack would be nonsense."""
        from deepmarkpy.benchmark import Benchmark
        from deepmarkpy.config import load_configs

        path = tmp_path / "na.json"
        path.write_text(json.dumps({
            "mode": "no_attacks", "models": ["DummyWatermarkModel"],
            "calculate_quality_metrics": False,
        }))
        benchmark = Benchmark(external_plugins_dir=plugins_dir)
        config = load_configs([str(path)], benchmark.attacks, benchmark.models)[0]

        assert run_module.resolved_attack_parameters(benchmark, config) == {}

    def test_detection_reliability_empty_selection_is_no_attacks(
        self, tmp_path, plugins_dir,
    ):
        """Empty means baseline-only here, not 'every attack'."""
        from deepmarkpy.benchmark import Benchmark
        from deepmarkpy.config import load_configs

        path = tmp_path / "dr.json"
        path.write_text(json.dumps({
            "mode": "detection_reliability", "models": ["DummyWatermarkModel"],
            "calculate_quality_metrics": False,
            "attacks": {"groups": [], "list": []},
        }))
        benchmark = Benchmark(external_plugins_dir=plugins_dir)
        config = load_configs([str(path)], benchmark.attacks, benchmark.models)[0]

        assert run_module.resolved_attack_parameters(benchmark, config) == {}

    def test_benchmark_empty_selection_is_every_attack(self, tmp_path, plugins_dir):
        from deepmarkpy.benchmark import Benchmark
        from deepmarkpy.config import load_configs

        path = tmp_path / "b.json"
        path.write_text(json.dumps({
            "mode": "benchmark", "models": ["DummyWatermarkModel"],
            "calculate_quality_metrics": False,
            "attacks": {"groups": [], "list": []},
        }))
        benchmark = Benchmark(external_plugins_dir=plugins_dir)
        config = load_configs([str(path)], benchmark.attacks, benchmark.models)[0]

        assert len(run_module.resolved_attack_parameters(benchmark, config)) > 20

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
    """Timings are measured only when the efficiency section asks for them.

    The section is separate from ``metrics`` because these numbers
    describe the machine rather than the watermarking method: they do not
    reproduce, and a run decides whether to take the measurement at all.
    """

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
        self, tmp_path, plugins_dir, audio_dir,
    ):
        report_dir = self._run(tmp_path, plugins_dir, audio_dir, "eff_on.json",
                               efficiency=self.EFFICIENCY)

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

    def test_nothing_is_measured_when_the_section_is_absent(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        report_dir = self._run(tmp_path, plugins_dir, audio_dir, "eff_off.json")

        results = json.loads((report_dir / "benchmark_results.json").read_text())
        entry = next(iter(next(iter(results.values()))["attacks"].values()))
        stats = json.loads((report_dir / "benchmark_stats.json").read_text())
        row = next(iter(stats.values()))

        # The raw results too, not only the aggregate: the timings used to
        # be taken regardless and merely filtered out when reduced.
        assert not [k for k in entry if "latency" in k], entry
        assert not [k for k in row if "latency" in k], row
        tex = (report_dir / "benchmark_report.tex").read_text()
        assert "tab:benchmark_efficiency" not in tex
        assert "Attack time" not in tex
        detailed = (report_dir / "detailed_report.tex").read_text()
        assert "tab:efficiency_" not in detailed
        assert "Attack time" not in detailed

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

    def test_the_terminal_lines_carry_the_efficiency_tag(
        self, tmp_path, plugins_dir, audio_dir, caplog,
    ):
        """Its own tag, so timing output can be read or filtered on its own."""
        from deepmarkpy.utils.efficiency import TERMINAL_TAG

        with caplog.at_level("INFO"):
            self._run(tmp_path, plugins_dir, audio_dir, "eff_log.json",
                      efficiency=self.EFFICIENCY)

        tagged = [r.message for r in caplog.records if TERMINAL_TAG in r.message]
        assert tagged, "no tagged efficiency output"
        assert any("Attack time" in line for line in tagged), tagged


class TestEmbeddingCostIsStatedOncePerPart:
    """Duration parts take a different code path from the flat report.

    The flat report states the embedding cost in its summary; the grouped
    one has no summary, so the statement has to be emitted per part or it
    disappears from exactly the configuration that uses duration groups.
    """

    def test_every_duration_part_states_it(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        config = write_config(
            tmp_path, "grouped.json",
            duration_groups={"boundaries": [1.2], "include_overall": True},
            efficiency={"enabled": True, "metrics": {
                "embed_latency": {"statistics": ["mean"]},
                "attack_latency": {"statistics": ["mean"]},
            }},
        )
        report_dir = tmp_path / "grouped"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)

        tex = (report_dir / "benchmark_report.tex").read_text()
        parts = tex.count("\\part{")
        assert parts >= 2, tex
        assert tex.count("Embedding cost per file") == parts


class TestReliabilityModeTimings:
    """The reliability mode has its own run loop, so it must time its own.

    It calls detect twice per file and twice per attack -- on clean audio
    for the false-positive rate and on watermarked audio for the false
    negative. Only the calls matching what the other modes time are
    recorded, so ``attack_latency`` means the same thing in all three.
    """

    @staticmethod
    def _write(tmp_path, name, **overrides):
        """BER is not a metric in this mode, so the shared base drops it."""
        config = write_config(tmp_path, name,
                              mode="detection_reliability", **overrides)
        raw = json.loads(pathlib.Path(config).read_text())
        raw["metrics"]["defaults"].pop("ber", None)
        pathlib.Path(config).write_text(json.dumps(raw))
        return config

    def _run(self, tmp_path, plugins_dir, audio_dir, **overrides):
        config = self._write(
            tmp_path, "reliability_eff.json",
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

    def test_nothing_is_measured_without_the_section(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        config = self._write(tmp_path, "reliability_off.json")
        report_dir = tmp_path / "reliability_off"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)

        result = json.loads(
            (report_dir / "detection_reliability.json").read_text()
        )
        attack = next(iter(result["attacks"].values()))
        assert not attack.get("timings"), attack.get("timings")
        assert not result["no_attack"].get("timings")

        record = next(iter(result["per_file"].values()))
        assert not [k for k in record if "latency" in k], record
        for attack_record in record["attacks"].values():
            assert not [k for k in attack_record if "latency" in k], attack_record

        tex = (report_dir / "detection_reliability_report.tex").read_text()
        assert "tab:dr_efficiency" not in tex
        assert "Attack time" not in tex

    def test_every_duration_part_states_the_embedding_cost(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """Each part is rebuilt from the per-file records, and that rebuild
        dropped the baseline timings, so the grouped report lost the line
        the flat one carries."""
        report_dir = self._run(
            tmp_path, plugins_dir, audio_dir,
            duration_groups={"boundaries": [1.2], "include_overall": True},
        )
        tex = (report_dir / "detection_reliability_report.tex").read_text()
        parts = tex.count("\\part{")
        assert parts >= 2, tex
        assert tex.count("Embedding cost per file") == parts

    def test_the_section_no_longer_warns_that_the_mode_ignores_it(
        self, tmp_path,
    ):
        """W014 said the mode did not record timings. It does now."""
        from deepmarkpy.config import load_config_data

        config = load_config_data(
            {"mode": "detection_reliability", "models": ["AudioSealModel"],
             "efficiency": {"enabled": True}},
            quiet=True,
        )
        assert "W014" not in [w.code for w in config.warnings]


class TestContainerMemorySection:
    """A snapshot of the services the run used, gated by the config.

    Not a per-attack metric: it covers models, dockerized attacks and the
    metric services, and it reports the whole container rather than the
    model. Only running containers appear; anything native, stopped or
    unreachable is absent rather than reported as zero.
    """

    ENABLED = {"enabled": True, "metrics": {
        "container_footprint": {"enabled": True},
        "embed_latency": {"enabled": False},
        "detect_latency": {"enabled": False},
        "attack_latency": {"enabled": False},
    }}

    def test_the_in_process_plugin_has_no_container_so_no_section(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """The model here runs in this process; there is nothing to report."""
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
        from deepmarkpy.utils import efficiency

        monkeypatch.setattr(efficiency.shutil, "which", lambda name: None)
        assert efficiency.container_snapshot(
            [("Model", "X", "http://localhost:5001")]
        ) == []

    def test_a_port_no_container_publishes_is_left_out(self):
        from deepmarkpy.utils import efficiency

        assert efficiency.container_snapshot(
            [("Model", "X", "http://localhost:59999")]
        ) == []

    def test_a_malformed_url_is_left_out(self):
        from deepmarkpy.utils import efficiency

        assert efficiency.container_snapshot(
            [("Model", "X", "not a url"), ("Model", "Y", None)]
        ) == []

    def test_it_is_not_collected_unless_the_config_asks(
        self, tmp_path, plugins_dir, audio_dir, monkeypatch,
    ):
        """Reading it shells out to docker, so a run that did not ask must
        not pay for it."""
        from deepmarkpy.utils import efficiency

        calls = []
        monkeypatch.setattr(efficiency, "_run_docker",
                            lambda args: calls.append(args))

        config = write_config(tmp_path, "no_containers.json")
        report_dir = tmp_path / "no_containers"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(report_dir), "--plugins_dir", plugins_dir)

        assert calls == [], calls


class TestContainerSectionCoversTheWholeRun:
    """The section is the deployment, not the report's own model.

    Two bugs sat here: each model's report listed only itself, and NISQA
    was looked up in ``metrics.defaults`` alone, so a config that enables
    it per attack group -- which is how the shipped templates are written
    -- reported it as unused while the service was being called.
    """

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
        """No selection there means no attacks, not every attack, so a
        running service it never called is not this run's footprint."""
        # detection_reliability measures one model per config.
        entries = self._rows_for(tmp_path, mode=mode, models=["AudioSealModel"])
        assert not any(kind == "Attack" for kind, _, _ in entries), entries


class TestTheFlagInterfaceStillWorks:
    """The pre-2.0 flags, and what they must still produce.

    An existing caller runs `--wm_model X --attack_types Y
    --calculate_quality_metrics` and gets its reports. The flags are
    assembled into a config mapping and validated by the same code a
    file goes through, so the run below this point is identical -- which
    is what these assert, by running both and comparing.
    """

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

    def test_a_flag_run_produces_its_reports(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        code, report_dir = self._flag_run(tmp_path, plugins_dir, audio_dir)
        assert code == run_module.EXIT_OK

        assert (report_dir / "benchmark_results.json").exists()
        assert (report_dir / "benchmark_report.tex").exists()
        # --calculate_quality_metrics has always meant the detailed report too.
        assert (report_dir / "detailed_report.tex").exists()

    def test_flags_and_the_equivalent_config_agree(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """The same run, expressed both ways, measures the same thing."""
        _code, flag_dir = self._flag_run(tmp_path, plugins_dir, audio_dir)

        config = write_config(
            tmp_path, "equivalent.json",
            metrics={"defaults": {}},  # no block: the built-in matrix applies
        )
        config_dir = tmp_path / "fromfile"
        run_cli("--config", config, "--wav_files_dir", audio_dir,
                "--report_dir", str(config_dir), "--plugins_dir", plugins_dir,
                "--seed", "3")

        def accuracy(report_dir):
            stats = json.loads(
                (report_dir / "benchmark_stats.json").read_text()
            )
            return {name: value.get("accuracy_mean")
                    for name, value in stats.items()}

        assert accuracy(flag_dir) == accuracy(config_dir)

    def test_an_attack_parameter_flag_reaches_the_attack(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        """The old CLI generated one flag per plugin parameter.

        It lands on the default preset only, exactly as the same key
        written bare in a config file does -- the flat namespace the
        flags had cannot name a version, so it cannot reach one.
        """
        report_dir = tmp_path / "params"
        assert run_cli(
            "--wm_model", "DummyWatermarkModel",
            "--attack_types", ATTACK_CLASS,
            f"--{ATTACK_PARAM}", "40",
            "--wav_files_dir", audio_dir,
            "--report_dir", str(report_dir),
            "--plugins_dir", plugins_dir, "--seed", "3",
        ) == run_module.EXIT_OK

        attacks = next(iter(json.loads(
            (report_dir / "benchmark_results.json").read_text()
        ).values()))["attacks"]

        assert 35 < attacks[f"{ATTACK_CLASS} (default)"]["attack_snr_db"] < 45
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

    def test_an_unknown_parameter_flag_is_refused(
        self, tmp_path, plugins_dir, audio_dir,
    ):
        with pytest.raises(SystemExit):
            run_cli("--wm_model", "DummyWatermarkModel",
                    "--not_a_parameter", "5",
                    "--wav_files_dir", audio_dir,
                    "--plugins_dir", plugins_dir, "--validate-only")

    def test_neither_a_config_nor_a_flag_still_asks_for_config(self):
        with pytest.raises(SystemExit):
            run_cli("--wav_files_dir", "whatever")
