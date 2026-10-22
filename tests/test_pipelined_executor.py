"""The pipelined executor returns what the sequential loop returns.

Both executors run the same plugins on the same files from the same seed, and
the results must be equal entry for entry and in order. The plugins installed
here stand in for the deployment: a model and an attack that ship a
Dockerfile, so they take the service lanes, next to packaged attacks that draw
from the host RNG on the ordered lane. They do their work in this process,
statelessly, as a container would.
"""

import json
import pathlib
import random

import numpy as np
import pytest
import requests
import soundfile as sf

from deepmarkpy import run as run_module
from deepmarkpy.benchmark import Benchmark, Parallelism
from deepmarkpy.config import ModeConfig
from deepmarkpy.core.wire import encode_audio
from deepmarkpy.execution import pipelined
from deepmarkpy.utils import metrics
from deepmarkpy.utils.metric_resolver import MetricResolver

SERVICE_MODEL = '''
import hashlib
import time

import numpy as np

from deepmarkpy.core.base_model import BaseModel


class ServiceStandInModel(BaseModel):
    """Stateless: one block per bit, nudged up or down, read back by sign."""

    def embed(self, audio, watermark_data, sampling_rate, **kwargs):
        out = np.array(audio, dtype=np.float64)
        block = len(out) // len(watermark_data)
        for i, bit in enumerate(watermark_data):
            out[i * block:(i + 1) * block] += 2e-3 if bit else -2e-3
        return out

    def detect(self, audio, sampling_rate, **kwargs):
        audio = np.asarray(audio, dtype=np.float64)
        # Answers take varying time, so detections finish out of order.
        time.sleep(hashlib.sha256(audio.tobytes()).digest()[0] / 255 * 0.01)
        block = len(audio) // 16
        return np.array([int(np.mean(audio[i * block:(i + 1) * block]) > 0)
                         for i in range(16)])
'''

SERVICE_ATTACK = '''
import hashlib
import time

import numpy as np

from deepmarkpy.core.base_attack import BaseAttack


class ServiceStandInAttack(BaseAttack):
    """Deterministic, and slow by an amount that depends on its input."""

    def apply(self, audio, **kwargs):
        audio = np.asarray(audio, dtype=np.float64)
        time.sleep(hashlib.sha256(audio.tobytes()).digest()[0] / 255 * 0.05)
        return np.convolve(audio, [0.5, 0.3, 0.2])[:len(audio)]
'''

SERVICE_RNG_ATTACK = '''
import numpy as np

from deepmarkpy.core.base_attack import BaseAttack


class ServiceRngAttack(BaseAttack):
    """Wrongly draws from the host RNG while claiming to run in a container."""

    def apply(self, audio, **kwargs):
        return np.asarray(audio) + np.random.normal(0, 1e-3, len(audio))
'''

IN_PLACE_ATTACK = '''
import numpy as np

from deepmarkpy.core.base_attack import BaseAttack


class InPlaceAttack(BaseAttack):
    """Changes the array it is given."""

    def apply(self, audio, **kwargs):
        audio *= 0.5
        return audio
'''

FAILING_ATTACK = '''
from deepmarkpy.core.base_attack import BaseAttack


class FailsOnLongFilesAttack(BaseAttack):
    def apply(self, audio, **kwargs):
        if len(audio) > 20000:
            raise ValueError("this attack fails on long files")
        return audio * 0.9
'''

MODEL_CONFIG = {
    "sampling_rate": 16000,
    "watermark_size": 16,
    "is_zero_bit": False,
    "returns_confidence": False,
}

# Packaged attacks on the ordered lane; the first three draw from np.random.
NATIVE_ATTACKS = [
    "GaussianNoiseAttack", "CropRandomAttack", "EchoAttack",
    "LowpassFilterAttack", "SignInversionAttack",
]


def _plugin(root, name, filename, source, config, dockerized):
    directory = pathlib.Path(root) / name
    directory.mkdir(parents=True)
    (directory / filename).write_text(source)
    (directory / "config.json").write_text(json.dumps(config))
    if dockerized:
        (directory / "Dockerfile").write_text("FROM scratch\n")


@pytest.fixture(autouse=True)
def _no_nisqa(monkeypatch):
    monkeypatch.setattr(metrics, "_NISQA_ENDPOINT", None)
    monkeypatch.setattr(metrics, "_nisqa_unavailable", True)
    monkeypatch.setattr(metrics, "_nisqa_reason", "disabled in tests")


@pytest.fixture(scope="module")
def plugins_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("plugins")
    _plugin(root, "service_model", "model.py", SERVICE_MODEL, MODEL_CONFIG, True)
    _plugin(root, "service_attack", "attack.py", SERVICE_ATTACK, {}, True)
    _plugin(root, "service_rng", "attack.py", SERVICE_RNG_ATTACK, {}, True)
    _plugin(root, "in_place", "attack.py", IN_PLACE_ATTACK, {}, False)
    _plugin(root, "failing", "attack.py", FAILING_ATTACK, {}, False)
    return str(root)


@pytest.fixture(scope="module")
def filepaths(tmp_path_factory):
    directory = tmp_path_factory.mktemp("audio")
    rng = np.random.default_rng(7)
    paths = []
    for index, seconds in enumerate((1.0, 1.5, 1.25)):
        t = np.arange(int(16000 * seconds)) / 16000
        audio = 0.3 * np.sin(2 * np.pi * (180 + 40 * index) * t) + 0.02 * rng.standard_normal(len(t))
        path = directory / f"clip{index}.wav"
        sf.write(str(path), audio, 16000)
        paths.append(str(path))
    return paths


@pytest.fixture(scope="module")
def benchmark(plugins_dir):
    return Benchmark(external_plugins_dir=plugins_dir)


def _run(benchmark, filepaths, attacks, executor, model="ServiceStandInModel",
         parallelism=None, seed=1234, on_file_complete=None):
    random.seed(seed)
    np.random.seed(seed)
    return benchmark.run(
        filepaths=filepaths, wm_model=model, attack_types=attacks,
        metric_resolver=MetricResolver.from_attack_groups(calculate_quality_metrics=False),
        executor=executor, on_file_complete=on_file_complete,
        parallelism=parallelism or Parallelism(workers=2, files_in_flight=2),
    )


class TestSameResultsAsSequential:
    def test_with_service_lanes(self, benchmark, filepaths):
        attacks = NATIVE_ATTACKS + ["ServiceStandInAttack"]
        sequential = _run(benchmark, filepaths, attacks, "sequential")
        completed = []
        piped = _run(benchmark, filepaths, attacks, "pipelined",
                     on_file_complete=lambda path, _: completed.append(path))
        assert json.dumps(piped) == json.dumps(sequential)
        assert completed == filepaths, "files must be handed on in order"

    def test_with_two_requests_in_flight_to_a_service(self, benchmark, filepaths):
        attacks = NATIVE_ATTACKS + ["ServiceStandInAttack"]
        sequential = _run(benchmark, filepaths, attacks, "sequential")
        piped = _run(benchmark, filepaths, attacks, "pipelined", parallelism=Parallelism(
            workers=1, files_in_flight=3, detect_concurrency=3,
            service_lanes=(("ServiceStandInAttack",),),
            service_concurrency={"ServiceStandInAttack": 2},
        ))
        assert json.dumps(piped) == json.dumps(sequential)

    def test_with_an_in_process_model(self, benchmark, filepaths, plugins_dir):
        """A model without a Dockerfile keeps every attack on the ordered lane."""
        attacks = NATIVE_ATTACKS + ["ServiceStandInAttack"]
        entry = benchmark.models["ServiceStandInModel"]
        model_dir = pathlib.Path(plugins_dir) / "service_model"
        (model_dir / "Dockerfile").rename(model_dir / "Dockerfile.off")
        try:
            run = pipelined.PipelinedRun.__new__(pipelined.PipelinedRun)
            run.bench, run.model_in_service = benchmark, pipelined.runs_in_service(entry["class"])
            assert run._is_ordered("ServiceStandInAttack")
            sequential = _run(benchmark, filepaths, attacks, "sequential")
            piped = _run(benchmark, filepaths, attacks, "pipelined")
        finally:
            (model_dir / "Dockerfile.off").rename(model_dir / "Dockerfile")
        assert json.dumps(piped) == json.dumps(sequential)


class TestLanes:
    def test_dockerized_plugins_are_recognised(self, benchmark):
        assert pipelined.runs_in_service(benchmark.attacks["ServiceStandInAttack"]["class"])
        assert pipelined.runs_in_service(benchmark.attacks["DiffusionAttack"]["class"])
        assert not pipelined.runs_in_service(benchmark.attacks["GaussianNoiseAttack"]["class"])
        assert not pipelined.runs_in_service(benchmark.attacks["InPlaceAttack"]["class"])

    def test_service_lanes_group_attacks(self, benchmark):
        run = pipelined.PipelinedRun.__new__(pipelined.PipelinedRun)
        run.entries = [((name,), False) for name in
                       ("DiffusionAttack", "VAEAttack", "EncodecAttack",
                        "NetworkTransmissionAttack", "NeuralVocoderAttack")]
        run.parallelism = Parallelism(
            service_lanes=(("NetworkTransmissionAttack",), ("NeuralVocoderAttack", "OpusCodecAttack")),
            service_concurrency={"NetworkTransmissionAttack": 2, "DiffusionAttack": 3},
        )
        lane_of, slots = run._plan_service_lanes()
        assert lane_of == {
            "DiffusionAttack": "services", "VAEAttack": "services", "EncodecAttack": "services",
            "NetworkTransmissionAttack": "NetworkTransmissionAttack",
            "NeuralVocoderAttack": "NeuralVocoderAttack",
        }
        # Diffusion shares its lane, so its concurrency cannot apply.
        assert slots == {"services": 1, "NetworkTransmissionAttack": 2, "NeuralVocoderAttack": 1}

    def test_an_attack_in_two_groups_is_refused(self, benchmark):
        run = pipelined.PipelinedRun.__new__(pipelined.PipelinedRun)
        run.entries = [(("VAEAttack",), False)]
        run.parallelism = Parallelism(service_lanes=(("VAEAttack",), ("VAEAttack", "EncodecAttack")))
        with pytest.raises(ValueError, match="more than one group"):
            run._plan_service_lanes()

    def test_attacks_that_draw_or_splice_stay_ordered(self, benchmark):
        run = pipelined.PipelinedRun.__new__(pipelined.PipelinedRun)
        run.bench, run.model_in_service = benchmark, True
        for name in ("CrossModelAttack", "ZeroBitCollusionAttack", "GaussianNoiseAttack",
                     "Codec2VocoderAttack", "SameModelAttack"):
            if name in benchmark.attacks:
                assert run._is_ordered(name), name
        assert not run._is_ordered("ServiceStandInAttack")


class TestGuards:
    def test_a_service_lane_drawing_from_the_host_rng_stops_the_run(self, benchmark, filepaths):
        with pytest.raises(RuntimeError, match="host RNG"):
            _run(benchmark, filepaths, NATIVE_ATTACKS + ["ServiceRngAttack"], "pipelined")

    def test_an_attack_changing_its_input_stops_the_run(self, benchmark, filepaths):
        with pytest.raises(RuntimeError, match="InPlaceAttack changed its input audio"):
            _run(benchmark, filepaths, ["InPlaceAttack", "LowpassFilterAttack"], "pipelined")


class TestFailures:
    def test_the_first_failure_is_raised_after_the_files_before_it(self, benchmark, filepaths):
        attacks = ["LowpassFilterAttack", "FailsOnLongFilesAttack", "ServiceStandInAttack"]
        outcomes = {}
        for executor in ("sequential", "pipelined"):
            completed = []
            with pytest.raises(ValueError, match="fails on long files"):
                _run(benchmark, filepaths, attacks, executor,
                     on_file_complete=lambda path, _: completed.append(path))
            outcomes[executor] = completed
        assert outcomes["pipelined"] == outcomes["sequential"] == filepaths[:1]

    def test_an_unknown_executor_is_refused_before_any_work(self, benchmark, filepaths):
        with pytest.raises(ValueError, match="Unknown executor"):
            benchmark.run(filepaths=filepaths, wm_model="ServiceStandInModel",
                          attack_types=["LowpassFilterAttack"], executor="threads")


class _Echo:
    """A service's answer: the audio it was sent, unchanged."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return {"audio": self._payload["audio"], "netem_active": True,
                "sampling_rate": self._payload.get("sampling_rate", 16000)}


def test_dockerized_attack_clients_draw_nothing_from_the_host_rng(monkeypatch, sample_audio):
    """What the pipelined executor's service lanes rely on, for every packaged client."""
    monkeypatch.setattr(requests, "post", lambda url, json=None, **kw: _Echo(json))
    bench = Benchmark()
    clients = sorted(name for name, entry in bench.attacks.items()
                     if pipelined.runs_in_service(entry["class"]))
    assert len(clients) >= 10, clients
    for name in clients:
        attack = bench.attacks[name]["class"]()
        random.seed(5)
        np.random.seed(5)
        before = (np.random.get_state()[1].copy(), random.getstate())
        attack.apply(np.array(sample_audio[0], dtype=np.float64), sampling_rate=sample_audio[1])
        assert np.array_equal(np.random.get_state()[1], before[0]), name
        assert random.getstate() == before[1], name


def test_metrics_do_not_change_their_inputs(sample_audio):
    """Metrics run in worker processes on copies; sequentially they share the arrays."""
    reference = np.array(sample_audio[0], dtype=np.float64)
    degraded = reference * 0.8 + 0.01
    kept = reference.copy(), degraded.copy()
    metrics.compute_metrics(reference, degraded, 16000,
                            metrics={"pesq", "stoi", "visqol", "psnr", "si_sdr", "sii", "ncm", "mcd"})
    assert np.array_equal(reference, kept[0]) and np.array_equal(degraded, kept[1])


class TestCommandLine:
    def _settings(self, argv):
        args, _ = run_module._build_parser().parse_known_args(argv)
        config = ModeConfig(mode="benchmark", source="c.json", general={})
        return run_module._resolve_settings(args, config)

    def test_defaults_are_sequential(self):
        settings = self._settings([])
        assert settings.executor == "sequential"
        assert settings.parallelism == Parallelism()

    def test_pipelined_flags(self):
        settings = self._settings([
            "--executor", "pipelined", "--workers", "3", "--files_in_flight", "5",
            "--service_lanes", "DiffusionAttack", "NeuralVocoderAttack+VAEAttack",
            "--service_concurrency", "DiffusionAttack=2", "VAEAttack=3",
        ])
        assert settings.executor == "pipelined"
        assert settings.parallelism == Parallelism(
            workers=3, files_in_flight=5,
            service_lanes=(("DiffusionAttack",), ("NeuralVocoderAttack", "VAEAttack")),
            service_concurrency={"DiffusionAttack": 2, "VAEAttack": 3},
        )

    @pytest.mark.parametrize("bad", [["--service_concurrency", "DiffusionAttack"],
                                     ["--service_lanes", "DiffusionAttack++VAEAttack"],
                                     ["--service_concurrency", "DiffusionAttack=0"],
                                     ["--workers", "0"]])
    def test_bad_values_are_refused(self, bad):
        with pytest.raises(SystemExit):
            run_module._build_parser().parse_known_args(bad)
