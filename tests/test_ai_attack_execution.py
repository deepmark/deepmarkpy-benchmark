"""Unit coverage for packaged AI attacks without downloading model weights."""

from types import SimpleNamespace

import numpy as np
import pytest


def _attack_without_init(cls, attack_key: str, config: dict):
    attack = cls.__new__(cls)
    attack.attack_key = attack_key
    attack._config = config
    return attack


def test_vae_executes_in_process_and_rejects_short_audio(monkeypatch):
    from deepmarkpy.plugins.attacks.vae.implementation import VAEImplementation

    attack = _attack_without_init(VAEImplementation, "vae", {"model_name": "test-model"})
    model = SimpleNamespace(inference=lambda audio: audio * 0.5)
    monkeypatch.setattr(attack, "_load_model", lambda model_name: model)
    monkeypatch.setattr(
        "deepmarkpy.plugins.attacks.vae.implementation.resample_audio",
        lambda audio, _input_sr, _target_sr: audio,
    )

    with pytest.raises(ValueError, match="shorter than one VAE block"):
        attack.apply(np.ones(1024, dtype=np.float32), sampling_rate=16000)

    result = attack.apply(np.ones(4096, dtype=np.float32), sampling_rate=16000)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 0.5)


def test_diffusion_uses_canonical_and_legacy_step_names(monkeypatch):
    from deepmarkpy.plugins.attacks.diffusion.implementation import (
        DiffusionImplementation,
    )

    attack = _attack_without_init(
        DiffusionImplementation,
        "diffusion",
        {"model_name": "test-model", "steps": 5},
    )
    calls = []
    model = SimpleNamespace(
        inference=lambda audio, sampling_rate, start_step: (
            calls.append((sampling_rate, start_step)) or audio
        )
    )
    monkeypatch.setattr(attack, "_load_model", lambda model_name: model)

    audio = np.ones(2048, dtype=np.float32)
    attack.apply(audio, sampling_rate=16000, steps=7)
    attack.apply(audio, sampling_rate=16000, diffusion_steps=9)

    assert calls == [(16000, 993), (16000, 991)]


@pytest.mark.parametrize(
    ("module_name", "class_name", "attack_key", "config"),
    [
        (
            "deepmarkpy.plugins.attacks.speech_tokenization.implementation",
            "SpeechTokenizationImplementation",
            "speech_tokenization",
            {"model_name": "test-tokenizer"},
        ),
        (
            "deepmarkpy.plugins.attacks.neural_vocoder.implementation",
            "NeuralVocoderImplementation",
            "neural_vocoder",
            {"model_name": "test-vocoder"},
        ),
    ],
)
def test_model_reconstruction_attacks_execute_in_process(
    monkeypatch,
    module_name,
    class_name,
    attack_key,
    config,
):
    module = __import__(module_name, fromlist=[class_name])
    attack = _attack_without_init(getattr(module, class_name), attack_key, config)
    model = SimpleNamespace(inference=lambda audio, sampling_rate: audio * 0.25)
    monkeypatch.setattr(attack, "_load_model", lambda model_name: model)

    result = attack.apply(np.ones(2048, dtype=np.float32), sampling_rate=32000)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 0.25)


def test_speech_enhancement_1_forwards_runtime_parameters(monkeypatch):
    from deepmarkpy.plugins.attacks.speech_enhancement_1.implementation import (
        SpeechEnhancement1Implementation,
    )

    attack = _attack_without_init(
        SpeechEnhancement1Implementation,
        "speech_enhancement_1",
        {"type": "waveform", "noise_strength": 0.01},
    )
    calls = []
    model = SimpleNamespace(
        inference=lambda audio, sampling_rate, noise_strength: (
            calls.append((sampling_rate, noise_strength)) or audio
        )
    )
    monkeypatch.setattr(attack, "_load_model", lambda enhancement_type: model)

    attack.apply(
        np.ones(2048, dtype=np.float32),
        sampling_rate=44100,
        noise_strength=0.005,
    )

    assert calls == [(44100, 0.005)]


def test_speech_enhancement_2_resamples_around_clearvoice(monkeypatch):
    from deepmarkpy.plugins.attacks.speech_enhancement_2.implementation import (
        SpeechEnhancement2Implementation,
    )

    attack = _attack_without_init(
        SpeechEnhancement2Implementation,
        "speech_enhancement_2",
        {"model_name": "test-clearvoice", "noise_strength": 0.0},
    )
    resamples = []

    def fake_resample(audio, input_sr, target_sr):
        resamples.append((input_sr, target_sr))
        return audio

    class FakeClearVoice:
        def __call__(self, *, input_path, online_write):
            assert input_path.endswith(".wav")
            assert online_write is False
            return {"output": np.ones(512, dtype=np.float32)}

    monkeypatch.setattr(
        "deepmarkpy.plugins.attacks.speech_enhancement_2.implementation.resample_audio",
        fake_resample,
    )
    monkeypatch.setattr(attack, "_load_model", lambda model_name: FakeClearVoice())

    result = attack.apply(np.ones(512, dtype=np.float32), sampling_rate=44100)

    assert result.dtype == np.float32
    assert resamples == [(44100, 16000), (16000, 44100)]
