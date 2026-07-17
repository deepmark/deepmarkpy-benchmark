import logging
import os
import tempfile

import numpy as np
import soundfile as sf

from deepmarkpy.core.base_attack import BaseAttack
from deepmarkpy.utils.utils import resample_audio


logger = logging.getLogger(__name__)


class SpeechEnhancement2Attack(BaseAttack):
    """Enhance speech with ClearVoice using its required 16 kHz input."""

    _cache: dict[str, object] = {}

    def _load_model(self, model_name: str):
        model = self._cache.get(model_name)
        if model is None:
            from clearvoice import ClearVoice

            logger.info("Loading ClearVoice enhancement model %s", model_name)
            model = ClearVoice(task="speech_enhancement", model_names=[model_name])
            self._cache[model_name] = model
        return model

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate")
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        model_name = str(kwargs.get("model_name", self.config.get("model_name")))
        noise_strength = float(
            kwargs.get("noise_strength", self.config.get("noise_strength", 0.0))
        )
        samples = np.asarray(audio, dtype=np.float32).squeeze()
        if noise_strength > 0:
            samples = samples + noise_strength * np.random.normal(0, 1, size=len(samples))

        target_sr = 16000
        samples = resample_audio(samples, int(sampling_rate), target_sr)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            sf.write(tmp_path, samples, target_sr)
            enhanced = self._load_model(model_name)(input_path=tmp_path, online_write=False)
            if isinstance(enhanced, dict):
                enhanced = next(iter(enhanced.values()))
            enhanced = np.asarray(enhanced, dtype=np.float32).squeeze()
            return np.asarray(
                resample_audio(enhanced, target_sr, int(sampling_rate)),
                dtype=np.float32,
            )
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
