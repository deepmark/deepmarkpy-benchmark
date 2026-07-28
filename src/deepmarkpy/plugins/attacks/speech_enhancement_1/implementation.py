import logging

import numpy as np

from deepmarkpy.core.base_attack import BaseAttack


logger = logging.getLogger(__name__)


class SpeechEnhancement1Implementation(BaseAttack):
    """Enhance noisy speech with a SpeechBrain model."""

    _cache: dict[str, object] = {}

    def _load_model(self, enhancement_type: str):
        model = self._cache.get(enhancement_type)
        if model is None:
            from deepmarkpy.plugins.attacks.speech_enhancement_1.speech_brain import SpeechBrain

            logger.info("Loading SpeechBrain enhancement model type=%s", enhancement_type)
            model = SpeechBrain(enhancement_type)
            self._cache[enhancement_type] = model
        return model

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate")
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        enhancement_type = str(kwargs.get("type", self.config.get("type", "waveform")))
        noise_strength = float(
            kwargs.get("noise_strength", self.config.get("noise_strength", 0.0))
        )
        result = self._load_model(enhancement_type).inference(
            np.asarray(audio, dtype=np.float32),
            int(sampling_rate),
            noise_strength,
        )
        return np.asarray(result, dtype=np.float32)
