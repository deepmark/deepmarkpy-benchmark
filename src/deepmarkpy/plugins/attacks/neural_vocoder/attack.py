import logging

import numpy as np

from deepmarkpy.core.base_attack import BaseAttack


logger = logging.getLogger(__name__)


class NeuralVocoderAttack(BaseAttack):
    """Reconstruct audio from a BigVGAN mel spectrogram."""

    _cache: dict[str, object] = {}

    def __init__(self):
        super().__init__()
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_name: str):
        model = self._cache.get(model_name)
        if model is None:
            from deepmarkpy.plugins.attacks.neural_vocoder.big_vgan import BigVGAN

            logger.info("Loading neural-vocoder model %s", model_name)
            model = BigVGAN(model_name, self.device)
            self._cache[model_name] = model
        return model

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate")
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        model_name = str(kwargs.get("model_name", self.config.get("model_name")))
        result = self._load_model(model_name).inference(
            np.asarray(audio, dtype=np.float32),
            int(sampling_rate),
        )
        return np.asarray(result, dtype=np.float32)
