import logging

import numpy as np

from deepmarkpy.core.base_attack import BaseAttack
from deepmarkpy.utils.utils import resample_audio


logger = logging.getLogger(__name__)


class VAEImplementation(BaseAttack):
    """Reconstruct audio through a pretrained RAVE VAE model."""

    _cache: dict[str, object] = {}

    def __init__(self):
        super().__init__()
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_name: str):
        model = self._cache.get(model_name)
        if model is None:
            from deepmarkpy.plugins.attacks.vae.vae import VAE

            logger.info("Loading VAE model %s", model_name)
            model = VAE(model_name, self.device)
            self._cache[model_name] = model
        return model

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate")
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        model_name = kwargs.get("model_name", self.config.get("model_name"))
        samples = np.asarray(audio, dtype=np.float32).squeeze()
        block_size = 2048
        new_length = (len(samples) // block_size) * block_size
        if new_length <= 0:
            raise ValueError("Audio payload is shorter than one VAE block")

        samples = samples[:new_length]
        samples = resample_audio(samples, int(sampling_rate), 48000)
        samples = self._load_model(str(model_name)).inference(samples)
        return np.asarray(
            resample_audio(samples, 48000, int(sampling_rate)),
            dtype=np.float32,
        )
