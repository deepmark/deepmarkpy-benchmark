import logging

import numpy as np
import torch

from deepmarkpy.core.base_attack import BaseAttack


logger = logging.getLogger(__name__)


class EncodecAttack(BaseAttack):
    """Round-trip audio through an Encodec model."""

    _cache: dict[tuple[str, float], object] = {}

    def __init__(self):
        super().__init__()
        self.model = None
        self._model_cache_key = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_name: str, bandwidth: float):
        cache_key = (model_name, bandwidth)
        if self.model is not None and self._model_cache_key == cache_key:
            return

        cached = self._cache.get(cache_key)
        if cached is not None:
            self.model = cached
            self._model_cache_key = cache_key
            return

        try:
            from encodec import EncodecModel
        except ImportError as exc:
            raise RuntimeError("encodec not found. Install deepmarkpy[encodec].") from exc

        logger.info("Loading Encodec model %s at bandwidth %s", model_name, bandwidth)
        if "24khz" in model_name:
            model = EncodecModel.encodec_model_24khz()
        else:
            model = EncodecModel.encodec_model_48khz()
        model.set_target_bandwidth(bandwidth)
        model = model.to(self.device)
        model.eval()

        self._cache[cache_key] = model
        self.model = model
        self._model_cache_key = cache_key

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = int(kwargs.get("sampling_rate", 16000))
        model_name = str(
            kwargs.get("model_name", self.config.get("model_name", "encodec_24khz"))
        )
        bandwidth = float(kwargs.get("bandwidth", self.config.get("bandwidth", 6.0)))
        target_sr = int(
            kwargs.get(
                "target_sampling_rate",
                self.config.get("target_sampling_rate", 24000),
            )
        )
        self._load_model(model_name, bandwidth)

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        waveform = waveform.to(self.device)

        if sampling_rate != target_sr:
            import torchaudio

            waveform = torchaudio.transforms.Resample(sampling_rate, target_sr).to(
                self.device
            )(waveform)

        with torch.no_grad():
            reconstructed = self.model.decode(self.model.encode(waveform))

        if sampling_rate != target_sr:
            reconstructed = torchaudio.transforms.Resample(target_sr, sampling_rate).to(
                self.device
            )(reconstructed)

        return np.asarray(reconstructed.squeeze().cpu().numpy(), dtype=np.float32)
