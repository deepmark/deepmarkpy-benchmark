import logging
import math

import numpy as np
import torch

from deepmarkpy.core.base_attack import BaseAttack


logger = logging.getLogger(__name__)


class DescriptAudioCodecAttack(BaseAttack):
    """Round-trip audio through a Descript Audio Codec model."""

    _cache: dict[str, dict] = {}

    def __init__(self):
        super().__init__()
        self.model = None
        self._model_cache_key = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.codec_sr = None
        self.n_codebooks = None
        self.supported_n_codebooks = None

    def _load_model(self, model_type: str):
        if self.model is not None and self._model_cache_key == model_type:
            return

        cached = self._cache.get(model_type)
        if cached is not None:
            self._apply_cache(cached)
            return

        try:
            import dac
        except ImportError as exc:
            raise RuntimeError("dac not found. Install deepmarkpy[dac].") from exc

        type_to_sr = {"44khz": 44100, "24khz": 24000, "16khz": 16000}
        if model_type not in type_to_sr:
            raise ValueError(f"Unsupported DAC model_type '{model_type}'")
        codec_sr = type_to_sr[model_type]

        logger.info("Downloading and loading DAC model %s", model_type)
        model_path = dac.utils.download(model_type=model_type)
        model = dac.DAC.load(model_path).to(self.device)
        model.eval()

        codebook_size = model.codebook_size
        downsampling_ratio = math.prod(
            block.block[-1].stride[0]
            for block in model.encoder.block
            if "EncoderBlock" in str(block.__class__)
        )
        supported_n_codebooks = list(range(1, model.n_codebooks + 1))
        supported_bandwidths = [
            codec_sr / downsampling_ratio * math.log2(codebook_size) * count
            for count in supported_n_codebooks
        ]
        cache = {
            "model": model,
            "model_type": model_type,
            "codec_sr": codec_sr,
            "n_codebooks": model.n_codebooks,
            "supported_n_codebooks": supported_n_codebooks,
            "supported_bandwidths": supported_bandwidths,
        }
        self._cache[model_type] = cache
        self._apply_cache(cache)

    def _apply_cache(self, cache: dict):
        self.model = cache["model"]
        self._model_cache_key = cache["model_type"]
        self.codec_sr = cache["codec_sr"]
        self.n_codebooks = cache["n_codebooks"]
        self.supported_n_codebooks = cache["supported_n_codebooks"]

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = int(kwargs.get("sampling_rate", 16000))
        model_type = str(
            kwargs.get("model_type", self.config.get("model_type", "44khz"))
        )
        self._load_model(model_type)

        target_sr = int(
            kwargs.get(
                "target_sampling_rate",
                self.config.get("target_sampling_rate", self.codec_sr),
            )
        )
        n_codebooks = int(
            kwargs.get("n_codebooks", self.config.get("n_codebooks", self.n_codebooks))
        )
        if n_codebooks not in self.supported_n_codebooks:
            logger.warning(
                "n_codebooks=%s not in supported range %s; using %s",
                n_codebooks,
                self.supported_n_codebooks,
                self.n_codebooks,
            )
            n_codebooks = self.n_codebooks

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        waveform = waveform.to(self.device)
        if sampling_rate != target_sr:
            import torchaudio

            waveform = torchaudio.transforms.Resample(sampling_rate, target_sr).to(
                self.device
            )(waveform)

        with torch.no_grad():
            original_length = waveform.shape[-1]
            reconstructed = self.model(waveform, n_quantizers=n_codebooks)["audio"]
            reconstructed = reconstructed[..., :original_length]

        if sampling_rate != target_sr:
            reconstructed = torchaudio.transforms.Resample(target_sr, sampling_rate).to(
                self.device
            )(reconstructed)

        return np.asarray(reconstructed.squeeze().cpu().numpy(), dtype=np.float32)
