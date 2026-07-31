"""Encodec neural-codec compression attack inference, HTTP-free.

Audio is resampled to the codec rate from the plugin config, passed through
an eval-mode encode/decode round trip at the configured bandwidth, and
resampled back to the request rate.
"""

import logging

import numpy as np
import torch
from encodec import EncodecModel

from deepmarkpy.core.inference import BaseAttackEngine

logger = logging.getLogger(__name__)


class EncodecEngine(BaseAttackEngine):
    """Encodec encode/decode round-trip at the configured bandwidth.

    The pre-trained codec (24 kHz or 48 kHz variant, from
    ``config['model_name_encodec']``) loads once at construction.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load the configured Encodec model onto ``device``."""
        self.config = config
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        self.device = device

        model_name = config.get("model_name_encodec", "encodec_24khz")
        bandwidth = config.get("bandwidth_encodec", 6.0)

        model = EncodecModel.encodec_model_24khz() if "24khz" in model_name else EncodecModel.encodec_model_48khz()
        model.set_target_bandwidth(bandwidth)
        model = model.to(self.device)
        model.eval()
        self.model = model

    def apply(self, audio: list, sampling_rate: int, **params) -> np.ndarray:
        """Run the Encodec compression round trip on ``audio``."""
        audio_arr = np.array(audio)

        waveform = torch.tensor(audio_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        waveform = waveform.to(self.device)

        target_sr = self.config.get("target_sampling_rate_encodec", 24000)
        if sampling_rate != target_sr:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=target_sr
            ).to(self.device)
            waveform = resampler(waveform)

        with torch.no_grad():
            encoded_frames = self.model.encode(waveform)
            reconstructed = self.model.decode(encoded_frames)

        if sampling_rate != target_sr:
            resampler_back = torchaudio.transforms.Resample(
                orig_freq=target_sr,
                new_freq=sampling_rate
            ).to(self.device)
            reconstructed = resampler_back(reconstructed)

        result = reconstructed.squeeze().cpu().numpy()

        return result


# Stable import alias.
Engine = EncodecEngine
