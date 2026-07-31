"""BigVGAN mel-vocoder resynthesis attack inference, HTTP-free.

``bigvgan``/``meldataset`` resolve from the image's /app/BigVGAN working
directory.
"""

import logging

import numpy as np
import torch
from deepmarkpy.utils.utils import resample_audio

import bigvgan
from meldataset import get_mel_spectrogram

from deepmarkpy.core.inference import BaseAttackEngine

logger = logging.getLogger(__name__)


class NeuralVocoderEngine(BaseAttackEngine):
    """BigVGAN mel-vocoder resynthesis through a 44.1 kHz round-trip.

    The vocoder loads once at construction.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load BigVGAN onto ``device`` (default: cuda if available)."""
        self.config = config
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        self.device = device
        self.model = bigvgan.BigVGAN.from_pretrained(config["model_name_neural_vocoder"])
        self.model.remove_weight_norm()
        self.model.eval().to(self.device)

    def apply(self, audio: list, sampling_rate: int, **params) -> np.ndarray:
        """Resynthesize ``audio`` through the 44.1 kHz mel round-trip."""
        audio_arr = np.array(audio)
        audio_arr = resample_audio(audio_arr, input_sr=sampling_rate, target_sr=44100)
        audio_arr = torch.FloatTensor(audio_arr).unsqueeze(0)

        mel = get_mel_spectrogram(audio_arr, self.model.h).to(self.device)

        with torch.inference_mode():
            output = self.model(mel)

        output = output.squeeze().cpu().numpy()

        return resample_audio(output, input_sr=44100, target_sr=sampling_rate)

