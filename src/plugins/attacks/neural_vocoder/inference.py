"""All inference for the neural_vocoder attack service (REORG_PLAN.md §5.1).

No FastAPI/HTTP imports. Logic moved verbatim from big_vgan.py, including
its import mechanics: the container-only ``app_utils`` alias and the
WORKDIR-dependent no-op ``sys.path.append("BigVGAN")`` stay intact until P2
(REORG_PLAN §4.1 — dead code moves verbatim; ``bigvgan``/``meldataset``
resolve from the image's /app/BigVGAN working directory).
"""

import logging
import os
import sys

import numpy as np
import torch
from app_utils.utils import resample_audio

sys.path.append("BigVGAN")

import bigvgan
from meldataset import get_mel_spectrogram

logger = logging.getLogger(__name__)


class Engine:
    """BigVGAN mel-vocoder resynthesis attack.

    The vocoder loads at construction (startup-loaded stays startup-loaded).
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
