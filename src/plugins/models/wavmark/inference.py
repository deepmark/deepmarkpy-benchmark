"""WavMark embed/detect inference, HTTP-free.

``embed`` feeds ``resample_audio`` the raw request list while ``detect``
feeds it the ndarray (docs/KNOWN_DEFECTS.md D11).
"""

import logging

import numpy as np
import torch
import wavmark

from utils.utils import resample_audio

logger = logging.getLogger(__name__)


class Engine:
    """WavMark embed/detect at the config sampling rate.

    Handles the request-rate resample round-trips around the wavmark
    encode/decode calls. The model loads once at construction.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load the WavMark model onto ``device`` (default: cuda if available)."""
        self.config = config
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        self.model = wavmark.load_model().to(device)

    def embed(self, audio: list, watermark_data: list, sampling_rate: int) -> np.ndarray:
        """Embed ``watermark_data`` into ``audio``; returns the watermarked signal.

        ``audio``/``watermark_data`` are the parsed request lists. When
        resampling engages, ``resample_audio`` receives the raw list (D11).
        """
        audio_arr = np.array(audio)
        watermark_arr = np.array(watermark_data)
        if sampling_rate != self.config["sampling_rate"]:
            audio_arr = resample_audio(audio, sampling_rate, self.config["sampling_rate"])

        watermarked_audio, _ = wavmark.encode_watermark(self.model, audio_arr, watermark_arr, show_progress=False)

        if sampling_rate != self.config["sampling_rate"]:
            watermarked_audio = resample_audio(watermarked_audio, self.config["sampling_rate"], sampling_rate)
        return watermarked_audio

    def detect(self, audio: list, sampling_rate: int) -> "np.ndarray | None":
        """Decode the watermark from ``audio``; ``None`` when decoding fails.

        When resampling engages, ``resample_audio`` receives the ndarray (D11).
        """
        audio_arr = np.array(audio)
        if sampling_rate != self.config["sampling_rate"]:
            audio_arr = resample_audio(audio_arr, sampling_rate, self.config["sampling_rate"])
        message, _ = wavmark.decode_watermark(self.model, audio_arr, show_progress=False)
        return message
