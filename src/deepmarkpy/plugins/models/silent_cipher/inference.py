"""SilentCipher embed/detect inference, HTTP-free.

The 44.1k checkpoint is loaded while the service config declares 16 kHz —
intentional, preserved behavior. Both endpoints intentionally feed
``resample_audio`` the raw request list. ``detect`` returns a JSON-able
bit list, or ``None`` when decoding fails — the bare except intentionally
swallows the failure.
"""

import logging

import numpy as np
import silentcipher
import torch

from deepmarkpy.utils.utils import resample_audio

from deepmarkpy.core.inference import BaseModelEngine

logger = logging.getLogger(__name__)


class SilentCipherEngine(BaseModelEngine):
    """SilentCipher embed/detect with byte-level message packing.

    The model loads once at construction. The request watermark is a flat
    bit array whose length must be a multiple of 8; it is packed into bytes
    for ``encode_wav`` and unpacked back to bits after ``decode_wav``.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load the 44.1k SilentCipher model onto ``device``."""
        self.config = config
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        self.model = silentcipher.get_model(model_type='44.1k', device=device)

    def embed(self, audio: list, watermark_data: list, sampling_rate: int) -> np.ndarray:
        """Embed ``watermark_data`` (bits, packed to bytes); returns the signal."""
        audio_arr = np.array(audio)
        watermark_arr = np.array(watermark_data)
        if sampling_rate != self.config["sampling_rate"]:
            audio_arr = resample_audio(audio, sampling_rate, self.config["sampling_rate"])

        watermark_arr = np.split(watermark_arr, len(watermark_arr) // 8)
        watermark_arr = [int("".join(map(str, arr)), 2) for arr in watermark_arr]
        watermarked_audio, _ = self.model.encode_wav(audio_arr, self.config["sampling_rate"], watermark_arr, calc_sdr=False)

        if sampling_rate != self.config["sampling_rate"]:
            watermarked_audio = resample_audio(watermarked_audio, self.config["sampling_rate"], sampling_rate)

        return watermarked_audio

    def detect(self, audio: list, sampling_rate: int):
        """Decode the watermark to a bit list; ``None`` when decoding fails."""
        audio_arr = np.array(audio)

        if sampling_rate != self.config["sampling_rate"]:
            audio_arr = resample_audio(audio, sampling_rate, self.config["sampling_rate"])

        message = self.model.decode_wav(audio_arr, self.config["sampling_rate"], phase_shift_decoding=self.config["phase_shift_decoding"])
        try:
            message = message['messages'][0]
            message = [np.array(list(f"{val:08b}"), dtype=np.int32) for val in message]
            message = np.concatenate(message)
            message = message.tolist()
        except:  # noqa: E722
            message = None

        return message


# Stable import alias.
Engine = SilentCipherEngine
