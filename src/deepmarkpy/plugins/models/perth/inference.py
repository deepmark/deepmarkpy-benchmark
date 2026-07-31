"""Perth zero-bit embed/detect inference, HTTP-free.

The request watermark bits are packed into bytes and then discarded —
``apply_watermark`` receives ``watermark=None`` and uses Perth's internal
watermark. The packing is kept: it also rejects a watermark whose length
is not a multiple of 8, which nothing else validates. ``detect`` returns
a JSON-able scalar or list.
"""

import logging

import numpy as np
import torch
from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

from deepmarkpy.utils.utils import resample_audio

from deepmarkpy.core.inference import BaseModelEngine

logger = logging.getLogger(__name__)


class PerthEngine(BaseModelEngine):
    """Perth zero-bit embed/detect at the config sampling rate.

    The watermarker loads once at construction. The computed ``device`` is
    logged and otherwise unused.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load the Perth implicit watermarker."""
        self.config = config
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        self.model = PerthImplicitWatermarker()

    def embed(self, audio: list, watermark_data: list, sampling_rate: int) -> np.ndarray:
        """Embed Perth's internal watermark; the request payload is unused."""
        audio_arr = np.array(audio)
        watermark_arr = np.array(watermark_data)
        if sampling_rate != self.config["sampling_rate"]:
            audio_arr = resample_audio(audio_arr, sampling_rate, self.config["sampling_rate"])

        # The packed value is intentionally unused: apply_watermark receives
        # watermark=None.
        watermark_arr = np.split(watermark_arr, len(watermark_arr) // 8)
        watermark_arr = [int("".join(map(str, arr)), 2) for arr in watermark_arr]
        watermarked_audio = self.model.apply_watermark(audio_arr,watermark=None,sample_rate=self.config["sampling_rate"])

        if sampling_rate != self.config["sampling_rate"]:
            watermarked_audio = resample_audio(watermarked_audio, self.config["sampling_rate"], sampling_rate)

        # Sanitize to ensure JSON serialization works
        watermarked_audio = np.nan_to_num(watermarked_audio, nan=0.0, posinf=0.0, neginf=0.0)

        return watermarked_audio

    def detect(self, audio: list, sampling_rate: int):
        """Return the detection score as a JSON-able scalar or list."""
        audio_arr = np.array(audio)

        if sampling_rate != self.config["sampling_rate"]:
            audio_arr = resample_audio(audio_arr, sampling_rate, self.config["sampling_rate"])

        message = self.model.get_watermark(audio_arr, self.config["sampling_rate"], round=True)
        if isinstance(message, np.ndarray) and message.ndim == 0:
            message = message.item() # Converts a 0-d NumPy array to its scalar equivalent

        # Sanitize to ensure JSON serialization works
        if isinstance(message, np.ndarray):
            message = np.nan_to_num(message, nan=0.0, posinf=0.0, neginf=0.0)
            message = message.tolist()
        elif isinstance(message, float):
            import math
            if math.isnan(message) or math.isinf(message):
                message = 0.0

        return message

