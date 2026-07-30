"""All inference for the Perth model service (REORG_PLAN.md §5.1).

No FastAPI/HTTP imports. Logic moved verbatim from app.py, including
defect D4 (docs/KNOWN_DEFECTS.md): the watermark bits are packed into
bytes and then ``apply_watermark`` is called with ``watermark=None`` —
the payload is discarded. **Never delete the packing as dead code.**
Also preserved: D11 (both endpoints feed ``resample_audio`` the raw
request list) and the unused device selection (dead, moves verbatim).
detect's sanitize block converts to JSON-able values in place (its
embedded ``.tolist()`` is part of the moved sequence), so app.py wraps
the returned value without further serialization.
"""

import logging

import numpy as np
import torch
from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

from utils.utils import resample_audio

logger = logging.getLogger(__name__)


class Engine:
    """Perth zero-bit embed/detect inference.

    The watermarker loads at construction (startup-loaded stays
    startup-loaded). The ``device`` computed here is unused by the current
    code — preserved verbatim, do not wire it up.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load the Perth implicit watermarker."""
        self.config = config
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        self.model = PerthImplicitWatermarker()

    def embed(self, audio: list, watermark_data: list, sampling_rate: int) -> np.ndarray:
        """Embed Perth's internal watermark (the request payload is discarded — D4)."""
        audio_arr = np.array(audio)
        watermark_arr = np.array(watermark_data)
        if sampling_rate != self.config["sampling_rate"]:
            audio_arr = resample_audio(audio, sampling_rate, self.config["sampling_rate"])

        # Preserved verbatim per KNOWN_DEFECTS D4 — the packed value is
        # discarded (watermark=None below); do not fix here.
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
            audio_arr = resample_audio(audio, sampling_rate, self.config["sampling_rate"])

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
