"""AWARE embed/detect inference, HTTP-free.

``resample_audio`` receives the ndarray on both endpoints, and embed
sanitizes before resampling back. The ``embed_watermark``/
``detect_watermark`` calls raise freely; app.py converts exceptions into
the service's error-in-200 responses (intentionally preserved shape).
"""

import numpy as np

from aware.service import embed_watermark, detect_watermark
from aware.utils.models import load

from deepmarkpy.utils.utils import resample_audio

from deepmarkpy.core.inference import BaseModelEngine


class AwareEngine(BaseModelEngine):
    """AWARE embed/detect via the ``aware`` package.

    The embedder/detector pair loads once at construction. ``device`` is
    accepted for interface uniformity and unused — the aware package
    manages its own placement.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load the AWARE embedder and detector."""
        self.config = config
        embedder, detector = load()
        self.model = {
            "embedder": embedder,
            "detector": detector,
        }

    def embed(self, audio: list, watermark_data: list, sampling_rate: int) -> np.ndarray:
        """Embed ``watermark_data``; sanitizes before the resample-back."""
        audio_arr = np.array(audio)
        watermark_arr = np.array(watermark_data, dtype=np.int32)

        if sampling_rate != self.config["sampling_rate"]:
            audio_arr = resample_audio(audio_arr, sampling_rate, self.config["sampling_rate"])

        embedder = self.model["embedder"]

        watermarked_audio = embed_watermark(
            audio_arr,
            self.config["sampling_rate"],
            watermark_arr,
            embedder
        )

        # Sanitize watermarked audio to ensure JSON serialization works
        # Replace NaN and Inf values with 0
        watermarked_audio = np.nan_to_num(watermarked_audio, nan=0.0, posinf=0.0, neginf=0.0)

        if sampling_rate != self.config["sampling_rate"]:
            watermarked_audio = resample_audio(watermarked_audio, self.config["sampling_rate"], sampling_rate)

        return watermarked_audio

    def detect(self, audio: list, sampling_rate: int) -> tuple:
        """Detect the watermark; returns ``(watermark_or_None, confidence)``."""
        audio_arr = np.array(audio)

        if sampling_rate != self.config["sampling_rate"]:
            audio_arr = resample_audio(audio_arr, sampling_rate, self.config["sampling_rate"])

        detector = self.model["detector"]

        detected_watermark, confidence = detect_watermark(
            audio_arr,
            self.config["sampling_rate"],
            detector
        )

        # Sanitize detected watermark and confidence to ensure JSON serialization works
        if detected_watermark is not None:
            detected_watermark = np.nan_to_num(detected_watermark, nan=0.0, posinf=1.0, neginf=0.0)


        return detected_watermark, confidence


# Stable import alias.
Engine = AwareEngine
