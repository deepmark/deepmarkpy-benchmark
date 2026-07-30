"""All inference for the AWARE model service (REORG_PLAN.md §5.1).

No FastAPI/HTTP imports. Logic moved verbatim from app.py: unlike the
other models, ``resample_audio`` receives the ndarray on both endpoints,
and embed sanitizes **before** resampling back — both preserved exactly.
The per-endpoint error-in-200 wrapping (D6, with its visible
``traceback.print_exc()`` stderr output) stays in app.py, which catches
what these methods raise; the ``embed_watermark``/``detect_watermark``
calls raise freely here.
"""

import numpy as np

from aware.service import embed_watermark, detect_watermark
from aware.utils.models import load

from utils.utils import resample_audio


class Engine:
    """AWARE embed/detect inference (logic inside the pip-installed package).

    The embedder/detector pair loads at construction — app.py wraps the
    construction in the current startup try/except (critical log +
    traceback + exit). ``device`` is accepted for signature uniformity but
    unused — the aware package manages its own placement.
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
        """Embed ``watermark_data``; sanitizes before the resample-back (as today)."""
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
