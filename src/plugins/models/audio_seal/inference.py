"""All inference for the AudioSeal model service (REORG_PLAN.md §5.1).

No FastAPI/HTTP imports. Logic moved verbatim from app.py, including
defect D11 (both endpoints feed ``resample_audio`` the raw request list)
and the detect-path behaviors: the 1000-sample minimum-length guard and
the RuntimeError fallback both yield the empty result ``([], 0.0)`` that
app.py serializes into the current empty-response body.
"""

import logging

import numpy as np
import torch
from audioseal import AudioSeal

from utils.utils import resample_audio

logger = logging.getLogger(__name__)


class Engine:
    """AudioSeal embed/detect inference.

    Generator and detector load at construction (startup-loaded stays
    startup-loaded). ``device`` is accepted for signature uniformity but
    unused — the current code performs no device placement, and that is
    preserved.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load the AudioSeal generator and detector checkpoints."""
        self.config = config
        self.model = {
            "generator": AudioSeal.load_generator("audioseal_wm_16bits"),
            "detector": AudioSeal.load_detector("audioseal_detector_16bits"),
        }

    def embed(self, audio: list, watermark_data: list, sampling_rate: int) -> np.ndarray:
        """Additively embed ``watermark_data``; returns the watermarked signal.

        When resampling engages, ``resample_audio`` receives the raw list (D11).
        """
        audio_arr = np.array(audio)
        watermark_arr = np.array(watermark_data)
        if sampling_rate != self.config["sampling_rate"]:
            audio_arr = resample_audio(audio, sampling_rate, self.config["sampling_rate"])

        generator = self.model["generator"]
        wav = torch.tensor(audio_arr, dtype=torch.float32)
        wav = wav.unsqueeze(0).unsqueeze(0)
        msg = torch.from_numpy(watermark_arr).unsqueeze(0)

        watermark = generator.get_watermark(
            wav, message=msg, sample_rate=self.config["sampling_rate"]
        )

        watermarked_audio = wav + watermark
        watermarked_audio = watermarked_audio.detach().numpy()
        watermarked_audio = np.squeeze(watermarked_audio)

        if sampling_rate != self.config["sampling_rate"]:
            watermarked_audio = resample_audio(watermarked_audio, self.config["sampling_rate"], sampling_rate)

        return watermarked_audio

    def detect(self, audio: list, sampling_rate: int) -> tuple:
        """Detect the watermark; returns ``(message, confidence)``.

        The minimum-length guard and the detector RuntimeError fallback
        return ``([], 0.0)`` — app.py maps that to the current empty
        response body. When resampling engages, ``resample_audio`` receives
        the raw list (D11).
        """
        audio_arr = np.array(audio)
        if sampling_rate != self.config["sampling_rate"]:
            audio_arr = resample_audio(audio, sampling_rate, self.config["sampling_rate"])

        # AudioSeal requires minimum audio length for the neural network
        # Kernel size is 7, but due to architecture we need more samples
        min_samples = 1000  # Safe minimum for AudioSeal
        if len(audio_arr) < min_samples:
            logger.warning(f"Audio too short for detection ({len(audio_arr)} samples), returning empty result")
            return [], 0.0

        detector = self.model["detector"]
        watermarked_audio = np.expand_dims(audio_arr, axis=[0, 1])
        watermarked_audio = torch.tensor(watermarked_audio, dtype=torch.float32)

        try:
            confidence, message = detector.detect_watermark(watermarked_audio, sampling_rate)
        except RuntimeError as e:
            logger.error(f"Detection failed: {e}")
            return [], 0.0

        message = message.squeeze().cpu().numpy()
        return message, confidence
