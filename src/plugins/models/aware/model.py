import logging
import os

import numpy as np

from core.base_model import BaseModel

logger = logging.getLogger(__name__)

class AwareModel(BaseModel):
    def __init__(self):
        super().__init__()

        # Determine the APP PORT from environment variables
        port = os.getenv("AWARE_PORT", "9004")

        if not port:
            logger.error("AWARE_PORT environment variable not set and no default provided.")
            raise ValueError("AWARE_PORT must be set")

        self.base_url = f"http://localhost:{port}"
        logger.info(f"AwareModel initialized. Target API: {self.base_url}")

    def embed(
        self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int
    ) -> np.ndarray:
        """Embeds a watermark into the audio using the AWARE service."""
        # Sanitize audio: replace NaN with 0 and clip Inf to valid float range
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        payload = {
            "audio": audio.tolist(),
            "watermark_data": watermark_data.tolist(),
            "sampling_rate": sampling_rate,
        }

        response_data = self._make_request(endpoint="/embed", json_data=payload, method="POST")

        if "error" in response_data:
            error_msg = response_data["error"]
            logger.error(f"AWARE API returned error during embedding: {error_msg}")
            raise RuntimeError(f"AWARE embedding failed: {error_msg}")

        if "watermarked_audio" not in response_data:
            logger.error("'/embed' response did not contain 'watermarked_audio' key.")
            raise KeyError("Missing 'watermarked_audio' in response from /embed")


        return np.array(response_data["watermarked_audio"])

    def is_watermarked(self, detect_output) -> bool:
        """AWARE returns (watermark, confidence). Compare confidence to threshold."""
        _watermark, confidence = detect_output
        threshold = self._config.get("detection_threshold", 0.5)
        return float(confidence) >= threshold

    def detect(self, audio: np.ndarray, sampling_rate: int):
        """Detects a watermark in the audio using the AWARE service."""
        # Sanitize audio: replace NaN with 0 and clip Inf to valid float range
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        payload = {"audio": audio.tolist(), "sampling_rate": sampling_rate}

        response_data = self._make_request(endpoint="/detect", json_data=payload, method="POST")

        if "error" in response_data:
            error_msg = response_data["error"]
            logger.error(f"AWARE API returned error during detection: {error_msg}")
            raise RuntimeError(f"AWARE detection failed: {error_msg}")

        if "watermark" not in response_data:
            logger.error("'/detect' response did not contain 'watermark' key.")
            raise KeyError("Missing 'watermark' in response from /detect")

        # Handle potential None value from the API
        watermark = response_data["watermark"]
        confidence = response_data["confidence"]


        watermark_array = np.array(watermark) if watermark is not None else None
        return watermark_array, confidence