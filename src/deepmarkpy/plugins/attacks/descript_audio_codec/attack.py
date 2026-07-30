import logging
import os

import numpy as np
import requests

from deepmarkpy.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)


class DescriptAudioCodecAttack(BaseAttack):
    """HTTP client for the containerized Descript Audio Codec attack."""

    def __init__(self):
        super().__init__()

        port = os.getenv("DESCRIPT_AUDIO_CODEC_PORT", "10008")
        self.endpoint = f"http://localhost:{port}"
        logger.info(f"DescriptAudioCodecAttack initialized. Target API: {self.endpoint}")

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """Run the DAC compression round trip via the descript_audio_codec service.

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters:
                - sampling_rate (int): The sampling rate of the audio signal in Hz.
                - n_codebooks_dac (int): Number of codebooks to use (optional,
                  from config if not provided).

        Returns:
            np.ndarray: The processed audio signal after DAC compression.
        """
        sampling_rate = kwargs.get("sampling_rate", 16000)
        n_codebooks = kwargs.get("n_codebooks_dac", self.config.get("n_codebooks_dac"))

        payload = {"audio": audio.tolist(), "sampling_rate": sampling_rate}
        if n_codebooks is not None:
            payload["n_codebooks_dac"] = n_codebooks

        try:
            response = requests.post(
                self.endpoint + "/attack",
                json=payload,
                timeout=600,
            )
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"DescriptAudioCodecAttack request failed: {e}")
            raise

        if "audio" not in response_data:
            logger.error("'/attack' response does not contain 'audio' key.")
            raise KeyError("Missing 'audio' in response from /attack")

        return np.array(response_data["audio"])
