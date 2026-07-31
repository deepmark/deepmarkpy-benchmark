import logging
import os

import numpy as np
import requests

from deepmarkpy.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)


class EncodecAttack(BaseAttack):
    """HTTP client for the containerized Encodec compression attack."""

    def __init__(self):
        super().__init__()

        port = os.getenv("ENCODEC_PORT", "10007")
        self.endpoint = f"http://localhost:{port}"
        logger.info(f"EncodecAttack initialized. Target API: {self.endpoint}")

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """Run the Encodec compression round trip via the encodec service.

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters:
                - sampling_rate (int): The sampling rate of the audio signal in Hz.

        Returns:
            np.ndarray: The processed audio signal after Encodec compression.
        """
        sampling_rate = kwargs.get("sampling_rate", 16000)

        try:
            response = requests.post(
                self.endpoint + "/attack",
                json={"audio": audio.tolist(), "sampling_rate": sampling_rate},
                timeout=600,
            )
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"EncodecAttack request failed: {e}")
            raise

        if response_data.get("audio") is None:
            raise RuntimeError(
                f"EncodecAttack: service returned no audio "
                f"({response_data.get('error', 'no error reported')})"
            )

        return np.array(response_data["audio"])
