import logging
import os

import numpy as np
import requests

from deepmarkpy.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)

class VAEAttack(BaseAttack):
    def __init__(self):
        super().__init__()

        host = "localhost" # Client always connects to localhost
        # Read the specific port variable for this attack service
        port = os.getenv("VAE_PORT", "10001") # Default specific to VAE
        if not port:
             logger.error("VAE_PORT environment variable not set.")
             raise ValueError("VAE_PORT must be set for VAEAttack")

        self.endpoint = f"http://{host}:{port}"
        logger.info(f"VAEAttack initialized. Target API: {self.endpoint}")

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """Applies the VAE attack using the backend service."""
        sampling_rate = kwargs.get("sampling_rate", None)
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        try:
            response = requests.post(
                self.endpoint + "/attack",
                json={
                    "audio": audio.tolist(),
                    "sampling_rate": sampling_rate
                },
                timeout=600,
            )
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"VAEAttack request failed: {e}")
            raise
        
        if response_data.get("audio") is None:
            raise RuntimeError(
                f"VAEAttack: service returned no audio "
                f"({response_data.get('error', 'no error reported')})"
            )
        return np.array(response_data["audio"])

