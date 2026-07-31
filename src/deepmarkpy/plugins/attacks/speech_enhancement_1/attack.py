import logging
import os

import numpy as np
import requests

from deepmarkpy.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)

class SpeechEnhancement1Attack(BaseAttack):
    def __init__(self):
        super().__init__()

        host = "localhost" # Client always connects to localhost
        # Read the specific port variable for this attack service
        port = os.getenv("SPEECH_ENHANCEMENT_PORT1", "10005") # Default specific to VAE
        if not port:
             logger.error("SPEECH_ENHANCEMENT_PORT1 environment variable not set.")
             raise ValueError("SPEECH_ENHANCEMENT_PORT1 must be set for SpeechEnhancement1Attack")

        self.endpoint = f"http://{host}:{port}"
        logger.info(f"SpeechEnhancement1Attack initialized. Target API: {self.endpoint}")

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate", 16000)
        noise_strength = kwargs.get("noise_strength_se1", self.config.get("noise_strength_se1"))
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        try:
            response = requests.post(
                self.endpoint + "/attack",
                json={
                    "audio": audio.tolist(),
                    "sampling_rate": sampling_rate,
                    "noise_strength": noise_strength,
                },
                timeout=600,
            )
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"SpeechEnhancement1Attack request failed: {e}")
            raise
        
        if response_data.get("audio") is None:
            raise RuntimeError(
                f"SpeechEnhancement1Attack: service returned no audio "
                f"({response_data.get('error', 'no error reported')})"
            )
        result = np.array(response_data["audio"])
        return result
