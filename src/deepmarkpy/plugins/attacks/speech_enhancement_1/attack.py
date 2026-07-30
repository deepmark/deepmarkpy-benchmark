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

        response = requests.post(
            self.endpoint + "/attack",
            json={
                "audio": audio.tolist(),
                "sampling_rate": sampling_rate,
                "noise_strength": noise_strength,
            },
        )
        response_data = response.json()
        
        if "audio" not in response_data:
             logger.error("'/apply' response does not contain 'audio' key.")
             raise KeyError("Missing 'audio' in response from /apply")
        result = np.array(response_data["audio"])
        return result
