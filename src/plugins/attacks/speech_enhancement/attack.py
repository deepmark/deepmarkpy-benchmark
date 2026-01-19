import logging
import os

import numpy as np
import requests

from core.base_attack import BaseAttack

logger = logging.getLogger(__name__)

class SpeechEnhancementAttack(BaseAttack):
    def __init__(self):
        super().__init__()

        host = "localhost" # Client always connects to localhost
        # Read the specific port variable for this attack service
        port = os.getenv("SPEECH_ENHANCEMENT_PORT", "10005") # Default specific to VAE
        if not port:
             logging.error("SPEECH_ENHANCEMENT_PORT environment variable not set.")
             raise ValueError("SPEECH_ENHANCEMENT_PORT must be set for SpeechEnhancementAttack")

        self.endpoint = f"http://{host}:{port}"
        logging.info(f"SpeechEnhancementAttack initialized. Target API: {self.endpoint}")

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate_se", None)
        noise_strength = kwargs.get("noise_strength", self.config.get("noise_strength"))
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
        return np.array(response_data["audio"])
