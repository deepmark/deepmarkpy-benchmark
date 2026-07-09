import logging
import os

import numpy as np
import requests

from deepmarkpy.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)

class SpeechEnhancement2Attack(BaseAttack):
    def __init__(self):
        super().__init__()

        host = "localhost" # Client always connects to localhost
        # Read the specific port variable for this attack service
        port = os.getenv("SPEECH_ENHANCEMENT_PORT2", "10006") # Default specific to VAE
        if not port:
             logger.error("SPEECH_ENHANCEMENT_PORT2 environment variable not set.")
             raise ValueError("SPEECH_ENHANCEMENT_PORT2 must be set for SpeechEnhancementAttack2")

        self.endpoint = f"http://{host}:{port}"
        logger.info(f"SpeechEnhancementAttack2 initialized. Target API: {self.endpoint}")

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate", 16000)
        model_name = kwargs.get("model_name", self.config.get("model_name"))
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        response = requests.post(
            self.endpoint + "/attack",
            json={
                "audio": audio.tolist(),
                "sampling_rate": sampling_rate,
                "model_name": model_name,
            },
        )
        response_data = response.json()
        
        if "audio" not in response_data:
             logger.error("'/apply' response does not contain 'audio' key.")
             raise KeyError("Missing 'audio' in response from /apply")
        result = np.array(response_data["audio"])
        return result
