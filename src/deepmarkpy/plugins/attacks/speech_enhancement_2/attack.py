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
        model_name = kwargs.get("model_name_se2", self.config.get("model_name_se2"))
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        try:
            response = requests.post(
                self.endpoint + "/attack",
                json={
                    "audio": audio.tolist(),
                    "sampling_rate": sampling_rate,
                    "model_name": model_name,
                },
                timeout=600,
            )
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"SpeechEnhancement2Attack request failed: {e}")
            raise
        
        if response_data.get("audio") is None:
            raise RuntimeError(
                f"SpeechEnhancement2Attack: service returned no audio "
                f"({response_data.get('error', 'no error reported')})"
            )
        result = np.array(response_data["audio"])
        return result
