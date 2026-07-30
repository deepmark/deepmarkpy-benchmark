import logging
import os

import numpy as np
import requests

from deepmarkpy.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)

class NeuralVocoderAttack(BaseAttack):
    def __init__(self):
        super().__init__()

        host = "localhost" # Client always connects to localhost
        # Read the specific port variable for this attack service
        port = os.getenv("NEURAL_VOCODER_PORT", "10004") # Default specific to NeuralVocoder
        if not port:
             logging.error("NEURAL_VOCODER_PORT environment variable not set.")
             raise ValueError("NEURAL_VOCODER_PORT must be set for NeuralVocoderAttack")

        self.endpoint = f"http://{host}:{port}"
        logging.info(f"NeuralVocoderAttack initialized. Target API: {self.endpoint}")

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate", None)
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        response = requests.post(
            self.endpoint + "/attack",
            json={"audio": audio.tolist(), "sampling_rate": sampling_rate},
        )
        response_data = response.json()
        
        if "audio" not in response_data:
             logger.error("'/apply' response does not contain 'audio' key.")
             raise KeyError("Missing 'audio' in response from /apply")
        return np.array(response_data["audio"])
