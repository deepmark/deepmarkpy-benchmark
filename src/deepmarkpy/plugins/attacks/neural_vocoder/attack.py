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

        try:
            response = requests.post(
                self.endpoint + "/attack",
                json={"audio": audio.tolist(), "sampling_rate": sampling_rate},
                timeout=600,
            )
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"NeuralVocoderAttack request failed: {e}")
            raise
        
        if response_data.get("audio") is None:
            raise RuntimeError(
                f"NeuralVocoderAttack: service returned no audio "
                f"({response_data.get('error', 'no error reported')})"
            )
        return np.array(response_data["audio"])
