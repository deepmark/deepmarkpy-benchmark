import logging
import os

import numpy as np
import requests

from core.base_attack import BaseAttack

logger = logging.getLogger(__name__)

class SpeechTokenizationAttack(BaseAttack):
    def __init__(self):
        super().__init__()

        host = "localhost" # Client always connects to localhost
        # Read the specific port variable for this attack service
        port = os.getenv("SPEECH_TOKENIZATION_PORT", "10003") # Default specific to SpeechTokenization
        if not port:
             logging.error("SPEECH_TOKENIZATION_PORT environment variable not set.")
             raise ValueError("SPEECH_TOKENIZATION_PORT must be set for SpeechTokenizationAttack")

        self.endpoint = f"http://{host}:{port}"
        logging.info(f"SpeechTokenizationAttack initialized. Target API: {self.endpoint}")

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate_st", None)
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        try:
            response = requests.post(
                self.endpoint + "/attack",
                json={"audio": audio.tolist(), "sampling_rate": sampling_rate},
                timeout=180,
            )
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"SpeechTokenizationAttack request failed: {e}")
            raise
        
        if "audio" not in response_data:
             logger.error("'/apply' response does not contain 'audio' key.")
             raise KeyError("Missing 'audio' in response from /apply")
        return np.array(response_data["audio"])

