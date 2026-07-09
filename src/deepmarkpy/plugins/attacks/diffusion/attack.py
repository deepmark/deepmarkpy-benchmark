import logging
import os

import numpy as np
import requests

from deepmarkpy.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)

class DiffusionAttack(BaseAttack):
    def __init__(self):
        super().__init__()

        host = "localhost" # Client always connects to localhost
        # Read the specific port variable for this attack service
        port = os.getenv("DIFFUSION_PORT", "10002") # Default specific to Diffusion
        if not port:
             logger.error("DIFFUSION_PORT environment variable not set.")
             raise ValueError("DIFFUSION_PORT must be set for DiffusionAttack")

        self.endpoint = f"http://{host}:{port}"
        logger.info(f"DiffusionAttack initialized. Target API: {self.endpoint}")

    def apply(self, audio, **kwargs):
        """Applies the Diffusion attack using the backend service."""
        sampling_rate = kwargs.get("sampling_rate", None)
        diffusion_steps = kwargs.get(
            "steps", self.config.get("steps")
        )
        assert diffusion_steps <= 150, "number of steps is too large."
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        response = requests.post(
            self.endpoint + "/attack",
            json={
                "audio": audio.tolist(),
                "sampling_rate": sampling_rate,
                "diffusion_steps": diffusion_steps,
            },
        )
        response_data = response.json()
        
        if "audio" not in response_data:
             logger.error("'/apply' response does not contain 'audio' key.")
             raise KeyError("Missing 'audio' in response from /apply")
        return np.array(response_data["audio"])
