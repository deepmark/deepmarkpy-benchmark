import numpy as np
import random

from deepmarkpy.core.base_attack import BaseAttack

import logging
logger = logging.getLogger(__name__)


class CropRandomAttack(BaseAttack):
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a crop attack by removing a percentage of audio from a random position.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - crop_percentage (float): Percentage of audio to remove. Default is 10.

        Returns:
            np.ndarray: Audio signal with a portion cropped from a random position.

        Raises:
            ValueError: If 'sampling_rate' is not provided in kwargs.
        """
        sampling_rate = kwargs.get("sampling_rate", None)
        crop_percentage = kwargs.get(
            "crop_percentage", self.config.get("crop_percentage", 10)
        )

        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        total_samples = len(audio)
        samples_to_crop = int(total_samples * (crop_percentage / 100.0))

        # Random start position 
        max_start = total_samples - samples_to_crop
        start_index = random.randint(0, max_start)
        end_index = start_index + samples_to_crop

        # Remove the cropped section by concatenating before and after
        cropped_audio = np.concatenate([audio[:start_index], audio[end_index:]])

        return cropped_audio
