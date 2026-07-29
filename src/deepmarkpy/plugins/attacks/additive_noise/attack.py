import numpy as np

from deepmarkpy.core.base_attack import BaseAttack


class AdditiveNoiseAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Additive Gaussian noise attack.

        Args:
            audio (np.ndarray): The input audio signal.
            noise_level (float): The standard deviation of the Gaussian noise.

        Returns:
            np.ndarray: The audio signal with noise added.
        """
        noise_level = kwargs.get("noise_level", self.config.get("noise_level"))
        noise = np.random.normal(0, noise_level, audio.shape)
        return audio + noise
