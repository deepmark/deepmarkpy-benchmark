import numpy as np

from core.base_attack import BaseAttack

class SmoothingAttack(BaseAttack):
    """
        Perform a smoothing attack on an audio signal with a uniform moving average filter.
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the smoothing attack:
                - window_size_smoothing (int): Smoothing window size.
        Returns:
            np.ndarray: The processed audio signal with the smoothing filtering applied.

    """

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        window_size = kwargs.get(
            "window_size_smoothing", self.config.get("window_size_smoothing")
        )
        window_size = int(window_size)
        kernel = np.ones(window_size) / window_size  # uniform smoothing kernel
        smoothed = np.convolve(audio, kernel, mode='same')
        return smoothed