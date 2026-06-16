import numpy as np
from scipy.signal import resample_poly, decimate
from core.base_attack import BaseAttack

class ResamplingPolyAttack(BaseAttack):
    """
        Perform a resampling attack on an audio signal, by downsampling it first, and then upsampling it
        to the starting sampling rate. This is done by using resample_poly and decimate function from scipy.
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the resampling attack:
                - down_factor_resampling_poly (int): The downsampling factor, which is the same as upsampling factor.
        Returns:
            np.ndarray: The processed audio signal with the high-pass filtering applied.

        Raises:
            TypeError: If the `down_factor_resampling_poly` is not int.

    """

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        down_factor = kwargs.get(
            "down_factor_resampling_poly", self.config.get("down_factor_resampling_poly")
        )
        if not isinstance(down_factor, int):
            raise TypeError("'down_factor' must be the integer value.")
        
        audio_down = decimate(audio, q=down_factor, ftype='fir', zero_phase=True)
        audio_up = resample_poly(audio_down, up=down_factor, down=1)
        return audio_up