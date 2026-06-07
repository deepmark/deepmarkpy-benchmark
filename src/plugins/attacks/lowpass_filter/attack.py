import numpy as np
from scipy import signal
from core.base_attack import BaseAttack

class LowpassFilterAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a low-pass filtering attack on an audio signal.

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the lowpass attack:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - cutoff_freq_lowpass (float): The cutoff frequency of the low-pass filter in Hz.
                - order_lowpass (int): The order of the Butterworth filter. Higher order means a steeper
                     roll-off but can introduce more phase distortion.
        Returns:
            np.ndarray: The processed audio signal with the low-pass filtering applied.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """
        sampling_rate = kwargs.get("sampling_rate", None)
        cutoff_freq = kwargs.get(
            "cutoff_freq_lowpass", self.config.get("cutoff_freq_lowpass")
        )
        order = kwargs.get("order_lowpass", self.config.get("order_lowpass"))

        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")
       
        nyquist = 0.5 * sampling_rate
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
        filtered_signal = signal.lfilter(b, a, audio)

        return filtered_signal