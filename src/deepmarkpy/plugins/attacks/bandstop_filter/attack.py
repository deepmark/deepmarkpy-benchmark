import numpy as np
from scipy import signal
from deepmarkpy.core.base_attack import BaseAttack

class BandstopFilterAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a band-stop filtering attack on an audio signal.
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the bandstop attack:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - freq_range_bandstop (float): Frequency range for the band stop filter.    
                - order_bandstop (int): The order of the Butterworth filter. Higher order means a steeper
                roll-off but can introduce more phase distortion.
        Returns:
            np.ndarray: The processed audio signal with the band-stop filtering applied.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """
        sampling_rate = kwargs.get("sampling_rate", None)
        freq_range = kwargs.get(
            "freq_range_bandstop", self.config.get("freq_range_bandstop")
        )
        order = kwargs.get("order_bandstop", self.config.get("order_bandstop"))

        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")
       
        nyquist = 0.5 * sampling_rate

        # Second-order sections rather than (b, a): at the low normalized
        # frequencies this band sits at, the transfer-function form loses
        # enough precision to drive the output to ~1e147 by 48 kHz.
        sos = signal.butter(
            order,
            [freq_range[0] / nyquist, freq_range[1] / nyquist],
            btype='bandstop',
            analog=False,
            output='sos',
        )

        filtered_signal = signal.sosfiltfilt(sos, audio)

        return filtered_signal