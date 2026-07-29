import numpy as np
from scipy import signal
from deepmarkpy.core.base_attack import BaseAttack

class LowpassFilterAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a low-pass filtering attack on an audio signal.

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the lowpass attack:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - cutoff_freq (float): Optional override of the cutoff frequency in Hz.
                  When not provided, the cutoff is looked up from
                  config["cutoff_freq_per_sr_lowpass"] using the sampling rate
                  (closest configured rate wins on a miss).
                - order (int): The order of the Butterworth filter. Higher order means a steeper
                     roll-off but can introduce more phase distortion.
        Returns:
            np.ndarray: The processed audio signal with the low-pass filtering applied.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """
        sampling_rate = kwargs.get("sampling_rate", None)
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        cutoff_override = kwargs.get("cutoff_freq", None)
        if cutoff_override is not None:
            cutoff_freq = cutoff_override
        else:
            cutoff_freq = self._lookup_cutoff(int(sampling_rate))

        order = kwargs.get("order", self.config.get("order"))

        nyquist = 0.5 * sampling_rate
        # Guard against cutoffs at or above nyquist, which would crash butter().
        cutoff_freq = min(cutoff_freq, nyquist * 0.99)

        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
        filtered_signal = signal.lfilter(b, a, audio)

        return filtered_signal

    def _lookup_cutoff(self, sampling_rate: int) -> float:
        """Pick the cutoff for ``sampling_rate`` from the per-rate map.

        Falls back to the closest configured rate when ``sampling_rate``
        is not an exact key in the map.
        """
        per_sr = self.config.get("cutoff_freq_per_sr", {})
        if not per_sr:
            raise ValueError(
                "Lowpass config missing 'cutoff_freq_per_sr_lowpass' map."
            )
        rates = {int(k): v for k, v in per_sr.items()}
        if sampling_rate in rates:
            return rates[sampling_rate]
        closest = min(rates.keys(), key=lambda r: (abs(r - sampling_rate), -r))
        return rates[closest]
