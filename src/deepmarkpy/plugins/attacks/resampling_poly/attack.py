from math import gcd

import numpy as np
from scipy.signal import resample_poly
from deepmarkpy.core.base_attack import BaseAttack

class ResamplingPolyAttack(BaseAttack):
    """
        Perform a resampling attack on an audio signal: resample it to a target sampling rate, then
        back to its own rate, so the detector still gets audio at its native rate. Both steps use
        resample_poly from scipy. A target below the audio's rate band-limits it to half the target;
        a target at or above it leaves the audio nearly unchanged.
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the resampling attack:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - target_sr_resampling_poly (int): The sampling rate in Hz to resample to and back from.
        Returns:
            np.ndarray: The resampled audio signal, the same length as the input.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

    """

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate", None)
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")
        target_sr = kwargs.get(
            "target_sr_resampling_poly", self.config.get("target_sr_resampling_poly")
        )

        sampling_rate, target_sr = int(sampling_rate), int(target_sr)
        g = gcd(sampling_rate, target_sr)
        up, down = target_sr // g, sampling_rate // g

        resampled = resample_poly(audio, up=up, down=down)
        restored = resample_poly(resampled, up=down, down=up)

        # The round trip can be a sample longer or shorter than the input.
        if len(restored) >= len(audio):
            return restored[:len(audio)]
        return np.pad(restored, (0, len(audio) - len(restored)))
