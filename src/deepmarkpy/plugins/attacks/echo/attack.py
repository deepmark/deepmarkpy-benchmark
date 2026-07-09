import numpy as np
from scipy.signal import fftconvolve

from deepmarkpy.core.base_attack import BaseAttack

class EchoAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform an echo attack on an audio signal. 
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the echo attack:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - volume_range (tuple): Min/max echo volume multiplier.
                - duration_range (tuple): Min/max delay duration in seconds.
        Returns:
            np.ndarray: The processed echoed audio signal.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """

        sampling_rate = kwargs.get("sampling_rate", None)
        volume_range = kwargs.get(
            "volume_range", self.config.get("volume_range")
        )
        duration_range = kwargs.get("duration_range", self.config.get("duration_range"))

        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")
        
        volume_range = tuple(volume_range)
        duration_range = tuple(duration_range)

        duration = np.random.uniform(*duration_range)
        volume = np.random.uniform(*volume_range)
        n_delay = int(sampling_rate * duration)

        # Build impulse response: [1.0, 0, 0, ..., 0, volume]
        impulse_response = np.zeros(n_delay + 1, dtype=np.float32)
        impulse_response[0] = 1.0  
        impulse_response[-1] = volume  

        echoed = fftconvolve(audio, impulse_response, mode='full')
        echoed = echoed / np.max(np.abs(echoed)) * np.max(np.abs(audio))
        echoed = echoed[:len(audio)]

        return echoed
