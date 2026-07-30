import numpy as np
import pyrubberband as pyrb
import librosa
import logging

from deepmarkpy.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)

class TimeStretchAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform time stretching on an audio signal using pyrubberband.
        Falls back to librosa if rubberband-cli is not installed.

        Args:
            audio (np.ndarray): Input audio signal to be stretched.
            **kwargs: Additional parameters for time stretching.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - stretch_rate_time_stretch (float): Stretching factor (>1 for slower, <1 for faster) (optional).

        Returns:
            np.ndarray: The time-stretched audio signal.

        Raises:
            ValueError: If `sampling_rate` is not provided in kwargs.

        Notes:
            - This function uses the pyrubberband library, which provides high-quality
            time-stretching capabilities while maintaining pitch integrity.
            - If rubberband-cli is not installed, falls back to librosa.effects.time_stretch
            - Ensure that pyrubberband and the Rubber Band Library are installed before use.
            - Stretch rate of `1.0` implies no change in speed.
            - Values greater than `1.0` slow down the audio, while values less than `1.0` speed it up.
        """
        sampling_rate = kwargs.get("sampling_rate", None)

        stretch_rate = kwargs.get("stretch_rate_time_stretch", self.config.get("stretch_rate_time_stretch"))

        if sampling_rate is None:
            raise ValueError(
                "'sampling_rate' must be provided in kwargs."
            )

        try:
            # Try using pyrubberband first
            return pyrb.time_stretch(audio, sampling_rate, stretch_rate)
        except Exception as e:
            logger.warning(f"Pyrubberband failed: {str(e)}. Falling back to librosa time_stretch.")
            # Use librosa as a fallback
            # Note: librosa uses rate=1/stretch_rate, so we need to invert it
            return librosa.effects.time_stretch(audio, rate=1/stretch_rate)
