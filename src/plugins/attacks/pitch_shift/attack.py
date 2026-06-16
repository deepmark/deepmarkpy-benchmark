import numpy as np
import pyrubberband as pyrb
import librosa
import logging

from core.base_attack import BaseAttack

logger = logging.getLogger(__name__)

class PitchShiftAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform pitch shifting using pyrubberband.
        Falls back to librosa if rubberband-cli is not installed.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters for pitch shifting.
                - sampling_rate (int): Sampling rate of the audio in Hz (optional).
                - cents_pitch_shift (float): Pitch shift in cents (1 cent = 1/100 of a semitone) (optional).

        Returns:
            np.ndarray: The pitch-shifted audio signal.

        Notes:
            - This function uses the pyrubberband library, which provides high-quality
            pitch shifting without altering the speed of the audio.
            - If rubberband-cli is not installed, falls back to librosa.effects.pitch_shift
            - Ensure that pyrubberband and the Rubber Band Library are installed before use.
        """

        sampling_rate = kwargs.get("sampling_rate", None)
        cents = kwargs.get("cents_pitch_shift", self.config.get("cents_pitch_shift"))

        if sampling_rate is None or cents is None:
            raise ValueError(
                "A sampling_rate and cents must be specified for PitchShiftAttack."
            )

        semitones = cents / 100
        
        try:
            # Try using pyrubberband first
            return pyrb.pitch_shift(audio, sampling_rate, semitones)
        except Exception as e:
            logger.warning(f"Pyrubberband failed: {str(e)}. Falling back to librosa pitch_shift.")
            # Use librosa as a fallback
            return librosa.effects.pitch_shift(audio, sr=sampling_rate, n_steps=semitones)
