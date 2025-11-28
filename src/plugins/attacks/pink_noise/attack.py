from scipy.signal import lfilter
import numpy as np

from core.base_attack import BaseAttack

class PinkNoiseAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a pink noise (Voss-McCartney) attack on an audio signal. 
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the pink noise attack:
                - amplitude (float): Controls how loud the sound is.
        Returns:
            np.ndarray: The processed audio signal with the pink noise applied.

        """

        amplitude = kwargs.get(
            "amplitude", self.config.get("amplitude")
        )
        n_samples = len(audio)
        white = np.random.normal(0, 1, n_samples)

        # Apply pink noise filter (from Julius O. Smith / Audio EQ Cookbook)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        pink = lfilter(b, a, white)

        pink = pink / np.max(np.abs(pink)) * amplitude
        pink = pink.astype(np.float32)

        noisy_audio=audio+pink
       
        return noisy_audio


