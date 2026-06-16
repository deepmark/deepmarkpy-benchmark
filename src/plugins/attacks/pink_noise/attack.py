from scipy.signal import lfilter
import numpy as np

from core.base_attack import BaseAttack

class PinkNoiseAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a pink noise attack on an audio signal using SNR control.
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the pink noise attack:
                - snr_db_pink_noise (float): Desired Signal-to-Noise Ratio in dB
        Returns:
            np.ndarray: The processed audio signal with the pink noise applied.
        """

        snr_db = kwargs.get(
            "snr_db_pink_noise", self.config.get("snr_db_pink_noise")
        )

        n_samples = len(audio)
        white = np.random.normal(0, 1, n_samples)

        # Apply pink noise filter (from Julius O. Smith / Audio EQ Cookbook)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        pink = lfilter(b, a, white)

        # Scale pink noise based on SNR
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        # Normalize pink noise to have the desired power
        pink_power = np.mean(pink ** 2)
        pink = pink * np.sqrt(noise_power / pink_power)
        pink = pink.astype(np.float32)

        noisy_audio = audio + pink

        return noisy_audio
