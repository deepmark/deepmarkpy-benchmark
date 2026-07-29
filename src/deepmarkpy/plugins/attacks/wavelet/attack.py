import numpy as np
import pywt

from deepmarkpy.core.base_attack import BaseAttack


class WaveletAttack(BaseAttack):
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform wavelet-based denoising on an audio signal.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters for the wavelet denoising.
                - wavelet (str): Wavelet type (e.g., 'db1', 'sym5'). Default is 'db1'.
                - wt_mode_wavelet (str): Thresholding mode ('soft' or 'hard'). Default is 'soft'.

        Returns:
            np.ndarray: The denoised audio signal.
        """
        wavelet = kwargs.get("wavelet", self.config.get("wavelet"))
        mode = kwargs.get("wt_mode", self.config.get("wt_mode"))
        threshold_factor = kwargs.get("threshold_factor", self.config.get("threshold_factor"))

        threshold = self.compute_threshold(audio, wavelet, threshold_factor)

        coeffs = pywt.wavedec(audio, wavelet)
        coeffs_denoised = [pywt.threshold(c, threshold, mode=mode) for c in coeffs]

        denoised_audio = pywt.waverec(coeffs_denoised, wavelet)

        return denoised_audio

    def compute_threshold(self, audio, wavelet, threshold_factor):
        """
        Compute the universal threshold for wavelet-based denoising.

        Args:
            audio (np.ndarray): Input audio signal.
            wavelet (str): Wavelet type (e.g., 'db1', 'sym5', etc.) used for decomposition.
            threshold_factor (float): Threshold factor for the universal threshold.

        Returns:
            float: The calculated threshold value.

        Notes:
            - This function uses the universal threshold formula:
              Threshold = sigma * sqrt(2 * log(n)),
              where sigma is the noise standard deviation estimated from the detail coefficients,
              and n is the length of the audio signal.
            - The estimation of sigma uses the robust formula:
              sigma = median(|coeffs[-1]|) / 0.6745,
              which is based on the assumption of Gaussian white noise.
            - The universal threshold is particularly effective for denoising signals corrupted by
              additive white Gaussian noise.

        """
        coeffs = pywt.wavedec(audio, wavelet)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(audio))) * threshold_factor
        return threshold
