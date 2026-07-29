import numpy as np
from scipy.signal import stft, istft
from deepmarkpy.core.base_attack import BaseAttack

class STFTQuantizationAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply STFT quantization to the audio signal.

            This function performs the following steps:
            1. Computes the Short-Time Fourier Transform (STFT) of the input signal.
            2. Quantizes the magnitude of the STFT coefficients to a specified number of levels,
            while preserving the phase information.
            3. Reconstructs the time-domain audio signal using the inverse STFT (ISTFT). 

            Args:
                audio (np.ndarray): Input audio signal.
                **kwargs: Additional parameters for the PCM attack:
                    - n_fft (int): Number of FFT points (window size). Default is 1024.
                    - hop_length (int): Hop length between windows. Default is 512.
                    - quantization_levels (int): Number of quantization levels for the STFT magnitude. Default is 256.

            Returns:
                np.ndarray: Time-domain audio signal after STFT magnitude quantization.
        """

        sr = kwargs.get("sampling_rate", None)
        n_fft = kwargs.get("n_fft", self.config.get("n_fft"))
        hop_length = kwargs.get("hop_length", self.config.get("hop_length"))
        quantization_levels = kwargs.get("quantization_levels", self.config.get("quantization_levels"))

        # Compute STFT
        _, _, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, window='hann')

        # Quantize magnitude while preserving phase
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        mag_min, mag_max = magnitude.min(), magnitude.max()
        mag_norm = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)
        mag_quant = np.round(mag_norm * (quantization_levels - 1)) / (quantization_levels - 1)
        mag_quant_rescaled = mag_quant * (mag_max - mag_min) + mag_min

        # Reconstruct complex STFT
        Zxx_quantized = mag_quant_rescaled * np.exp(1j * phase)

        _, signal_quantized = istft(Zxx_quantized, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, window='hann')

        return signal_quantized
    
