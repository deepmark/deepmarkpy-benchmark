import logging

import numpy as np

from deepmarkpy.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)


class AdditiveNoiseAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Additive Gaussian noise attack at a fixed absolute amplitude.

        ``noise_level`` is a standard deviation in signal units, independent of
        the input's loudness -- a fixed noise floor. This differs from
        ``GaussianNoiseAttack`` and ``PinkNoiseAttack``, which hold a constant
        SNR instead, and it means this attack's effective SNR moves dB for dB
        with input level. Across an ordinary corpus it can land either side of
        its SNR-parameterized siblings, so which attack is harsher varies by
        file; the effective SNR is logged per call, and recorded per file as
        ``attack_snr_db`` in the results.

        Args:
            audio (np.ndarray): The input audio signal.
            noise_level (float): The standard deviation of the Gaussian noise.

        Returns:
            np.ndarray: The audio signal with noise added.
        """
        noise_level = kwargs.get("noise_level", self.config.get("noise_level"))
        noise = np.random.normal(0, noise_level, audio.shape)

        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms > 0 and noise_level > 0:
            logger.info(
                "additive_noise: effective SNR %.1f dB "
                "(noise_level=%g, signal RMS=%.4f)",
                20 * np.log10(rms / noise_level), noise_level, rms,
            )

        return audio + noise
