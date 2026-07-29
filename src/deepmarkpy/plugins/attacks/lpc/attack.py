import numpy as np
from scipy import signal
from deepmarkpy.core.base_attack import BaseAttack

class LPCAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a LPC (linear predictive coding) attack via Burg's method on an audio signal.        

        This function applies Burg's method to estimate coefficients of a linear
        filter on ``audio`` of order ``order``.  Burg's method is an extension to the
        Yule-Walker approach, which are both sometimes referred to as LPC parameter
        estimation by autocorrelation. Then, it synthesizes the audio signal using these
        coefficients.

        It follows the description and implementation approach described in the
        introduction by Marple, and this implementation is taken from the librosa library.
        [#] Larry Marple.
           A New Autoregressive Spectrum Analysis Algorithm.
           IEEE Transactions on Acoustics, Speech, and Signal Processing
           vol 28, no. 4, 1980.


        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the lowpass attack:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - order (int): Order of the linear filter, should be a positive integer.
                - axis (int): Axis along which to compute the coefficients.
        Returns:
            np.ndarray: The processed audio signal with the lpc attack applied.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """

        sampling_rate = kwargs.get("sampling_rate", None)
        order = kwargs.get("order", self.config.get("order"))
        axis = kwargs.get("axis", self.config.get("axis", -1))


        audio = audio.swapaxes(axis, 0)

        dtype = audio.dtype

        shape = list(audio.shape)
        shape[0] = order + 1

        ar_coeffs = np.zeros(tuple(shape), dtype=dtype)
        ar_coeffs[0] = 1

        ar_coeffs_prev = ar_coeffs.copy()

        shape[0] = 1
        reflect_coeff = np.zeros(shape, dtype=dtype)
        den = reflect_coeff.copy()

        dtype_ = den.dtype
        epsilon = np.finfo(dtype_).tiny

        # Call the helper, and swap the results back to the target axis position
        a = np.swapaxes(
            self._lpc(audio, order, ar_coeffs, ar_coeffs_prev, reflect_coeff, den, epsilon), 0, axis
        )
        #synthesize the audio signal using the LPC coefficients
        b = np.hstack([[0], -1 * a[1:]])
        y_hat = signal.lfilter(b, [1], audio)
        return y_hat


    def _lpc(
    self,
    y: np.ndarray,
    order: int,
    ar_coeffs: np.ndarray,
    ar_coeffs_prev: np.ndarray,
    reflect_coeff: np.ndarray,
    den: np.ndarray,
    epsilon: float,
    ) -> np.ndarray:
        """Linear Prediction Coefficients via Burg's method

        This function applies Burg's method to estimate coefficients of a linear
        filter on ``audio`` of order ``order``.  
        """
        fwd_pred_error = y[1:]
        bwd_pred_error = y[:-1]

        den[0] = np.sum(fwd_pred_error**2 + bwd_pred_error**2, axis=0)

        for i in range(order):
            reflect_coeff[0] = np.sum(bwd_pred_error * fwd_pred_error, axis=0)
            reflect_coeff[0] *= -2
            reflect_coeff[0] /= den[0] + epsilon

            ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
            for j in range(1, i + 2):
                ar_coeffs[j] = (
                    ar_coeffs_prev[j] + reflect_coeff[0] * ar_coeffs_prev[i - j + 1]
                )

            fwd_pred_error_tmp = fwd_pred_error
            fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
            bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp

            q = 1.0 - reflect_coeff[0] ** 2
            den[0] = q * den[0] - bwd_pred_error[-1] ** 2 - fwd_pred_error[0] ** 2

            fwd_pred_error = fwd_pred_error[1:]
            bwd_pred_error = bwd_pred_error[:-1]

        return ar_coeffs
