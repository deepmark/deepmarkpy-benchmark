import numpy as np

from deepmarkpy.core.base_attack import BaseAttack

class QuantizationAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a quantization attack on an audio signal. 
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the quantization attack:
                - bit_quantization (tuple): Number of quantization levels (e.g., 256 for 8-bit).
        Returns:
            np.ndarray: The processed quantized audio signal.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """
        quantization_bit = kwargs.get(
            "bit_quantization", self.config.get("bit_quantization")
        )

        # Normalize to [0, 1]
        min_val = np.min(audio)
        max_val = np.max(audio)
        normalized = (audio - min_val) / (max_val - min_val + 1e-8)  

        # Quantize to levels
        quantized = np.round(normalized * (quantization_bit - 1))

        # Rescale to original range
        rescaled = (quantized / (quantization_bit - 1)) * (max_val - min_val) + min_val

        return rescaled