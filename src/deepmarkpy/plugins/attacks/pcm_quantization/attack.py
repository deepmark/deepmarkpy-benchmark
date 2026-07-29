import numpy as np
from deepmarkpy.core.base_attack import BaseAttack

class PCMQuantizationAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Pulse Code Modulation (PCM) style quantization to an audio signal.

            This method simulates lossy digitization by mapping the continuous audio 
            values into a finite set of discrete levels. The number of levels is 
            determined by the specified quantization bit depth. 
            Args:
                audio (np.ndarray): Input audio signal.
                **kwargs: Additional parameters for the PCM attack:
                    - pcm_quantization (int): Number of quantization levels (e.g., 256 for 8-bit).
                    Defaults to the class configuration if not provided.

            Returns:
                np.ndarray: The quantized audio signal with reduced resolution.

            Raises:
                ValueError: If `quantization_bit` is not provided and missing from the configuration.
        """

        sr = kwargs.get("sampling_rate", None)
        pcm = kwargs.get("bit_depth", self.config.get("bit_depth"))
        # Convert to specified PCM bit depth and back (simulates quantization)
        if pcm == 8:
            # 8-bit signed: -128 to 127
            audio_int = np.clip(audio * 127.0, -128, 127).astype(np.int8)
            audio = audio_int.astype(np.float32) / 127.0
        elif pcm == 16:   
            # 16-bit signed: -32768 to 32767
            audio_int = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
            audio = audio_int.astype(np.float32) / 32767.0
        elif pcm == 24:
            # 24-bit signed: -8388608 to 8388607
            audio_int = np.clip(audio * 8388607.0, -8388608, 8388607).astype(np.int32)
            audio = audio_int.astype(np.float32) / 8388607.0
        else:
            raise ValueError(f"Unsupported PCM bit depth: {pcm}")
        return audio
    
