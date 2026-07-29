import numpy as np

from deepmarkpy.core.base_attack import BaseAttack


class SameModelAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform multiple watermarking using the same model repeatedly.

        Args:
            audio (np.ndarray): The input audio signal to watermark.
            **kwargs: Additional parameters for the watermarking process.
                - model (BaseModel): BaseModel instance to use.
                - sampling_rate (int): The sampling rate of the audio signal.

        Returns:
            np.ndarray: The watermarked audio signal.
        """
        model = kwargs.get("model", None)
        sampling_rate = kwargs.get("sampling_rate", None)

        if model is None or sampling_rate is None:
            raise ValueError(
                "A model and sampling_rate must be specified for same_model_watermarking."
            )

        return model.embed(
            audio=audio,
            watermark_data=model.generate_watermark(),
            sampling_rate=sampling_rate,
        )
