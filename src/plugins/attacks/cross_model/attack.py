import numpy as np

from core.base_attack import BaseAttack


class CrossModelAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> tuple:
        """
        Perform cross-model watermarking by re-embedding with a different model.

        Args:
            audio (np.ndarray): The input audio signal to watermark.
            **kwargs: Additional parameters for the watermarking process.
                - model (BaseModel): Model currently benchmarked.
                - models (dict): Dictionary of all available models.
                - different_model_name_cross_model (str): Name of the model to use for re-embedding.
                - sampling_rate (int): The sampling rate of the audio signal.

        Returns:
            tuple: (watermarked_audio, watermark_data) - the re-embedded audio and
                   the watermark used by the different model.
        """
        models = kwargs.get("models", None)
        different_model_name = kwargs.get("different_model_name_cross_model", None)
        sampling_rate = kwargs.get("sampling_rate", None)

        if models is None or sampling_rate is None:
            raise ValueError(
                "Both 'models' and 'sampling_rate' must be specified for cross_model_watermarking."
            )

        model = self._get_different_model(different_model_name, models)

        watermark_data_ = model.generate_watermark()

        watermarked_audio = model.embed(
            audio=audio,
            watermark_data=watermark_data_,
            sampling_rate=sampling_rate,
        )
        return watermarked_audio, watermark_data_

    def _get_different_model(self, different_model_name, models):
        """
        Instantiate the specified model from available models.

        Raises:
            ValueError: If the model name is not found in available models.
        """
        if different_model_name and different_model_name in models:
            model_cls = models[different_model_name]['class']
            return model_cls()

        available = list(models.keys())
        raise ValueError(
            f"Model '{different_model_name}' not found. Available models: {available}"
        )