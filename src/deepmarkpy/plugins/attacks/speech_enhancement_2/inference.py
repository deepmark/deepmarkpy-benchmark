"""ClearVoice speech-enhancement attack inference, HTTP-free.

ClearVoice is constructed per request inside ``apply``; construction
placement is part of the service's observable behavior. The config-driven
noise term is unseeded, so the service is stochastic. app.py converts
exceptions from ``apply`` into the error-in-200 response (intentionally
preserved shape).
"""

import logging
import os
import tempfile

import numpy as np
import soundfile as sf
from clearvoice import ClearVoice

from deepmarkpy.utils.utils import resample_audio

from deepmarkpy.core.inference import BaseAttackEngine

logger = logging.getLogger(__name__)


# The ClearVoice speech_enhancement networks, from its own dispatch table.
# Anything else either does not exist or belongs to a different task.
SPEECH_ENHANCEMENT_MODELS = frozenset({
    "MossFormerGAN_SE_16K",
    "MossFormer2_SE_48K",
    "FRCRN_SE_16K",
})


class SpeechEnhancement2Engine(BaseAttackEngine):
    """ClearVoice speech-enhancement attack.

    ``__init__`` loads nothing — ClearVoice is constructed per request
    inside ``apply``.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Store config; no weights load here by design (per-request model)."""
        self.config = config

    def apply(self, audio: list, sampling_rate: int, **params) -> np.ndarray:
        """Enhance ``audio`` with the request's ClearVoice ``model_name``.

        Raises on processing failure; app.py builds the error-in-200 response.
        """
        model_name = params["model_name"]
        if model_name not in SPEECH_ENHANCEMENT_MODELS:
            # ClearVoice interpolates this straight into a config path and only
            # checks the name after opening the file, so an unchecked value
            # reads arbitrary YAML off the container and returns what it finds
            # through the error response.
            raise ValueError(
                f"Unknown speech-enhancement model {model_name!r}; "
                f"expected one of {sorted(SPEECH_ENHANCEMENT_MODELS)}"
            )
        audio_arr = np.array(audio)

        # Create a temporary file with .wav extension (not .mp3)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Add noise before saving
            noise_strength = self.config.get("noise_strength_se2", 0.0)
            if noise_strength > 0:
                noisy = audio_arr + noise_strength * np.random.normal(0, 1, size=(len(audio_arr)))
                audio_arr = noisy

            # Resample to 16kHz for the model
            target_sr = 16000
            if sampling_rate != target_sr:
                audio_arr = resample_audio(audio_arr, sampling_rate, target_sr)

            # Save the audio array to the temporary WAV file

            sf.write(tmp_path, audio_arr, target_sr)

            # Pass the temporary file path to ClearVoice
            logger.info(f"Processing with ClearVoice model: {model_name}")
            myClearVoice = ClearVoice(task='speech_enhancement', model_names=[model_name])
            audio_cv = myClearVoice(input_path=tmp_path, online_write=False)


            # ClearVoice returns a dict {filename: audio_array} when online_write=False
            if isinstance(audio_cv, dict):
                audio_cv = list(audio_cv.values())[0]
            audio_cv = np.array(audio_cv).squeeze()
            if sampling_rate != target_sr:
                audio_cv = resample_audio(audio_cv, target_sr, sampling_rate)
            return audio_cv

        finally:
            # Delete the temporary file
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    logger.info(f"Deleted temporary file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {str(e)}")

