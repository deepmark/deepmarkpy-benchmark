"""RAVE variational-autoencoder reconstruction attack inference, HTTP-free.

Input is truncated to a multiple of 2048 samples and passed through the
model at 48 kHz, so output length generally differs from input length.
Some RAVE exports sample latents internally, making the service stochastic
across calls.
"""

import logging
import os
import shutil

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from deepmarkpy.utils.utils import renormalize_audio, resample_audio

from deepmarkpy.core.inference import BaseAttackEngine

logger = logging.getLogger(__name__)

# Pinned so a rebuild cannot silently pick up a different upstream checkpoint.
WEIGHTS_REVISION = "c25a03d625840c40cd3a48779ed72f2a1947d7b4"


class VAEEngine(BaseAttackEngine):
    """RAVE reconstruction through a 48 kHz round-trip.

    The TorchScript export loads once at construction, downloading into the
    local ``models`` cache on first use.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Load the RAVE export named by ``config['model_name_vae']``."""
        self.config = config
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        model_name = config["model_name_vae"]
        model_path = os.path.join('models', model_name)
        repo_id = "Intelligent-Instruments-Lab/rave-models"
        local_model_dir = "models"
        os.makedirs(local_model_dir, exist_ok=True)

        model_filename = os.path.basename(model_path)
        local_model_path = os.path.join(local_model_dir, model_filename)

        if not os.path.exists(local_model_path):
            logger.info(
                f"Model '{model_filename}' not found. Downloading from Hugging Face..."
            )
            downloaded_path = hf_hub_download(
                repo_id=repo_id, filename=model_filename, revision=WEIGHTS_REVISION
            )
            shutil.copy2(downloaded_path, local_model_path)
            model_path = local_model_path
        else:
            model_path = local_model_path

        self.model = torch.jit.load(model_path).eval().to(device)
        self.device = device

    def apply(self, audio: list, sampling_rate: int, **params) -> np.ndarray:
        """Reconstruct ``audio`` through the RAVE model."""
        audio_arr = np.array(audio)
        audio_arr = np.squeeze(audio_arr)

        block_size = 2048
        original_length = len(audio_arr)
        new_length = (original_length // block_size) * block_size
        audio_arr = audio_arr[:new_length]

        audio_arr = resample_audio(audio_arr, sampling_rate, target_sr=48000)

        audio_arr = self._reconstruct(audio_arr)

        audio_arr = resample_audio(audio_arr, 48000, sampling_rate)

        return audio_arr

    def _reconstruct(self, audio):
        """Peak-normalize, run the TorchScript forward, renormalize."""
        waveform = torch.from_numpy(audio).float()
        waveform = waveform.to(self.device)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform / waveform.abs().max()

        with torch.no_grad():
            reconstructed = self.model.forward(waveform.unsqueeze(0))
        reconstructed = reconstructed.squeeze()

        reconstructed = reconstructed.cpu().numpy()
        return renormalize_audio(audio, reconstructed)

