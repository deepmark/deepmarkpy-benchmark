import logging
import os
import shutil

import torch
from huggingface_hub import hf_hub_download

from deepmarkpy.utils.utils import renormalize_audio

logger = logging.getLogger(__name__)


class VAE():
    def __init__(self, model_name, device):
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
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
            shutil.copy2(downloaded_path, local_model_path)
            model_path = local_model_path
        else:
            model_path = local_model_path

        self.model = torch.jit.load(model_path).eval().to(device)
        self.device = device

    def inference(self, audio):
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
