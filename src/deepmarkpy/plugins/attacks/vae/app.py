import logging
import os
import sys
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from vae import VAE

from deepmarkpy.utils.utils import load_config, resample_audio

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = VAE(config["model_name"], device)


class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int


@app.post("/attack")
async def attack(request: AttackRequest):
    """
    Applies a VAE-based watermarking attack on the given audio signal.

    Args:
        audio (np.ndarray): The input audio signal.
        **kwargs: Additional parameters.
            - sampling_rate (int): The original sampling rate of the audio (required).

    Returns:
        np.ndarray: The attacked audio signal.
    """
    sampling_rate = request.sampling_rate
    audio = np.array(request.audio)
    audio = np.squeeze(audio)

    block_size = 2048
    original_length = len(audio)
    new_length = (original_length // block_size) * block_size
    audio = audio[:new_length]

    audio = resample_audio(audio, sampling_rate, target_sr=48000)

    audio = model.inference(audio)

    audio = resample_audio(audio, 48000, sampling_rate)

    return {"audio": audio.tolist()}


if __name__ == "__main__":
    # Use the default as a fallback if APP_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 10001))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    uvicorn.run(app, host=host, port=app_port)
