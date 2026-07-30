import logging
import os
import sys
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from inference import Engine
from deepmarkpy.utils.utils import load_config

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

engine = Engine(config)


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
    audio = engine.apply(request.audio, request.sampling_rate)

    return {"audio": audio.tolist()}


if __name__ == "__main__":
    # Use the default as a fallback if APP_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 10001))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    uvicorn.run(app, host=host, port=app_port)
