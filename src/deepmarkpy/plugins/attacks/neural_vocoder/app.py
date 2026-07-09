import logging
import os
import sys
from typing import List

import numpy as np
import torch
import uvicorn
from big_vgan import BigVGAN
from fastapi import FastAPI
from pydantic import BaseModel

from app_utils.utils import load_config

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = BigVGAN(config["model_name"], device)


class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int


@app.post("/attack")
async def attack(request: AttackRequest):
    sampling_rate = request.sampling_rate
    audio = np.array(request.audio)

    audio = model.inference(audio, sampling_rate)

    return {"audio": audio.tolist()}


if __name__ == "__main__":
    # Use the default as a fallback if APP_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 10004))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    uvicorn.run(app, host=host, port=app_port)
