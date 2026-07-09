import logging
import os
import sys
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from speech_brain import SpeechBrain

from deepmarkpy.utils.utils import load_config

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

model = SpeechBrain(config["type"])


class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int
    noise_strength: float


@app.post("/attack")
async def attack(request: AttackRequest):
    sampling_rate = request.sampling_rate
    audio = np.array(request.audio)
    noise_strength = request.noise_strength

    audio = model.inference(audio, sampling_rate, noise_strength)

    return {"audio": audio.tolist()}
