import logging
import os
import sys
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference import SpeechEnhancement1Engine
from deepmarkpy.core.inference import MAX_AUDIO_SAMPLES
from deepmarkpy.utils.utils import load_config

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

engine = SpeechEnhancement1Engine(config)


class AttackRequest(BaseModel):
    audio: List[float] = Field(..., max_length=MAX_AUDIO_SAMPLES)
    sampling_rate: int
    noise_strength: float


@app.post("/attack")
async def attack(request: AttackRequest):
    audio = engine.apply(
        request.audio, request.sampling_rate, noise_strength=request.noise_strength
    )

    return {"audio": audio.tolist()}
