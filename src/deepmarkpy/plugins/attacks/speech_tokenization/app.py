import logging
import os
import sys
import traceback
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from inference import SpeechTokenizationEngine

from deepmarkpy.core.inference import MAX_AUDIO_B64_CHARS
from deepmarkpy.core.wire import decode_audio, encode_audio
from deepmarkpy.utils.utils import load_config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

engine = SpeechTokenizationEngine(config, device)


class AttackRequest(BaseModel):
    audio: str = Field(..., max_length=MAX_AUDIO_B64_CHARS)
    sampling_rate: int


@app.post("/attack")
async def attack(request: AttackRequest):
    try:
        sampling_rate = request.sampling_rate
        audio_arr = decode_audio(request.audio)
        logger.info(f"Received request: sampling_rate={sampling_rate}, audio_length={len(audio_arr)}")
        audio = audio_arr

        # The engine resamples to the model's 16 kHz and back, so the request
        # rate goes straight through. Resampling here as well left the engine
        # converting 16 kHz to 16 kHz, which resample_audio short-circuits.
        logger.info("Starting model inference...")
        audio = engine.apply(audio, sampling_rate)
        logger.info(f"Inference complete. Output length: {len(audio)}")

        return JSONResponse({"audio": encode_audio(audio)})
    except Exception as e:
        logger.error(f"Error in /attack: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    app_port = int(os.getenv("APP_PORT") or os.getenv("SPEECH_TOKENIZATION_PORT", "10003"))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{app_port}")
    uvicorn.run(app, host=host, port=app_port)
