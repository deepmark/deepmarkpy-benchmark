import logging
import os
import sys
import traceback
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from xcodec import XCodec

from utils.utils import load_config, resample_audio

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

model = XCodec(config["model_name"], device)


class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int


@app.post("/attack")
async def attack(request: AttackRequest):
    try:
        sampling_rate = request.sampling_rate
        logger.info(f"Received request: sampling_rate={sampling_rate}, audio_length={len(request.audio)}")
        audio = np.array(request.audio)

        target_sr = 16000
        if sampling_rate != target_sr:
            logger.info(f"Resampling from {sampling_rate} to {target_sr}")
            audio = resample_audio(audio, sampling_rate, target_sr)
            logger.info(f"Resampled audio length: {len(audio)}")

        logger.info("Starting model inference...")
        audio = model.inference(audio, target_sr)
        logger.info(f"Inference complete. Output length: {len(audio)}")

        if sampling_rate != target_sr:
            logger.info(f"Resampling output from {target_sr} to {sampling_rate}")
            audio = resample_audio(audio, target_sr, sampling_rate)
            logger.info(f"Final output length: {len(audio)}")

        return {"audio": audio.tolist()}
    except Exception as e:
        logger.error(f"Error in /attack: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    app_port = int(os.getenv("APP_PORT") or os.getenv("SPEECH_TOKENIZATION_PORT", "10003"))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{app_port}")
    uvicorn.run(app, host=host, port=app_port)
