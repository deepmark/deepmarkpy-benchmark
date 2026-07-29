import logging
import os
import sys
from typing import List

import numpy as np
import silentcipher
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from deepmarkpy.utils.utils import load_config, resample_audio

logger = logging.getLogger(__name__)

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
            
model = silentcipher.get_model(model_type='44.1k', device=device)

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

class EmbedRequest(BaseModel):
    audio: List[float]
    watermark_data: List[int]
    sampling_rate: int

class DetectRequest(BaseModel):
    audio: List[float]
    sampling_rate: int

@app.post("/embed")
async def embed(request: EmbedRequest):
    """Embed a watermark in an audio file."""
    audio = np.array(request.audio)
    watermark_data = np.array(request.watermark_data)
    sampling_rate = request.sampling_rate
    if sampling_rate != config["sampling_rate"]:
        audio = resample_audio(request.audio, sampling_rate, config["sampling_rate"])

    watermark_data = np.split(watermark_data, len(watermark_data) // 8)
    watermark_data = [int("".join(map(str, arr)), 2) for arr in watermark_data]
    watermarked_audio, _ = model.encode_wav(audio, config["sampling_rate"], watermark_data, calc_sdr=False)

    if sampling_rate != config["sampling_rate"]:
        watermarked_audio = resample_audio(watermarked_audio, config["sampling_rate"], sampling_rate)

    return {"watermarked_audio": watermarked_audio.tolist()}


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detect a watermark from an audio file."""
    audio = np.array(request.audio)
    sampling_rate = request.sampling_rate

    if sampling_rate != config["sampling_rate"]:
        audio = resample_audio(request.audio, sampling_rate, config["sampling_rate"])

    message = model.decode_wav(audio, config["sampling_rate"], phase_shift_decoding=config["phase_shift_decoding"])
    try:
        message = message['messages'][0]
        message = [np.array(list(f"{val:08b}"), dtype=np.int32) for val in message]
        message = np.concatenate(message)
        message = message.tolist()
    except:  # noqa: E722
        message = None
    
    return {"watermark": message}

if __name__ == "__main__":
    # Use the default as a fallback if APP_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 7001))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    uvicorn.run(app, host=host, port=app_port)