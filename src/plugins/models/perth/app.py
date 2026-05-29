import logging
import os
import sys
from typing import List
import torch

import numpy as np
from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

from fastapi import FastAPI
from pydantic import BaseModel

from utils.utils import load_config, resample_audio

logger = logging.getLogger(__name__)

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
            
model =  PerthImplicitWatermarker()

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
    watermarked_audio = model.apply_watermark(audio,watermark=None,sample_rate=config["sampling_rate"])

    if sampling_rate != config["sampling_rate"]:
        watermarked_audio = resample_audio(watermarked_audio, config["sampling_rate"], sampling_rate)

    # Sanitize to ensure JSON serialization works
    watermarked_audio = np.nan_to_num(watermarked_audio, nan=0.0, posinf=0.0, neginf=0.0)

    return {"watermarked_audio": watermarked_audio.tolist()}


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detect a watermark from an audio file."""
    audio = np.array(request.audio)
    sampling_rate = request.sampling_rate

    if sampling_rate != config["sampling_rate"]:
        audio = resample_audio(request.audio, sampling_rate, config["sampling_rate"])

    message = model.get_watermark(audio, config["sampling_rate"], round=True)
    if isinstance(message, np.ndarray) and message.ndim == 0:
        message = message.item() # Converts a 0-d NumPy array to its scalar equivalent

    # Sanitize to ensure JSON serialization works
    if isinstance(message, np.ndarray):
        message = np.nan_to_num(message, nan=0.0, posinf=0.0, neginf=0.0)
        message = message.tolist()
    elif isinstance(message, float):
        import math
        if math.isnan(message) or math.isinf(message):
            message = 0.0

    return {"watermark": message}