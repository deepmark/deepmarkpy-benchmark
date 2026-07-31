import logging
import os
import sys
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference import SilentCipherEngine
from deepmarkpy.core.inference import MAX_AUDIO_SAMPLES, MAX_WATERMARK_BITS
from deepmarkpy.utils.utils import load_config

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

engine = SilentCipherEngine(config)

class EmbedRequest(BaseModel):
    audio: List[float] = Field(..., max_length=MAX_AUDIO_SAMPLES)
    watermark_data: List[int] = Field(..., max_length=MAX_WATERMARK_BITS)
    sampling_rate: int

class DetectRequest(BaseModel):
    audio: List[float] = Field(..., max_length=MAX_AUDIO_SAMPLES)
    sampling_rate: int

@app.post("/embed")
async def embed(request: EmbedRequest):
    """Embed a watermark in an audio file."""
    watermarked_audio = engine.embed(
        request.audio, request.watermark_data, request.sampling_rate
    )
    return {"watermarked_audio": watermarked_audio.tolist()}


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detect a watermark from an audio file."""
    # SilentCipherEngine.detect returns a JSON-able bit list, or None on decode failure.
    message = engine.detect(request.audio, request.sampling_rate)
    return {"watermark": message}

if __name__ == "__main__":
    # Use the default as a fallback if APP_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 7001))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    uvicorn.run(app, host=host, port=app_port)
