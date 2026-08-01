import logging
import os
import sys
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference import AudioSealEngine
from deepmarkpy.core.inference import MAX_AUDIO_B64_CHARS, MAX_WATERMARK_BITS
from deepmarkpy.core.wire import decode_audio, encode_audio
from deepmarkpy.utils.utils import load_config

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

engine = AudioSealEngine(config)

class EmbedRequest(BaseModel):
    audio: str = Field(..., max_length=MAX_AUDIO_B64_CHARS)
    watermark_data: List[int] = Field(..., max_length=MAX_WATERMARK_BITS)
    sampling_rate: int

class DetectRequest(BaseModel):
    audio: str = Field(..., max_length=MAX_AUDIO_B64_CHARS)
    sampling_rate: int


@app.post("/embed")
async def embed(request: EmbedRequest):
    """Embed a watermark in an audio file."""
    watermarked_audio = engine.embed(
        decode_audio(request.audio), request.watermark_data, request.sampling_rate
    )
    return JSONResponse({"watermarked_audio": encode_audio(watermarked_audio)})


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detect a watermark from an audio file."""
    message, confidence = engine.detect(decode_audio(request.audio), request.sampling_rate)
    if isinstance(message, list):
        # Empty detection result (audio too short or detector failure).
        return JSONResponse({"watermark": message, "confidence": confidence})
    return JSONResponse({"watermark": message if message is None else message.tolist(),
                         "confidence": float(confidence)})

if __name__ == "__main__":
    # Use the default as a fallback if APP_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 5001))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    uvicorn.run(app, host=host, port=app_port)
