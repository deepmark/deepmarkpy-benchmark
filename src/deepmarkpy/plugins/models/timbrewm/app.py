import logging
import os
import sys
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from deepmarkpy.core.inference import MAX_AUDIO_B64_CHARS, MAX_WATERMARK_BITS
from deepmarkpy.core.wire import decode_audio, encode_audio
from deepmarkpy.utils.utils import load_config

# Configure more detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from inference import TimbreWMEngine

app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

engine = TimbreWMEngine(config)

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
    try:
        audio_arr = decode_audio(request.audio)
        logger.debug(f"Received embed request. Audio length: {len(audio_arr)}, Watermark length: {len(request.watermark_data)}")
        watermarked_audio = engine.embed(
            audio_arr, request.watermark_data, request.sampling_rate
        )
        logger.debug("Returning watermarked audio")
        return JSONResponse({"watermarked_audio": encode_audio(watermarked_audio)})
    except Exception as e:
        logger.error(f"Error in embed endpoint: {str(e)}", exc_info=True)
        raise


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detect a watermark from an audio file."""
    try:
        audio_arr = decode_audio(request.audio)
        logger.debug(f"Received detect request. Audio length: {len(audio_arr)}")
        message = engine.detect(audio_arr, request.sampling_rate)
        logger.debug("Returning watermark")
        return JSONResponse({"watermark": message if message is None else message.tolist()})
    except Exception as e:
        logger.error(f"Error in detect endpoint: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Use the default as a fallback if APP_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 9001))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    # Add timeout settings to handle large payloads
    uvicorn.run(app, host=host, port=app_port, timeout_keep_alive=120)
