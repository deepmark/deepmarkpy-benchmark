import logging
import os
import sys
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from inference import Engine
from deepmarkpy.utils.utils import load_config


logger = logging.getLogger(__name__)

app = FastAPI()

# Load AWARE models
try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

try:
    logger.info("Loading AWARE models...")
    engine = Engine(config)
    logger.info("AWARE models loaded successfully")
except Exception as e:
    logger.critical(f"Failed to load AWARE models: {e}. Application cannot start.")
    import traceback
    traceback.print_exc()
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
    """Embed a watermark in an audio file using AWARE."""
    try:
        watermarked_audio = engine.embed(
            request.audio, request.watermark_data, request.sampling_rate
        )
    except Exception as e:
        logger.error(f"Error embedding watermark: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    return {
        "watermarked_audio": watermarked_audio.tolist(),
    }


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detect a watermark from an audio file using AWARE."""
    try:
        detected_watermark, confidence = engine.detect(
            request.audio, request.sampling_rate
        )
    except Exception as e:
        logger.error(f"Error detecting watermark: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    return {
        "watermark": detected_watermark.tolist() if detected_watermark is not None else None,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    # Use the default as a fallback if APP_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 9004))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting AWARE server on {host}:{app_port}")
    uvicorn.run(app, host=host, port=app_port)
