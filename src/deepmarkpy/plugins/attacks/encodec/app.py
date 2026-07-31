import logging
import os
import sys
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference import EncodecEngine
from deepmarkpy.core.inference import MAX_AUDIO_SAMPLES
from deepmarkpy.utils.utils import load_config

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

engine = EncodecEngine(config)


class AttackRequest(BaseModel):
    audio: List[float] = Field(..., max_length=MAX_AUDIO_SAMPLES)
    sampling_rate: int


@app.post("/attack")
async def attack(request: AttackRequest):
    audio = engine.apply(request.audio, request.sampling_rate)

    return {"audio": audio.tolist()}


if __name__ == "__main__":
    # Use the default as a fallback if APP_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 10007))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    uvicorn.run(app, host=host, port=app_port)
