import logging
import os
import sys
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from inference import DiffusionEngine
from deepmarkpy.utils.utils import load_config

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

engine = DiffusionEngine(config)


class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int
    diffusion_steps: int


@app.post("/attack")
async def attack(request: AttackRequest):
    audio = engine.apply(
        request.audio, request.sampling_rate, diffusion_steps=request.diffusion_steps
    )

    return {"audio": audio.tolist()}


if __name__ == "__main__":
    # Use the default as a fallback if APP_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 10002))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    uvicorn.run(app, host=host, port=app_port)
