import importlib.util
import logging
import os
import sys
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from deepmarkpy.utils.utils import load_config

# The upstream BigVGAN clone ships its own inference.py, and the service's
# working directory (/app/BigVGAN) shadows PYTHONPATH for a bare
# `import inference` — load this plugin's inference.py by explicit path.
_spec = importlib.util.spec_from_file_location(
    "plugin_inference",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference.py"),
)
_inference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inference)
Engine = _inference.Engine

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

engine = Engine(config)


class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int


@app.post("/attack")
async def attack(request: AttackRequest):
    audio = engine.apply(request.audio, request.sampling_rate)

    return {"audio": audio.tolist()}


if __name__ == "__main__":
    # Use the default as a fallback if APP_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 10004))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    uvicorn.run(app, host=host, port=app_port)
