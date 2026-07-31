import logging
import sys
from typing import List
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from inference import SpeechEnhancement2Engine
from deepmarkpy.utils.utils import load_config

logger = logging.getLogger(__name__)
app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

engine = SpeechEnhancement2Engine(config)


class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int
    model_name: str


@app.post("/attack")
async def attack(request: AttackRequest):
    try:
        audio_cv = engine.apply(
            request.audio, request.sampling_rate, model_name=request.model_name
        )
        return {"audio": audio_cv.tolist()}

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        return {"error": str(e), "audio": None}
