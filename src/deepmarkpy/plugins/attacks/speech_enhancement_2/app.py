import logging
import sys
from typing import List
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from inference import SpeechEnhancement2Engine
from deepmarkpy.core.inference import MAX_AUDIO_B64_CHARS
from deepmarkpy.core.wire import decode_audio, encode_audio
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
    audio: str = Field(..., max_length=MAX_AUDIO_B64_CHARS)
    sampling_rate: int
    model_name: str


@app.post("/attack")
async def attack(request: AttackRequest):
    try:
        audio_cv = engine.apply(
            decode_audio(request.audio), request.sampling_rate, model_name=request.model_name
        )
        return JSONResponse({"audio": encode_audio(audio_cv)})

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e), "audio": None})
