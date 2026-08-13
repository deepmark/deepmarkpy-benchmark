"""NISQA scoring service — non-intrusive MOS prediction."""

import logging
import os

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from deepmarkpy.core.wire import decode_audio, encode_audio
from deepmarkpy.core.inference import MAX_AUDIO_SAMPLES
from inference import NISQAEngine

logger = logging.getLogger(__name__)

app = FastAPI()

engine = NISQAEngine(weights_path="/app/weights/nisqa.tar")

host = os.environ.get("HOST", "0.0.0.0")
app_port = int(os.environ.get("APP_PORT", "10030"))


class PredictRequest(BaseModel):
    audio: str
    sampling_rate: int


@app.post("/predict")
def predict(request: PredictRequest):
    audio = decode_audio(request.audio)
    if audio is None or len(audio) == 0:
        return {"error": "empty audio", "scores": None}
    if len(audio) > MAX_AUDIO_SAMPLES:
        return {
            "error": f"audio too long ({len(audio)} samples, max {MAX_AUDIO_SAMPLES})",
            "scores": None,
        }

    result = engine.predict(audio, request.sampling_rate)
    if result is None:
        return {"error": "prediction failed", "scores": None}

    return {"error": None, "scores": result}


@app.get("/health")
def health():
    return {"status": "ok", "service": "nisqa"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=host, port=app_port)
