"""
Pure Opus Codec Attack - FastAPI Server

A lightweight service that only runs opusenc/opusdec -- no WebRTC noise
suppression, no tc netem, no UDP packet simulation. The client sends
audio at the model's sampling rate. Opus handles everything internally:
opusenc remaps to its supported rate for compression, opusdec decodes
and outputs. The client checks the output sampling rate and resamples
back to the model's rate with librosa if needed.
"""

import logging
import os
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from deepmarkpy.core.inference import MAX_AUDIO_B64_CHARS
from deepmarkpy.core.wire import decode_audio, encode_audio

from inference import OpusCodecEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

engine = OpusCodecEngine({})


class AttackRequest(BaseModel):
    audio: str = Field(..., max_length=MAX_AUDIO_B64_CHARS)
    sampling_rate: int
    bitrate: int = 16
    framesize: float = 20


@app.post("/attack")
async def attack(request: AttackRequest):
    """Run a pure Opus encode/decode round trip on the supplied audio."""
    try:
        result, output_sr = engine.apply(
            decode_audio(request.audio),
            request.sampling_rate,
            bitrate=request.bitrate,
            framesize=request.framesize,
        )
        return JSONResponse({
            "audio": encode_audio(result.astype(np.float32)),
            "sampling_rate": output_sr,
        })
    except Exception as e:
        logger.error(f"Attack failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy", "service": "opus_codec"})


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "10023"))
    uvicorn.run(app, host=host, port=port)
