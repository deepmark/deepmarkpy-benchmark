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
import subprocess
import tempfile
from typing import List

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int
    bitrate: int = 16
    framesize: float = 20


def encode_with_opus(input_wav: str, output_opus: str, bitrate: int, framesize: float) -> None:
    """Encode a WAV file to Opus with the given bitrate and frame size."""
    framesize_str = str(int(framesize)) if framesize >= 5 else "2.5"
    cmd = [
        "opusenc",
        "--bitrate", str(int(bitrate)),
        "--framesize", framesize_str,
        "--quiet",
        input_wav,
        output_opus,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"opusenc failed: {result.stderr}")


def decode_with_opus(input_opus: str, output_wav: str) -> None:
    """Decode an Opus file back to WAV. No --rate flag: let Opus decide."""
    cmd = [
        "opusdec",
        "--quiet",
        input_opus,
        output_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"opusdec failed: {result.stderr}")


def process_opus_codec(
    audio: np.ndarray, sampling_rate: int, bitrate: int, framesize: float,
):
    """Pure Opus encode -> decode round trip.

    Opus handles all internal resampling. Returns (decoded_audio, output_sr)
    where output_sr is whatever rate opusdec chose to output at.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
        input_wav = f_in.name
    with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as f_opus:
        opus_file = f_opus.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        output_wav = f_out.name

    try:
        sf.write(input_wav, audio, sampling_rate)
        encode_with_opus(input_wav, opus_file, bitrate, framesize)
        decode_with_opus(opus_file, output_wav)

        decoded_audio, output_sr = sf.read(output_wav)

        logger.info(
            f"Opus codec pass complete: bitrate={bitrate}k, framesize={framesize}ms, "
            f"input_sr={sampling_rate}Hz, output_sr={output_sr}Hz"
        )
        return decoded_audio.astype(np.float32), int(output_sr)
    finally:
        for f in [input_wav, opus_file, output_wav]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


@app.post("/attack")
async def attack(request: AttackRequest):
    """Run a pure Opus encode/decode round trip on the supplied audio."""
    try:
        audio = np.array(request.audio, dtype=np.float32)
        result, output_sr = process_opus_codec(
            audio=audio,
            sampling_rate=request.sampling_rate,
            bitrate=request.bitrate,
            framesize=request.framesize,
        )
        return {
            "audio": result.astype(np.float32).tolist(),
            "sampling_rate": output_sr,
        }
    except Exception as e:
        logger.error(f"Attack failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "opus_codec"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "10023"))
    uvicorn.run(app, host=host, port=port)
