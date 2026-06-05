"""
Pure Opus Codec Attack - FastAPI Server

A lightweight service that only runs opusenc/opusdec -- no WebRTC noise
suppression, no tc netem, no UDP packet simulation. The client sends
audio at the model's sampling rate; ``opusenc`` silently remaps to its
nearest internal rate (8/12/16/24/48 kHz) for compression, and
``opusdec --rate <sampling_rate>`` decodes back to the original rate so
the caller never sees a different sampling rate on the way out.
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


def decode_with_opus(input_opus: str, output_wav: str, sampling_rate: int) -> None:
    """Decode an Opus file back to WAV at the given sample rate."""
    cmd = [
        "opusdec",
        "--rate", str(sampling_rate),
        "--quiet",
        input_opus,
        output_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"opusdec failed: {result.stderr}")


def process_opus_codec(
    audio: np.ndarray, sampling_rate: int, bitrate: int, framesize: float,
) -> np.ndarray:
    """Pure Opus encode -> decode round trip (no network, no preprocessing)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
        input_wav = f_in.name
    with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as f_opus:
        opus_file = f_opus.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        output_wav = f_out.name

    try:
        sf.write(input_wav, audio, sampling_rate)
        encode_with_opus(input_wav, opus_file, bitrate, framesize)
        decode_with_opus(opus_file, output_wav, sampling_rate)

        decoded_audio, _ = sf.read(output_wav)

        if len(decoded_audio) > len(audio):
            decoded_audio = decoded_audio[:len(audio)]
        elif len(decoded_audio) < len(audio):
            decoded_audio = np.pad(decoded_audio, (0, len(audio) - len(decoded_audio)))

        logger.info(
            f"Opus codec pass complete: bitrate={bitrate}k, framesize={framesize}ms, "
            f"sr={sampling_rate}Hz"
        )
        return decoded_audio.astype(np.float32)
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
        result = process_opus_codec(
            audio=audio,
            sampling_rate=request.sampling_rate,
            bitrate=request.bitrate,
            framesize=request.framesize,
        )
        return {"audio": result.astype(np.float32).tolist()}
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
