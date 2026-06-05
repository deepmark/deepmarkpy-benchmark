"""
Pure Opus Codec Attack - FastAPI Server

A lightweight service that only runs opusenc/opusdec -- no WebRTC noise
suppression, no tc netem, no UDP packet simulation. The client sends
audio at the model's sampling rate; ``opusenc`` silently remaps to its
nearest internal rate (8/12/16/24/48 kHz) for compression, and
``opusdec`` decodes at that same internal rate. Resampling back to the
model's sampling rate happens on the client side via librosa so the SR
conversion stays under our control instead of being delegated to opusdec.
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


OPUS_INTERNAL_RATES = (8000, 12000, 16000, 24000, 48000)


def _opus_internal_rate(input_sr: int) -> int:
    """Return the Opus internal rate that ``opusenc`` will pick for ``input_sr``.

    Opus only supports those five rates internally, so any other input
    is silently remapped to the nearest supported one. We mirror that
    choice here so the server can ask ``opusdec`` to output at the same
    rate the codec actually ran at, instead of letting opusdec resample
    back to the originally-tagged input SR.
    """
    return min(OPUS_INTERNAL_RATES, key=lambda r: (abs(r - input_sr), -r))


def decode_with_opus(input_opus: str, output_wav: str, decoded_sr: int) -> None:
    """Decode an Opus file at the codec's internal sample rate.

    ``decoded_sr`` should be the Opus internal rate the encoder used
    (one of 8/12/16/24/48 kHz). Asking opusdec for that rate lets it
    skip the WAV-header-driven resample to the originally-tagged input
    SR, so the librosa resample on the client side is the only SR
    conversion that happens after compression.
    """
    cmd = [
        "opusdec",
        "--rate", str(decoded_sr),
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
    """Pure Opus encode -> decode round trip (no network, no preprocessing).

    Returns ``(decoded_audio, decoded_sr)``. ``decoded_sr`` is the Opus
    internal rate the codec actually ran at (one of 8/12/16/24/48 kHz),
    not the input rate. Resampling back to the model's SR is left to
    the client.
    """
    decoded_sr = _opus_internal_rate(sampling_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
        input_wav = f_in.name
    with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as f_opus:
        opus_file = f_opus.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        output_wav = f_out.name

    try:
        sf.write(input_wav, audio, sampling_rate)
        encode_with_opus(input_wav, opus_file, bitrate, framesize)
        decode_with_opus(opus_file, output_wav, decoded_sr)

        decoded_audio, file_sr = sf.read(output_wav)

        logger.info(
            f"Opus codec pass complete: bitrate={bitrate}k, framesize={framesize}ms, "
            f"input_sr={sampling_rate}Hz, decoded_sr={file_sr}Hz "
            f"(opus internal rate)"
        )
        return decoded_audio.astype(np.float32), int(file_sr)
    finally:
        for f in [input_wav, opus_file, output_wav]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


@app.post("/attack")
async def attack(request: AttackRequest):
    """Run a pure Opus encode/decode round trip on the supplied audio.

    Response includes ``sampling_rate`` so the client knows what rate the
    decoded audio comes back at (always 48 kHz, Opus's native output).
    """
    try:
        audio = np.array(request.audio, dtype=np.float32)
        result, decoded_sr = process_opus_codec(
            audio=audio,
            sampling_rate=request.sampling_rate,
            bitrate=request.bitrate,
            framesize=request.framesize,
        )
        return {
            "audio": result.astype(np.float32).tolist(),
            "sampling_rate": decoded_sr,
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
