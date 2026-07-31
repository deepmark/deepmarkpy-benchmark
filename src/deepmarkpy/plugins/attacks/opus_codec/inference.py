"""Opus encode/decode round-trip attack inference, HTTP-free.

``Engine.apply`` returns ``(audio, output_sr)``: the response carries the
decoder's output rate and the client resamples back.
"""

import logging
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from deepmarkpy.core.inference import BaseAttackEngine

# Named "app" so the INFO line in process_opus_codec keeps its existing
# format in the service log output ("INFO:app:...").
logger = logging.getLogger("app")


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


class OpusCodecEngine(BaseAttackEngine):
    """Opus encode/decode round-trip attack (external opusenc/opusdec tools).

    Nothing loads at construction — the codec is a subprocess tool pair.
    """

    def __init__(self, config: dict, device: str | None = None):
        """Store config; no model to load (subprocess-based codec)."""
        self.config = config

    def apply(self, audio: list, sampling_rate: int, **params) -> "tuple[np.ndarray, int]":
        """Run the Opus round trip; returns ``(decoded_audio, output_sr)``.

        ``params`` requires ``bitrate`` and ``framesize``. The output rate
        rides back to the client, which owns the resample-back.
        """
        audio_arr = np.array(audio, dtype=np.float32)
        return process_opus_codec(
            audio=audio_arr,
            sampling_rate=sampling_rate,
            bitrate=params["bitrate"],
            framesize=params["framesize"],
        )


# Stable import alias.
Engine = OpusCodecEngine
