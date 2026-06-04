"""
Pure Opus Codec Attack - Client

Applies only Opus encode/decode at 16 kHz, with no WebRTC noise suppression
and no network emulation. Resamples the input from the benchmark's sampling
rate down to 16 kHz before the Opus pipeline and back up afterwards, so the
codec actually operates on a 16 kHz signal regardless of the benchmark rate.

Talks to its own dedicated Docker service (port OPUS_CODEC_PORT, default
10023) -- a lightweight server that only does opusenc/opusdec, no WebRTC
APM and no tc netem.
"""

import logging
import os

import librosa
import numpy as np
import requests

from core.base_attack import BaseAttack

logger = logging.getLogger(__name__)

# Opus runs at a fixed 16 kHz wideband rate for this attack. This is an
# intrinsic property of the codec stage, not a tunable knob, so it lives
# here as a constant rather than in config.json.
OPUS_WORKING_RATE = 16000


class OpusCodecAttack(BaseAttack):
    """
    Pure Opus codec attack — encode and decode only, no NS, no network.

    Resampling to/from the 16 kHz Opus working rate happens inside the
    attack based on the benchmark's runtime sampling rate; no fixed rate
    is read from config.

    Config parameters:
        - bitrate_opus_codec (int): Opus bitrate in kbps (default: 16)
        - framesize_opus_codec (float): Frame size in ms (default: 20)
    """

    def __init__(self):
        super().__init__()
        host = "localhost"
        port = os.getenv("OPUS_CODEC_PORT", "10023")
        if not port:
            raise ValueError("OPUS_CODEC_PORT must be set for OpusCodecAttack")
        self.endpoint = f"http://{host}:{port}"
        logger.info(f"OpusCodecAttack initialized. Target API: {self.endpoint}")

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        original_sr = kwargs.get("sampling_rate")
        if original_sr is None:
            raise ValueError(
                "'sampling_rate' must be provided (benchmark rate, e.g. 44100)."
            )
        original_sr = int(original_sr)

        bitrate = kwargs.get(
            "bitrate_opus_codec",
            self.config.get("bitrate_opus_codec", 16),
        )
        framesize = kwargs.get(
            "framesize_opus_codec",
            self.config.get("framesize_opus_codec", 20),
        )

        audio = np.asarray(audio).astype(np.float32, copy=False)
        original_len = len(audio)

        # Resample down to the Opus working rate inside the attack.
        if original_sr != OPUS_WORKING_RATE:
            audio_16k = librosa.resample(
                audio, orig_sr=original_sr, target_sr=OPUS_WORKING_RATE
            )
        else:
            audio_16k = audio

        try:
            response = requests.post(
                self.endpoint + "/attack",
                json={
                    "audio": audio_16k.tolist(),
                    "sampling_rate": OPUS_WORKING_RATE,
                    "bitrate": bitrate,
                    "framesize": framesize,
                },
                timeout=120,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to opus_codec service: {e}")
            logger.error(
                "Ensure the container is running with: docker-compose up opus_codec"
            )
            raise RuntimeError(f"Opus codec service unavailable: {e}")

        response_data = response.json()
        if "audio" not in response_data:
            raise KeyError("Missing 'audio' in response from service")

        attacked_16k = np.asarray(response_data["audio"], dtype=np.float32)

        # Resample back up to the benchmark rate inside the attack.
        if original_sr != OPUS_WORKING_RATE:
            attacked = librosa.resample(
                attacked_16k, orig_sr=OPUS_WORKING_RATE, target_sr=original_sr
            )
        else:
            attacked = attacked_16k

        # Lock to the caller's length so resample drift doesn't leak out.
        if len(attacked) > original_len:
            attacked = attacked[:original_len]
        elif len(attacked) < original_len:
            attacked = np.pad(attacked, (0, original_len - len(attacked)))

        logger.info(
            f"OpusCodec attack: sr {original_sr}->{OPUS_WORKING_RATE}->{original_sr}, "
            f"bitrate={bitrate}k, framesize={framesize}ms"
        )
        return attacked.astype(np.float32)
