"""
Pure Opus Codec Attack - Client

Applies only Opus encode/decode, with no WebRTC noise suppression and no
network emulation. Resamples the input from the benchmark's sampling rate
down to the configured Opus working rate before the codec pipeline and
back up afterwards, so the codec actually operates at that rate regardless
of the benchmark rate.

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


class OpusCodecAttack(BaseAttack):
    """
    Pure Opus codec attack — encode and decode only, no NS, no network.

    Resampling to/from the Opus working rate happens inside the attack:
    the input (at the benchmark's runtime sampling rate) is resampled down
    to ``sampling_rate_opus_codec``, run through the codec, then resampled
    back up to the original rate.

    Config parameters:
        - bitrate_opus_codec (int): Opus bitrate in kbps (default: 16)
        - framesize_opus_codec (float): Frame size in ms (default: 20)
        - sampling_rate_opus_codec (int): Opus working rate in Hz (default: 16000)
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
        opus_sr = kwargs.get(
            "sampling_rate_opus_codec",
            self.config.get("sampling_rate_opus_codec"),
        )
        if opus_sr is None:
            raise ValueError(
                "'sampling_rate_opus_codec' must be set in config.json "
                "(the Opus working rate, e.g. 16000)."
            )
        opus_sr = int(opus_sr)

        audio = np.asarray(audio).astype(np.float32, copy=False)
        original_len = len(audio)

        # Resample down to the configured Opus working rate inside the attack.
        if original_sr != opus_sr:
            audio_opus = librosa.resample(
                audio, orig_sr=original_sr, target_sr=opus_sr
            )
        else:
            audio_opus = audio

        try:
            response = requests.post(
                self.endpoint + "/attack",
                json={
                    "audio": audio_opus.tolist(),
                    "sampling_rate": opus_sr,
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

        attacked_opus = np.asarray(response_data["audio"], dtype=np.float32)

        # Resample back up to the benchmark rate inside the attack.
        if original_sr != opus_sr:
            attacked = librosa.resample(
                attacked_opus, orig_sr=opus_sr, target_sr=original_sr
            )
        else:
            attacked = attacked_opus

        # Lock to the caller's length so resample drift doesn't leak out.
        if len(attacked) > original_len:
            attacked = attacked[:original_len]
        elif len(attacked) < original_len:
            attacked = np.pad(attacked, (0, original_len - len(attacked)))

        logger.info(
            f"OpusCodec attack: sr {original_sr}->{opus_sr}->{original_sr}, "
            f"bitrate={bitrate}k, framesize={framesize}ms"
        )
        return attacked.astype(np.float32)
