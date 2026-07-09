"""
Pure Opus Codec Attack - Client

Applies only Opus encode/decode, with no WebRTC noise suppression and no
network emulation. Audio is forwarded to the codec service at the
benchmark's runtime sampling rate. The server uses ``opusenc`` (which
silently remaps to its nearest supported internal rate -- 8/12/16/24/48
kHz -- for compression) and ``opusdec`` to decode at that same Opus
internal rate. The decoded audio comes back at the codec's internal rate
and this client resamples it to the model's sampling rate with librosa,
so the SR conversion uses our resampler instead of opusdec's.

Talks to its own dedicated Docker service (port OPUS_CODEC_PORT, default
10023) -- a lightweight server that only does opusenc/opusdec, no WebRTC
APM and no tc netem.
"""

import logging
import os

import librosa
import numpy as np
import requests

from deepmarkpy.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)


class OpusCodecAttack(BaseAttack):
    """
    Pure Opus codec attack — encode and decode only, no NS, no network.

    Audio is sent to the service at the benchmark's sampling rate;
    ``opusenc`` handles the remap to its internal rate; ``opusdec``
    returns the decoded signal at that same internal rate; this client
    resamples it back to the benchmark sampling rate via librosa.

    Config parameters:
        - bitrate (int): Opus bitrate in kbps (default: 16)
        - framesize (float): Frame size in ms (default: 20)
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
        sampling_rate = kwargs.get("sampling_rate")
        if sampling_rate is None:
            raise ValueError(
                "'sampling_rate' must be provided (benchmark rate, e.g. 44100)."
            )
        sampling_rate = int(sampling_rate)

        bitrate = kwargs.get(
            "bitrate",
            self.config.get("bitrate", 16),
        )
        framesize = kwargs.get(
            "framesize",
            self.config.get("framesize", 20),
        )
    
        audio = np.asarray(audio).astype(np.float32, copy=False)
        original_len = len(audio)

        try:
            response = requests.post(
                self.endpoint + "/attack",
                json={
                    "audio": audio.tolist(),
                    "sampling_rate": sampling_rate,
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

        attacked = np.asarray(response_data["audio"], dtype=np.float32)
        decoded_sr = int(response_data.get("sampling_rate", sampling_rate))

        # opusdec emits audio at its native 48 kHz output. Resample back
        # to the benchmark sampling rate here (instead of letting opusdec
        # do it), so the SR conversion uses librosa's polyphase resampler
        # and stays the same algorithm we use everywhere else.
        if decoded_sr != sampling_rate:
            attacked = librosa.resample(
                attacked, orig_sr=decoded_sr, target_sr=sampling_rate
            )

        # Lock to caller's length so any drift from resampling doesn't
        # leak out to downstream metrics that compare sample-by-sample.
        if len(attacked) > original_len:
            attacked = attacked[:original_len]
        elif len(attacked) < original_len:
            attacked = np.pad(attacked, (0, original_len - len(attacked)))

        logger.info(
            f"OpusCodec attack: sr {sampling_rate}Hz -> codec -> "
            f"{decoded_sr}Hz -> {sampling_rate}Hz, "
            f"bitrate={bitrate}k, framesize={framesize}ms"
        )
        return attacked.astype(np.float32)
