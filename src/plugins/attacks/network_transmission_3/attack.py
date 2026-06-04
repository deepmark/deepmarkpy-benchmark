"""
RTP VoIP Network Transmission Attack v3 - Client

Combines the realistic VoIP pipeline of v2 (RTP framing, deadline-based
jitter buffer, FEC-overwrites-PLC) with library-backed audio processing
of v1 (WebRTC APM for noise suppression and VAD, pyloudnorm for AGC),
instead of the custom NLMS / spectral-gating / RMS implementations of v2.

The pipeline is locked to 16 kHz: one resample on entry, one on exit.
No nearest-rate selection, no four-step resample dance.

Requires Docker container with --cap-add=NET_ADMIN.
"""

import logging
import os

import numpy as np
import requests

from core.base_attack import BaseAttack

logger = logging.getLogger(__name__)


class NetworkTransmission3Attack(BaseAttack):
    """
    Realistic RTP-based VoIP attack with library-backed audio processing.

    Audio is processed at a fixed 16 kHz pipeline rate:
        input_sr -> resample to 16k -> WebRTC APM (NS+VAD) -> Opus encode
        -> RTP/netem -> jitter buffer -> Opus decode (FEC+PLC) -> AGC
        (pyloudnorm) -> resample back to input_sr

    Config parameters:
        - bitrate_bps_netem3 (int): Opus bitrate in bps (default: 24000)
        - frame_duration_ms_netem3 (int): Frame size in ms (default: 20)
        - delay_ms_netem3 (int): Base network delay in ms (default: 100)
        - jitter_ms_netem3 (int): Delay variation in ms (default: 30)
        - packet_loss_netem3 (int): Packet loss percentage (default: 10)
        - duplication_netem3 (int): Packet duplication % (default: 0)
        - reorder_netem3 (int): Packet reorder % (default: 3)
        - corruption_netem3 (int): Bit corruption % (default: 0)
        - fec_enabled_netem3 (bool): Enable Opus in-band FEC (default: true)
        - expected_loss_netem3 (int): Expected loss hint for encoder (default: 10)
        - ns_enabled_netem3 (bool): Enable WebRTC noise suppression (default: true)
        - vad_enabled_netem3 (bool): Enable WebRTC VAD (default: true)
        - agc_enabled_netem3 (bool): Enable pyloudnorm AGC (default: true)
        - agc_target_lufs_netem3 (float): AGC target LUFS (default: -18)
        - playout_delay_ms_netem3 (int): Jitter buffer depth in ms (default: 60)
    """

    def __init__(self):
        super().__init__()

        host = "localhost"
        port = os.getenv("NETWORK_TRANSMISSION_3_PORT", "10022")
        if not port:
            logger.error("NETWORK_TRANSMISSION_3_PORT environment variable not set.")
            raise ValueError(
                "NETWORK_TRANSMISSION_3_PORT must be set for NetworkTransmission3Attack"
            )

        self.endpoint = f"http://{host}:{port}"
        logger.info(f"NetworkTransmission3Attack initialized. Target API: {self.endpoint}")

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate")
        if sampling_rate is None:
            raise ValueError(
                "NetworkTransmission3Attack requires 'sampling_rate' in kwargs"
            )

        params = {}
        for key in [
            "bitrate_bps_netem3",
            "frame_duration_ms_netem3",
            "delay_ms_netem3",
            "jitter_ms_netem3",
            "packet_loss_netem3",
            "duplication_netem3",
            "reorder_netem3",
            "corruption_netem3",
            "fec_enabled_netem3",
            "expected_loss_netem3",
            "ns_enabled_netem3",
            "vad_enabled_netem3",
            "agc_enabled_netem3",
            "agc_target_lufs_netem3",
            "playout_delay_ms_netem3",
        ]:
            params[key] = kwargs.get(key, self.config.get(key))

        try:
            response = requests.post(
                self.endpoint + "/attack",
                json={
                    "audio": audio.tolist(),
                    "sampling_rate": sampling_rate,
                    **params,
                },
                timeout=180,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to network_transmission_3 service: {e}")
            logger.error(
                "Ensure the container is running with: "
                "docker-compose up network_transmission_3"
            )
            raise RuntimeError(f"Network transmission 3 service unavailable: {e}")

        response_data = response.json()
        if "audio" not in response_data:
            logger.error("Response does not contain 'audio' key.")
            raise KeyError("Missing 'audio' in response from network_transmission_3 service")

        logger.info(
            f"NetworkTransmission3 attack: bitrate={params['bitrate_bps_netem3']}bps, "
            f"delay={params['delay_ms_netem3']}ms, jitter={params['jitter_ms_netem3']}ms, "
            f"loss={params['packet_loss_netem3']}%, fec={params['fec_enabled_netem3']}"
        )
        return np.array(response_data["audio"], dtype=np.float32)
