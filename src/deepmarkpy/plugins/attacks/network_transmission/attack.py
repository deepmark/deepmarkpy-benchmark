"""
RTP VoIP Network Transmission Attack - Client

Realistic VoIP simulation: WebRTC APM (NS + VAD) on the sender side,
Opus encode with in-band FEC, RTP framing with drift-free pacing
through tc netem, deadline-based jitter buffer, Opus decode (FEC
overwrites PLC), and pyloudnorm AGC on the receiver side.

The pipeline is locked to 16 kHz: one resample on entry, one on exit.

Requires Docker container with --cap-add=NET_ADMIN.
"""

import logging
import os

import numpy as np
import requests

from deepmarkpy.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)


class NetworkTransmissionAttack(BaseAttack):
    """
    Realistic RTP-based VoIP attack with library-backed audio processing.

    Audio is processed at a fixed 16 kHz pipeline rate:
        input_sr -> resample to 16k -> WebRTC APM (NS+VAD) -> Opus encode
        -> RTP/netem -> jitter buffer -> Opus decode (FEC+PLC) -> AGC
        (pyloudnorm) -> resample back to input_sr

    Config parameters:
        - bitrate_bps_netem (int): Opus bitrate in bps (default: 24000)
        - frame_duration_ms_netem (int): Frame size in ms (default: 20)
        - delay_ms_netem (int): Base network delay in ms (default: 100)
        - jitter_ms_netem (int): Delay variation in ms (default: 30)
        - packet_loss_netem (int): Packet loss percentage (default: 10)
        - duplication_netem (int): Packet duplication % (default: 0)
        - reorder_netem (int): Packet reorder % (default: 3)
        - corruption_netem (int): Bit corruption % (default: 0)
        - fec_enabled_netem (bool): Enable Opus in-band FEC (default: true)
        - expected_loss_netem (int): Expected loss hint for encoder (default: 10)
        - ns_enabled_netem (bool): Enable WebRTC noise suppression (default: true)
        - vad_enabled_netem (bool): Enable WebRTC VAD (default: true)
        - agc_enabled_netem (bool): Enable pyloudnorm AGC (default: true)
        - agc_target_lufs_netem (float): AGC target LUFS (default: -18)
        - playout_delay_ms_netem (int): Jitter buffer depth in ms (default: 60)
    """

    def __init__(self):
        super().__init__()

        host = "localhost"
        port = os.getenv("NETWORK_TRANSMISSION_PORT", "10020")
        if not port:
            logger.error("NETWORK_TRANSMISSION_PORT environment variable not set.")
            raise ValueError(
                "NETWORK_TRANSMISSION_PORT must be set for NetworkTransmissionAttack"
            )

        self.endpoint = f"http://{host}:{port}"
        logger.info(f"NetworkTransmissionAttack initialized. Target API: {self.endpoint}")

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate")
        if sampling_rate is None:
            raise ValueError(
                "NetworkTransmissionAttack requires 'sampling_rate' in kwargs"
            )

        params = {}
        for key in [
            "bitrate_bps_netem",
            "frame_duration_ms_netem",
            "delay_ms_netem",
            "jitter_ms_netem",
            "packet_loss_netem",
            "duplication_netem",
            "reorder_netem",
            "corruption_netem",
            "fec_enabled_netem",
            "expected_loss_netem",
            "ns_enabled_netem",
            "vad_enabled_netem",
            "agc_enabled_netem",
            "agc_target_lufs_netem",
            "playout_delay_ms_netem",
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
            logger.error(f"Failed to connect to network_transmission service: {e}")
            logger.error(
                "Ensure the container is running with: "
                "docker-compose up network_transmission"
            )
            raise RuntimeError(f"Network transmission service unavailable: {e}")

        response_data = response.json()
        if "audio" not in response_data:
            logger.error("Response does not contain 'audio' key.")
            raise KeyError("Missing 'audio' in response from network_transmission service")

        logger.info(
            f"NetworkTransmission attack: bitrate={params['bitrate_bps_netem']}bps, "
            f"delay={params['delay_ms_netem']}ms, jitter={params['jitter_ms_netem']}ms, "
            f"loss={params['packet_loss_netem']}%, fec={params['fec_enabled_netem']}"
        )
        return np.array(response_data["audio"], dtype=np.float32)
