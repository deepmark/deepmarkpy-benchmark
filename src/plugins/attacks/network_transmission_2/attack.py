"""
RTP VoIP Network Transmission Attack - Client

Realistic VoIP call simulation with:
- RTP packet framing (RFC 3550)
- Real-time pacing with sender/receiver threads
- opuslib codec with in-band FEC and VoIP application mode
- Jitter buffer with deadline-based playout
- Packet Loss Concealment (PLC)
- Optional sender-side AEC, noise suppression, AGC
- Optional receiver-side denoise and AGC
- tc netem for network impairments (delay, jitter, loss, reorder, duplication)

Requires Docker container with --cap-add=NET_ADMIN.
"""

import logging
import os

import numpy as np
import requests

from core.base_attack import BaseAttack

logger = logging.getLogger(__name__)


class NetworkTransmission2Attack(BaseAttack):
    """
    Realistic RTP-based VoIP network transmission attack.

    Simulates a complete VoIP call pipeline including RTP packetization,
    real-time pacing, jitter buffer, Opus FEC, and optional audio
    processing (AEC, denoise, AGC) on both sender and receiver sides.

    Config parameters:
        - bitrate_bps_netem2 (int): Opus bitrate in bps (default: 24000)
        - frame_duration_ms_netem2 (int): Frame size in ms (default: 20)
        - delay_ms_netem2 (int): Base network delay in ms (default: 50)
        - jitter_ms_netem2 (int): Delay variation in ms (default: 10)
        - packet_loss_netem2 (int): Packet loss percentage (default: 5)
        - duplication_netem2 (int): Packet duplication % (default: 0)
        - reorder_netem2 (int): Packet reorder % (default: 0)
        - corruption_netem2 (int): Bit corruption % (default: 0)
        - fec_enabled_netem2 (bool): Enable Opus in-band FEC (default: true)
        - expected_loss_netem2 (int): Expected loss hint for encoder (default: 5)
        - aec_enabled_netem2 (bool): Enable sender AEC (default: false)
        - denoise_method_netem2 (str): Denoise method: none/noisereduce (default: none)
        - agc_target_lufs_netem2 (int): AGC target loudness (default: -18)
        - playout_delay_ms_netem2 (int): Jitter buffer depth in ms (default: 60).
            Packets arriving after first_arrival + playout_delay + seq*frame
            are discarded as "late", matching real VoIP playout behavior.
    """

    def __init__(self):
        super().__init__()

        host = "localhost"
        port = os.getenv("NETWORK_TRANSMISSION_2_PORT", "10021")
        if not port:
            logger.error("NETWORK_TRANSMISSION_2_PORT environment variable not set.")
            raise ValueError("NETWORK_TRANSMISSION_2_PORT must be set for NetworkTransmission2Attack")

        self.endpoint = f"http://{host}:{port}"
        logger.info(f"NetworkTransmission2Attack initialized. Target API: {self.endpoint}")

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply RTP VoIP network transmission attack.

        Args:
            audio (np.ndarray): Input audio signal
            **kwargs: Override config parameters

        Returns:
            np.ndarray: Audio after full VoIP pipeline simulation
        """
        sampling_rate = kwargs.get("sampling_rate")
        if sampling_rate is None:
            raise ValueError(
                "NetworkTransmission2Attack requires 'sampling_rate' in kwargs "
                "(propagated from the benchmark runner)."
            )

        params = {}
        for key in [
            "bitrate_bps_netem2",
            "frame_duration_ms_netem2",
            "delay_ms_netem2",
            "jitter_ms_netem2",
            "packet_loss_netem2",
            "duplication_netem2",
            "reorder_netem2",
            "corruption_netem2",
            "fec_enabled_netem2",
            "expected_loss_netem2",
            "aec_enabled_netem2",
            "denoise_method_netem2",
            "agc_target_lufs_netem2",
            "playout_delay_ms_netem2",
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
            logger.error(f"Failed to connect to network_transmission_2 service: {e}")
            logger.error(
                "Ensure the container is running with: "
                "docker-compose up network_transmission_2"
            )
            raise RuntimeError(f"Network transmission 2 service unavailable: {e}")

        response_data = response.json()

        if "audio" not in response_data:
            logger.error("Response does not contain 'audio' key.")
            raise KeyError("Missing 'audio' in response from network_transmission_2 service")

        logger.info(
            f"NetworkTransmission2 attack: bitrate={params['bitrate_bps_netem2']}bps, "
            f"delay={params['delay_ms_netem2']}ms, jitter={params['jitter_ms_netem2']}ms, "
            f"loss={params['packet_loss_netem2']}%, fec={params['fec_enabled_netem2']}"
        )

        return np.array(response_data["audio"], dtype=np.float32)
