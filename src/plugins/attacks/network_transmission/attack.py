"""
Opus Network Emulation Attack - Client

Simulates VoIP/WebRTC audio transmission with realistic network impairments:
- audio preporcessing with WebRTC APM
- Opus codec compression
- Network delay and jitter (via tc netem)

This attack calls a Docker container that runs with --cap-add=NET_ADMIN
to enable tc netem for network emulation.
"""

import logging
import os

import numpy as np
import requests

from core.base_attack import BaseAttack

logger = logging.getLogger(__name__)


class NetworkTransmissionAttack(BaseAttack):
    """
    Audio attack simulating Opus codec transmission over an impaired network.

    Uses Docker container with tc netem for realistic network conditions
    including delay, jitter, burst packet loss, and jitter buffer overflow.

    Config parameters:
        - bitrate_opus_network (int): Opus bitrate in kbps (default: 16)
        - framesize_opus_network (float): Frame size in ms (default: 20)
        - delay_ms_opus_network (int): Base network delay in ms (default: 50)
        - jitter_ms_opus_network (int): Delay variation in ms (default: 20)
        - packet_loss_opus_network (int): Packet loss percentage (default: 5)

    The signal sampling rate is taken from the benchmark runner via the
    ``sampling_rate`` kwarg; the server resamples internally to the
    nearest Opus-supported rate so the codec stage stays consistent
    regardless of which watermarking model is in use.
    """

    def __init__(self):
        super().__init__()

        host = "localhost"
        port = os.getenv("NETWORK_TRANSMISSION_PORT", "10020")
        if not port:
            logger.error("NETWORK_TRANSMISSION_PORT environment variable not set.")
            raise ValueError("NETWORK_TRANSMISSION_PORT must be set for NetworkTransmissionAttack")

        self.endpoint = f"http://{host}:{port}"
        logger.info(f"OpusNetworkAttack initialized. Target API: {self.endpoint}")

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Opus network emulation attack.

        Args:
            audio (np.ndarray): Input audio signal
            **kwargs: Override config parameters

        Returns:
            np.ndarray: Audio after Opus encoding/decoding with network effects
        """
        # Use the actual signal SR propagated by the benchmark, not the
        # config-only attack-specific override (which was historically
        # hardcoded to 16 kHz and silently misrepresented every other
        # model's audio).
        sampling_rate = kwargs.get("sampling_rate")
        if sampling_rate is None:
            raise ValueError(
                "NetworkTransmissionAttack requires 'sampling_rate' in kwargs "
                "(propagated from the benchmark runner)."
            )

        # Basic parameters
        bitrate = kwargs.get(
            "bitrate_opus_network",
            self.config.get("bitrate_opus_network", 16)
        )
        framesize = kwargs.get(
            "framesize_opus_network",
            self.config.get("framesize_opus_network", 20)
        )
        delay_ms = kwargs.get(
            "delay_ms_opus_network",
            self.config.get("delay_ms_opus_network", 50)
        )
        jitter_ms = kwargs.get(
            "jitter_ms_opus_network",
            self.config.get("jitter_ms_opus_network", 20)
        )
        packet_loss = kwargs.get(
            "packet_loss_opus_network",
            self.config.get("packet_loss_opus_network", 5)
        )

        # Make request to Docker service
        try:
            response = requests.post(
                self.endpoint + "/attack",
                json={
                    "audio": audio.tolist(),
                    "sampling_rate": sampling_rate,
                    "bitrate": bitrate,
                    "framesize": framesize,
                    "delay_ms": delay_ms,
                    "jitter_ms": jitter_ms,
                    "packet_loss": packet_loss,
                },
                timeout=120
            )
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to opus_network service: {e}")
            logger.error("Ensure the container is running with: "
                        "docker-compose up opus_network")
            raise RuntimeError(f"Opus network service unavailable: {e}")

        response_data = response.json()

        if "audio" not in response_data:
            logger.error("Response does not contain 'audio' key.")
            raise KeyError("Missing 'audio' in response from opus_network service")

        logger.info(f"OpusNetwork attack: bitrate={bitrate}k, delay={delay_ms}ms, "
                   f"jitter={jitter_ms}ms, loss={packet_loss}%, ")

        return np.array(response_data["audio"], dtype=np.float32)
