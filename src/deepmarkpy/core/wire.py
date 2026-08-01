"""Audio encoding for the HTTP hop between clients and services.

Audio used to cross every hop as a JSON array of decimal floats, which cost
about 21.6 bytes per sample in each direction and spent real CPU on both ends
formatting and parsing decimals. A detect call pushed megabytes up to receive
a few dozen bytes back.

Samples now travel as base64 of the raw little-endian float64 buffer: 8 bytes
per sample, no decimal formatting, and ``np.frombuffer`` on the far side.

float64 is deliberate. float32 would halve the payload again, but it is lossy
-- it would change attacked audio and therefore the accuracy numbers the
benchmark reports. Decimal JSON round-tripped float64 exactly, so anything
narrower here would be a silent behavior change rather than a transport one.
"""

import base64

import numpy as np

# Sent alongside the payload so a service can tell a v2 client from a v1 one
# and fail clearly rather than misreading bytes as decimals.
AUDIO_ENCODING = "base64-float64"


def encode_audio(audio) -> str:
    """Serialize a 1-D signal for transport, losslessly."""
    arr = np.ascontiguousarray(np.asarray(audio, dtype=np.float64).ravel())
    return base64.b64encode(arr.tobytes()).decode("ascii")


def decode_audio(payload) -> np.ndarray:
    """Inverse of :func:`encode_audio`.

    Accepts the JSON list form as well, so a service still understands a
    client that has not been upgraded.
    """
    if payload is None:
        return None
    if isinstance(payload, str):
        return np.frombuffer(base64.b64decode(payload), dtype=np.float64)
    return np.asarray(payload, dtype=np.float64)
