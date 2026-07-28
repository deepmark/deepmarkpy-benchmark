from __future__ import annotations

import os
from typing import Any

import numpy as np
import requests

from deepmarkpy.core.base_attack import BaseAttack
from deepmarkpy.utils.param_aliases import normalize_attack_config


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class RemoteAttack(BaseAttack):
    """Base class for attacks executed by an isolated HTTP service."""

    port_env = ""
    default_port = 0
    timeout_seconds = 600.0

    def __init__(self):
        super().__init__()
        if not self.port_env or not self.default_port:
            raise TypeError("RemoteAttack subclasses must configure port_env and default_port")

        host = os.getenv(f"{self.port_env}_HOST") or os.getenv(
            "DEEPMARK_ATTACK_HOST", "localhost"
        )
        port = os.getenv(self.port_env, str(self.default_port))
        if not port:
            raise ValueError(f"{self.port_env} must not be empty")
        self.endpoint = f"http://{host}:{port}"

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate")
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        normalized_kwargs = normalize_attack_config(self.attack_key, kwargs)
        params = {
            key: _json_safe(value)
            for key, value in normalized_kwargs.items()
            if key != "sampling_rate"
        }
        response = requests.post(
            f"{self.endpoint}/attack",
            json={
                "audio": np.asarray(audio, dtype=np.float32).tolist(),
                "sampling_rate": int(sampling_rate),
                **params,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        response_data = response.json()
        if response_data.get("audio") is None:
            detail = response_data.get("error") or "Missing 'audio' in /attack response"
            raise RuntimeError(str(detail))
        return np.asarray(response_data["audio"], dtype=np.float32)
