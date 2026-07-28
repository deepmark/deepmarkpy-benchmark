from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import numpy as np

from deepmarkpy.core.base_attack import BaseAttack


def run_attack_payload(attack: BaseAttack, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one packaged attack from the transport-neutral service payload."""

    if "audio" not in payload:
        raise ValueError("Request payload must contain 'audio'")
    if "sampling_rate" not in payload:
        raise ValueError("Request payload must contain 'sampling_rate'")

    sampling_rate = int(payload["sampling_rate"])
    params = {
        key: value
        for key, value in payload.items()
        if key not in {"audio", "sampling_rate"}
    }
    result = attack.apply(
        np.asarray(payload["audio"], dtype=np.float32),
        sampling_rate=sampling_rate,
        **params,
    )
    return {
        "audio": np.asarray(result, dtype=np.float32).tolist(),
        "sampling_rate": sampling_rate,
    }


def create_attack_app(
    attack_factory: Callable[[], BaseAttack],
    *,
    attack_name: str,
):
    """Create a lazy-loading FastAPI service for one packaged attack."""

    from fastapi import FastAPI, HTTPException

    app = FastAPI(title=f"deepmarkpy {attack_name} attack")

    @lru_cache(maxsize=1)
    def get_attack() -> BaseAttack:
        return attack_factory()

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"status": "ok", "attack": attack_name}

    @app.post("/attack")
    def attack(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return run_attack_payload(get_attack(), payload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app
