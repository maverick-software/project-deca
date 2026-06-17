"""Helpers for stub stage implementations."""

from __future__ import annotations

from typing import Any

import numpy as np

from decadic.config import STAGE_LATENT_DIM
from decadic.cycle.types import CycleContext, StageTrace


def latent_seed(ctx: CycleContext, key: str) -> np.ndarray:
    arr = ctx.latents.setdefault(key, np.zeros((STAGE_LATENT_DIM,), dtype=np.float32))
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
        ctx.latents[key] = arr
    return arr


def nudge_latent(vec: np.ndarray, scale: float = 0.02) -> None:
    vec[:] = np.tanh(vec + scale * np.random.standard_normal(vec.shape).astype(np.float32))


def trace(stage: int, name: str, **payload: Any) -> StageTrace:
    return StageTrace(stage=stage, name=name, payload=dict(payload))
