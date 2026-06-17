"""Turn slot-attention output into egocentric object proposals.

Pure geometry/bookkeeping (no torch): each present slot becomes a proposal with
an appearance fingerprint and a coarse egocentric placement derived from where
its mask sits in the image (bearing from the camera FOV) and how big it looks
(an inverse-size range proxy). These proposals are what working memory's data
association binds into persistent, anonymously-named object files.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np


def vision_fovy_deg() -> float:
    """Vertical FOV of the egocentric camera (assets/humanoid_body.xml: fovy=80)."""
    return float(os.environ.get("DECADIC_VISION_FOVY", "80"))


def uv_to_bearing(u: float, v: float, fovy_deg: float) -> tuple[float, float]:
    """Image (u,v) in [0,1] -> (azimuth, elevation) radians; center looks forward.

    u is horizontal (0 left, 1 right), v vertical (0 top, 1 bottom). A square
    sensor is assumed (fovx == fovy). Up is positive elevation.
    """
    half = math.radians(fovy_deg) * 0.5
    az = (u - 0.5) * 2.0 * half
    el = -(v - 0.5) * 2.0 * half
    return az, el


def range_from_spread(spread: float) -> float:
    """Inverse-size depth proxy: a bigger (more spread-out) mask looks closer."""
    return float(min(8.0, 0.35 / (spread + 0.05)))


def bearing_to_relative(az: float, el: float, rng: float) -> list[float]:
    """Egocentric relative vector: +x forward, +y left, +z up; |vec| == range."""
    x = rng * math.cos(el) * math.cos(az)
    y = rng * math.cos(el) * math.sin(az)
    z = rng * math.sin(el)
    return [float(x), float(y), float(z)]


def extract_proposals(
    slots: np.ndarray,
    presence: np.ndarray,
    centroids: np.ndarray,
    *,
    threshold: float,
    fovy_deg: float | None = None,
) -> list[dict[str, Any]]:
    """Build object proposals from per-slot appearance, presence, and mask centroid.

    ``slots``: [K, slot_dim]; ``presence``: [K]; ``centroids``: [K, 3] = (u, v, spread).
    Only slots whose presence clears ``threshold`` become proposals.
    """
    fovy = vision_fovy_deg() if fovy_deg is None else fovy_deg
    proposals: list[dict[str, Any]] = []
    k = slots.shape[0]
    for i in range(k):
        pres = float(presence[i])
        if pres < threshold:
            continue
        u, v, spread = (
            float(centroids[i, 0]),
            float(centroids[i, 1]),
            float(centroids[i, 2]),
        )
        az, el = uv_to_bearing(u, v, fovy)
        rng = range_from_spread(spread)
        proposals.append(
            {
                "idx": i,
                "appearance": [float(x) for x in slots[i].tolist()],
                "presence": pres,
                "uv": [u, v],
                "spread": spread,
                "bearing": [float(az), float(el)],
                "relative": bearing_to_relative(az, el, rng),
            }
        )
    return proposals
