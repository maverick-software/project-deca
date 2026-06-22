"""Fly/human-inspired pre-cognitive perception organ.

The organ enriches anonymous visual proposals with local retinotopic features,
frame-difference motion, looming, stuff/background hints, and body-coupled agency
signals. It never emits semantic labels; live cognition receives only anonymous
object-file fields.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


DEFAULT_GRID = 16
LOW_CONTRAST = 0.035
STUFF_SPREAD = 0.34
BODY_MOTION = 0.08
LOOMING_DELTA = 0.04


@dataclass(frozen=True)
class RetinotopicMap:
    width: int
    height: int
    intensity: list[list[float]]
    contrast: list[list[float]]
    frame_delta: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerceptionOrganDiagnostics:
    frame_seen: bool
    stale_frame: bool
    grid_size: int
    flow_confidence: float
    global_motion: float
    local_motion_max: float
    local_motion_mean: float
    looming_count: int
    stuff_count: int
    body_candidate_count: int
    foreground_count: int
    checkpoint_status: str = "online_lightweight"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PerceptionOrganState:
    prev_gray: np.ndarray | None = None
    prev_timestamp: str | None = None
    prev_spread_by_idx: dict[int, float] = field(default_factory=dict)


def _decode_gray(obs: dict[str, Any] | None, size: int) -> np.ndarray | None:
    vis = (obs or {}).get("vision") or {}
    raw_b64 = vis.get("data")
    if not (isinstance(raw_b64, str) and raw_b64.strip()):
        return None
    try:
        from PIL import Image

        blob = base64.b64decode(raw_b64)
        im = Image.open(io.BytesIO(blob)).convert("L").resize((size, size))
        return np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        return None


def _decode_rgb(obs: dict[str, Any] | None, size: int) -> np.ndarray | None:
    vis = (obs or {}).get("vision") or {}
    raw_b64 = vis.get("data")
    if not (isinstance(raw_b64, str) and raw_b64.strip()):
        return None
    try:
        from PIL import Image

        blob = base64.b64decode(raw_b64)
        im = Image.open(io.BytesIO(blob)).convert("RGB").resize((size, size))
        return np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        return None


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _sample(map_: np.ndarray, uv: list[float] | None, radius: int = 1) -> float:
    if uv is None or len(uv) < 2:
        return 0.0
    h, w = map_.shape
    x = int(max(0, min(w - 1, round(float(uv[0]) * (w - 1)))))
    y = int(max(0, min(h - 1, round(float(uv[1]) * (h - 1)))))
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    return float(map_[y0:y1, x0:x1].mean())


def _patch_rgb(rgb: np.ndarray | None, uv: list[float] | None, radius: int = 1) -> dict[str, Any]:
    if rgb is None or uv is None or len(uv) < 2:
        return {}
    h, w, _ = rgb.shape
    x = int(max(0, min(w - 1, round(float(uv[0]) * (w - 1)))))
    y = int(max(0, min(h - 1, round(float(uv[1]) * (h - 1)))))
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    patch = rgb[y0:y1, x0:x1].reshape(-1, 3)
    if patch.size == 0:
        return {}
    mean_rgb = patch.mean(axis=0)
    brightness = float(mean_rgb.mean())
    # Coarse hue-family vector without semantic color names. Six bins are enough
    # to preserve repeatable appearance evidence without turning it into labels.
    bins = np.zeros(6, dtype=np.float32)
    for r, g, b in patch:
        mx = max(float(r), float(g), float(b))
        mn = min(float(r), float(g), float(b))
        if mx - mn < 1e-6:
            idx = 0
        elif mx == float(r):
            idx = int(((float(g) - float(b)) / (mx - mn) / 6.0 + 1.0) * 3.0) % 6
        elif mx == float(g):
            idx = int((((float(b) - float(r)) / (mx - mn) + 2.0) / 6.0) * 6.0) % 6
        else:
            idx = int((((float(r) - float(g)) / (mx - mn) + 4.0) / 6.0) * 6.0) % 6
        bins[idx] += 1.0
    bins = bins / max(1.0, float(bins.sum()))
    return {
        "rgb_mean": [round(float(x), 6) for x in mean_rgb.tolist()],
        "hue_histogram": [round(float(x), 6) for x in bins.tolist()],
        "brightness": round(brightness, 6),
    }


def _motion_vector(delta: np.ndarray, uv: list[float] | None) -> list[float]:
    if uv is None or len(uv) < 2:
        return [0.0, 0.0]
    h, w = delta.shape
    x = int(max(0, min(w - 1, round(float(uv[0]) * (w - 1)))))
    y = int(max(0, min(h - 1, round(float(uv[1]) * (h - 1)))))
    y0, y1 = max(0, y - 1), min(h, y + 2)
    x0, x1 = max(0, x - 1), min(w, x + 2)
    patch = delta[y0:y1, x0:x1]
    if patch.size == 0 or float(patch.sum()) <= 1e-8:
        return [0.0, 0.0]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx = float((patch * xx).sum() / patch.sum())
    cy = float((patch * yy).sum() / patch.sum())
    return [float((cx - x) / max(1, w)), float((cy - y) / max(1, h))]


def _motor_rms(prev_motor: Any) -> float:
    if prev_motor is None:
        return 0.0
    try:
        arr = np.asarray(prev_motor.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    except Exception:
        try:
            arr = np.asarray(prev_motor, dtype=np.float32).reshape(-1)
        except Exception:
            return 0.0
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def _touch_active(obs: dict[str, Any] | None) -> bool:
    contacts = ((obs or {}).get("proprioception") or {}).get("contacts")
    if not isinstance(contacts, list) or not contacts:
        return False
    try:
        return max(abs(float(c)) for c in contacts) > 50.0
    except (TypeError, ValueError):
        return False


class PerceptionOrgan:
    """Stateful, label-free visual/body feature enricher."""

    def __init__(self, *, grid_size: int = DEFAULT_GRID) -> None:
        self.grid_size = max(4, int(grid_size))
        self.state = PerceptionOrganState()
        self.last_retinotopic_map: dict[str, Any] | None = None
        self.last_diagnostics: dict[str, Any] | None = None

    def process(
        self,
        obs: dict[str, Any] | None,
        proposals: list[dict[str, Any]],
        *,
        prev_motor: Any = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
        gray = _decode_gray(obs, self.grid_size)
        rgb = _decode_rgb(obs, self.grid_size)
        timestamp = str((obs or {}).get("timestamp", "") or "")
        stale = bool(timestamp and timestamp == self.state.prev_timestamp)
        if gray is None:
            diag = PerceptionOrganDiagnostics(
                frame_seen=False,
                stale_frame=stale,
                grid_size=self.grid_size,
                flow_confidence=0.0,
                global_motion=0.0,
                local_motion_max=0.0,
                local_motion_mean=0.0,
                looming_count=0,
                stuff_count=0,
                body_candidate_count=0,
                foreground_count=0,
            ).to_dict()
            self.last_diagnostics = diag
            return proposals, diag, self.last_retinotopic_map

        contrast = _grad_mag(gray)
        if self.state.prev_gray is None or stale:
            delta = np.zeros_like(gray, dtype=np.float32)
        else:
            delta = np.abs(gray - self.state.prev_gray).astype(np.float32)

        motor = _motor_rms(prev_motor)
        touch = _touch_active(obs)
        enriched: list[dict[str, Any]] = []
        stuff_count = 0
        body_count = 0
        looming_count = 0
        foreground_count = 0
        local_vals: list[float] = []
        next_spreads: dict[int, float] = {}

        for raw in proposals:
            p = dict(raw)
            uv = p.get("uv") if isinstance(p.get("uv"), list) else None
            try:
                idx = int(p.get("idx", -1) or -1)
            except (TypeError, ValueError):
                idx = -1
            local_contrast = _sample(contrast, uv)
            local_motion = _sample(delta, uv)
            local_vals.append(local_motion)
            mv = _motion_vector(delta, uv)
            try:
                spread = float(p.get("spread")) if p.get("spread") is not None else 0.0
            except (TypeError, ValueError):
                spread = 0.0
            prev_spread = self.state.prev_spread_by_idx.get(idx)
            looming = float(spread - prev_spread) if prev_spread is not None else 0.0
            if idx >= 0:
                next_spreads[idx] = spread

            p["retina_contrast"] = local_contrast
            p["local_motion"] = local_motion
            p["flow"] = mv
            p["motion"] = mv
            p["looming"] = looming
            prop_ev = dict(p.get("property_evidence") or {})
            prop_ev.update(_patch_rgb(rgb, uv))
            prop_ev["edge_strength"] = local_contrast
            prop_ev["local_motion"] = local_motion
            prop_ev["looming"] = looming
            prop_ev["size_proxy"] = max(0.0, min(1.0, spread))
            p["property_evidence"] = prop_ev

            kind = str(p.get("kind_hint", "object"))
            if spread >= STUFF_SPREAD and local_motion < BODY_MOTION:
                kind = "stuff"
            elif local_contrast < LOW_CONTRAST and local_motion < BODY_MOTION:
                kind = "stuff"
            elif (motor > 0.03 or touch) and local_motion >= BODY_MOTION:
                kind = "body_part_candidate"
                p["agency"] = max(float(p.get("agency", 0.0) or 0.0), min(1.0, local_motion + motor))

            if looming >= LOOMING_DELTA:
                looming_count += 1
            if kind == "stuff":
                stuff_count += 1
            elif kind == "body_part_candidate":
                body_count += 1
                foreground_count += 1
            else:
                foreground_count += 1
            p["kind_hint"] = kind
            enriched.append(p)

        global_motion = float(delta.mean())
        local_max = float(max(local_vals) if local_vals else 0.0)
        local_mean = float(sum(local_vals) / len(local_vals)) if local_vals else 0.0
        flow_conf = float(max(0.0, min(1.0, local_max - global_motion)))
        ret = RetinotopicMap(
            width=self.grid_size,
            height=self.grid_size,
            intensity=np.round(gray, 4).tolist(),
            contrast=np.round(contrast, 4).tolist(),
            frame_delta=np.round(delta, 4).tolist(),
        ).to_dict()
        diag = PerceptionOrganDiagnostics(
            frame_seen=True,
            stale_frame=stale,
            grid_size=self.grid_size,
            flow_confidence=round(flow_conf, 6),
            global_motion=round(global_motion, 6),
            local_motion_max=round(local_max, 6),
            local_motion_mean=round(local_mean, 6),
            looming_count=looming_count,
            stuff_count=stuff_count,
            body_candidate_count=body_count,
            foreground_count=foreground_count,
        ).to_dict()
        self.state.prev_gray = gray
        self.state.prev_timestamp = timestamp
        self.state.prev_spread_by_idx = next_spreads
        self.last_retinotopic_map = ret
        self.last_diagnostics = diag
        return enriched, diag, ret
