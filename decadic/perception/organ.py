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

from decadic import config as C


DEFAULT_GRID = 16
LOW_CONTRAST = 0.035
STUFF_SPREAD = 0.34
BODY_MOTION = 0.08
LOOMING_DELTA = 0.04
MAX_BOOTSTRAP_PROPOSALS = 7
BOOTSTRAP_DEDUP_UV = 0.12


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


def _uv_to_bearing(u: float, v: float, fovy_deg: float = 80.0) -> tuple[float, float]:
    half = math.radians(fovy_deg) * 0.5
    return (u - 0.5) * 2.0 * half, -(v - 0.5) * 2.0 * half


def _range_from_spread(spread: float) -> float:
    return float(min(8.0, 0.35 / (float(spread) + 0.05)))


def _bearing_to_relative(az: float, el: float, rng: float) -> list[float]:
    x = rng * math.cos(el) * math.cos(az)
    y = rng * math.cos(el) * math.sin(az)
    z = rng * math.sin(el)
    return [float(x), float(y), float(z)]


def _hue_histogram_from_patch(patch: np.ndarray) -> list[float]:
    if patch.size == 0:
        return [0.0] * 6
    bins = np.zeros(6, dtype=np.float32)
    for r, g, b in patch.reshape(-1, 3):
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
    total = float(bins.sum())
    if total > 0.0:
        bins /= total
    return [round(float(x), 6) for x in bins.tolist()]


def _retinotopic_bootstrap_proposals(
    gray: np.ndarray,
    rgb: np.ndarray | None,
    contrast: np.ndarray,
    delta: np.ndarray,
    existing: list[dict[str, Any]],
    *,
    max_count: int = MAX_BOOTSTRAP_PROPOSALS,
) -> list[dict[str, Any]]:
    """Create anonymous region proposals directly from image-space primitives.

    This is a bootstrap scaffold for early perception, not a semantic detector:
    it uses only contrast, brightness discontinuity, and frame-difference motion.
    The resulting proposals flow through the same object-file health gates as
    learned slots, so bad regions are still allowed to be skipped downstream.
    """
    h, w = gray.shape
    if h == 0 or w == 0:
        return []
    brightness = np.abs(gray - float(np.median(gray))).astype(np.float32)
    sal = 0.55 * contrast.astype(np.float32) + 0.35 * brightness + 0.55 * delta.astype(np.float32)
    if not np.isfinite(sal).all() or float(sal.max()) <= 1e-6:
        return []
    floor = max(0.035, float(np.mean(sal) + 0.35 * np.std(sal)), float(np.percentile(sal, 72)))
    mask = sal >= floor
    visited = np.zeros_like(mask, dtype=bool)
    existing_uvs: list[list[float]] = []
    for p in existing:
        uv = p.get("uv") if isinstance(p, dict) else None
        if isinstance(uv, list) and len(uv) >= 2:
            try:
                existing_uvs.append([float(uv[0]), float(uv[1])])
            except (TypeError, ValueError):
                pass

    comps: list[dict[str, Any]] = []
    for sy in range(h):
        for sx in range(w):
            if visited[sy, sx] or not mask[sy, sx]:
                continue
            stack = [(sy, sx)]
            visited[sy, sx] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            area = len(cells)
            area_frac = area / float(h * w)
            if area < 1 or area_frac > 0.55:
                continue
            yy = np.asarray([c[0] for c in cells], dtype=np.float32)
            xx = np.asarray([c[1] for c in cells], dtype=np.float32)
            weights = np.asarray([max(float(sal[y, x]), 1e-4) for y, x in cells], dtype=np.float32)
            total = float(weights.sum()) or 1.0
            cx = float((xx * weights).sum() / total)
            cy = float((yy * weights).sum() / total)
            u = float((cx + 0.5) / max(1, w))
            v = float((cy + 0.5) / max(1, h))
            if any(math.hypot(u - eu, v - ev) < BOOTSTRAP_DEDUP_UV for eu, ev in existing_uvs):
                continue
            var = float((((xx - cx) ** 2 + (yy - cy) ** 2) * weights).sum() / total)
            spread = float(max(math.sqrt(var) / max(1.0, float(max(h, w))), math.sqrt(area_frac) * 0.5))
            mean_gray = float(np.mean([gray[y, x] for y, x in cells]))
            std_gray = float(np.std([gray[y, x] for y, x in cells]))
            mean_contrast = float(np.mean([contrast[y, x] for y, x in cells]))
            mean_motion = float(np.mean([delta[y, x] for y, x in cells]))
            max_sal = float(np.max([sal[y, x] for y, x in cells]))
            if rgb is not None:
                patch = np.asarray([rgb[y, x] for y, x in cells], dtype=np.float32).reshape(-1, 3)
                rgb_mean = patch.mean(axis=0)
                hue = _hue_histogram_from_patch(patch)
            else:
                rgb_mean = np.asarray([mean_gray, mean_gray, mean_gray], dtype=np.float32)
                hue = [0.0] * 6
            az, el = _uv_to_bearing(u, v)
            rng = _range_from_spread(spread)
            presence = max(0.22, min(1.0, 0.30 + max_sal * 1.4 + mean_contrast * 1.2 + mean_motion * 1.5))
            prop_ev = {
                "area": round(float(area_frac), 6),
                "size_proxy": round(float(spread), 6),
                "edge_strength": round(mean_contrast, 6),
                "brightness_contrast": round(max_sal, 6),
                "local_motion": round(mean_motion, 6),
                "rgb_mean": [round(float(x), 6) for x in rgb_mean.tolist()],
                "hue_histogram": hue,
            }
            comps.append(
                {
                    "idx": 1000 + len(comps),
                    "appearance": [
                        round(mean_gray, 6),
                        round(std_gray, 6),
                        round(mean_contrast, 6),
                        round(mean_motion, 6),
                        round(float(area_frac), 6),
                        round(u, 6),
                        round(v, 6),
                        *[round(float(x), 6) for x in rgb_mean.tolist()],
                        *hue,
                    ],
                    "presence": presence,
                    "uv": [u, v],
                    "spread": spread,
                    "bearing": [float(az), float(el)],
                    "relative": _bearing_to_relative(az, el, rng),
                    "retina_contrast": mean_contrast,
                    "local_motion": mean_motion,
                    "motion": _motion_vector(delta, [u, v]),
                    "flow": _motion_vector(delta, [u, v]),
                    "mask_entropy": 0.0,
                    "kind_hint": "object",
                    "property_evidence": prop_ev,
                    "proposal_source": "retinotopic_bootstrap",
                }
            )
    comps.sort(key=lambda p: float(p.get("presence", 0.0)), reverse=True)
    return comps[:max_count]


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
        bootstrap = _retinotopic_bootstrap_proposals(
            gray,
            rgb,
            contrast,
            delta,
            proposals,
            max_count=max(0, C.perception_candidate_capacity() - len(proposals)),
        )
        proposals = [dict(p) for p in proposals] + bootstrap

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
            local_contrast = max(_sample(contrast, uv), float(p.get("retina_contrast", 0.0) or 0.0))
            local_motion = max(_sample(delta, uv), float(p.get("local_motion", 0.0) or 0.0))
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
        diag["bootstrap_proposal_count"] = len(bootstrap)
        diag["candidate_count"] = len(enriched)
        diag["candidate_capacity"] = C.perception_candidate_capacity()
        self.state.prev_gray = gray
        self.state.prev_timestamp = timestamp
        self.state.prev_spread_by_idx = next_spreads
        self.last_retinotopic_map = ret
        self.last_diagnostics = diag
        return enriched, diag, ret
