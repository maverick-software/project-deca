"""Anonymous object files and discovery-health gates.

This module is deliberately label-free. It can consume scaffolded/training
signals upstream, but the records emitted into working memory are anonymous
``object`` / ``stuff`` / ``body_part_candidate`` perceptual units only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


COLLAPSE_MIN_OBJECTS = 3
CENTROID_COLLAPSE_THRESHOLD = 0.035
APPEARANCE_COLLAPSE_THRESHOLD = 0.985
STUFF_SPREAD_THRESHOLD = 0.34
LOW_CONFIDENCE_THRESHOLD = 0.2
PROVISIONAL_CONFIDENCE_FLOOR = 0.05
HEALTH_STATES = ("healthy", "low_confidence", "collapsed", "no_objects", "teacher_only", "stale_frame")
FORBIDDEN_PROPERTY_TOKENS = (
    "label",
    "class",
    "kind_name",
    "food",
    "water",
    "floor",
    "hand",
    "wall",
    "building",
    "ball",
    "bear",
)


@dataclass(frozen=True)
class ObjectFile:
    object_id: str | None
    idx: int
    centroid_uv: list[float] | None
    relative: list[float] | None
    bearing: list[float] | None
    appearance: list[float] | None
    motion: list[float] | None
    depth: float | None
    persistence: float
    agency: float
    kind_hint: str
    confidence: float
    presence: float
    spread: float | None
    mask_entropy: float | None = None
    flow: list[float] | None = None
    local_motion: float | None = None
    retina_contrast: float | None = None
    looming: float | None = None
    property_evidence: dict[str, Any] | None = None
    entity_role: str = "compact_entity"
    provisional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_working_memory_proposal(self) -> dict[str, Any]:
        """Compatibility adapter for ``WorkingMemory.integrate_discovered``."""
        return {
            "object_id": self.object_id,
            "idx": self.idx,
            "appearance": self.appearance,
            "presence": self.presence,
            "confidence": self.confidence,
            "uv": self.centroid_uv,
            "relative": self.relative,
            "bearing": self.bearing,
            "spread": self.spread,
            "motion": self.motion,
            "persistence": self.persistence,
            "agency": self.agency,
            "kind_hint": self.kind_hint,
            "mask_entropy": self.mask_entropy,
            "flow": self.flow,
            "local_motion": self.local_motion,
            "retina_contrast": self.retina_contrast,
            "looming": self.looming,
            "property_evidence": dict(self.property_evidence or {}),
            "entity_role": self.entity_role,
            "provisional": self.provisional,
        }


@dataclass(frozen=True)
class DiscoveryHealth:
    status: str
    collapsed: bool
    reason: str
    active_proposals: int
    object_files: int
    stable_tracked_objects: int
    centroid_spread: float
    appearance_cosine_mean: float | None
    appearance_cosine_max: float | None
    mask_entropy_mean: float | None
    stuff_count: int
    body_candidate_count: int
    looming_count: int
    flow_confidence: float
    low_confidence_count: int
    ltm_write: str = "not_evaluated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_ltm_write(self, status: str) -> "DiscoveryHealth":
        data = self.to_dict()
        data["ltm_write"] = status
        return DiscoveryHealth(**data)


def _as_float_list(value: Any, n: int | None = None) -> list[float] | None:
    if not isinstance(value, list):
        return None
    if n is not None and len(value) < n:
        return None
    try:
        vals = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return vals[:n] if n is not None else vals


def _depth_from_relative(relative: list[float] | None) -> float | None:
    if not relative or len(relative) < 3:
        return None
    return float(math.sqrt(sum(float(x) * float(x) for x in relative[:3])))


def _kind_hint(spread: float | None, confidence: float) -> str:
    if spread is not None and spread >= STUFF_SPREAD_THRESHOLD:
        return "stuff"
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "stuff"
    return "object"


def _entity_role(kind_hint: str, spread: float | None, local_motion: float | None = None) -> str:
    if kind_hint == "body_part_candidate":
        return "body_coupled_entity"
    if kind_hint == "stuff":
        return "extended_entity"
    if spread is not None and spread >= STUFF_SPREAD_THRESHOLD:
        return "extended_entity"
    if local_motion is not None and abs(float(local_motion)) > 0.12:
        return "compact_entity"
    return "compact_entity"


def _clean_property_evidence(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        key = str(k).strip()
        low = key.lower()
        if not key or any(tok in low for tok in FORBIDDEN_PROPERTY_TOKENS):
            continue
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[key] = float(v)
        elif isinstance(v, list):
            vals: list[float] = []
            ok = True
            for x in v[:32]:
                try:
                    fx = float(x)
                except (TypeError, ValueError):
                    ok = False
                    break
                if not math.isfinite(fx):
                    ok = False
                    break
                vals.append(fx)
            if ok and vals:
                out[key] = vals
    return out


def _property_evidence_from_proposal(
    p: dict[str, Any],
    *,
    spread: float | None,
    depth: float | None,
    local_motion: float | None,
    retina_contrast: float | None,
    looming: float | None,
    bearing: list[float] | None,
    confidence: float,
) -> dict[str, Any]:
    """Build anonymous primitive evidence from a proposal.

    This intentionally contains only perceptual/interoceptive numeric evidence.
    Semantic labels, object names, and simulator kinds are ignored.
    """
    ev: dict[str, Any] = _clean_property_evidence(p.get("property_evidence"))
    if spread is not None:
        area = max(0.0, min(1.0, float(spread) * float(spread) * 4.0))
        ev.setdefault("area", area)
        ev.setdefault("size_proxy", max(0.0, min(1.0, float(spread))))
        # Roundness/compactness are weak priors from the available region extent.
        ev.setdefault("compactness", max(0.0, min(1.0, 1.0 - abs(float(spread) - 0.16) / 0.34)))
        ev.setdefault("roundness", max(0.0, min(1.0, 1.0 - abs(float(spread) - 0.12) / 0.30)))
    if depth is not None:
        ev.setdefault("depth", float(depth))
    if bearing is not None:
        ev.setdefault("bearing_azimuth", float(bearing[0]))
        ev.setdefault("bearing_elevation", float(bearing[1]))
    if retina_contrast is not None:
        ev.setdefault("edge_strength", max(0.0, min(1.0, float(retina_contrast))))
        ev.setdefault("brightness_contrast", max(0.0, min(1.0, float(retina_contrast))))
    if local_motion is not None:
        lm = max(0.0, min(1.0, float(local_motion)))
        ev.setdefault("local_motion", lm)
        ev.setdefault("static_confidence", max(0.0, min(1.0, 1.0 - 3.0 * lm)))
    if looming is not None:
        ev.setdefault("looming", max(-1.0, min(1.0, float(looming))))
    ev.setdefault("perceptual_confidence", max(0.0, min(1.0, float(confidence))))
    return ev


def object_files_from_proposals(proposals: list[dict[str, Any]]) -> list[ObjectFile]:
    """Convert raw slot proposals into anonymous object files.

    Large near-uniform regions are marked as ``stuff`` so floor/background-like
    proposals do not poison object memory as foreground entities.
    """
    out: list[ObjectFile] = []
    for p in proposals:
        uv = _as_float_list(p.get("uv"), 2)
        rel = _as_float_list(p.get("relative"), 3)
        bearing = _as_float_list(p.get("bearing"), 2)
        appearance = _as_float_list(p.get("appearance"))
        try:
            presence = float(p.get("presence", 0.0) or 0.0)
        except (TypeError, ValueError):
            presence = 0.0
        try:
            spread = float(p.get("spread")) if p.get("spread") is not None else None
        except (TypeError, ValueError):
            spread = None
        try:
            entropy = float(p.get("mask_entropy")) if p.get("mask_entropy") is not None else None
        except (TypeError, ValueError):
            entropy = None
        flow = _as_float_list(p.get("flow"), 2)
        try:
            local_motion = float(p.get("local_motion")) if p.get("local_motion") is not None else None
        except (TypeError, ValueError):
            local_motion = None
        try:
            retina_contrast = float(p.get("retina_contrast")) if p.get("retina_contrast") is not None else None
        except (TypeError, ValueError):
            retina_contrast = None
        try:
            looming = float(p.get("looming")) if p.get("looming") is not None else None
        except (TypeError, ValueError):
            looming = None
        stuff_penalty = 0.75 if spread is not None and spread >= STUFF_SPREAD_THRESHOLD else 0.0
        confidence = max(0.0, min(1.0, presence * (1.0 - stuff_penalty)))
        kind = str(p.get("kind_hint") or _kind_hint(spread, confidence))
        if kind == "stuff":
            confidence = min(confidence, 0.19)
        role = _entity_role(kind, spread, local_motion)
        provisional = bool(confidence < LOW_CONFIDENCE_THRESHOLD)
        depth = _depth_from_relative(rel)
        prop_ev = _property_evidence_from_proposal(
            p,
            spread=spread,
            depth=depth,
            local_motion=local_motion,
            retina_contrast=retina_contrast,
            looming=looming,
            bearing=bearing,
            confidence=confidence,
        )
        out.append(
            ObjectFile(
                object_id=p.get("object_id") if isinstance(p.get("object_id"), str) else None,
                idx=int(p.get("idx", -1) or -1),
                centroid_uv=uv,
                relative=rel,
                bearing=bearing,
                appearance=appearance,
                motion=_as_float_list(p.get("motion"), 2),
                depth=depth,
                persistence=float(p.get("persistence", confidence) or 0.0),
                agency=float(p.get("agency", 0.0) or 0.0),
                kind_hint=kind,
                confidence=confidence,
                presence=max(0.0, min(1.0, presence)),
                spread=spread,
                mask_entropy=entropy,
                flow=flow,
                local_motion=local_motion,
                retina_contrast=retina_contrast,
                looming=looming,
                property_evidence=prop_ev,
                entity_role=role,
                provisional=provisional,
            )
        )
    return out


def _centroid_spread(files: list[ObjectFile]) -> float:
    pts = np.asarray([f.centroid_uv for f in files if f.centroid_uv is not None], dtype=np.float32)
    if pts.shape[0] < 2:
        return 0.0
    mu = pts.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(pts - mu, axis=1).mean())


def _appearance_cosines(files: list[ObjectFile]) -> tuple[float | None, float | None]:
    vecs = [np.asarray(f.appearance, dtype=np.float32).reshape(-1) for f in files if f.appearance]
    vals: list[float] = []
    for i in range(len(vecs)):
        ai = vecs[i]
        nai = float(np.linalg.norm(ai))
        if nai < 1e-8:
            continue
        for j in range(i + 1, len(vecs)):
            bj = vecs[j]
            nbj = float(np.linalg.norm(bj))
            if bj.size != ai.size or nbj < 1e-8:
                continue
            vals.append(float(np.dot(ai, bj) / (nai * nbj)))
    if not vals:
        return None, None
    return float(sum(vals) / len(vals)), float(max(vals))


def evaluate_discovery_health(
    files: list[ObjectFile],
    *,
    tracked_count: int = 0,
    stable_tracked_objects: int = 0,
) -> DiscoveryHealth:
    entity_like = [f for f in files if f.confidence >= PROVISIONAL_CONFIDENCE_FLOOR]
    object_like = [
        f for f in files if f.kind_hint != "stuff" and f.confidence >= LOW_CONFIDENCE_THRESHOLD
    ]
    spread = _centroid_spread(object_like)
    cos_mean, cos_max = _appearance_cosines(object_like)
    entropies = [float(f.mask_entropy) for f in files if f.mask_entropy is not None]
    stuff = sum(1 for f in files if f.kind_hint == "stuff")
    body_candidates = sum(1 for f in files if f.kind_hint == "body_part_candidate")
    looming_count = sum(1 for f in files if (f.looming or 0.0) > 0.04)
    flow_vals = [float(f.local_motion) for f in files if f.local_motion is not None]
    flow_conf = max(flow_vals) - (sum(flow_vals) / len(flow_vals)) if len(flow_vals) >= 2 else 0.0
    low_conf = sum(1 for f in files if f.confidence < LOW_CONFIDENCE_THRESHOLD)
    collapsed = False
    reason = "healthy"
    if not files:
        reason = "skipped_no_objects"
    elif not entity_like:
        reason = "skipped_low_confidence"
    elif not object_like:
        reason = "recorded_provisional_evidence"
    elif len(object_like) >= COLLAPSE_MIN_OBJECTS and spread < CENTROID_COLLAPSE_THRESHOLD:
        collapsed = True
        reason = "skipped_perception_collapsed"
    elif (
        len(object_like) >= COLLAPSE_MIN_OBJECTS
        and cos_mean is not None
        and cos_mean >= APPEARANCE_COLLAPSE_THRESHOLD
    ):
        collapsed = True
        reason = "skipped_perception_collapsed"
    if collapsed:
        status = "collapsed"
    elif not files:
        status = "no_objects"
    elif reason == "healthy":
        status = "healthy"
    elif reason == "recorded_provisional_evidence":
        status = "low_confidence"
    else:
        status = "low_confidence"
    return DiscoveryHealth(
        status=status,
        collapsed=collapsed,
        reason=reason,
        active_proposals=len(files),
        object_files=len(object_like),
        stable_tracked_objects=int(stable_tracked_objects),
        centroid_spread=round(spread, 6),
        appearance_cosine_mean=round(cos_mean, 6) if cos_mean is not None else None,
        appearance_cosine_max=round(cos_max, 6) if cos_max is not None else None,
        mask_entropy_mean=round(sum(entropies) / len(entropies), 6) if entropies else None,
        stuff_count=stuff,
        body_candidate_count=body_candidates,
        looming_count=looming_count,
        flow_confidence=round(max(0.0, min(1.0, flow_conf)), 6),
        low_confidence_count=low_conf,
        ltm_write="not_evaluated",
    )
