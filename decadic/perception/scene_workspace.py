"""Persistent anonymous scene workspace.

This module sits between object-file perception and the Decadic cognition loop.
It maintains a live egocentric scene model from anonymous perceptual evidence:
objects, stuff/background regions, body-part candidates, spatial relations, and
attention focus candidates. It deliberately rejects semantic labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from decadic import config as C

FORBIDDEN_SCENE_TOKENS = (
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


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _vec(value: Any, n: int | None = None) -> list[float] | None:
    if not isinstance(value, list):
        return None
    if n is not None and len(value) < n:
        return None
    vals: list[float] = []
    for item in value[:n]:
        x = _finite_float(item, float("nan"))
        if not math.isfinite(x):
            return None
        vals.append(x)
    return vals


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def _clean_property_evidence(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for raw_key, raw_val in raw.items():
        key = str(raw_key).strip()
        low = key.lower()
        if not key or any(tok in low for tok in FORBIDDEN_SCENE_TOKENS):
            continue
        if isinstance(raw_val, (int, float)):
            x = _finite_float(raw_val, float("nan"))
            if math.isfinite(x):
                out[key] = x
        elif isinstance(raw_val, list):
            vals: list[float] = []
            ok = True
            for item in raw_val[:32]:
                x = _finite_float(item, float("nan"))
                if not math.isfinite(x):
                    ok = False
                    break
                vals.append(x)
            if ok and vals:
                out[key] = vals
    return out


def _entity_role(kind_hint: str, spread: Any = None) -> str:
    if kind_hint == "body_part_candidate":
        return "body_coupled_entity"
    if kind_hint == "stuff":
        return "extended_entity"
    try:
        sp = float(spread)
    except (TypeError, ValueError):
        sp = 0.0
    if sp >= 0.34:
        return "extended_entity"
    return "compact_entity"


def _blend(old: list[float] | None, new: list[float] | None, alpha: float) -> list[float] | None:
    if new is None:
        return old
    if old is None or len(old) != len(new):
        return list(new)
    return [(1.0 - alpha) * a + alpha * b for a, b in zip(old, new)]


def _blend_properties(
    old: dict[str, Any], new: dict[str, Any], alpha: float
) -> dict[str, Any]:
    out = dict(old)
    for key, value in new.items():
        prev = out.get(key)
        if isinstance(prev, (int, float)) and isinstance(value, (int, float)):
            out[key] = (1.0 - alpha) * float(prev) + alpha * float(value)
        elif (
            isinstance(prev, list)
            and isinstance(value, list)
            and len(prev) == len(value)
        ):
            out[key] = [(1.0 - alpha) * float(a) + alpha * float(b) for a, b in zip(prev, value)]
        else:
            out[key] = value
    return out


def _property_signal(evidence: dict[str, Any], key: str) -> float:
    value = evidence.get(key)
    if isinstance(value, list):
        vals = [_finite_float(v, 0.0) for v in value[:8]]
        return max(0.0, min(1.0, float(sum(vals) / max(1, len(vals)))))
    return max(0.0, min(1.0, _finite_float(value, 0.0)))


def _attention_context(
    homeostasis: Any | None = None,
    state_bus: Any | None = None,
) -> dict[str, Any]:
    def deficit(name: str) -> float:
        if homeostasis is None:
            return 0.0
        return max(0.0, min(1.0, (100.0 - _finite_float(getattr(homeostasis, name, 100.0), 100.0)) / 100.0))

    return {
        "energy_deficit": deficit("energy"),
        "hydration_deficit": deficit("hydration"),
        "integrity_deficit": deficit("integrity"),
        "pain": max(0.0, min(1.0, _finite_float(getattr(state_bus, "pain_scalar", 0.0), 0.0))),
        "pleasure": max(0.0, min(1.0, _finite_float(getattr(state_bus, "pleasure_scalar", 0.0), 0.0))),
        "priority": str(getattr(state_bus, "priority_label", "explore") or "explore"),
    }


@dataclass
class SceneEntity:
    entity_id: str
    object_id: str | None = None
    kind_hint: str = "object"
    visible: bool = True
    occluded: bool = False
    occlusion_age: int = 0
    centroid_uv: list[float] | None = None
    relative: list[float] | None = None
    depth: float | None = None
    motion: list[float] | None = None
    appearance: list[float] | None = None
    confidence: float = 0.0
    persistence: float = 0.0
    salience: float = 0.0
    attention_score: float = 0.0
    attention_reasons: dict[str, float] = field(default_factory=dict)
    drive_match: dict[str, float] = field(default_factory=dict)
    agency: float = 0.0
    looming: float = 0.0
    local_motion: float = 0.0
    retina_contrast: float = 0.0
    predicted_centroid_uv: list[float] | None = None
    predicted_relative: list[float] | None = None
    prediction_visibility: float | None = None
    prediction_uncertainty: float | None = None
    prediction_error: float | None = None
    property_evidence: dict[str, Any] = field(default_factory=dict)
    first_cycle: int = 0
    last_seen_cycle: int = 0
    seen_count: int = 0
    uv_history: list[list[float]] = field(default_factory=list)
    relative_history: list[list[float]] = field(default_factory=list)
    entity_role: str = "compact_entity"
    provisional: bool = True

    def predicted_uv(self) -> list[float] | None:
        if self.predicted_centroid_uv is not None:
            return list(self.predicted_centroid_uv)
        if self.centroid_uv is None:
            return None
        if len(self.uv_history) < 2:
            return list(self.centroid_uv)
        a, b = self.uv_history[-2], self.uv_history[-1]
        return [2.0 * b[0] - a[0], 2.0 * b[1] - a[1]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "object_id": self.object_id,
            "kind_hint": self.kind_hint,
            "visible": self.visible,
            "occluded": self.occluded,
            "occlusion_age": self.occlusion_age,
            "centroid_uv": list(self.centroid_uv) if self.centroid_uv else None,
            "relative": list(self.relative) if self.relative else None,
            "depth": self.depth,
            "motion": list(self.motion) if self.motion else None,
            "confidence": round(float(self.confidence), 4),
            "persistence": round(float(self.persistence), 4),
            "salience": round(float(self.salience), 4),
            "attention_score": round(float(self.attention_score), 4),
            "attention_reasons": {k: round(float(v), 4) for k, v in self.attention_reasons.items()},
            "drive_match": {k: round(float(v), 4) for k, v in self.drive_match.items()},
            "agency": round(float(self.agency), 4),
            "looming": round(float(self.looming), 4),
            "local_motion": round(float(self.local_motion), 4),
            "retina_contrast": round(float(self.retina_contrast), 4),
            "predicted_centroid_uv": list(self.predicted_centroid_uv) if self.predicted_centroid_uv else None,
            "predicted_relative": list(self.predicted_relative) if self.predicted_relative else None,
            "prediction_visibility": (
                round(float(self.prediction_visibility), 4)
                if self.prediction_visibility is not None
                else None
            ),
            "prediction_uncertainty": (
                round(float(self.prediction_uncertainty), 4)
                if self.prediction_uncertainty is not None
                else None
            ),
            "prediction_error": (
                round(float(self.prediction_error), 4)
                if self.prediction_error is not None
                else None
            ),
            "property_evidence": dict(self.property_evidence),
            "first_cycle": self.first_cycle,
            "last_seen_cycle": self.last_seen_cycle,
            "seen_count": self.seen_count,
            "entity_role": self.entity_role,
            "provisional": bool(self.provisional),
        }


@dataclass
class SceneRelation:
    src: str
    dst: str
    kind: str
    confidence: float
    last_cycle: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind,
            "confidence": round(float(self.confidence), 4),
            "last_cycle": self.last_cycle,
        }


class SceneWorkspace:
    """Anonymous egocentric scene graph with persistence and focus selection."""

    def __init__(self, *, ttl_cycles: int = 12, relation_enabled: bool = True) -> None:
        self.ttl_cycles = max(1, int(ttl_cycles))
        self.relation_enabled = bool(relation_enabled)
        self.cycle = 0
        self._next_id = 0
        self.entities: dict[str, SceneEntity] = {}
        self.relations: dict[tuple[str, str, str], SceneRelation] = {}
        self.focus_ids: list[str] = []
        self.prediction_error: float | None = None
        self.last_update: dict[str, Any] = {}
        self.attention_context: dict[str, Any] = _attention_context()
        self.attention_weights: dict[str, float] = C.scene_attention_weights()

    def update(
        self,
        object_files: list[dict[str, Any]],
        *,
        focus_capacity: int = 7,
        entity_capacity: int | None = None,
        attention_context: dict[str, Any] | None = None,
        attention_weights: dict[str, float] | None = None,
        predictions: list[dict[str, Any]] | None = None,
        prediction_match_threshold: float = 0.35,
    ) -> None:
        self.cycle += 1
        self.attention_context = dict(attention_context or self.attention_context or _attention_context())
        self.attention_weights = dict(attention_weights or self.attention_weights or C.scene_attention_weights())
        self._apply_predictions(predictions or [])
        previous_predictions = {
            eid: ent.predicted_uv()
            for eid, ent in self.entities.items()
            if ent.predicted_uv() is not None
        }
        for ent in self.entities.values():
            ent.visible = False
            ent.occluded = True
            ent.occlusion_age += 1
            ent.persistence = max(0.0, ent.persistence * 0.85)
            ent.salience = max(0.0, ent.salience * 0.75)

        errors: list[float] = []
        seen: set[str] = set()
        matched_existing = 0
        prediction_assisted = 0
        duplicate_prevented = 0
        for raw in object_files:
            if not isinstance(raw, dict):
                continue
            kind_hint = str(raw.get("kind_hint", "object"))
            conf = _finite_float(raw.get("confidence", raw.get("presence", 0.0)))
            if conf < 0.05:
                continue
            matched, used_prediction, blocked = self._match(
                raw,
                used=seen,
                prediction_match_threshold=prediction_match_threshold,
            )
            duplicate_prevented += blocked
            if matched is None:
                matched = self._new_entity_id()
                self.entities[matched] = SceneEntity(
                    entity_id=matched,
                    first_cycle=self.cycle,
                )
            else:
                matched_existing += 1
                if used_prediction:
                    prediction_assisted += 1
            ent = self.entities[matched]
            pred = previous_predictions.get(matched)
            uv = _vec(raw.get("centroid_uv") or raw.get("uv"), 2)
            if pred is not None and uv is not None:
                err = math.hypot(uv[0] - pred[0], uv[1] - pred[1])
                errors.append(err)
                ent.prediction_error = err
            self._refresh(ent, raw, kind_hint=kind_hint, confidence=conf)
            seen.add(matched)

        self._evict_expired()
        self._enforce_capacity(entity_capacity)
        self._rebuild_relations()
        self.focus_ids = self.select_focus(focus_capacity)
        self.prediction_error = float(sum(errors) / len(errors)) if errors else None
        self.last_update = {
            "cycle": self.cycle,
            "visible": len(seen),
            "entity_count": len(self.entities),
            "prediction_error": self.prediction_error,
            "prediction_count": len(predictions or []),
            "reidentified_count": matched_existing,
            "prediction_assisted_count": prediction_assisted,
            "duplicate_prevention_count": duplicate_prevented,
            "candidate_count": len(object_files),
        }

    def _apply_predictions(self, predictions: list[dict[str, Any]]) -> None:
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            eid = str(pred.get("entity_id", ""))
            ent = self.entities.get(eid)
            if ent is None:
                continue
            ent.predicted_centroid_uv = _vec(pred.get("centroid_uv"), 2)
            ent.predicted_relative = _vec(pred.get("relative"), 3)
            ent.prediction_visibility = _finite_float(pred.get("visibility"), 0.5)
            ent.prediction_uncertainty = _finite_float(pred.get("uncertainty"), 1.0)

    def _new_entity_id(self) -> str:
        eid = f"scene-{self._next_id:05d}"
        self._next_id += 1
        return eid

    def _match(
        self,
        raw: dict[str, Any],
        *,
        used: set[str],
        prediction_match_threshold: float,
    ) -> tuple[str | None, bool, int]:
        oid = raw.get("object_id")
        if oid:
            oid_s = str(oid)
            for eid, ent in self.entities.items():
                if eid not in used and ent.object_id == oid_s:
                    return eid, False, 0
        uv = _vec(raw.get("centroid_uv") or raw.get("uv"), 2)
        app = _vec(raw.get("appearance"))
        best_id: str | None = None
        best = 0.45
        best_used_prediction = False
        duplicate_blocked = 0
        for eid, ent in self.entities.items():
            if eid in used:
                duplicate_blocked += 1
                continue
            app_score = 0.5 * (_cosine(app, ent.appearance) + 1.0)
            pos_score = 0.0
            pred = ent.predicted_uv()
            if uv is not None and pred is not None:
                d = math.hypot(uv[0] - pred[0], uv[1] - pred[1])
                pos_score = max(0.0, 1.0 - d / 0.45)
            score = 0.6 * app_score + 0.4 * pos_score
            pred_ok = ent.predicted_centroid_uv is not None and pos_score >= prediction_match_threshold
            if score >= best:
                best = score
                best_id = eid
                best_used_prediction = bool(pred_ok)
        return best_id, best_used_prediction, duplicate_blocked

    def _refresh(
        self,
        ent: SceneEntity,
        raw: dict[str, Any],
        *,
        kind_hint: str,
        confidence: float,
    ) -> None:
        uv = _vec(raw.get("centroid_uv") or raw.get("uv"), 2)
        rel = _vec(raw.get("relative"), 3)
        app = _vec(raw.get("appearance"))
        motion = _vec(raw.get("motion") or raw.get("flow"), 2)
        depth = raw.get("depth")
        if depth is None and rel is not None:
            depth = math.sqrt(sum(x * x for x in rel[:3]))
        ent.object_id = str(raw.get("object_id")) if raw.get("object_id") else ent.object_id
        ent.kind_hint = kind_hint
        ent.entity_role = str(raw.get("entity_role") or _entity_role(kind_hint, raw.get("spread")))
        ent.provisional = confidence < 0.2
        ent.visible = True
        ent.occluded = False
        ent.occlusion_age = 0
        ent.centroid_uv = uv or ent.centroid_uv
        ent.relative = rel or ent.relative
        ent.depth = _finite_float(depth, ent.depth if ent.depth is not None else 0.0)
        ent.motion = motion or ent.motion
        ent.appearance = _blend(ent.appearance, app, 0.35)
        ent.confidence = confidence
        ent.persistence = min(1.0, max(ent.persistence, confidence))
        ent.agency = 0.8 * ent.agency + 0.2 * _finite_float(raw.get("agency"), 0.0)
        ent.looming = _finite_float(raw.get("looming"), 0.0)
        ent.local_motion = _finite_float(raw.get("local_motion"), 0.0)
        ent.retina_contrast = _finite_float(raw.get("retina_contrast"), 0.0)
        ent.last_seen_cycle = self.cycle
        ent.seen_count += 1
        if uv is not None:
            ent.uv_history.append(list(uv))
            ent.uv_history = ent.uv_history[-8:]
        if rel is not None:
            ent.relative_history.append(list(rel))
            ent.relative_history = ent.relative_history[-8:]
        ev = _clean_property_evidence(raw.get("property_evidence"))
        ent.property_evidence = _blend_properties(ent.property_evidence, ev, 0.25)
        ent.salience = self._salience(ent)
        ent.attention_score = ent.salience

    def _salience(self, ent: SceneEntity) -> float:
        score, reasons, drive_match = self._attention_score(ent)
        ent.attention_reasons = reasons
        ent.drive_match = drive_match
        return score

    def _attention_score(self, ent: SceneEntity) -> tuple[float, dict[str, float], dict[str, float]]:
        if ent.kind_hint == "stuff":
            base = 0.15 * ent.confidence
        else:
            base = 0.35 * ent.confidence + 0.20 * ent.persistence
        weights = self.attention_weights or {}
        novelty = (0.20 if ent.seen_count <= 2 else 0.0) * float(weights.get("novelty", 1.0))
        motion = min(0.20, abs(ent.local_motion) * 2.0)
        looming = min(0.25, max(0.0, ent.looming))
        surprise = min(0.20, max(0.0, ent.prediction_error or 0.0))
        agency = min(0.15, max(0.0, ent.agency) * 0.15)
        proximity = 0.0
        if ent.depth is not None and ent.depth > 0:
            proximity = min(0.15, 0.15 / max(1.0, ent.depth))
        drive_match: dict[str, float] = {}
        relief = 0.0
        threat = 0.0
        curiosity = 0.0
        if C.drive_attention_enabled():
            ctx = self.attention_context or {}
            energy_def = max(0.0, min(1.0, _finite_float(ctx.get("energy_deficit"), 0.0)))
            hydration_def = max(0.0, min(1.0, _finite_float(ctx.get("hydration_deficit"), 0.0)))
            integrity_def = max(0.0, min(1.0, _finite_float(ctx.get("integrity_deficit"), 0.0)))
            pain = max(0.0, min(1.0, _finite_float(ctx.get("pain"), 0.0)))
            ev = ent.property_evidence or {}
            energy_relief = _property_signal(ev, "predicts_energy_relief")
            hydration_relief = _property_signal(ev, "predicts_hydration_relief")
            pain_risk = max(
                _property_signal(ev, "predicts_pain"),
                _property_signal(ev, "predicts_integrity_loss"),
            )
            drive_w = float(weights.get("drive", 1.0))
            relief = drive_w * float(weights.get("relief", 1.0)) * (
                energy_def * energy_relief + hydration_def * hydration_relief
            )
            threat = drive_w * float(weights.get("threat", 1.0)) * max(integrity_def, pain) * pain_risk
            if str(ctx.get("priority", "")).lower() == "investigate":
                curiosity = 0.12 * max(novelty, surprise, min(0.2, _finite_float(ent.prediction_uncertainty, 0.0)))
            drive_match = {
                "energy_deficit": energy_def,
                "hydration_deficit": hydration_def,
                "integrity_deficit": integrity_def,
                "energy_relief": energy_def * energy_relief,
                "hydration_relief": hydration_def * hydration_relief,
                "threat": max(integrity_def, pain) * pain_risk,
                "curiosity": curiosity,
            }
        reasons = {
            "base": base,
            "novelty": novelty,
            "motion": motion,
            "looming": looming,
            "surprise": surprise,
            "agency": agency,
            "proximity": proximity,
            "relief": relief,
            "threat": threat,
            "curiosity": curiosity,
        }
        score = sum(reasons.values())
        return max(0.0, min(1.0, score)), reasons, drive_match

    def _evict_expired(self) -> None:
        expired = [
            eid
            for eid, ent in self.entities.items()
            if ent.occlusion_age > self.ttl_cycles or ent.persistence <= 0.01
        ]
        for eid in expired:
            del self.entities[eid]
        self.relations = {
            k: r
            for k, r in self.relations.items()
            if r.src in self.entities and r.dst in self.entities
        }

    def _enforce_capacity(self, capacity: int | None) -> None:
        cap = max(1, int(capacity or C.scene_entity_capacity()))
        if len(self.entities) <= cap:
            return
        ranked = sorted(
            self.entities.values(),
            key=lambda e: (
                e.visible,
                e.attention_score,
                e.persistence,
                e.confidence,
                e.seen_count,
                -e.occlusion_age,
            ),
            reverse=True,
        )
        keep = {e.entity_id for e in ranked[:cap]}
        self.entities = {eid: ent for eid, ent in self.entities.items() if eid in keep}
        self.relations = {
            k: r
            for k, r in self.relations.items()
            if r.src in self.entities and r.dst in self.entities
        }

    def _rebuild_relations(self) -> None:
        if not self.relation_enabled:
            self.relations.clear()
            return
        visible = [e for e in self.entities.values() if e.visible]
        new_rel: dict[tuple[str, str, str], SceneRelation] = {}
        for i, a in enumerate(visible):
            for b in visible[i + 1 :]:
                for kind, conf in self._pair_relations(a, b):
                    src, dst = (a.entity_id, b.entity_id)
                    key = (src, dst, kind)
                    new_rel[key] = SceneRelation(src, dst, kind, conf, self.cycle)
        self.relations = new_rel

    def _pair_relations(self, a: SceneEntity, b: SceneEntity) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = [("co_visible", min(a.confidence, b.confidence))]
        if a.relative is not None and b.relative is not None:
            d = math.sqrt(sum((x - y) ** 2 for x, y in zip(a.relative[:3], b.relative[:3])))
            if d < 1.5:
                out.append(("near", max(0.0, 1.0 - d / 1.5)))
            elif d > 4.0:
                out.append(("far", min(1.0, d / 8.0)))
        if a.centroid_uv is not None and b.centroid_uv is not None:
            dx = a.centroid_uv[0] - b.centroid_uv[0]
            dy = a.centroid_uv[1] - b.centroid_uv[1]
            if abs(dx) > 0.05:
                out.append(("left_of" if dx < 0 else "right_of", min(1.0, abs(dx) * 2.0)))
            if abs(dy) > 0.05:
                out.append(("above" if dy < 0 else "below", min(1.0, abs(dy) * 2.0)))
        return out

    def select_focus(self, capacity: int) -> list[str]:
        candidates = [e for e in self.entities.values() if e.attention_score > 0.0 or e.salience > 0.0]
        candidates.sort(key=lambda e: (e.attention_score, e.salience, e.confidence, e.seen_count), reverse=True)
        return [e.entity_id for e in candidates[: max(1, int(capacity))]]

    def focus_entities(self) -> list[SceneEntity]:
        return [self.entities[eid] for eid in self.focus_ids if eid in self.entities]

    def focus_proposals(self) -> list[dict[str, Any]]:
        proposals: list[dict[str, Any]] = []
        for ent in self.focus_entities():
            proposals.append(
                {
                    "object_id": ent.object_id or ent.entity_id,
                    "idx": -1,
                    "appearance": list(ent.appearance) if ent.appearance else None,
                    "presence": float(ent.persistence),
                    "confidence": float(ent.confidence),
                    "uv": list(ent.centroid_uv) if ent.centroid_uv else None,
                    "relative": list(ent.relative) if ent.relative else None,
                    "motion": list(ent.motion) if ent.motion else None,
                    "persistence": float(ent.persistence),
                    "agency": float(ent.agency),
                    "kind_hint": ent.kind_hint,
                    "entity_role": ent.entity_role,
                    "provisional": bool(ent.provisional),
                    "flow": list(ent.motion) if ent.motion else None,
                    "local_motion": float(ent.local_motion),
                    "retina_contrast": float(ent.retina_contrast),
                    "looming": float(ent.looming),
                    "prediction_error": float(ent.prediction_error or 0.0),
                    "prediction_uncertainty": float(ent.prediction_uncertainty or 0.0),
                    "occlusion_age": int(ent.occlusion_age),
                    "surprise": float(ent.prediction_error or 0.0),
                    "attention_score": float(ent.attention_score),
                    "attention_reasons": dict(ent.attention_reasons),
                    "drive_match": dict(ent.drive_match),
                    "property_evidence": dict(ent.property_evidence),
                    "scene_entity_id": ent.entity_id,
                }
            )
        return proposals

    def relation_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.relations.values()]

    def snapshot(self) -> dict[str, Any]:
        visible = [e for e in self.entities.values() if e.visible]
        occluded = [e for e in self.entities.values() if e.occluded]
        stable = [e for e in self.entities.values() if e.seen_count >= 2]
        foreground = [e for e in self.entities.values() if e.kind_hint != "stuff"]
        duplicate_identity_count = max(0, len(foreground) - len({e.object_id or e.entity_id for e in foreground}))
        unstable = [
            e for e in foreground
            if e.prediction_error is not None and e.prediction_error > 0.35
        ]
        return {
            "cycle": self.cycle,
            "entity_count": len(self.entities),
            "visible_count": len(visible),
            "occluded_count": len(occluded),
            "stable_count": len(stable),
            "stuff_count": sum(1 for e in self.entities.values() if e.kind_hint == "stuff"),
            "body_candidate_count": sum(1 for e in self.entities.values() if e.kind_hint == "body_part_candidate"),
            "duplicate_identity_count": duplicate_identity_count,
            "focus_ids": list(self.focus_ids),
            "prediction_error": self.prediction_error,
            "prediction_unstable_count": len(unstable),
            "prediction_count": int(self.last_update.get("prediction_count", 0) or 0),
            "reidentified_count": int(self.last_update.get("reidentified_count", 0) or 0),
            "prediction_assisted_count": int(self.last_update.get("prediction_assisted_count", 0) or 0),
            "duplicate_prevention_count": int(self.last_update.get("duplicate_prevention_count", 0) or 0),
            "candidate_count": int(self.last_update.get("candidate_count", 0) or 0),
            "focus_capacity": len(self.focus_ids),
            "active_drive_deficits": {
                "energy": round(float((self.attention_context or {}).get("energy_deficit", 0.0) or 0.0), 4),
                "hydration": round(float((self.attention_context or {}).get("hydration_deficit", 0.0) or 0.0), 4),
                "integrity": round(float((self.attention_context or {}).get("integrity_deficit", 0.0) or 0.0), 4),
                "pain": round(float((self.attention_context or {}).get("pain", 0.0) or 0.0), 4),
                "priority": str((self.attention_context or {}).get("priority", "explore")),
            },
            "attention_top": [
                {
                    "entity_id": e.entity_id,
                    "attention_score": round(float(e.attention_score), 4),
                    "attention_reasons": {k: round(float(v), 4) for k, v in e.attention_reasons.items()},
                    "drive_match": {k: round(float(v), 4) for k, v in e.drive_match.items()},
                }
                for e in sorted(self.entities.values(), key=lambda x: x.attention_score, reverse=True)[:8]
            ],
            "entities": [e.to_dict() for e in sorted(self.entities.values(), key=lambda x: x.salience, reverse=True)],
            "relations": self.relation_dicts(),
        }


def attention_context_from_state(homeostasis: Any | None = None, state_bus: Any | None = None) -> dict[str, Any]:
    return _attention_context(homeostasis, state_bus)
