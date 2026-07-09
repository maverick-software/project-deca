"""Bounded, decaying working memory - the substrate for temporal persistence.

The egocentric graph is no longer rebuilt from scratch every observation; instead
entities are written into bounded slots that decay over time. An object that
leaves the field of view persists (object permanence) at falling salience until
it is either re-seen or evicted. This is condition 3 (temporal persistence) and
the integration substrate for condition 4 of the paper's logical layer.
See docs/logical_layer_conformance.md.
"""

from __future__ import annotations

import math
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from decadic.config import (
    DEFAULT_WM_DECAY,
    DEFAULT_WM_MIN_SALIENCE,
    DEFAULT_WM_SCENE_ALPHA,
    DEFAULT_WM_SCENE_BLEND,
    DEFAULT_WORKING_MEMORY_SLOTS,
    entity_entry_confidence_floor,
    entity_precision_eta,
    entity_promotion_precision,
    entity_provisional_decay_boost,
    provisional_entry_enabled,
)
from decadic.state.body_map import BODY_PARTS

# Audio/event salience is transient - it fades faster than spatial salience.
AUDIO_DECAY = 0.6
POS_HISTORY_LEN = 8
SCENE_PREVIEW_BUCKETS = 32

# --- WS5-M0 slot-tensor layout (FROZEN; see docs/ws5_m0_wm_inventory.md) -----
# Fixed-dim per-slot projection so the neural stack can receive slots as a
# (K, D_slot) token matrix instead of a pooled vector. Layout, in order:
#   [ appearance 0:16 | spatial 16:27 | scalars 27:40 ]
# spatial = relative(3, tanh-scaled) + bearing(2, /pi) + uv(2) + motion(2)
#           + heading(sin, cos)
# scalars = salience, tanh(affect), audio, agency, confidence, precision,
#           evidence(cap 8), contradiction, looming, prediction_error,
#           prediction_uncertainty, staleness(cap 32 cycles), in_view
SLOT_TENSOR_APPEARANCE_DIM = 16
SLOT_TENSOR_SPATIAL_DIM = 11
SLOT_TENSOR_SCALAR_DIM = 13
SLOT_TENSOR_DIM = (
    SLOT_TENSOR_APPEARANCE_DIM + SLOT_TENSOR_SPATIAL_DIM + SLOT_TENSOR_SCALAR_DIM
)
SLOT_APPEARANCE_SLICE = slice(0, SLOT_TENSOR_APPEARANCE_DIM)
SLOT_SPATIAL_SLICE = slice(
    SLOT_TENSOR_APPEARANCE_DIM, SLOT_TENSOR_APPEARANCE_DIM + SLOT_TENSOR_SPATIAL_DIM
)
SLOT_SCALAR_SLICE = slice(
    SLOT_TENSOR_APPEARANCE_DIM + SLOT_TENSOR_SPATIAL_DIM, SLOT_TENSOR_DIM
)
SLOT_POS_SCALE = 20.0  # tanh(x / scale): patrol-range positions stay linear-ish
SLOT_STALENESS_HORIZON = 32.0  # cycles-unseen that saturate the staleness scalar
SLOT_EVIDENCE_CAP = 8.0


def _body_part_from_event(ev: dict[str, Any]) -> str | None:
    raw = str(ev.get("source") or ev.get("body_part") or ev.get("sensor") or "").lower()
    if raw.startswith("touch_"):
        raw = raw[len("touch_") :]
    return raw if raw in BODY_PARTS else None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _fin(x: float | None) -> float | None:
    """Snapshot-only: NaN/Inf -> 0 so a transient bad value can't serialize to null
    and crash numeric UI widgets. None passes through unchanged."""
    if x is None:
        return None
    return x if isinstance(x, (int, float)) and math.isfinite(x) else 0.0


def _fin_seq(xs: list[float] | None) -> list[float] | None:
    if xs is None:
        return None
    return [v if isinstance(v, (int, float)) and math.isfinite(v) else 0.0 for v in xs]


@dataclass
class MemorySlot:
    """One remembered entity, refreshed on sight and decayed otherwise."""

    entity_id: str
    kind: str = "entity"
    position: list[float] | None = None
    relative: list[float] | None = None
    salience: float = 1.0
    affective_weight: float = 0.0
    last_seen_cycle: int = 0
    seen_count: int = 1
    audio_intensity: float = 0.0
    last_event: str | None = None
    pos_history: deque[list[float]] = field(
        default_factory=lambda: deque(maxlen=POS_HISTORY_LEN)
    )
    # --- discovered-perception object file (unused in oracle mode) -----------
    # Appearance fingerprint (the slot's latent), used for re-identification.
    appearance: list[float] | None = None
    # WS5-M4.1: the graph entity's STORED appearance when identity-matched --
    # the stable key the slot tensor exposes to the network (the live
    # ``appearance`` above EMA-drifts per sighting; this one anchors identity
    # across occlusion gaps). None -> the slot keys on its own appearance.
    key_appearance: list[float] | None = None
    # Image-space centroid (u, v) in [0,1] and its short history, for predicting
    # where a tracked object should re-appear (constant-velocity association).
    uv: list[float] | None = None
    uv_history: deque[list[float]] = field(
        default_factory=lambda: deque(maxlen=POS_HISTORY_LEN)
    )
    bearing: list[float] | None = None  # (azimuth, elevation) radians
    # Sense of agency: how much this object's motion is explained by efference.
    agency: float = 0.0
    agency_seen: int = 0
    # Anonymous object-file metadata. ``kind_hint`` is perceptual only: no semantic
    # label enters cognition.
    confidence: float = 1.0
    kind_hint: str = "object"
    motion: list[float] | None = None
    local_motion: float = 0.0
    retina_contrast: float = 0.0
    looming: float = 0.0
    prediction_error: float = 0.0
    prediction_uncertainty: float = 0.0
    occlusion_age: int = 0
    property_evidence: dict[str, Any] = field(default_factory=dict)
    scene_entity_id: str | None = None
    entity_role: str = "compact_entity"
    precision: float = 0.0
    provisional: bool = True
    evidence_count: float = 0.0
    contradiction_pressure: float = 0.0
    event_links: list[str] = field(default_factory=list)
    relationship_links: list[str] = field(default_factory=list)

    def heading(self) -> float | None:
        """Planar heading (radians) inferred from remembered position history."""
        if len(self.pos_history) < 2:
            return None
        x0, y0 = self.pos_history[0][0], self.pos_history[0][1]
        x1, y1 = self.pos_history[-1][0], self.pos_history[-1][1]
        dx, dy = x1 - x0, y1 - y0
        if dx * dx + dy * dy < 1e-6:
            return None
        return math.atan2(dy, dx)

    def predicted_uv(self) -> list[float] | None:
        """Constant-velocity prediction of the next image-space centroid."""
        if self.uv is None:
            return None
        if len(self.uv_history) < 2:
            return list(self.uv)
        a = self.uv_history[-2]
        b = self.uv_history[-1]
        return [2.0 * b[0] - a[0], 2.0 * b[1] - a[1]]

    def as_node(self) -> dict[str, Any]:
        node: dict[str, Any] = {
            "id": self.entity_id,
            "role": "entity",
            "kind": self.kind,
            "salience": round(self.salience, 4),
            "last_seen_cycle": self.last_seen_cycle,
        }
        if self.position is not None:
            node["position"] = list(self.position)
        if self.relative is not None:
            node["relative"] = list(self.relative)
        if self.bearing is not None:
            node["bearing"] = [round(float(x), 4) for x in self.bearing]
        if self.agency_seen > 0:
            node["agency"] = round(float(self.agency), 4)
        node["confidence"] = round(float(self.confidence), 4)
        node["kind_hint"] = self.kind_hint
        node["entity_role"] = self.entity_role
        node["precision"] = round(float(self.precision), 4)
        node["provisional"] = bool(self.provisional)
        node["evidence_count"] = round(float(self.evidence_count), 4)
        node["contradiction_pressure"] = round(float(self.contradiction_pressure), 4)
        if self.scene_entity_id is not None:
            node["scene_entity_id"] = self.scene_entity_id
        if self.prediction_error:
            node["prediction_error"] = round(float(self.prediction_error), 4)
        if self.prediction_uncertainty:
            node["prediction_uncertainty"] = round(float(self.prediction_uncertainty), 4)
        return node


@dataclass
class WorkingMemory:
    """Capacity-bounded set of decaying entity slots (Global Workspace store)."""

    capacity: int = DEFAULT_WORKING_MEMORY_SLOTS
    decay: float = DEFAULT_WM_DECAY
    min_salience: float = DEFAULT_WM_MIN_SALIENCE
    cycle: int = 0
    slots: dict[str, MemorySlot] = field(default_factory=dict)
    # Monotonic counter for coining anonymous discovered-object ids (obj-NNNN).
    _next_obj_id: int = 0
    # Persisting scene latent: EMA of the pooled fused percept, carried across
    # cycles. This is the latent "image in the mind" (object permanence at the
    # sub-symbolic level), distinct from the symbolic entity slots above.
    scene_latent: list[float] | None = None
    scene_alpha: float = field(
        default_factory=lambda: _env_float("DECADIC_WM_SCENE_ALPHA", DEFAULT_WM_SCENE_ALPHA)
    )
    scene_blend: float = field(
        default_factory=lambda: _env_float("DECADIC_WM_SCENE_BLEND", DEFAULT_WM_SCENE_BLEND)
    )

    def deposit_scene(self, latent: list[float], alpha: float | None = None) -> None:
        """EMA-blend a freshly pooled percept into the persisting scene latent."""
        try:
            vals = [float(v) for v in latent]
        except (TypeError, ValueError):
            return
        if not vals:
            return
        a = self.scene_alpha if alpha is None else float(alpha)
        a = min(1.0, max(0.0, a))
        if self.scene_latent is None or len(self.scene_latent) != len(vals):
            self.scene_latent = vals
        else:
            self.scene_latent = [
                (1.0 - a) * old + a * new for old, new in zip(self.scene_latent, vals)
            ]

    def _fold_scene(self, dim: int) -> list[float]:
        """Fold the (large) scene latent into `dim` buckets via chunk means."""
        out = [0.0] * dim
        if dim <= 0 or not self.scene_latent:
            return out
        counts = [0] * dim
        n = len(self.scene_latent)
        for i, v in enumerate(self.scene_latent):
            b = min(dim - 1, i * dim // n)
            out[b] += v
            counts[b] += 1
        return [s / c if c else 0.0 for s, c in zip(out, counts)]

    def scene_rms(self) -> float | None:
        if not self.scene_latent:
            return None
        return math.sqrt(sum(v * v for v in self.scene_latent) / len(self.scene_latent))

    def integrate(
        self,
        observed_nodes: list[dict[str, Any]],
        affect: dict[str, float] | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        """Decay every slot, then assimilate/accommodate the freshly seen entities."""
        affect = affect or {}
        self.cycle += 1

        for slot in self.slots.values():
            slot.salience *= self.decay
            slot.audio_intensity *= AUDIO_DECAY

        for node in observed_nodes:
            if node.get("role") != "entity":
                continue
            eid = str(node.get("id", ""))
            if not eid:
                continue
            pos = _as_vec(node.get("position"))
            # WS5-M0.4: oracle nodes may carry a controlled appearance vector
            # (binding-probe injection); byte-identical when absent.
            node_app = _as_floats(node.get("appearance"))
            existing = self.slots.get(eid)
            if existing is None:
                slot = MemorySlot(
                    entity_id=eid,
                    kind=str(node.get("kind", "entity")),
                    position=pos,
                    relative=_as_vec(node.get("relative")),
                    salience=1.0,
                    affective_weight=float(affect.get(eid, 0.0)),
                    last_seen_cycle=self.cycle,
                    seen_count=1,
                )
                if node_app is not None:
                    slot.appearance = node_app
                if pos is not None:
                    slot.pos_history.append(pos)
                self.slots[eid] = slot
            else:
                existing.kind = str(node.get("kind", existing.kind))
                rel = _as_vec(node.get("relative"))
                if pos is not None:
                    existing.position = pos
                    existing.pos_history.append(pos)
                if rel is not None:
                    existing.relative = rel
                if node_app is not None:
                    existing.appearance = node_app
                existing.salience = 1.0
                existing.affective_weight = float(affect.get(eid, existing.affective_weight))
                existing.last_seen_cycle = self.cycle
                existing.seen_count += 1

        # Bind events to their source entity (the bear's growl, a food chime).
        self._bind_events(events)

        # Forget entities that have decayed past usefulness, then enforce capacity.
        for eid in [e for e, s in self.slots.items() if s.salience < self.min_salience]:
            del self.slots[eid]
        if len(self.slots) > self.capacity:
            ranked = sorted(self.slots.values(), key=lambda s: s.salience, reverse=True)
            keep = {s.entity_id for s in ranked[: self.capacity]}
            self.slots = {e: s for e, s in self.slots.items() if e in keep}

    def integrate_discovered(
        self,
        proposals: list[dict[str, Any]],
        *,
        events: list[dict[str, Any]] | None = None,
        appearance_weight: float = 0.6,
        match_threshold: float = 0.35,
        appearance_ema: float = 0.5,
        reidentify: Callable[[list[float]], str | None] | None = None,
        key_lookup: Callable[[str], list[float] | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Bind camera-derived proposals to persistent object files (data association).

        Unlike :meth:`integrate` (which keys on an oracle id), discovered proposals
        carry no identity: each is matched to an existing slot by appearance cosine
        and predicted image position, or else coins a fresh anonymous id. Returns,
        for each proposal that matched an *existing* slot with image history, a
        record ``{idx, entity_id, prev_uv, cur_uv}`` so the caller can compute the
        realized image-motion that drives the agency (self vs other) signal.

        ``reidentify`` is the optional long-term-memory read path: before coining a
        brand-new anonymous id, the appearance is offered to the long-term graph;
        on a hit the slot adopts that persistent ``ent-NNNNN`` id so the bounded
        "now" graph and the unbounded long-term graph share identities. When it is
        ``None`` (oracle mode / no LTM) the behavior is byte-identical to before.

        ``key_lookup`` (WS5-M4.1) fetches the graph entity's STORED appearance
        for an identity-matched slot, which becomes the slot's stable
        ``key_appearance`` -- the identity anchor the slot tensor exposes to
        the network across occlusion gaps. ``None`` -> unchanged behavior.
        """

        def _apply_graph_key(slot: MemorySlot, sid_: str) -> None:
            if key_lookup is None or slot.key_appearance is not None:
                return
            try:
                ka = key_lookup(sid_)
            except Exception:
                ka = None
            if ka:
                slot.key_appearance = [float(v) for v in ka]

        self.cycle += 1
        for slot in self.slots.values():
            decay = self.decay / entity_provisional_decay_boost() if slot.provisional else self.decay
            slot.salience *= max(0.0, min(0.9999, decay))
            slot.audio_intensity *= AUDIO_DECAY

        matched: list[dict[str, Any]] = []
        used: set[str] = set()
        clean: list[dict[str, Any]] = []
        entry_floor = entity_entry_confidence_floor() if provisional_entry_enabled() else 0.2
        for prop in proposals:
            try:
                conf = float(prop.get("confidence", prop.get("presence", 0.0)) or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            if conf < entry_floor:
                continue
            clean.append(prop)
        ordered = sorted(clean, key=lambda p: float(p.get("confidence", p.get("presence", 0.0)) or 0.0), reverse=True)
        for prop in ordered:
            appearance = _as_floats(prop.get("appearance"))
            uv = _as_uv(prop.get("uv"))
            best_id, best_score = None, match_threshold
            for sid, slot in self.slots.items():
                if sid in used:
                    continue
                score = self._match_score(
                    appearance, uv, slot, appearance_weight=appearance_weight
                )
                if score >= best_score:
                    best_score, best_id = score, sid
            if best_id is not None:
                slot = self.slots[best_id]
                prev_uv = list(slot.uv) if slot.uv is not None else None
                self._refresh_slot(slot, prop, appearance, uv, appearance_ema)
                used.add(best_id)
                if prev_uv is not None and uv is not None:
                    matched.append(
                        {
                            "idx": int(prop.get("idx", -1)),
                            "entity_id": best_id,
                            "prev_uv": prev_uv,
                            "cur_uv": uv,
                        }
                    )
            else:
                # Reinstatement (LTM -> WM): ask the long-term graph whether this
                # appearance is a known object before minting a fresh anonymous id.
                # A hit rebinds the slot to the persistent ent-NNNNN identity.
                sid = (
                    reidentify(appearance)
                    if (reidentify is not None and appearance is not None)
                    else None
                )
                if sid is not None and sid in self.slots and sid not in used:
                    slot = self.slots[sid]
                    prev_uv = list(slot.uv) if slot.uv is not None else None
                    self._refresh_slot(slot, prop, appearance, uv, appearance_ema)
                    _apply_graph_key(slot, sid)
                    used.add(sid)
                    if prev_uv is not None and uv is not None:
                        matched.append(
                            {
                                "idx": int(prop.get("idx", -1)),
                                "entity_id": sid,
                                "prev_uv": prev_uv,
                                "cur_uv": uv,
                            }
                        )
                else:
                    reinstated = sid is not None
                    if sid is None:
                        sid = self._coin_id()
                    slot = self._new_slot(sid, prop, appearance, uv)
                    if reinstated:
                        # LTM hit: this is a KNOWN entity re-entering the now.
                        _apply_graph_key(slot, sid)
                    self.slots[sid] = slot
                    used.add(sid)

        self._bind_events_discovered(events)

        for eid in [e for e, s in self.slots.items() if s.salience < self.min_salience]:
            del self.slots[eid]
        if len(self.slots) > self.capacity:
            ranked = sorted(self.slots.values(), key=lambda s: s.salience, reverse=True)
            keep = {s.entity_id for s in ranked[: self.capacity]}
            self.slots = {e: s for e, s in self.slots.items() if e in keep}
        return matched

    def _coin_id(self) -> str:
        sid = f"obj-{self._next_obj_id:04d}"
        self._next_obj_id += 1
        return sid

    def _match_score(
        self,
        appearance: list[float] | None,
        uv: list[float] | None,
        slot: MemorySlot,
        *,
        appearance_weight: float,
    ) -> float:
        app_score = _cosine(appearance, slot.appearance)
        # Map cosine [-1,1] -> [0,1] so it composes with the position score.
        app_score = 0.5 * (app_score + 1.0)
        pos_score = 0.0
        pred = slot.predicted_uv()
        if uv is not None and pred is not None:
            d = math.hypot(uv[0] - pred[0], uv[1] - pred[1])
            pos_score = max(0.0, 1.0 - d / 0.5)  # within ~half the frame counts
        return appearance_weight * app_score + (1.0 - appearance_weight) * pos_score

    def _new_slot(
        self,
        sid: str,
        prop: dict[str, Any],
        appearance: list[float] | None,
        uv: list[float] | None,
    ) -> MemorySlot:
        slot = MemorySlot(
            entity_id=sid,
            kind="unknown",
            relative=_as_vec(prop.get("relative")),
            salience=1.0,
            last_seen_cycle=self.cycle,
            seen_count=1,
            appearance=appearance,
            uv=list(uv) if uv is not None else None,
            bearing=_as_uv(prop.get("bearing")),
            confidence=_as_conf(prop),
            kind_hint=str(prop.get("kind_hint", "object")),
            motion=_as_uv(prop.get("motion")),
            local_motion=_as_float(prop.get("local_motion")),
            retina_contrast=_as_float(prop.get("retina_contrast")),
            looming=_as_float(prop.get("looming")),
            prediction_error=_as_float(prop.get("prediction_error")),
            prediction_uncertainty=_as_float(prop.get("prediction_uncertainty")),
            occlusion_age=int(_as_float(prop.get("occlusion_age"))),
            property_evidence=_as_property_evidence(prop.get("property_evidence")),
            scene_entity_id=str(prop.get("scene_entity_id")) if prop.get("scene_entity_id") else None,
            entity_role=_entity_role(prop),
            precision=max(0.0, min(1.0, _as_float(prop.get("precision")) or _as_conf(prop))),
            provisional=True,
            evidence_count=max(0.0, _as_conf(prop)),
        )
        slot.provisional = not _slot_promoted(slot)
        if uv is not None:
            slot.uv_history.append(list(uv))
        rel = slot.relative
        if rel is not None:
            slot.position = list(rel)
            slot.pos_history.append(list(rel))
        return slot

    def _refresh_slot(
        self,
        slot: MemorySlot,
        prop: dict[str, Any],
        appearance: list[float] | None,
        uv: list[float] | None,
        appearance_ema: float,
    ) -> None:
        if appearance is not None:
            prior_app = list(slot.appearance) if slot.appearance is not None else None
            if slot.appearance is None or len(slot.appearance) != len(appearance):
                slot.appearance = appearance
            else:
                a = appearance_ema
                slot.appearance = [
                    (1.0 - a) * o + a * n for o, n in zip(slot.appearance, appearance)
                ]
            if prior_app is not None and len(prior_app) == len(appearance):
                sim = _cosine(prior_app, appearance)
                if sim < 0.25:
                    slot.contradiction_pressure = min(
                        1.0,
                        slot.contradiction_pressure + (0.25 - sim),
                    )
        if uv is not None:
            slot.uv = list(uv)
            slot.uv_history.append(list(uv))
        rel = _as_vec(prop.get("relative"))
        if rel is not None:
            slot.relative = rel
            slot.position = list(rel)
            slot.pos_history.append(list(rel))
        bearing = _as_uv(prop.get("bearing"))
        if bearing is not None:
            slot.bearing = bearing
        slot.salience = 1.0
        slot.last_seen_cycle = self.cycle
        slot.seen_count += 1
        slot.confidence = _as_conf(prop)
        slot.kind_hint = str(prop.get("kind_hint", slot.kind_hint or "object"))
        slot.entity_role = _entity_role(prop)
        eta = entity_precision_eta()
        signal = max(slot.confidence, 0.05)
        slot.precision = min(1.0, max(0.0, slot.precision + eta * signal * (1.0 - slot.precision)))
        slot.evidence_count += signal
        slot.provisional = not _slot_promoted(slot)
        motion = _as_uv(prop.get("motion"))
        if motion is not None:
            slot.motion = motion
        slot.local_motion = _as_float(prop.get("local_motion"))
        slot.retina_contrast = _as_float(prop.get("retina_contrast"))
        slot.looming = _as_float(prop.get("looming"))
        slot.prediction_error = _as_float(prop.get("prediction_error"))
        slot.prediction_uncertainty = _as_float(prop.get("prediction_uncertainty"))
        slot.occlusion_age = int(_as_float(prop.get("occlusion_age")))
        slot.property_evidence = _as_property_evidence(prop.get("property_evidence"))
        if prop.get("scene_entity_id"):
            slot.scene_entity_id = str(prop.get("scene_entity_id"))

    def update_agency(
        self,
        agency_scores: dict[str, float],
        *,
        ema: float,
        threshold: float,
        min_seen: int,
        touch_active: bool = False,
        touch_boost: float = 0.05,
    ) -> None:
        """EMA-blend per-slot agency and promote persistent high-agency slots to body parts.

        ``touch_active`` cross-checks the efference signal with proprioceptive
        touch: a contact coincident with an already-agentic slot strengthens the
        "mine" edge (a hand you can both command *and* feel when it touches).
        """
        for sid, score in agency_scores.items():
            slot = self.slots.get(sid)
            if slot is None:
                continue
            slot.agency = (1.0 - ema) * slot.agency + ema * float(score)
            slot.agency_seen += 1
            if touch_active and slot.agency >= threshold:
                slot.agency = min(1.0, slot.agency + touch_boost)
            if slot.agency_seen >= min_seen and slot.agency >= threshold:
                slot.kind = "self_part"
            elif slot.kind == "self_part" and slot.agency < threshold:
                slot.kind = "unknown"

    def _bind_events_discovered(self, events: list[dict[str, Any]] | None) -> None:
        """Affect has no oracle id here: bind each event to the most salient in-view slot.

        You feel the pain/pleasure and attribute it to whatever you are attending
        to (the salient thing in view), not to a labeled entity.
        """
        in_view = [s for s in self.slots.values() if s.last_seen_cycle == self.cycle]
        if not in_view:
            return
        target = max(in_view, key=lambda s: s.salience)
        # WS-IND I4: metacognition-gated belief updates — evidence written onto
        # a slot is TEMPERED by that slot's perceptual confidence (reliability
        # of the percept the attribution lands on). gain = 1 - w*(1 - conf):
        # never blocks a first observation, only slows learning from junk
        # percepts; w=0 or flag off -> exact parity. The affective weight is
        # left untempered (you still FEEL the event; you just hold the causal
        # attribution more lightly).
        _ev_gain = 1.0
        try:
            from decadic.config import belief_temper_enabled, belief_temper_weight

            if belief_temper_enabled():
                _conf = max(0.0, min(1.0, float(target.confidence)))
                _ev_gain = 1.0 - belief_temper_weight() * (1.0 - _conf)
        except Exception:
            _ev_gain = 1.0
        for ev in events or []:
            if not isinstance(ev, dict):
                continue
            try:
                intensity = float(ev.get("intensity", 0.0))
            except (TypeError, ValueError):
                intensity = 0.0
            raw_intensity = intensity  # affect path uses the untempered value
            intensity = intensity * _ev_gain  # evidence path uses the tempered one
            et = str(ev.get("type", "")).lower()
            sign = 0.0
            if et in ("collision", "damage", "environment_damage", "fall", "combat_hit"):
                sign = -1.0
                target.property_evidence["predicts_integrity_loss"] = max(
                    float(target.property_evidence.get("predicts_integrity_loss", 0.0) or 0.0),
                    max(0.0, min(1.0, intensity)),
                )
                target.property_evidence["predicts_pain"] = max(
                    float(target.property_evidence.get("predicts_pain", 0.0) or 0.0),
                    max(0.0, min(1.0, intensity)),
                )
                part = _body_part_from_event(ev)
                if part:
                    key = f"predicts_{part}_pain"
                    target.property_evidence[key] = max(
                        float(target.property_evidence.get(key, 0.0) or 0.0),
                        max(0.0, min(1.0, intensity)),
                    )
            elif et == "threat_near":
                sign = -0.5
                target.property_evidence["predicts_pain"] = max(
                    float(target.property_evidence.get("predicts_pain", 0.0) or 0.0),
                    0.5 * max(0.0, min(1.0, intensity)),
                )
            elif et in ("food", "eat", "nourish"):
                sign = 1.0
                target.property_evidence["predicts_energy_relief"] = max(
                    float(target.property_evidence.get("predicts_energy_relief", 0.0) or 0.0),
                    max(0.0, min(1.0, intensity)),
                )
                target.property_evidence["predicts_pleasure"] = max(
                    float(target.property_evidence.get("predicts_pleasure", 0.0) or 0.0),
                    max(0.0, min(1.0, intensity)),
                )
            elif et in ("water", "drink", "hydrate"):
                sign = 1.0
                target.property_evidence["predicts_hydration_relief"] = max(
                    float(target.property_evidence.get("predicts_hydration_relief", 0.0) or 0.0),
                    max(0.0, min(1.0, intensity)),
                )
                target.property_evidence["predicts_pleasure"] = max(
                    float(target.property_evidence.get("predicts_pleasure", 0.0) or 0.0),
                    max(0.0, min(1.0, intensity)),
                )
            elif et == "offer":
                sign = 0.5
            if sign != 0.0:
                target.affective_weight = max(
                    -5.0, min(5.0, target.affective_weight + sign * raw_intensity)
                )
                target.audio_intensity = max(target.audio_intensity, max(0.0, min(1.0, raw_intensity)))
                target.last_event = et or target.last_event
                link = f"event:{_anonymous_event_class(et)}:{self.cycle}"
                if link not in target.event_links:
                    target.event_links.append(link)
                    target.event_links = target.event_links[-16:]

    def _bind_events(self, events: list[dict[str, Any]] | None) -> None:
        """Annotate a slot with the latest event keyed to its entity id."""
        for ev in events or []:
            if not isinstance(ev, dict):
                continue
            src = ev.get("entity") or ev.get("source") or ev.get("id")
            if not isinstance(src, str):
                continue
            slot = self.slots.get(src)
            if slot is None:
                continue
            try:
                intensity = float(ev.get("intensity", 0.0))
            except (TypeError, ValueError):
                intensity = 0.0
            slot.audio_intensity = max(slot.audio_intensity, max(0.0, min(1.0, intensity)))
            slot.last_event = str(ev.get("type", "")) or slot.last_event

    def active_slots(self) -> list[MemorySlot]:
        return sorted(self.slots.values(), key=lambda s: s.salience, reverse=True)

    # ---------------------------------------------------------- WS5-M0.2
    def _slot_row(self, s: MemorySlot) -> list[float]:
        """One slot -> the frozen SLOT_TENSOR_DIM feature row (pure read)."""

        def _t(x: float) -> float:  # bounded, finite
            x = _as_float(x)
            return math.tanh(x)

        row = [0.0] * SLOT_TENSOR_DIM
        # appearance [0:16] -- the identity KEY: the graph entity's stored
        # appearance when identity-matched (M4.1, stable across occlusion),
        # else the slot's own latent fingerprint.
        key = s.key_appearance or s.appearance
        if key:
            for i, v in enumerate(key[:SLOT_TENSOR_APPEARANCE_DIM]):
                row[i] = _as_float(v)
        # spatial [16:27]
        base = SLOT_TENSOR_APPEARANCE_DIM
        rel = s.relative if s.relative is not None else s.position
        if rel is not None:
            for i in range(min(3, len(rel))):
                row[base + i] = math.tanh(_as_float(rel[i]) / SLOT_POS_SCALE)
        if s.bearing is not None:
            row[base + 3] = _as_float(s.bearing[0]) / math.pi
            row[base + 4] = _as_float(s.bearing[1]) / math.pi
        if s.uv is not None:
            row[base + 5] = _as_float(s.uv[0])
            row[base + 6] = _as_float(s.uv[1])
        if s.motion is not None:
            row[base + 7] = _t(s.motion[0])
            row[base + 8] = _t(s.motion[1])
        h = s.heading()
        if h is not None:
            row[base + 9] = math.sin(h)
            row[base + 10] = math.cos(h)
        # scalars [27:40]
        sc = SLOT_TENSOR_APPEARANCE_DIM + SLOT_TENSOR_SPATIAL_DIM
        age = max(0.0, float(self.cycle - s.last_seen_cycle))
        scalars = (
            min(1.0, max(0.0, _as_float(s.salience))),
            _t(s.affective_weight),
            min(1.0, max(0.0, _as_float(s.audio_intensity))),
            min(1.0, max(0.0, _as_float(s.agency))),
            min(1.0, max(0.0, _as_float(s.confidence))),
            min(1.0, max(0.0, _as_float(s.precision))),
            min(1.0, _as_float(s.evidence_count) / SLOT_EVIDENCE_CAP),
            min(1.0, max(0.0, _as_float(s.contradiction_pressure))),
            min(1.0, max(0.0, _as_float(s.looming))),
            min(1.0, max(0.0, _as_float(s.prediction_error))),
            min(1.0, max(0.0, _as_float(s.prediction_uncertainty))),
            min(1.0, age / SLOT_STALENESS_HORIZON),
            1.0 if s.last_seen_cycle == self.cycle else 0.0,
        )
        row[sc : sc + SLOT_TENSOR_SCALAR_DIM] = scalars
        return row

    def slot_tensor(self, k_max: int | None = None, d_slot: int = SLOT_TENSOR_DIM):
        """WS5-M0.2: the slots as a (K, D_slot) float32 token matrix + mask.

        Read-side adapter -- a pure function of existing slot state; slot
        lifecycle is untouched. Ordering is salience-ranked with a stable,
        deterministic tie-break on entity_id, so identical WM state always
        yields an identical tensor (the parity culture's requirement for
        anything that feeds the stack). Rows past the live slot count are
        zero with mask False; ``d_slot`` other than the frozen layout dim
        truncates/zero-pads the row (projection stays fixed).
        """
        import numpy as np

        k = int(self.capacity if k_max is None else max(0, k_max))
        out = np.zeros((k, int(d_slot)), dtype=np.float32)
        mask = np.zeros((k,), dtype=bool)
        if k == 0:
            return out, mask
        ordered = sorted(
            self.slots.values(), key=lambda s: (-float(s.salience), s.entity_id)
        )[:k]
        for i, s in enumerate(ordered):
            row = self._slot_row(s)
            n = min(len(row), int(d_slot))
            out[i, :n] = row[:n]
            mask[i] = True
        return out, mask

    def entity_nodes(self) -> list[dict[str, Any]]:
        """Persisted entity nodes (includes recently-unseen ones still in memory)."""
        return [s.as_node() for s in self.active_slots()]

    def attention_vector(self, dim: int) -> list[float]:
        """Salience-weighted, affect-signed summary for the State Bus A blend.

        Mixes two components: hashed entity slots (who is out there) and the
        folded persisting scene latent (what the scene feels like), weighted by
        `scene_blend`.
        """
        out = [0.0] * dim
        if dim <= 0:
            return out
        entity = [0.0] * dim
        if self.slots:
            total = sum(s.salience for s in self.slots.values()) or 1.0
            for slot in self.slots.values():
                weight = slot.salience / total
                # Hash entity id into the vector so distinct objects occupy distinct channels.
                idx = (hash(slot.entity_id) % dim + dim) % dim
                signed = weight * (1.0 + math.tanh(slot.affective_weight))
                entity[idx] += signed
        if not self.scene_latent:
            return entity
        # tanh-squash the folded latent so raw encoder magnitudes stay bounded.
        scene = [math.tanh(v) for v in self._fold_scene(dim)]
        g = min(1.0, max(0.0, self.scene_blend))
        if not self.slots:
            return scene
        return [(1.0 - g) * e + g * s for e, s in zip(entity, scene)]

    def workspace_candidates(self, dim: int) -> tuple[list[list[float]], list[float]]:
        """Per-candidate (vector, salience) pairs for global-workspace competition.

        Decomposes :meth:`attention_vector` into the individual coalitions that
        compete for ignition: one candidate per active entity slot (its signed,
        affect-weighted hashed channel) plus, when present, the persisting scene
        latent as an ambient candidate (weighted by ``scene_blend`` so it competes
        on the same footing as the slots rather than swamping them). Returns
        ``(vectors, saliences)``; empty lists when there is nothing to broadcast.
        """
        vectors: list[list[float]] = []
        saliences: list[float] = []
        if dim <= 0:
            return vectors, saliences
        for slot in self.slots.values():
            sal = max(0.0, float(slot.salience))
            if sal <= 0.0:
                continue
            v = [0.0] * dim
            idx = (hash(slot.entity_id) % dim + dim) % dim
            v[idx] = 1.0 + math.tanh(slot.affective_weight)
            vectors.append(v)
            saliences.append(sal)
        if self.scene_latent:
            scene = [math.tanh(v) for v in self._fold_scene(dim)]
            rms = self.scene_rms() or 0.0
            scene_sal = float(min(1.0, max(0.0, self.scene_blend)) * min(1.0, rms))
            if scene_sal > 0.0:
                vectors.append(scene)
                saliences.append(scene_sal)
        return vectors, saliences

    def snapshot(self) -> dict[str, Any]:
        rms = self.scene_rms()
        preview = (
            _fin_seq([round(math.tanh(v), 4) for v in self._fold_scene(SCENE_PREVIEW_BUCKETS)])
            if self.scene_latent
            else None
        )
        return {
            "capacity": self.capacity,
            "decay": self.decay,
            "cycle": self.cycle,
            "scene_latent_rms": _fin(round(rms, 4)) if rms is not None else None,
            "scene_preview": preview,
            "slots": [
                {
                    "entity_id": s.entity_id,
                    "kind": s.kind,
                    "position": _fin_seq(list(s.position)) if s.position is not None else None,
                    "relative": _fin_seq(list(s.relative)) if s.relative is not None else None,
                    "heading": _fin(s.heading()),
                    "salience": _fin(round(s.salience, 4)),
                    "affective_weight": _fin(round(s.affective_weight, 4)),
                    "audio_intensity": _fin(round(s.audio_intensity, 4)),
                    "last_event": s.last_event,
                    "last_seen_cycle": s.last_seen_cycle,
                    "seen_count": s.seen_count,
                    "in_view": s.last_seen_cycle == self.cycle,
                    "uv": _fin_seq(list(s.uv)) if s.uv is not None else None,
                    "bearing": _fin_seq(list(s.bearing)) if s.bearing is not None else None,
                    "agency": _fin(round(s.agency, 4)) if s.agency_seen > 0 else None,
                    "confidence": _fin(round(s.confidence, 4)),
                    "kind_hint": s.kind_hint,
                    "entity_role": s.entity_role,
                    "precision": _fin(round(s.precision, 4)),
                    "provisional": bool(s.provisional),
                    "evidence_count": _fin(round(s.evidence_count, 4)),
                    "contradiction_pressure": _fin(round(s.contradiction_pressure, 4)),
                    "event_links": list(s.event_links[-8:]),
                    "relationship_links": list(s.relationship_links[-8:]),
                    "motion": _fin_seq(list(s.motion)) if s.motion is not None else None,
                    "local_motion": _fin(round(s.local_motion, 4)),
                    "retina_contrast": _fin(round(s.retina_contrast, 4)),
                    "looming": _fin(round(s.looming, 4)),
                    "prediction_error": _fin(round(s.prediction_error, 4)),
                    "prediction_uncertainty": _fin(round(s.prediction_uncertainty, 4)),
                    "occlusion_age": int(s.occlusion_age),
                    "property_evidence": dict(s.property_evidence),
                    "scene_entity_id": s.scene_entity_id,
                }
                for s in self.active_slots()
            ],
        }


def _as_vec(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    return None


def _as_floats(value: Any) -> list[float] | None:
    if isinstance(value, list) and value:
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None
    return None


def _as_uv(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) >= 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None
    return None


def _entity_role(prop: dict[str, Any]) -> str:
    role = str(prop.get("entity_role", "") or "").strip()
    if role in ("compact_entity", "extended_entity", "body_coupled_entity", "boundary_entity", "field_entity"):
        return role
    kind = str(prop.get("kind_hint", "object"))
    if kind == "body_part_candidate":
        return "body_coupled_entity"
    if kind == "stuff":
        return "extended_entity"
    try:
        spread = float(prop.get("spread", 0.0) or 0.0)
    except (TypeError, ValueError):
        spread = 0.0
    if spread >= 0.34:
        return "extended_entity"
    return "compact_entity"


def _slot_promoted(slot: MemorySlot) -> bool:
    precision = float(slot.precision if slot.precision is not None else slot.confidence)
    return precision >= entity_promotion_precision() and int(slot.seen_count) >= 2


def _anonymous_event_class(event_type: str) -> str:
    et = str(event_type or "").lower()
    if et in ("collision", "damage", "environment_damage", "fall", "combat_hit", "threat_near"):
        return "aversive_state_change"
    if et in ("food", "eat", "nourish", "water", "drink", "hydrate", "offer"):
        return "interoceptive_relief"
    if et:
        return "sensory_state_change"
    return "state_change"


def _as_conf(prop: dict[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(prop.get("confidence", prop.get("presence", 0.0)) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _as_float(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _as_property_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    forbidden = (
        "label",
        "class",
        "kind_name",
        "semantic_label",
        "oracle",
        "sim_kind",
        "food",
        "water",
        "floor",
        "hand",
        "wall",
        "building",
        "ball",
        "bear",
    )
    out: dict[str, Any] = {}
    for k, v in value.items():
        key = str(k).strip()
        low = key.lower()
        if not key or any(tok in low for tok in forbidden):
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


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))
