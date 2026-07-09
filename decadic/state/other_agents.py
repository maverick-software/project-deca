"""WS-EXPAND E10 — other-agent modeling: adaptivity gate + predicted-state registry.

The evidence verdict was unambiguous: agent-modeling pays only when ADAPTIVE
others are present, and is pure overhead when solo — so the gate is the design,
not a config knob. Every perceived entity gets a cheap ballistic
(constant-velocity) motion prior; only when an entity's movement repeatedly
DEFEATS that prior (prediction-error EMA above threshold after a warmup) is it
classified adaptive and given a predicted-state model. Scripted props and
static resources never spawn models; a solo scene carries zero models
(regression-tested).

Honest scope for this milestone: the predicted-state model is the ballistic
prior plus an adaptive-error channel — enough to (a) prove the gate on the rig,
(b) surface `predicts_other_next` positions, and (c) feed the E10.3 policy
ingress later. Reusing the full self-model architecture per entity and the
observation-imitation path (E10.4) land with the two-agent arena (E10.5).
Pure python, fully defensive.
"""

from __future__ import annotations

import math
from typing import Any

from decadic import config as C


def _pos(entity: Any) -> "tuple[float, float] | None":
    try:
        if isinstance(entity, dict):
            p = entity.get("position")
        else:
            p = getattr(entity, "position", None)
        if p is None or len(p) < 2:
            return None
        x, y = float(p[0]), float(p[1])
        if x != x or y != y:
            return None
        return x, y
    except Exception:
        return None


def _eid(entity: Any) -> "str | None":
    try:
        if isinstance(entity, dict):
            v = entity.get("entity_id") or entity.get("id")
        else:
            v = getattr(entity, "entity_id", None) or getattr(entity, "id", None)
        return str(v) if v else None
    except Exception:
        return None


class _Track:
    __slots__ = ("prev", "cur", "err_ema", "obs", "adaptive", "last_seen")

    def __init__(self) -> None:
        self.prev: "tuple[float, float] | None" = None
        self.cur: "tuple[float, float] | None" = None
        self.err_ema = 0.0
        self.obs = 0
        self.adaptive = False
        self.last_seen = 0  # ingest tick, for stalest-eviction


class OtherAgentRegistry:
    """Per-entity ballistic priors; models activate only on evidenced adaptivity."""

    def __init__(
        self,
        *,
        err_threshold: float | None = None,
        warmup_obs: int | None = None,
        ema_alpha: float | None = None,
        max_tracks: int | None = None,
    ) -> None:
        self.err_threshold = float(
            err_threshold if err_threshold is not None else C.other_err_threshold()
        )
        self.warmup_obs = int(warmup_obs if warmup_obs is not None else C.other_warmup_obs())
        self.ema_alpha = float(ema_alpha if ema_alpha is not None else C.other_ema_alpha())
        self.max_tracks = int(max_tracks if max_tracks is not None else C.other_max_tracks())
        self._tracks: dict[str, _Track] = {}
        self._tick = 0

    def ingest(self, entities: "list[Any] | None") -> None:
        """One perception cycle's entities (dicts or objects with .position)."""
        if not entities:
            return
        self._tick += 1
        for e in entities:
            eid, pos = _eid(e), _pos(e)
            if eid is None or pos is None:
                continue
            t = self._tracks.get(eid)
            if t is None:
                if len(self._tracks) >= self.max_tracks:
                    # Evict the STALEST track rather than refusing new ids
                    # forever (2026-07-07 audit: discovered-mode entity ids
                    # churn — a silted-up table would blind the registry to
                    # every newcomer, including a genuinely adaptive one).
                    stalest = min(self._tracks, key=lambda k: self._tracks[k].last_seen)
                    del self._tracks[stalest]
                t = _Track()
                self._tracks[eid] = t
            t.last_seen = self._tick
            # Score the ballistic prior BEFORE updating it: predicted = cur +
            # (cur - prev); a static prop predicts itself perfectly forever.
            if t.cur is not None and t.prev is not None:
                px = 2.0 * t.cur[0] - t.prev[0]
                py = 2.0 * t.cur[1] - t.prev[1]
                err = math.hypot(pos[0] - px, pos[1] - py)
                t.err_ema = (1.0 - self.ema_alpha) * t.err_ema + self.ema_alpha * err
            t.prev, t.cur = t.cur, pos
            t.obs += 1
            # The adaptivity gate: enough observations AND the simple motion
            # prior keeps failing -> something is CHOOSING its movement.
            t.adaptive = t.obs >= self.warmup_obs and t.err_ema > self.err_threshold

    def predicted_next(self, eid: str) -> "tuple[float, float] | None":
        """predicts_other_next: ballistic extrapolation for a tracked entity."""
        t = self._tracks.get(eid)
        if t is None or t.cur is None or t.prev is None:
            return None
        return 2.0 * t.cur[0] - t.prev[0], 2.0 * t.cur[1] - t.prev[1]

    def current_pos(self, eid: str) -> "tuple[float, float] | None":
        t = self._tracks.get(eid)
        return t.cur if t is not None else None

    def dominant_adaptive(self) -> "str | None":
        """E10.3: the adaptive entity most worth modeling right now — the one
        whose movement defeats the ballistic prior hardest."""
        best, best_err = None, 0.0
        for k, t in self._tracks.items():
            if t.adaptive and t.err_ema > best_err:
                best, best_err = k, t.err_ema
        return best

    def adaptive_ids(self) -> "list[str]":
        return [k for k, t in self._tracks.items() if t.adaptive]

    def telemetry(self) -> dict[str, int | float]:
        adaptive = self.adaptive_ids()
        return {
            "other_tracks": len(self._tracks),
            "other_models_active": len(adaptive),  # MUST be 0 in solo scenes
            "other_max_err_ema": round(
                max((t.err_ema for t in self._tracks.values()), default=0.0), 6
            ),
        }


# --- E10.3: the other-vector fed to the policy ---------------------------------

# Frozen layout (the zero-init other_ingress and its tests depend on it):
# [0] presence (1.0 iff an adaptive other is modeled — all-zero when solo, so
#     the channel is inert exactly when the adaptivity gate says it should be)
# [1:3] egocentric bearing (cos az, sin az) to the other, [3] norm distance
# [4:6] egocentric bearing to its PREDICTED NEXT position, [6] norm distance
# [7] adaptivity strength (prior-defeat magnitude, clamped)
OTHER_VEC_DIM = 8


def encode_other_vec(
    registry: "OtherAgentRegistry | None",
    self_pos: Any,
    self_yaw: float,
    *,
    max_dist: float,
) -> "list[float]":
    """Build the policy's view of the dominant adaptive other. Defensive:
    anything missing -> all zeros (solo parity)."""
    v = [0.0] * OTHER_VEC_DIM
    try:
        if registry is None or self_pos is None or len(self_pos) < 2:
            return v
        eid = registry.dominant_adaptive()
        if eid is None:
            return v
        cur = registry.current_pos(eid)
        if cur is None:
            return v
        from decadic.state.spatial_recall import egocentric_bearing

        c, s, d = egocentric_bearing(
            list(self_pos), float(self_yaw), [cur[0], cur[1], 0.0], max_dist=max_dist
        )
        v[0] = 1.0
        v[1], v[2], v[3] = c, s, d
        nxt = registry.predicted_next(eid)
        if nxt is not None:
            cn, sn, dn = egocentric_bearing(
                list(self_pos), float(self_yaw), [nxt[0], nxt[1], 0.0], max_dist=max_dist
            )
            v[4], v[5], v[6] = cn, sn, dn
        t = registry._tracks.get(eid)
        if t is not None:
            v[7] = min(1.0, t.err_ema / max(1e-9, 4.0 * registry.err_threshold))
        return v
    except Exception:
        return [0.0] * OTHER_VEC_DIM
