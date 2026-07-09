"""WS-FORAGE M4 — spatial recall: turn a remembered resource into an egocentric goal.

The M3 goal vector says *which* need is active; this module answers *where* the
thing that relieves it was last seen, expressed as an egocentric bearing the
policy can steer by. That is the piece that lets incentive salience pull the
agent toward a resource it does NOT currently see -- "recall it, turn toward it,
go" -- rather than only reacting to a visible cue.

Two pure helpers (no torch):

- ``resolve_goal_target(goal_id, graph)`` queries the long-term graph for the
  entity that best predicts relief for the active need (the ``predicts_*_relief``
  property beliefs) and returns its last-known position. Landmark fallback: if
  the target itself has no stored position, borrow a spatially-related
  neighbour's position, so the agent can navigate to the remembered CONTEXT and
  re-acquire the resource by cue there (robust to a non-stationary world).
- ``egocentric_bearing(self_pos, self_yaw, target_pos)`` transforms an allocentric
  target into (cos az, sin az, normalized distance) relative to the agent's
  heading.

Fully defensive: any failure (no graph, no belief, no position, a concurrent
mutation) returns ``None`` / a masked-off bearing, so M4 degrades to "no bearing"
-- a no-op leaving the M3 vector's [4:8] slots zero. It can never perturb the
cycle.
"""

from __future__ import annotations

import math
from typing import Any

# Active need -> the property-belief key that marks an entity as relieving it.
# (Formed in working_memory as predicts_hydration_relief / predicts_energy_relief;
# integrity has no relief belief today -> naturally yields no target.)
_NEED_RELIEF_KEY: dict[str, str] = {
    "hydration": "predicts_hydration_relief",
    "energy": "predicts_energy_relief",
    "integrity": "predicts_integrity_relief",
}


def resolve_goal_target(
    goal_id: str | None,
    graph: Any,
    *,
    min_confidence: float = 0.0,
) -> "tuple[str, list[float]] | None":
    """Last-known position of the entity that best predicts relief for ``goal_id``.

    Returns ``(entity_id, [x, y, z])`` or ``None``. Best-effort and defensive.
    """
    key = _NEED_RELIEF_KEY.get(goal_id or "")
    if key is None or graph is None:
        return None
    try:
        beliefs = getattr(graph, "_beliefs", None)
        nodes = getattr(graph, "_nodes", None)
        if not beliefs or not nodes:
            return None
        best_id, best_score = None, -1.0
        for (node_id, prop_key), b in list(beliefs.items()):
            if prop_key != key:
                continue
            conf = float(b.get("confidence", 0.0) or 0.0)
            mean = float(b.get("mean", 0.0) or 0.0)
            score = conf * max(0.0, mean)
            if conf >= min_confidence and score > best_score:
                best_id, best_score = node_id, score
        if best_id is None:
            return None
        node = nodes.get(best_id)
        pos = node.get("position") if node else None
        if not pos or len(pos) < 2:
            pos = _landmark_position(graph, best_id)  # M4.4 fallback
        if not pos or len(pos) < 2:
            return None
        z = float(pos[2]) if len(pos) > 2 else 0.0
        return best_id, [float(pos[0]), float(pos[1]), z]
    except Exception:
        return None


def _landmark_position(graph: Any, entity_id: str) -> "list[float] | None":
    """Position of the highest-weight spatially-related neighbour of ``entity_id``
    (M4.4): if the resource's own position is unknown/stale, steer to a
    remembered landmark near it and re-cue locally."""
    try:
        edges = getattr(graph, "_edges", None)
        nodes = getattr(graph, "_nodes", None)
        if not edges or not nodes:
            return None
        best_pos, best_w = None, -1.0
        for (src, dst, _kind), e in list(edges.items()):
            if src == entity_id:
                nb = dst
            elif dst == entity_id:
                nb = src
            else:
                continue
            node = nodes.get(nb)
            pos = node.get("position") if node else None
            w = float(e.get("weight", 0.0) or 0.0)
            if pos and len(pos) >= 2 and w > best_w:
                best_pos, best_w = pos, w
        return best_pos
    except Exception:
        return None


_THREAT_KEYS: tuple[str, ...] = ("predicts_pain", "predicts_integrity_loss")


def resolve_threat_target(
    graph: Any,
    *,
    min_confidence: float = 0.2,
) -> "tuple[str, list[float], float] | None":
    """WS-EXPAND E5.1: the strongest remembered threat with a known position.

    Scans the ``predicts_pain`` / ``predicts_integrity_loss`` property beliefs
    (the association store the aversive channel rides — same rail as relief)
    and returns ``(entity_id, [x, y, z], strength)`` for the highest
    confidence*mean entity, or ``None``. Strength is the belief score in
    [0, 1], so a fading threat memory produces a fading avoidance signal.
    Defensive like ``resolve_goal_target``: any failure -> None.
    """
    if graph is None:
        return None
    try:
        beliefs = getattr(graph, "_beliefs", None)
        nodes = getattr(graph, "_nodes", None)
        if not beliefs or not nodes:
            return None
        # Strongest threat WITH a usable position: a position-less belief must
        # not mask a locatable one (caught by test_resolve_threat_target).
        best_id, best_score, best_pos = None, 0.0, None
        for (node_id, prop_key), b in list(beliefs.items()):
            if prop_key not in _THREAT_KEYS:
                continue
            conf = float(b.get("confidence", 0.0) or 0.0)
            mean = float(b.get("mean", 0.0) or 0.0)
            score = conf * max(0.0, mean)
            if conf < min_confidence or score <= best_score:
                continue
            node = nodes.get(node_id)
            pos = node.get("position") if node else None
            if not pos or len(pos) < 2:
                continue  # unlocatable threat: skip, keep scanning
            best_id, best_score, best_pos = node_id, score, pos
        if best_id is None or best_pos is None:
            return None
        z = float(best_pos[2]) if len(best_pos) > 2 else 0.0
        return best_id, [float(best_pos[0]), float(best_pos[1]), z], min(1.0, best_score)
    except Exception:
        return None


def egocentric_bearing(
    self_pos: "list[float]",
    self_yaw: float,
    target_pos: "list[float]",
    *,
    max_dist: float = 10.0,
) -> "tuple[float, float, float]":
    """(cos az, sin az, normalized distance) from agent to target in the XY plane,
    relative to the agent's heading. ``cos az > 0`` -> target ahead; ``sin az > 0``
    -> to the agent's left. Distance normalized by ``max_dist`` and clamped to
    [0, 1]. Pure / torch-free."""
    dx = float(target_pos[0]) - float(self_pos[0])
    dy = float(target_pos[1]) - float(self_pos[1])
    az = math.atan2(dy, dx) - float(self_yaw)
    az = (az + math.pi) % (2.0 * math.pi) - math.pi  # wrap to [-pi, pi]
    dist = math.hypot(dx, dy)
    norm = min(1.0, dist / max(1e-6, float(max_dist)))
    return math.cos(az), math.sin(az), norm
