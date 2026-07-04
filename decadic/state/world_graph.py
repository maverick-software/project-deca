"""Parse observation ``world_state`` into a self-indexed egocentric graph.

Nodes are entities (self, perceived objects, context blobs); edges are the typed
relations the paper's logical layer requires - spatial (self->entity distance),
proximity (entity<->entity), affective (self->entity survival valence), and
context membership. See docs/logical_layer_conformance.md.
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_NODE_CAP = 48
DEFAULT_PROXIMITY_RADIUS = 3.0
DEFAULT_MAX_PROXIMITY_EDGES = 32
AFFECT_EDGE_MIN = 1e-3
AFFECT_DECAY = 0.95
AFFECT_CLAMP = 5.0


def _float_vec3(val: Any) -> list[float] | None:
    if not isinstance(val, list) or len(val) < 3:
        return None
    try:
        return [float(val[0]), float(val[1]), float(val[2])]
    except (TypeError, ValueError):
        return None


def _append_context_nodes(world_state: dict[str, Any], out: list[dict[str, Any]], cap: int) -> None:
    """Attach lightweight ``region`` / ``near_landmark`` / ``game`` / ``body`` blobs as graph nodes."""
    for key in ("region", "near_landmark", "game", "body"):
        if len(out) >= cap:
            return
        blob = world_state.get(key)
        if not isinstance(blob, dict):
            continue
        node: dict[str, Any] = {"role": "context", "kind": key}
        cid = blob.get("id")
        if cid is not None:
            node["id"] = str(cid)
        else:
            name = blob.get("name") or blob.get("display_name") or key
            node["id"] = str(name)
        if isinstance(blob.get("display_name"), str):
            node["display_name"] = blob["display_name"]
        if isinstance(blob.get("route"), str):
            node["route"] = blob["route"]
        if isinstance(blob.get("control_mode"), str):
            node["control_mode"] = blob["control_mode"]
        if isinstance(blob.get("title"), str):
            node["title"] = blob["title"]
        if "moving" in blob:
            node["moving"] = blob["moving"]
        if "standing" in blob:
            node["standing"] = blob["standing"]
        out.append(node)


def egocentric_nodes_from_world_state(
    world_state: Any,
    *,
    cap: int = DEFAULT_NODE_CAP,
) -> list[dict[str, Any]]:
    """Build nodes with ``role`` self vs entity; prefers ``relative`` position when present."""
    if not isinstance(world_state, dict):
        return []
    out: list[dict[str, Any]] = []
    agent = world_state.get("agent")
    if isinstance(agent, dict):
        node: dict[str, Any] = {"role": "self", "id": str(agent.get("id", "self"))}
        pos = _float_vec3(agent.get("position"))
        if pos:
            node["position"] = pos
        ori = _float_vec3(agent.get("orientation"))
        if ori:
            node["orientation"] = ori
        out.append(node)

    entities = world_state.get("entities")
    entity_list: list[Any] = entities if isinstance(entities, list) else []

    for raw in entity_list:
        if len(out) >= cap:
            break
        if not isinstance(raw, dict):
            continue
        ent: dict[str, Any] = {
            "role": "entity",
            "id": str(raw.get("id", "")),
            "kind": str(raw.get("kind", "unknown")),
        }
        rel = _float_vec3(raw.get("relative"))
        if rel:
            ent["relative"] = rel
        pos = _float_vec3(raw.get("position"))
        if pos:
            ent["position"] = pos
        # WS5-M0.4: controlled appearance vectors ride the oracle seam so the
        # binding probe can inject entities with known fingerprints without
        # the (synthetic-starved) discovery pipeline. Absent -> unchanged.
        app = raw.get("appearance")
        if isinstance(app, list) and app:
            try:
                ent["appearance"] = [float(v) for v in app[:64]]
            except (TypeError, ValueError):
                pass
        if ent.get("relative") or ent.get("position"):
            out.append(ent)

    _append_context_nodes(world_state, out, cap)
    return out[:cap]


def egocentric_nodes_from_perception(
    self_node: dict[str, Any],
    wm_nodes: list[dict[str, Any]],
    *,
    cap: int = DEFAULT_NODE_CAP,
) -> list[dict[str, Any]]:
    """Assemble the discovered graph: a sensed self node + working-memory object files.

    Mirrors the schema of :func:`egocentric_nodes_from_world_state` (``role``,
    ``id``, ``kind``, ``relative``, plus ``salience``/``agency`` carried from the
    object files) so the dashboard keeps working, but nothing here comes from the
    oracle: the self is sensed from proprioception and the entities are coined,
    appearance/motion-associated object files from the agent's own camera.
    """
    node = dict(self_node)
    node.setdefault("role", "self")
    node.setdefault("id", "self")
    out: list[dict[str, Any]] = [node]
    for n in wm_nodes:
        if len(out) >= cap:
            break
        if n.get("role") == "entity":
            out.append(n)
    return out[:cap]


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(c * c for c in vec))


def _rel_vec(node: dict[str, Any], self_pos: list[float] | None) -> list[float] | None:
    """Vector from self to this node (prefers explicit ``relative``)."""
    rel = node.get("relative")
    if isinstance(rel, list) and len(rel) >= 3:
        return [float(rel[0]), float(rel[1]), float(rel[2])]
    pos = node.get("position")
    if isinstance(pos, list) and len(pos) >= 3 and self_pos is not None:
        return [float(pos[i]) - float(self_pos[i]) for i in range(3)]
    return None


def _abs_pos(node: dict[str, Any], self_pos: list[float] | None) -> list[float] | None:
    """Best-effort absolute position (falls back to self_pos + relative)."""
    pos = node.get("position")
    if isinstance(pos, list) and len(pos) >= 3:
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    rel = node.get("relative")
    if isinstance(rel, list) and len(rel) >= 3 and self_pos is not None:
        return [float(self_pos[i]) + float(rel[i]) for i in range(3)]
    return None


def edges_from_nodes(
    nodes: list[dict[str, Any]],
    *,
    affect: dict[str, float] | None = None,
    proximity_radius: float = DEFAULT_PROXIMITY_RADIUS,
    max_proximity_edges: int = DEFAULT_MAX_PROXIMITY_EDGES,
) -> list[dict[str, Any]]:
    """Typed edges over an egocentric node list, all indexed to the self-node."""
    affect = affect or {}
    self_node = next((n for n in nodes if n.get("role") == "self"), None)
    if self_node is None:
        return []
    self_id = str(self_node.get("id", "self"))
    self_pos = self_node.get("position") if isinstance(self_node.get("position"), list) else None

    entities = [n for n in nodes if n.get("role") == "entity"]
    contexts = [n for n in nodes if n.get("role") == "context"]
    edges: list[dict[str, Any]] = []

    for ent in entities:
        eid = str(ent.get("id", ""))
        rel = _rel_vec(ent, self_pos)
        if rel is not None:
            dist = _norm(rel)
            edges.append(
                {
                    "source": self_id,
                    "target": eid,
                    "kind": "spatial",
                    "weight": round(1.0 / (1.0 + dist), 4),
                    "distance": round(dist, 4),
                }
            )
        valence = float(affect.get(eid, 0.0))
        if abs(valence) >= AFFECT_EDGE_MIN:
            edges.append(
                {
                    "source": self_id,
                    "target": eid,
                    "kind": "affective",
                    "weight": round(valence, 4),
                }
            )
        # Discovered body schema: a slot whose motion the agent has learned to
        # command (promoted to "self_part") is bound to the self by an "agency"
        # edge - the learned "this is mine" relation.
        if ent.get("kind") == "self_part":
            edges.append(
                {
                    "source": self_id,
                    "target": eid,
                    "kind": "agency",
                    "weight": round(float(ent.get("agency", 1.0)), 4),
                }
            )

    proximity: list[tuple[float, dict[str, Any]]] = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            pa = _abs_pos(entities[i], self_pos)
            pb = _abs_pos(entities[j], self_pos)
            if pa is None or pb is None:
                continue
            dist = _norm([pa[k] - pb[k] for k in range(3)])
            if dist <= proximity_radius:
                proximity.append(
                    (
                        dist,
                        {
                            "source": str(entities[i].get("id", "")),
                            "target": str(entities[j].get("id", "")),
                            "kind": "proximity",
                            "weight": round(1.0 / (1.0 + dist), 4),
                            "distance": round(dist, 4),
                        },
                    )
                )
    proximity.sort(key=lambda t: t[0])
    edges.extend(e for _, e in proximity[:max_proximity_edges])

    for ctx in contexts:
        edges.append(
            {
                "source": self_id,
                "target": str(ctx.get("id", ctx.get("kind", "context"))),
                "kind": "context",
                "weight": 1.0,
            }
        )

    return edges


def egocentric_graph_from_world_state(
    world_state: Any,
    *,
    affect: dict[str, float] | None = None,
    cap: int = DEFAULT_NODE_CAP,
    proximity_radius: float = DEFAULT_PROXIMITY_RADIUS,
    max_proximity_edges: int = DEFAULT_MAX_PROXIMITY_EDGES,
) -> dict[str, Any]:
    """Self-indexed relational graph: ``{"nodes": [...], "edges": [...]}``."""
    nodes = egocentric_nodes_from_world_state(world_state, cap=cap)
    edges = edges_from_nodes(
        nodes,
        affect=affect,
        proximity_radius=proximity_radius,
        max_proximity_edges=max_proximity_edges,
    )
    return {"nodes": nodes, "edges": edges}


def update_entity_affect(
    affect: dict[str, float],
    events: list[dict[str, Any]] | None,
    *,
    decay: float = AFFECT_DECAY,
) -> dict[str, float]:
    """Decay then update per-entity survival valence from event ``source`` ids.

    Negative for harm (collision/damage/fall/threat), positive for nourishment.
    Events keyed to sensors rather than entities simply find no matching node
    later, so they contribute no affective edge.
    """
    for key in list(affect.keys()):
        affect[key] *= decay
        if abs(affect[key]) < 1e-4:
            del affect[key]

    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        ent = ev.get("entity") or ev.get("source") or ev.get("id")
        if not isinstance(ent, str) or not ent:
            continue
        et = str(ev.get("type", "")).lower()
        try:
            intensity = float(ev.get("intensity", 0.0))
        except (TypeError, ValueError):
            intensity = 0.0
        if et in ("collision", "damage", "environment_damage", "fall", "combat_hit"):
            affect[ent] = affect.get(ent, 0.0) - intensity
        elif et == "threat_near":
            affect[ent] = affect.get(ent, 0.0) - 0.5 * intensity
        elif et in ("food", "eat", "nourish"):
            affect[ent] = affect.get(ent, 0.0) + intensity
        elif et == "offer":
            # A parent/NPC offering nourishment builds a positive social bond.
            affect[ent] = affect.get(ent, 0.0) + 0.5 * intensity
        else:
            continue
        affect[ent] = max(-AFFECT_CLAMP, min(AFFECT_CLAMP, affect[ent]))

    return affect
