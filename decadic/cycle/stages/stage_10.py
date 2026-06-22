"""Stage 10 — Normative memory mapping (stub write to episodic store)."""

from __future__ import annotations

import numpy as np

from decadic.config import gwt_salience_boost, ltm_consolidate_min_seen
from decadic.cycle.stages._helpers import trace
from decadic.cycle.types import CycleContext, StageTrace
from decadic.memory.embeddings import episode_embedding_from_cycle
from decadic.memory.episodic_store import EpisodicRecord


def run(ctx: CycleContext) -> StageTrace:
    traces = ctx.latents.get("stage_traces", [])
    summary = {
        "priority": ctx.state_bus.priority_label,
        "viability": ctx.viability.value,
        "pain": ctx.state_bus.pain_scalar,
        "pleasure": ctx.state_bus.pleasure_scalar,
        "action": ctx.latents.get("action"),
        "compressed_trace_ids": [t.stage for t in traces],
    }
    scene_ws = getattr(ctx.perceptual, "scene_workspace", None)
    focus = getattr(ctx.perceptual, "focus", None)
    if scene_ws is not None:
        scene_snap = scene_ws.snapshot()
        summary["scene"] = {
            "entity_count": scene_snap.get("entity_count", 0),
            "visible_count": scene_snap.get("visible_count", 0),
            "occluded_count": scene_snap.get("occluded_count", 0),
            "focus_ids": list(scene_snap.get("focus_ids", [])),
            "prediction_error": scene_snap.get("prediction_error"),
        }
    elif isinstance(focus, dict):
        summary["scene"] = {"focus_ids": list(focus.get("ids", []))}
    ws_summary = ctx.latents.get("workspace")
    if isinstance(ws_summary, dict):
        summary["workspace"] = {
            "ignited": bool(ws_summary.get("ignited")),
            "share": ws_summary.get("share"),
            "winners": list(ws_summary.get("winners", [])),
            "focus_ids": list(ws_summary.get("focus_ids", [])),
        }
    salience = abs(ctx.state_bus.pain_scalar - ctx.state_bus.pleasure_scalar) + (
        1.0 - ctx.viability.value / 100.0
    )
    # Global-workspace broadcast (Phase 2): a strong ignition lifts the stored
    # episode's salience, so globally-broadcast content is preferentially
    # remembered. No-op when GWT is off (no workspace block) or it did not ignite.
    ws = ctx.latents.get("workspace")
    if isinstance(ws, dict) and ws.get("ignited"):
        salience += gwt_salience_boost() * float(ws.get("share", 0.0))
    z5_raw = ctx.latents.get("z5_snapshot")
    z5_np = np.asarray(z5_raw, dtype=np.float32).reshape(-1) if z5_raw else None
    # Perceptual key (compression of this cycle's z0); None when the loop is off,
    # in which case the stored embedding's perceptual tail is zeros (parity).
    key_raw = ctx.latents.get("percept_key")
    key_np = np.asarray(key_raw, dtype=np.float32).reshape(-1) if key_raw else None
    emb = episode_embedding_from_cycle(ctx.state_bus, z5_np, key_np)
    ctx.episodic.append(
        EpisodicRecord(
            cycle_index=ctx.state_bus.cycle_index,
            summary=summary,
            salience=float(salience),
            embedding=emb.astype(float).tolist(),
        )
    )
    # Consolidation (WM -> LTM): commit stable working-memory object files into the
    # persistent, unbounded long-term graph and link co-present ones. Discovered
    # mode only (oracle slots carry no appearance fingerprint, so this is a no-op
    # there); skipped entirely when the long-term graph is disabled.
    if ctx.ltm_graph is not None and ctx.perception_mode == "discovered":
        wm = getattr(ctx.perceptual, "working_memory", None)
        if wm is not None:
            health = getattr(ctx.perceptual, "discovery_health", None)
            reason = str((health or {}).get("reason", "healthy")) if isinstance(health, dict) else "healthy"
            scene_health = getattr(ctx.perceptual, "scene_health", None)
            if (
                isinstance(scene_health, dict)
                and int(scene_health.get("prediction_unstable_count", 0) or 0) > 0
            ):
                reason = "skipped_prediction_unstable"
            status = "accepted"
            relationship_update = reason != "skipped_perception_collapsed"
            property_update = reason not in (
                "skipped_no_objects",
                "skipped_low_confidence",
                "skipped_prediction_unstable",
            )
            if reason in ("skipped_no_objects", "skipped_low_confidence", "skipped_prediction_unstable"):
                status = reason
                ids = []
            else:
                slots = [
                    s
                    for s in wm.active_slots()
                    if str(getattr(s, "kind_hint", "object")) != "stuff"
                    and getattr(s, "appearance", None)
                    and float(getattr(s, "confidence", 1.0) or 0.0) >= 0.2
                    and int(getattr(s, "seen_count", 0)) >= ltm_consolidate_min_seen()
                ]
                if not slots:
                    status = "skipped_no_objects"
                    ids = []
                else:
                    ids = ctx.ltm_graph.consolidate(
                        slots,
                        cycle=int(ctx.state_bus.cycle_index),
                        min_seen=ltm_consolidate_min_seen(),
                        property_update=property_update,
                        relationship_update=relationship_update,
                    )
                    if relationship_update and scene_ws is not None and ids:
                        scene_to_ltm: dict[str, str] = {}
                        for slot, node_id in zip(slots, ids):
                            sid = getattr(slot, "scene_entity_id", None)
                            if sid:
                                scene_to_ltm[str(sid)] = str(node_id)
                        for rel in scene_ws.relation_dicts():
                            src = scene_to_ltm.get(str(rel.get("src")))
                            dst = scene_to_ltm.get(str(rel.get("dst")))
                            kind = str(rel.get("kind", "scene_relation"))
                            if src and dst and src != dst:
                                ctx.ltm_graph.bump_edge(
                                    src,
                                    dst,
                                    kind=f"scene_{kind}",
                                    weight=float(rel.get("confidence", 1.0) or 1.0),
                                    cycle=int(ctx.state_bus.cycle_index),
                                )
                    if not relationship_update:
                        status = "accepted_properties"
            report = {
                "status": status,
                "reason": reason if status == "accepted" else status,
                "accepted_ids": list(ids),
                "identity_refresh": status in ("accepted", "accepted_properties"),
                "property_update": bool(property_update and ids),
                "relationship_update": bool(relationship_update and status == "accepted"),
                "relationship_updates_skipped": 0 if relationship_update else 1,
            }
            stats = getattr(ctx.ltm_graph, "belief_stats", lambda: {})()
            if isinstance(stats, dict):
                report.update(stats)
            if hasattr(ctx.perceptual, "ltm_consolidation"):
                ctx.perceptual.ltm_consolidation = report
            if isinstance(health, dict):
                health = dict(health)
                health["ltm_write"] = status
                ctx.perceptual.discovery_health = health
    return trace(10, "normative_memory_mapping", salience=float(salience))
