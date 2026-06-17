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
            ctx.ltm_graph.consolidate(
                wm.active_slots(),
                cycle=int(ctx.state_bus.cycle_index),
                min_seen=ltm_consolidate_min_seen(),
            )
    return trace(10, "normative_memory_mapping", salience=float(salience))
