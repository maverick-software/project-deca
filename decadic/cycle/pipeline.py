"""Ordered Decadic Cycle execution (Phase 1 stubs)."""

from __future__ import annotations

import time
from dataclasses import asdict
from datetime import UTC, datetime

from decadic.cycle.stages import (
    stage_01,
    stage_02,
    stage_03,
    stage_04,
    stage_05,
    stage_06,
    stage_07,
    stage_08,
    stage_09,
    stage_10,
)
from decadic.cycle.types import CycleContext
from decadic.state.viability import (
    apply_pain_pleasure_to_B,
    ema_affect,
    reward_success_stub,
    stub_prediction_error_penalty,
    viability_delta_to_signals,
)

_RUNNERS_1_9 = [
    stage_01.run,
    stage_02.run,
    stage_03.run,
    stage_04.run,
    stage_05.run,
    stage_06.run,
    stage_07.run,
    stage_08.run,
    stage_09.run,
]


def _utc_ts() -> str:
    return datetime.now(UTC).isoformat()


def run_cycle(ctx: CycleContext) -> dict:
    """Execute stages 1–10 synchronously; returns outbound action message dict."""
    ctx.state_bus.cycle_index += 1

    traces = []
    ctx.latents.clear()
    stage_timing_ms_total = 0.0
    for runner in _RUNNERS_1_9:
        t0 = time.perf_counter()
        tr = runner(ctx)
        ms = (time.perf_counter() - t0) * 1000.0
        stage_timing_ms_total += ms
        tr.payload["timing_ms"] = round(ms, 4)
        traces.append(tr)

    ctx.latents["stage_traces"] = traces

    pe_delta = stub_prediction_error_penalty(
        ctx.perceptual.integration_ticks, ctx.state_bus.cycle_index
    )
    # Prediction error no longer drains viability (homeostatic reservoirs own
    # survival); it still produces affect signals into element B.
    p_pe, pl_pe = viability_delta_to_signals(pe_delta)
    ctx.state_bus.emotion_physio = apply_pain_pleasure_to_B(
        ctx.state_bus.emotion_physio, p_pe, pl_pe
    )

    reward_stub = reward_success_stub(ctx.state_bus.cycle_index)
    p_rw, pl_rw = viability_delta_to_signals(reward_stub)
    ctx.state_bus.emotion_physio = apply_pain_pleasure_to_B(
        ctx.state_bus.emotion_physio, p_rw, pl_rw
    )

    ctx.state_bus.pain_scalar = ema_affect(ctx.state_bus.pain_scalar, p_pe + p_rw)
    ctx.state_bus.pleasure_scalar = ema_affect(ctx.state_bus.pleasure_scalar, pl_pe + pl_rw)

    t10 = time.perf_counter()
    tr10 = stage_10.run(ctx)
    ms10 = (time.perf_counter() - t10) * 1000.0
    stage_timing_ms_total += ms10
    tr10.payload["timing_ms"] = round(ms10, 4)
    traces.append(tr10)

    action = ctx.latents.get("action", {"type": "noop", "parameters": {}})
    predicted = ctx.latents.get(
        "predicted_outcome",
        {"embedding": [], "expected_position": [0.0, 0.0, 0.0]},
    )

    ctx.state_bus.action_history.append(
        {"cycle": ctx.state_bus.cycle_index, "action": action}
    )

    diagnostics = {
        "stub_prediction_error_delta": pe_delta,
        "stub_reward_delta": reward_stub,
        "stage_timing_ms_total": round(stage_timing_ms_total, 4),
        "viability_value": ctx.viability.value,
        "salience_hint": float(tr10.payload.get("salience", 0.0)),
    }

    return {
        "timestamp": _utc_ts(),
        "action": action,
        "predicted_outcome": predicted,
        "trace": [asdict(t) for t in traces],
        "_diagnostics": diagnostics,
    }
