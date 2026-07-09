"""WS-IND I1 — attention schema: a predictive model of the system's own attention.

The stack has attention everywhere (the gate's skip/escalate arbitration,
workspace ignition, slot routing) but until now no representation OF it. This
module adds the schema's pure parts: encode the gate's current state, encode
the realized outcome one cycle later (the self-supervised target — outcome as
target, never a reward), track rolling prediction accuracy, and turn a
predicted-escalation probability into a BOUNDED anticipatory gate bias.

Functional payoff (the reason it ships, indicator AST-1 second): a controller
does better with a predictive model of the thing it controls — anticipatory
gating means deliberating BEFORE the surprise instead of one cycle after. The
same head, run later over the E10 other-agent tracks, becomes modeling of
OTHERS' attention (joint-attention substrate; waits on the two-agent arena).

The head itself lives in ``neural_stack`` (zero-init output layer -> a fresh
agent predicts nothing and biases nothing: birth-identical); its prediction
re-enters the stack through a zero-init ingress (the schema informing control).
"""

from __future__ import annotations

from typing import Any

# Canonical gate-reason classes the schema predicts. Order is FROZEN (the
# schema head's output layout and the CE target indices depend on it).
GATE_REASONS: tuple[str, ...] = (
    "skip",
    "score",
    "hysteresis",
    "fast_path",
    "type2_memory_search",
)

# Gate-state input vector layout (all already in gate telemetry / decision):
# [0] score, [1] threshold_effective, [2] escalated (0/1), [3] latch remaining
# (normalized by hysteresis_k), [4] type2 cooldown (normalized by refractory),
# [5] trailing escalation rate.
GATE_STATE_DIM = 6

# Schema prediction vector re-entering the stack: [0] p(escalate),
# [1:6] reason-class probabilities, [6] predicted next gate score,
# [7] p(workspace ignites) — WS-DEPTH D4, [8] predicted ignition share — D4.
SCHEMA_VEC_DIM = 4 + len(GATE_REASONS)


def encode_ws_target(workspace_block: "dict | None") -> "tuple[float, float] | None":
    """WS-DEPTH D4: (ignited, coalition share) realized targets from the
    pipeline's workspace block; None when GWT was off this cycle."""
    if not isinstance(workspace_block, dict) or not workspace_block.get("enabled"):
        return None
    try:
        share = float(workspace_block.get("share", 0.0) or 0.0)
    except (TypeError, ValueError):
        share = 0.0  # junk share degrades, but the valid ignited bit survives
    if share != share:
        share = 0.0
    return (
        1.0 if workspace_block.get("ignited") else 0.0,
        max(0.0, min(1.0, share)),
    )


def build_gate_state_vec(decision: Any, gate: Any) -> list[float]:
    """The schema's view of this cycle's attention state (pure floats)."""
    try:
        hk = max(1, int(getattr(gate, "hysteresis_k", 1)))
        refr = max(1, int(getattr(gate, "type2_refractory", 1) or 1))
        return [
            float(getattr(decision, "score", 0.0)),
            float(getattr(decision, "threshold_effective", 0.0)),
            1.0 if getattr(decision, "escalate", False) else 0.0,
            float(getattr(gate, "_latch_remaining", 0)) / hk,
            float(getattr(gate, "_type2_cooldown", 0)) / refr,
            float(getattr(gate, "escalation_rate", 0.0)),
        ]
    except Exception:
        return [0.0] * GATE_STATE_DIM


def encode_realized_target(decision: Any) -> "tuple[float, int, float]":
    """(escalated, reason index, score) — the self-supervised training target."""
    reason = str(getattr(decision, "reason", "skip"))
    # "no_precedent" forces deliberation on the first cycles; treat it as the
    # score-driven class (the schema needn't learn a bootstrap artifact).
    if reason == "no_precedent":
        reason = "score"
    try:
        idx = GATE_REASONS.index(reason)
    except ValueError:
        idx = 0
    return (
        1.0 if getattr(decision, "escalate", False) else 0.0,
        idx,
        float(getattr(decision, "score", 0.0)),
    )


class SchemaAccuracy:
    """Rolling accuracy of predicted-escalation vs realized (telemetry only)."""

    def __init__(self, window: int = 256) -> None:
        self.window = max(8, int(window))
        self._hits: list[int] = []
        self._base: list[int] = []  # realized escalations, for the base rate
        self.total = 0

    def note(self, p_escalate: float, realized: bool) -> None:
        pred = p_escalate >= 0.5
        self._hits.append(1 if pred == bool(realized) else 0)
        self._base.append(1 if realized else 0)
        if len(self._hits) > self.window:
            self._hits.pop(0)
            self._base.pop(0)
        self.total += 1

    def telemetry(self) -> dict[str, float | int]:
        n = max(1, len(self._hits))
        esc_rate = sum(self._base) / n
        # Accuracy of always predicting the majority class -- the bar to beat.
        base_acc = max(esc_rate, 1.0 - esc_rate)
        return {
            "schema_samples": self.total,
            "schema_accuracy": round(sum(self._hits) / n, 6),
            "schema_base_accuracy": round(base_acc, 6),
        }


def schema_gate_bias(p_escalate: float, *, gain: float, cap: float) -> float:
    """Anticipatory threshold bias from predicted escalation.

    Only above-chance predictions bias (p <= 0.5 -> exactly 0, which is also
    the zero-init head's output -> birth-identical). Bounded by ``cap``; the
    gate's setter re-clamps against the SHARED modulation cap, so the schema
    and the learning-control channels can never compound past it.
    """
    try:
        p = float(p_escalate)
    except (TypeError, ValueError):
        return 0.0
    if p != p or p <= 0.5:
        return 0.0
    return min(float(cap), float(gain) * (p - 0.5) * 2.0)
