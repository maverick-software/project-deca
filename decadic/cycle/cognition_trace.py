"""Cognitive Trace - a human-readable "why" for each cognitive cycle.

This is the interpretability/monitoring layer. It does not change behaviour: it
reads tensors the cycle already computed and translates them into a structured,
human-readable record of *why* the agent acted - in the agent's own terms:

- intent: the survival objective decomposed per goal dimension. The root
  motivation is the homeostatic interoceptive drive (keep the reservoirs -
  hydration / energy / integrity - near full). Using the agent's own world
  model, we ask what the emitted motor command is predicted to do to each
  reservoir the innate prior cares about, versus standing still. This is the
  literal "what is it trying to achieve," read straight off the free-energy
  objective, not a post-hoc story.
- self_surprise: how well last cycle's forward-model prediction matched the
  realized body state (the agent's own prediction error, per dimension).
- affect / recalled_episode: the felt pain/pleasure/risk and the most similar
  past episode (grounding the opaque internal-narrative latent in a real memory).
- salient / counterfactuals / probes / narrative: optional, gated extras filled
  by later phases (input attribution, action rollouts, latent probes, prose).

Everything is read-only. The collect step runs once per cycle under no_grad on
the pre-optimizer-step weights (the ones that produced the action), so the
explanation is faithful to the decision that was actually made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from decadic.config import INTERO_PRED_DIM
from decadic.nn.frozen_encoders import (
    controllable_intero_vector,
    intero_preference_weights,
    preferred_intero_vector,
)

if TYPE_CHECKING:  # avoid a hard torch/runtime import at module load
    import torch

    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.nn.bundle import NeuralBundle

_PROPRIO_BASE = ["roll", "pitch", "yaw", "height", "vx", "vy", "vz"]
_INTERO_LABELS = ["hydration", "energy", "integrity"]


def _proprio_labels(n: int) -> list[str]:
    return [_PROPRIO_BASE[i] if i < len(_PROPRIO_BASE) else f"joint_{i - len(_PROPRIO_BASE)}" for i in range(n)]


def _np1(t: "torch.Tensor") -> np.ndarray:
    return t.detach().float().cpu().reshape(-1).numpy()


@dataclass
class CognitiveTrace:
    """One cycle's structured explanation; serialized to JSON for the dashboard."""

    cycle: int
    intent: dict[str, Any]
    self_surprise: dict[str, Any]
    affect: dict[str, Any]
    recalled_episode: dict[str, Any] | None = None
    salient: dict[str, Any] | None = None
    counterfactuals: dict[str, Any] | None = None
    probes: dict[str, Any] | None = None
    self_model: dict[str, Any] | None = None
    workspace: dict[str, Any] | None = None
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "intent": self.intent,
            "self_surprise": self.self_surprise,
            "affect": self.affect,
            "recalled_episode": self.recalled_episode,
            "salient": self.salient,
            "counterfactuals": self.counterfactuals,
            "probes": self.probes,
            "self_model": self.self_model,
            "workspace": self.workspace,
            "narrative": self.narrative,
        }

    def compact(self) -> dict[str, Any]:
        """A tiny per-cycle summary for the temporal-trace ring buffer."""
        top = (self.intent.get("drivers") or [{}])[0]
        return {
            "cycle": self.cycle,
            "intent": self.intent.get("summary", ""),
            "top_goal": top.get("goal"),
            "pain": self.affect.get("pain"),
            "pleasure": self.affect.get("pleasure"),
            "risk": self.affect.get("risk"),
            "surprise": self.self_surprise.get("mean_abs_residual"),
        }


def collect_cognition_inputs(
    *,
    bundle: "NeuralBundle",
    z5: "torch.Tensor",
    motor_u: "torch.Tensor",
    ctx: "CycleContext",
    fwd_dim: int,
    drive_on: bool,
    s_target: "torch.Tensor | None",
) -> dict[str, Any]:
    """Capture the raw arrays for the trace under no_grad (call pre-optimizer-step).

    Uses the agent's own (frozen-for-this-call) world models to predict the effect
    of the emitted command vs standing still, plus last cycle's realized
    prediction error. Returns plain numpy so :func:`build` needs no torch.
    """
    import torch

    stack = bundle.stack
    zero_u = torch.zeros_like(motor_u)
    raw: dict[str, Any] = {}
    with torch.no_grad():
        # Homeostatic (drive) intent: the root motivation - what the emitted
        # action is predicted to do to the reservoirs vs standing still, scored
        # by the innate preference weights. This is the live survival objective.
        if drive_on and getattr(stack, "has_intero_model", False):
            idim = int(INTERO_PRED_DIM)
            now = torch.as_tensor(
                [controllable_intero_vector(ctx.homeostasis, idim)], device=z5.device, dtype=z5.dtype
            )
            pred_i = _np1(stack.forward_predict_intero(z5, motor_u, now, detach_params=True))
            pred_i_still = _np1(stack.forward_predict_intero(z5, zero_u, now, detach_params=True))
            raw["drive"] = {
                "pred": pred_i,
                "pred_still": pred_i_still,
                "pref": np.asarray(preferred_intero_vector(idim), dtype=np.float32),
                "w": np.asarray(intero_preference_weights(idim), dtype=np.float32),
                "now": _np1(now),
            }
        # Self-model surprise: last cycle's forward prediction vs the realized
        # current state. prev_state/prev_motor are still the previous cycle's at
        # this point (they are overwritten later in the cycle).
        if bundle.prev_state is not None and bundle.prev_motor is not None and s_target is not None:
            pred_prev = _np1(stack.forward_predict(bundle.prev_state, bundle.prev_motor, detach_params=True))
            raw["surprise"] = {"pred": pred_prev, "target": _np1(s_target)}
    return raw


def _salient_node(wm: Any | None) -> dict[str, Any] | None:
    """The working-memory slot currently dominating the attention blend into A."""
    if wm is None:
        return None
    slots = getattr(wm, "slots", {}) or {}
    if not slots:
        return None
    top = max(slots.values(), key=lambda s: getattr(s, "salience", 0.0))
    return {
        "node_id": getattr(top, "entity_id", None),
        "kind": getattr(top, "kind", None),
        "salience": round(float(getattr(top, "salience", 0.0)), 4),
        "affective_weight": round(float(getattr(top, "affective_weight", 0.0)), 4),
    }


def attribution_pass(
    *,
    bundle: "NeuralBundle",
    z0: "torch.Tensor",
    ep: "torch.Tensor",
    mem_t: "torch.Tensor",
    wm: Any | None = None,
) -> dict[str, Any]:
    """Gradient attribution of the emitted motor command to its three inputs.

    Re-runs the stack on requires-grad copies of (perception z0, affect proxy ep,
    episodic-recall context mem_t) and takes d|motor_u|/d(input). The stack's
    recurrent buffers are snapshotted and restored so this stays read-only (no
    extra step of the LSTM/GRU state). Gated/sampled by the caller so the default
    cycle rate is preserved.
    """
    import torch

    stack = bundle.stack
    g0 = stack.gru_h.detach().clone()
    h0 = stack.lstm_h.detach().clone()
    c0 = stack.lstm_c.detach().clone()
    chan = {"perception": 0.0, "affect": 0.0, "memory": 0.0}
    try:
        z0v = z0.detach().clone().requires_grad_(True)
        epv = ep.detach().clone().requires_grad_(True)
        memv = mem_t.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            out2 = stack(z0v, epv, memv)
            target = out2["motor_u"].norm()
            grads = torch.autograd.grad(
                target, [z0v, epv, memv], retain_graph=False, allow_unused=True
            )

        def _gnorm(g: "torch.Tensor | None") -> float:
            return float(g.detach().norm().cpu()) if g is not None else 0.0

        chan = {
            "perception": _gnorm(grads[0]),
            "affect": _gnorm(grads[1]),
            "memory": _gnorm(grads[2]),
        }
    finally:  # always restore the recurrent state - purity guarantee
        stack.gru_h.copy_(g0)
        stack.lstm_h.copy_(h0)
        stack.lstm_c.copy_(c0)

    tot = sum(chan.values()) or 1.0
    return {
        "target": "|motor_u|",
        "channels": {k: round(v, 5) for k, v in chan.items()},
        "fractions": {k: round(v / tot, 4) for k, v in chan.items()},
        "node": _salient_node(wm),
    }


def counterfactual_rollout(
    *,
    bundle: "NeuralBundle",
    z5: "torch.Tensor",
    base_motor: "torch.Tensor | None",
    homeostasis: Any | None,
    fwd_dim: int,
    drive_on: bool,
    n_babble: int = 2,
    sigma: float = 0.3,
) -> dict[str, Any] | None:
    """Predict the survival-objective landscape for alternative motor commands.

    Runs the frozen forward models over a small candidate set (the emitted action,
    standing still, and a couple of babble perturbations) and reports the
    predicted reservoir deviation ("drive cost") each would incur - the decision
    landscape the policy is implicitly optimizing. On-demand / no_grad only.
    """
    import torch

    if base_motor is None:
        return None
    stack = bundle.stack
    device, dtype = z5.device, z5.dtype
    base = base_motor.detach().to(device=device, dtype=dtype)
    candidates: dict[str, "torch.Tensor"] = {"emitted": base, "still": torch.zeros_like(base)}
    for k in range(max(0, n_babble)):
        candidates[f"babble_{k}"] = base + sigma * torch.randn_like(base)

    idim = int(INTERO_PRED_DIM)
    use_drive = bool(drive_on and getattr(stack, "has_intero_model", False))
    pref_i = w_i = now_i = None
    if use_drive:
        pref_i = torch.as_tensor([preferred_intero_vector(idim)], device=device, dtype=dtype)
        w_i = torch.as_tensor([intero_preference_weights(idim)], device=device, dtype=dtype)
        now_i = torch.as_tensor(
            [controllable_intero_vector(homeostasis, idim)], device=device, dtype=dtype
        )

    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, u in candidates.items():
            entry: dict[str, Any] = {"action": name}
            if use_drive:
                pred_i = stack.forward_predict_intero(z5, u, now_i, detach_params=True)
                entry["drive_cost"] = round(float((w_i * (pred_i - pref_i).pow(2)).mean().cpu()), 6)
                entry["intero_pred"] = [round(float(x), 4) for x in _np1(pred_i)]
            pred_p = stack.forward_predict(z5, u, detach_params=True)
            entry["proprio_pred"] = [round(float(x), 4) for x in _np1(pred_p)[:7]]
            n = max(1, u.numel())
            entry["motor_rms"] = round(float(u.detach().pow(2).mean().sqrt().cpu()) if n else 0.0, 4)
            results.append(entry)
    if use_drive:
        results.sort(key=lambda r: r.get("drive_cost", 0.0))
    return {
        "candidates": results,
        "objective": "min drive_cost (predicted reservoir deviation from full)" if use_drive else "world-model preview only",
    }


def _intent_drivers(group: str, labels: list[str], blk: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    pred, pred_still, pref, w = blk["pred"], blk["pred_still"], blk["pref"], blk["w"]
    drivers: list[dict[str, Any]] = []
    n = min(len(pred), len(pref), len(w))
    for i in range(n):
        weight = float(w[i])
        if weight <= 0.0:  # the prior is indifferent to this dim
            continue
        dev_act = abs(float(pred[i]) - float(pref[i]))
        dev_still = abs(float(pred_still[i]) - float(pref[i]))
        drivers.append(
            {
                "goal": labels[i] if i < len(labels) else f"{group}_{i}",
                "group": group,
                "weight": round(weight, 4),
                "predicted": round(float(pred[i]), 4),
                "preferred": round(float(pref[i]), 4),
                "current": round(float(blk["now"][i]), 4) if "now" in blk and i < len(blk["now"]) else None,
                "deviation": round(dev_act, 4),
                # >0 means the chosen action is predicted to reduce the deviation
                # from the prior relative to standing still (it "helps").
                "action_delta": round(dev_still - dev_act, 4),
                "contribution": round(weight * dev_act * dev_act, 6),
            }
        )
    return drivers


def _intent(raw: dict[str, Any]) -> dict[str, Any]:
    drivers: list[dict[str, Any]] = []
    if "drive" in raw:
        drivers += _intent_drivers("drive", _INTERO_LABELS, raw["drive"])
    drivers.sort(key=lambda d: d["contribution"], reverse=True)

    if not drivers:
        return {
            "summary": "no active survival objective (homeostatic drive off); motion is exploratory",
            "drivers": [],
        }
    helps = sorted(
        (d for d in drivers if d["action_delta"] > 1e-4), key=lambda d: d["action_delta"], reverse=True
    )
    top_need = drivers[0]
    if helps:
        mover = helps[0]
        sign = "+" if mover["action_delta"] > 0 else ""
        summary = (
            f"acting to raise {mover['goal']} (predicted {sign}{mover['action_delta']} vs. standing still); "
            f"most depleted: {top_need['goal']} (dev {top_need['deviation']})"
        )
    else:
        summary = f"conserving; most depleted: {top_need['goal']} (dev {top_need['deviation']})"
    return {"summary": summary, "drivers": drivers[:10]}


def _surprise(raw: dict[str, Any], fwd_dim: int) -> dict[str, Any]:
    blk = raw.get("surprise")
    if not blk:
        return {"dims": [], "mean_abs_residual": None, "summary": "no prior prediction to compare"}
    pred, target = blk["pred"], blk["target"]
    labels = _proprio_labels(fwd_dim)
    dims: list[dict[str, Any]] = []
    n = min(len(pred), len(target))
    for i in range(n):
        res = abs(float(pred[i]) - float(target[i]))
        dims.append(
            {
                "name": labels[i] if i < len(labels) else f"dim_{i}",
                "predicted": round(float(pred[i]), 4),
                "actual": round(float(target[i]), 4),
                "residual": round(res, 4),
            }
        )
    dims.sort(key=lambda d: d["residual"], reverse=True)
    mean_res = round(float(np.mean([d["residual"] for d in dims])) if dims else 0.0, 6)
    worst = dims[0]["name"] if dims else "?"
    return {
        "dims": dims[:8],
        "mean_abs_residual": mean_res,
        "summary": f"body-model surprise {mean_res} (worst: {worst})",
    }


def _recalled_episode(
    episodic: "EpisodicStore | None", qv: np.ndarray | None, current_cycle: int
) -> dict[str, Any] | None:
    if episodic is None or qv is None:
        return None
    try:
        hits = episodic.search_similar(qv, top_k=2, exclude_cycle=current_cycle)
    except Exception:  # interpretability must never break the cycle
        return None
    if not hits:
        return None
    top = hits[0]
    summary = top.get("summary") or {}
    action = summary.get("action") or {}
    return {
        "cycle": top.get("cycle_index"),
        "similarity": round(float(top.get("similarity", 0.0)), 4),
        "salience": round(float(top.get("salience", 0.0)), 4),
        "priority": summary.get("priority"),
        "pain": summary.get("pain"),
        "pleasure": summary.get("pleasure"),
        "viability": summary.get("viability"),
        "action_type": action.get("type") if isinstance(action, dict) else None,
    }


def build(
    *,
    cycle: int,
    raw: dict[str, Any] | None,
    fwd_dim: int,
    affect: dict[str, Any],
    episodic: "EpisodicStore | None" = None,
    qv: np.ndarray | None = None,
    salient: dict[str, Any] | None = None,
    counterfactuals: dict[str, Any] | None = None,
    probes: dict[str, Any] | None = None,
    self_model: dict[str, Any] | None = None,
    workspace: dict[str, Any] | None = None,
) -> CognitiveTrace:
    """Assemble the structured trace from captured raw arrays + cycle scalars."""
    raw = raw or {}
    return CognitiveTrace(
        cycle=cycle,
        intent=_intent(raw),
        self_surprise=_surprise(raw, fwd_dim),
        affect=affect,
        recalled_episode=_recalled_episode(episodic, qv, cycle),
        salient=salient if salient is not None else raw.get("attribution"),
        counterfactuals=counterfactuals,
        probes=probes,
        self_model=self_model,
        workspace=workspace,
    )
