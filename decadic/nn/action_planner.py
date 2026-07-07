"""WS-EXPAND E1.6 — online rollout action selection (the deliberate path).

On an ESCALATED cycle, before the motor command leaves the pipeline: sample K
perturbations of the chosen action, roll the agent's own interoceptive world
model a short horizon for each, score every candidate by the same deficit-gated
successor value the Layer-2 shaping term uses, and return a BOUNDED bias toward
the best candidate. "Before I move, imagine a few variations of this move and
lean toward the one my own model says feeds me."

Evidence-review guardrails, all structural here:
- SHORT horizon truncated by the successor-features value (compounding-error
  control; the rollout cannot fabricate long-range fantasies).
- BIAS, never override: the returned delta is clamped to ``bias_max`` and
  scaled by ``bias_gain`` — the reactive policy stays in charge.
- Runs ONLY on the (already refractory-bounded) deliberate path, under
  ``no_grad`` (a plan is not a gradient path the policy can edit).
- Trust-ramped by the caller via the SF value ramp: a naive agent (ramp 0)
  never plans -> byte-identical at birth.
- Deterministic: candidate noise comes from a dedicated generator seeded with
  the cycle index, so runs replay exactly and the global RNG stream is
  untouched (no parity drift for anything downstream).

Honest scope (mirrors imagination.py): the stack has no latent-dynamics model,
so z5 is held fixed across the rollout; only the interoceptive consequence of
holding each candidate action is imagined, plus a terminal psi(z5, u) value.
That is a bounded approximation — sufficient for ranking action variations,
not for long-horizon trajectory planning.
"""

from __future__ import annotations

import torch


def plan_action_bias(
    stack,
    z5: torch.Tensor,
    motor_u: torch.Tensor,
    intero_now: torch.Tensor,
    w_gated: torch.Tensor,
    *,
    k: int,
    horizon: int,
    gamma: float,
    sigma: float,
    bias_gain: float,
    bias_max: float,
    cycle: int,
) -> "tuple[torch.Tensor, dict] | None":
    """Bounded additive motor bias toward the best imagined candidate, or None.

    Returns ``(delta, telemetry)`` where ``delta`` is ready to add to the motor
    command (already gain-scaled and clamped), or ``None`` when planning is a
    no-op (k<1, non-finite inputs, model failure, or the chosen action already
    scored best). Never raises; never mutates its inputs; all model calls are
    ``no_grad``.
    """
    try:
        k = int(k)
        horizon = max(1, int(horizon))
        if k < 1 or bias_gain <= 0.0 or bias_max <= 0.0:
            return None
        with torch.no_grad():
            base_u = motor_u.detach()
            if base_u.dim() == 1:
                base_u = base_u.unsqueeze(0)
            if not torch.isfinite(base_u).all():
                return None
            n_act = base_u.shape[-1]
            # Deterministic candidate noise: dedicated CPU generator seeded by
            # the cycle index -> replayable, and the global RNG is untouched.
            gen = torch.Generator(device="cpu")
            gen.manual_seed(0x5EED ^ (int(cycle) * 2654435761 & 0x7FFFFFFF))
            noise = torch.randn(k, n_act, generator=gen) * float(sigma)
            noise = noise.to(device=base_u.device, dtype=base_u.dtype)
            cands = torch.cat([base_u, (base_u + noise).clamp(-1.0, 1.0)], dim=0)
            n = cands.shape[0]  # k+1, candidate 0 == the policy's own choice

            z5_rep = z5.detach().expand(n, -1)
            cur = intero_now.detach().expand(n, -1).clone()
            w = w_gated.detach().reshape(1, -1)
            ret = torch.zeros(n, device=base_u.device, dtype=base_u.dtype)
            disc = 1.0
            for _ in range(horizon):
                nxt = stack.forward_predict_intero(z5_rep, cands, cur)
                if not torch.isfinite(nxt).all():
                    return None
                ret = ret + disc * (w * (nxt - cur)).sum(dim=1)
                cur = nxt
                disc *= float(gamma)
            # Terminal truncation by the learned successor value (the guardrail
            # against compounding model error on longer horizons).
            psi = stack.successor_predict(z5_rep, cands, detach_params=True)
            if not torch.isfinite(psi).all():
                return None
            ret = ret + disc * (w * psi).sum(dim=1)
            if not torch.isfinite(ret).all():
                return None
            best = int(ret.argmax().item())
            if best == 0:
                return None  # the chosen action already scored best: no bias
            delta = (cands[best : best + 1] - base_u).clamp(-float(bias_max), float(bias_max))
            delta = float(bias_gain) * delta
            telemetry = {
                "planner_candidates": n,
                "planner_best_gain": float((ret[best] - ret[0]).item()),
                "planner_bias_linf": float(delta.abs().max().item()),
            }
            return delta.reshape(motor_u.shape), telemetry
    except Exception:
        return None  # planning can never break the cycle
