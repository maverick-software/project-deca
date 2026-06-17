"""Return-based credit assignment over an ordered episode.

The live policy objective is one-step (greedy drive reduction), so the food/water
relief never reaches the long run of postural/locomotor commands that PRECEDED
it. These pure helpers compute forward-view lambda-returns over an ordered
episode -- the "respect the journey" credit smear -- so a distal reward propagates
backward and lands on the actions that led there.

Two flavors, sharing the same recursion:

- ``lambda_returns(rewards)`` -> scalar lambda-return per step (the value target).
- ``lambda_returns_vec(feats)`` -> vector lambda-return per step (the successor-
  features target psi_t ~= sum_k (gamma*lam)^k phi_{t+k}).

Forward-view recursion (Sutton & Barto):
    G_t = r_t + gamma * [ (1 - lam) * V(s_{t+1}) + lam * G_{t+1} ]
With no learned value available at accumulation time (``values=None`` -> zeros)
this reduces to the truncated, (gamma*lam)-weighted return -- exactly the
eligibility-trace smear where distal credit decays as it travels back. Pass a
``values`` array to bootstrap (e.g. TD(lambda) inside the consolidator). The
terminal bootstrap is 0 (episodes always close: achieved, abandoned, truncated,
or died), so returns are well-defined and need no external value at the boundary.

Pure Python (lists/floats), no torch / numpy, so it is trivially unit-testable.
"""

from __future__ import annotations

from typing import Sequence


def lambda_returns(
    rewards: Sequence[float],
    *,
    gamma: float,
    lam: float,
    values: "Sequence[float] | None" = None,
    bootstrap: float = 0.0,
) -> list[float]:
    """Forward-view lambda-returns for a scalar reward sequence.

    ``values[t]`` approximates ``V(s_t)`` for bootstrapping; ``None`` -> zeros.
    ``bootstrap`` is ``V(s_n)`` past the last step (0 for a terminal episode).
    """
    n = len(rewards)
    if n == 0:
        return []
    gamma = float(gamma)
    lam = float(lam)
    vals = [0.0] * n if values is None else [float(v) for v in values]
    out = [0.0] * n
    next_val = float(bootstrap)
    next_g = float(bootstrap)
    for t in range(n - 1, -1, -1):
        g = float(rewards[t]) + gamma * ((1.0 - lam) * next_val + lam * next_g)
        out[t] = g
        next_val = vals[t]
        next_g = g
    return out


def lambda_returns_vec(
    feats: Sequence[Sequence[float]],
    *,
    gamma: float,
    lam: float,
    values: "Sequence[Sequence[float]] | None" = None,
    bootstrap: "Sequence[float] | None" = None,
) -> list[list[float]]:
    """Vector forward-view lambda-returns (successor-features targets).

    ``feats[t]`` is the per-step feature vector phi_t (all equal length).
    Returns ``psi[t]``, the discounted future-feature target for each step.
    """
    n = len(feats)
    if n == 0:
        return []
    dim = len(feats[0])
    gamma = float(gamma)
    lam = float(lam)
    vals = (
        [[0.0] * dim for _ in range(n)]
        if values is None
        else [[float(x) for x in v] for v in values]
    )
    boot = [0.0] * dim if bootstrap is None else [float(x) for x in bootstrap]
    out: list[list[float]] = [[0.0] * dim for _ in range(n)]
    next_val = list(boot)
    next_g = list(boot)
    for t in range(n - 1, -1, -1):
        ft = feats[t]
        g = [
            float(ft[i]) + gamma * ((1.0 - lam) * next_val[i] + lam * next_g[i])
            for i in range(dim)
        ]
        out[t] = g
        next_val = vals[t]
        next_g = g
    return out
