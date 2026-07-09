"""WS-DEPTH D2 — the unified self-vector and its workspace candidacy.

Deca represents itself in silos (represented self, felt body-state, attention
schema, metacognitive confidence). This module binds the self-relevant state
each cycle already exposes into ONE frozen-layout vector, packs it into a
workspace-competition candidate, and prices its salience by interoceptive
urgency — so the self competes for conscious access like any other content,
and deprivation turns attention inward.

Parity: candidate salience carries a birth ramp (0 at cycle 0) and the whole
lane is flag-gated; at ramp-zero the self can never win an ignition, so the
baseline is untouched. Pure python/numpy; no torch.

SELF_VEC frozen layout (SELF_VEC_DIM = 16):
    [0:4]   metacognition summary (element E pooled to 4)
    [4]     pain scalar          [5]  pleasure scalar
    [6]     interoceptive urgency (max reservoir deficit, [0,1])
    [7]     viability / 100
    [8:12]  attention-schema prediction pooled to 4 (zeros pre-schema)
    [12]    calibrated next-error prediction (squashed)
    [13]    calibrated P(drive improves)
    [14]    rest state (0/1)
    [15]    reserved
"""

from __future__ import annotations

from typing import Any

import numpy as np

SELF_VEC_DIM = 16


def _pool(vec: Any, k: int) -> "list[float]":
    """Mean-pool an arbitrary-length sequence into k chunks (zeros if empty)."""
    try:
        arr = [float(x) for x in (vec if vec is not None else [])]
    except (TypeError, ValueError):
        arr = []
    if not arr:
        return [0.0] * k
    n = len(arr)
    out = []
    for i in range(k):
        lo, hi = (i * n) // k, max((i * n) // k + 1, ((i + 1) * n) // k)
        chunk = arr[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if v == v else default


def build_self_vec(
    *,
    metacog: Any,
    pain: float,
    pleasure: float,
    urgency: float,
    viability: float,
    schema_pred: Any = None,
    cal_next_err: float = 0.0,
    cal_p_improve: float = 0.0,
    rest_active: bool = False,
) -> "list[float]":
    v = [0.0] * SELF_VEC_DIM
    v[0:4] = _pool(metacog, 4)
    v[4] = max(-1.0, min(1.0, _f(pain)))
    v[5] = max(-1.0, min(1.0, _f(pleasure)))
    v[6] = max(0.0, min(1.0, _f(urgency)))
    v[7] = max(0.0, min(1.0, _f(viability) / 100.0))
    v[8:12] = _pool(schema_pred, 4)
    v[12] = max(-1.0, min(1.0, _f(cal_next_err)))
    v[13] = max(0.0, min(1.0, _f(cal_p_improve)))
    v[14] = 1.0 if rest_active else 0.0
    return v


def pack_candidate(self_vec: "list[float]", dim: int) -> np.ndarray:
    """Deterministically tile the self-vec into a workspace-candidate row."""
    base = np.asarray(self_vec, dtype=np.float32)
    reps = int(np.ceil(dim / max(1, base.size)))
    return np.tile(base, reps)[:dim]


def candidate_salience(urgency: float, cycle: int, *, gain: float, ramp_cycles: int) -> float:
    """Urgency-priced, birth-ramped salience for the self candidate.

    0 at cycle 0 (parity: the self cannot win an ignition at birth); grows
    linearly to gain x urgency over the ramp — the drive-gain pattern.
    """
    r = min(1.0, max(0.0, float(cycle) / max(1, int(ramp_cycles))))
    return max(0.0, float(gain)) * max(0.0, min(1.0, _f(urgency))) * r
