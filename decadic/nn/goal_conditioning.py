"""Goal conditioning for the motor policy (WS-FORAGE M3/M4).

The policy is otherwise a reactive function of the *current* percept, so it can
approach a resource it SEES but cannot pursue one it merely NEEDS-and-remembers.
This module builds a small fixed-width **goal vector** describing the active
homeostatic goal (which need, how deprived) -- and, from M4, an egocentric
bearing to the remembered resource -- which is folded into the policy input via a
zero-initialized ingress in the neural stack. Zero-init means a fresh agent is
byte-identical until experience trains the ingress; the capability then emerges.

The vector layout is FROZEN here (house rule G5): M3 owns [0:4], M4 reserves
[4:8]. Pure python (no torch) so it is trivially unit-testable and cannot become
a hot-path cost.

Layout (``GOAL_VEC_DIM`` = 12):
    [0:3]  need one-hot over GOAL_LABELS (hydration, energy, integrity)
    [3]    deficit magnitude of the active need in [0, 1]
    [4:6]  egocentric bearing (cos az, sin az) to the remembered target   -- M4
    [6]    normalized distance to the remembered target                    -- M4
    [7]    target-valid mask (1.0 if a remembered target exists, else 0.0) -- M4
    [8:10] positional code: world (x, y) / pos-scale, clamped to [-1, 1]  -- E1.3
    [10]   sin(yaw)                                                        -- E1.3
    [11]   cos(yaw)  (pose-valid <=> [10]^2+[11]^2 == 1; all-zero = none)  -- E1.3

The E1.3 slots grew the layout 8 -> 12 (WS-EXPAND). Old checkpoints carry an
8-wide ``goal_ingress`` weight; the bundle load pads it with ZERO columns
(function-preserving: the new inputs contribute nothing until trained), so a
trained M3/M4 ingress survives the migration.
"""

from __future__ import annotations

GOAL_VEC_DIM = 12

# Canonical need order. Mirrors decadic.state.goal_lifecycle.GOAL_LABELS; a
# consistency test pins them together so the one-hot never silently drifts (we
# keep a local copy so this nn module stays decoupled from state/).
GOAL_LABELS: tuple[str, ...] = ("hydration", "energy", "integrity")

# Frozen slice boundaries (referenced by M4/E1.3 and the layout test).
NEED_ONEHOT = slice(0, 3)
DEFICIT_IDX = 3
BEARING = slice(4, 6)
DISTANCE_IDX = 6
TARGET_MASK_IDX = 7
POSITION = slice(8, 10)  # E1.3
YAW_SIN_IDX = 10  # E1.3
YAW_COS_IDX = 11  # E1.3


def encode_goal(
    goal_id: str | None,
    deficit: float,
    *,
    bearing_cos: float | None = None,
    bearing_sin: float | None = None,
    distance: float | None = None,
    pos_nx: float | None = None,
    pos_ny: float | None = None,
    yaw: float | None = None,
) -> list[float]:
    """Build the goal vector for the active goal.

    ``goal_id`` is the active need label (or ``None`` -> all-zero vector, i.e. no
    conditioning). ``deficit`` is that need's depletion in [0, 1]. The M4 bearing
    fields default to a zero, mask-off target (so M3 alone leaves [4:8] zero and
    the target mask off). The E1.3 positional code (``pos_nx``/``pos_ny`` already
    normalized by the caller, plus ``yaw`` radians) defaults to all-zero -> "no
    pose", so pre-E1 callers are unchanged. All values are finite and clamped.
    """
    v = [0.0] * GOAL_VEC_DIM
    if not goal_id:
        return v
    try:
        idx = GOAL_LABELS.index(goal_id)
    except ValueError:
        # Unknown label -> encode only the deficit magnitude (no need one-hot),
        # never crash the cycle.
        idx = -1
    if 0 <= idx < 3:
        v[idx] = 1.0
    v[DEFICIT_IDX] = _clamp01(deficit)
    if bearing_cos is not None and bearing_sin is not None and distance is not None:
        v[4] = _clampf(bearing_cos, -1.0, 1.0)
        v[5] = _clampf(bearing_sin, -1.0, 1.0)
        v[DISTANCE_IDX] = _clamp01(distance)
        v[TARGET_MASK_IDX] = 1.0
    if pos_nx is not None and pos_ny is not None and yaw is not None:
        import math as _math

        v[8] = _clampf(pos_nx, -1.0, 1.0)
        v[9] = _clampf(pos_ny, -1.0, 1.0)
        yf = _clampf(yaw, -1e9, 1e9)
        v[YAW_SIN_IDX] = _math.sin(yf)
        v[YAW_COS_IDX] = _math.cos(yf)
    return v


def _clamp01(x: float) -> float:
    return _clampf(x, 0.0, 1.0)


def _clampf(x: float, lo: float, hi: float) -> float:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return lo
    if xf != xf:  # NaN
        return lo
    return lo if xf < lo else hi if xf > hi else xf
