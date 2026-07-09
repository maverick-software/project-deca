"""WS-EXPAND E4 — cached (habit) vs deliberate dual control: the pure parts.

The gate already arbitrates skip-vs-escalate every cycle; E4 routes that
decision to two motor sources. On ESCALATED cycles the full goal-conditioned
policy acts and its (input, action) pair is banked as TEACHER data. On SKIP
cycles the motor command blends toward a small cached head's action — but only
in proportion to a TRUST weight earned by distillation quality. A habit is a
deliberation that has proven it can be compressed.

Guardrails (evidence review, E4):
- **Teacher outputs only.** The buffer accepts pairs exclusively from
  escalated cycles, so the cached head can never train on its own actions
  (the distillation-collapse failure).
- **Trust is earned, never assumed.** The blend weight is 0 until the
  distillation-loss EMA falls below threshold — a fresh agent is
  byte-identical, and a stale habit (teacher moved, loss rises) automatically
  loses the body again.
- **Staleness/thrash guards live elsewhere by design:** the E2.3 surprise
  channel raises escalation propensity when the world shifts (forcing the
  deliberate path to retake control and re-teach), and the gate's
  hysteresis + Type-2 refractory already bound arbitration flapping.
- **Deterministic distillation batches** (most-recent window, no RNG): runs
  replay exactly, and recency makes the distillation continually track the
  evolving teacher.

This module holds the torch-light machinery (ring buffer + trust math) so it
is unit-testable without a stack; the cached head itself lives in
``neural_stack`` beside the policy.
"""

from __future__ import annotations

from typing import Any


class DistillBuffer:
    """Ring buffer of (input, teacher-action) pairs from escalated cycles."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._items: list[tuple[Any, Any]] = []
        self._next = 0
        self.total_pushed = 0

    def __len__(self) -> int:
        return len(self._items)

    def push(self, z: Any, u: Any) -> None:
        """Bank one teacher pair (caller detaches; escalated cycles only)."""
        if len(self._items) < self.capacity:
            self._items.append((z, u))
        else:
            self._items[self._next] = (z, u)
            self._next = (self._next + 1) % self.capacity
        self.total_pushed += 1

    def recent(self, n: int) -> "list[tuple[Any, Any]]":
        """The most recently pushed n pairs (deterministic; no RNG).

        Recency keeps the distillation tracking the evolving teacher — the
        continual-distillation requirement from the evidence review.
        """
        n = max(0, int(n))
        if n == 0 or not self._items:
            return []
        if len(self._items) < self.capacity:
            return self._items[-n:]
        # Full ring: order by push time ending at the newest slot.
        newest_first: list[tuple[Any, Any]] = []
        idx = (self._next - 1) % self.capacity
        for _ in range(min(n, self.capacity)):
            newest_first.append(self._items[idx])
            idx = (idx - 1) % self.capacity
        newest_first.reverse()
        return newest_first

    def window(self, offset: int, n: int) -> "list[tuple[Any, Any]]":
        """A deterministic ROTATING window of n pairs starting at ``offset``.

        Trust-overfit guard (caught by the distillation test, and the same
        exposure exists live during long skip stretches when no new teacher
        pairs arrive): training always on the most-recent window lets the head
        memorize those 32 pairs — loss EMA collapses, trust opens, but the
        habit hasn't compressed the teacher. Rotating the window by the cycle
        index sweeps the WHOLE buffer over time, so a low loss EMA means fit
        across everything the teacher has demonstrated. Still RNG-free.
        """
        n = max(0, int(n))
        total = len(self._items)
        if n == 0 or total == 0:
            return []
        if n >= total:
            return list(self._items)
        start = int(offset) % total
        out = []
        for i in range(n):
            out.append(self._items[(start + i) % total])
        return out

    def clear(self) -> None:
        self._items = []
        self._next = 0


def trust_weight(distill_loss_ema: "float | None", *, threshold: float, max_w: float) -> float:
    """Blend weight toward the cached action, earned by distillation quality.

    None (never distilled) or EMA >= threshold -> 0.0 (deliberate policy keeps
    the body; birth-identity). EMA at 0 -> max_w. Linear in between — trust
    grows exactly as the habit demonstrates it reproduces the teacher, and
    decays the moment it stops (stale-habit guard works both directions).
    """
    if distill_loss_ema is None:
        return 0.0
    try:
        ema = float(distill_loss_ema)
    except (TypeError, ValueError):
        return 0.0
    if ema != ema or ema < 0.0:  # NaN / nonsense -> no trust
        return 0.0
    threshold = max(1e-12, float(threshold))
    if ema >= threshold:
        return 0.0
    return max(0.0, min(1.0, float(max_w))) * (1.0 - ema / threshold)
