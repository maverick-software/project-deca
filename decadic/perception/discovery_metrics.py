"""Evaluation-only scoring of the discovered world graph against oracle truth.

In ``discovered`` perception mode the agent builds its egocentric graph from its
own camera + proprioception, and the simulator's ``world_state.entities`` are
*never* fed into cognition. They are kept here purely as ground truth so we can
measure how well perception is recovering the world: detection precision/recall
(by egocentric direction), identity stability (how much coined ids churn), and
body-part agency accuracy (do the slots flagged ``self_part`` line up with the
real hands/feet). Nothing in this module influences the agent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _rel_of(node: dict[str, Any], self_pos: list[float] | None) -> list[float] | None:
    rel = node.get("relative")
    if isinstance(rel, list) and len(rel) >= 3:
        try:
            return [float(rel[0]), float(rel[1]), float(rel[2])]
        except (TypeError, ValueError):
            return None
    pos = node.get("position")
    if isinstance(pos, list) and len(pos) >= 3 and self_pos is not None:
        try:
            return [float(pos[i]) - float(self_pos[i]) for i in range(3)]
        except (TypeError, ValueError):
            return None
    return None


def _direction_cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(c * c for c in a))
    nb = math.sqrt(sum(c * c for c in b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(3))
    return max(-1.0, min(1.0, dot / (na * nb)))


def _greedy_match(
    discovered: list[list[float]],
    truth: list[list[float]],
    *,
    min_cos: float,
) -> int:
    """Greedy 1:1 matches between discovered and truth egocentric directions."""
    used: set[int] = set()
    tp = 0
    for d in discovered:
        best_j, best_cos = -1, min_cos
        for j, t in enumerate(truth):
            if j in used:
                continue
            c = _direction_cos(d, t)
            if c >= best_cos:
                best_cos, best_j = c, j
        if best_j >= 0:
            used.add(best_j)
            tp += 1
    return tp


@dataclass
class DiscoveryEvaluator:
    """Running discovery-quality stats; updated each discovered-mode graph build."""

    ema: float = 0.1
    match_min_cos: float = 0.85  # ~31 deg cone counts as the same egocentric direction
    updates: int = 0
    precision_ema: float = 0.0
    recall_ema: float = 0.0
    id_churn_ema: float = 0.0
    body_part_accuracy_ema: float = 0.0
    last_detected: int = 0
    last_oracle: int = 0
    last_matched: int = 0
    last_body_parts_found: int = 0
    last_body_parts_truth: int = 0
    _prev_ids: set[str] = field(default_factory=set)

    def update(
        self,
        discovered_nodes: list[dict[str, Any]],
        oracle_truth: list[dict[str, Any]] | None,
        *,
        self_pos: list[float] | None = None,
        body_parts_truth: dict[str, list[float]] | None = None,
    ) -> None:
        ent = [n for n in discovered_nodes if n.get("role") == "entity"]
        disc_dirs = [r for n in ent if (r := _rel_of(n, self_pos)) is not None]
        truth = oracle_truth or []
        truth_dirs = [r for t in truth if (r := _rel_of(t, self_pos)) is not None]

        tp = _greedy_match(disc_dirs, truth_dirs, min_cos=self.match_min_cos)
        precision = tp / max(1, len(disc_dirs))
        recall = tp / max(1, len(truth_dirs))

        cur_ids = {str(n.get("id", "")) for n in ent if n.get("id")}
        new_ids = cur_ids - self._prev_ids
        churn = len(new_ids) / max(1, len(cur_ids))
        self._prev_ids = cur_ids

        # Body-part agency accuracy: self_part directions vs real limb directions.
        bp_found = 0
        bp_truth = 0
        if body_parts_truth:
            self_parts = [
                r
                for n in discovered_nodes
                if n.get("kind") == "self_part" and (r := _rel_of(n, self_pos)) is not None
            ]
            truth_bp = [list(v) for v in body_parts_truth.values() if isinstance(v, list) and len(v) >= 3]
            # truth limbs are world positions; convert to egocentric relative.
            truth_bp_rel = []
            for v in truth_bp:
                if self_pos is not None and len(self_pos) >= 3:
                    truth_bp_rel.append([float(v[i]) - float(self_pos[i]) for i in range(3)])
                else:
                    truth_bp_rel.append([float(v[0]), float(v[1]), float(v[2])])
            bp_truth = len(truth_bp_rel)
            bp_found = _greedy_match(self_parts, truth_bp_rel, min_cos=self.match_min_cos)
            bp_acc = bp_found / max(1, bp_truth)
            self.body_part_accuracy_ema = self._blend(self.body_part_accuracy_ema, bp_acc)

        a = self.ema
        self.precision_ema = self._blend(self.precision_ema, precision)
        self.recall_ema = self._blend(self.recall_ema, recall)
        self.id_churn_ema = self._blend(self.id_churn_ema, churn)
        self.last_detected = len(disc_dirs)
        self.last_oracle = len(truth_dirs)
        self.last_matched = tp
        self.last_body_parts_found = bp_found
        self.last_body_parts_truth = bp_truth
        self.updates += 1

    def _blend(self, prev: float, new: float) -> float:
        if self.updates == 0:
            return float(new)
        return (1.0 - self.ema) * prev + self.ema * float(new)

    def snapshot(self) -> dict[str, Any]:
        return {
            "updates": self.updates,
            "precision": round(self.precision_ema, 4),
            "recall": round(self.recall_ema, 4),
            "id_churn": round(self.id_churn_ema, 4),
            "id_stability": round(1.0 - self.id_churn_ema, 4),
            "body_part_accuracy": round(self.body_part_accuracy_ema, 4),
            "last_detected": self.last_detected,
            "last_oracle": self.last_oracle,
            "last_matched": self.last_matched,
            "last_body_parts_found": self.last_body_parts_found,
            "last_body_parts_truth": self.last_body_parts_truth,
        }
