"""The represented self: the agent's own body/affect/capability as graph content.

Self-model program, Phase 5. The architecture already has an implicit self -- a
proprioceptive ``role: self`` node, an agency ("this is mine") signal, and the
A‖C‖E self-report carried by the Phase-1 spine. But the self is not yet a
*represented object*: its interoceptive state, affect, and capabilities are not
written as content on the self-node, and the self-node's own summary is not fed
back as a thing the agent models.

This module assembles a compact ``RepresentedSelf`` from the live homeostatic
reservoirs (interoception), the affect scalars, and the discovered body schema
(capability = how much of the world the agent has learned to command). It exposes:

- ``embedding()``  -- a fixed 8-D vector fed back through the self-model spine, so
  the represented self conditions the next cycle (alongside the A‖C‖E report).
- ``node_content()`` -- intero/affect/capability fields merged onto the self-node.
- ``semantic_edges()`` -- typed "controls" edges from the self to its body parts,
  so the self's relation to its own effectors is explicit in the relational graph.

Pure + dependency-light (numpy only) so it is trivially testable and stays well
under the file-size budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

REPSELF_DIM = 8  # [viability, hydration, energy, integrity, pain, pleasure, priority, capability]


def _u(x: float) -> float:
    """Clamp to a unit interval (defensive; the inputs are already ~[0,1])."""
    return float(min(1.0, max(0.0, x)))


@dataclass
class RepresentedSelf:
    """A compact, structured snapshot of the agent as an object it models."""

    intero: tuple[float, float, float, float]  # viability, hydration, energy, integrity (each /100)
    affect: tuple[float, float, float]  # pain, pleasure, priority
    capability: float  # mean agency over learned body parts (or part fraction)
    n_parts: int = 0  # how many slots have been promoted to self_part

    def embedding(self) -> np.ndarray:
        """Fixed REPSELF_DIM vector fed back through the spine (the modelled self)."""
        v = np.array([*self.intero, *self.affect, self.capability], dtype=np.float32)
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    def node_content(self) -> dict[str, Any]:
        """Intero/affect/capability content to merge onto the egocentric self-node."""
        hy, en, it = self.intero[1], self.intero[2], self.intero[3]
        pa, pl, pr = self.affect
        return {
            "intero": {
                "viability": round(self.intero[0], 4),
                "hydration": round(hy, 4),
                "energy": round(en, 4),
                "integrity": round(it, 4),
            },
            "affect": {
                "pain": round(pa, 4),
                "pleasure": round(pl, 4),
                "priority": round(pr, 4),
            },
            "capability": round(self.capability, 4),
            "n_parts": int(self.n_parts),
        }

    def semantic_edges(
        self, self_id: str, entity_nodes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Typed "controls" edges from the self to each learned body part."""
        edges: list[dict[str, Any]] = []
        for n in entity_nodes:
            if n.get("kind") == "self_part":
                edges.append(
                    {
                        "source": self_id,
                        "target": str(n.get("id", "")),
                        "kind": "controls",
                        "weight": round(float(n.get("agency", 1.0)), 4),
                    }
                )
        return edges


def build_represented_self(
    *,
    viability: float,
    homeostasis: Any | None,
    pain: float,
    pleasure: float,
    priority: float,
    working_memory: Any | None,
) -> RepresentedSelf:
    """Assemble the represented self from live interoception, affect, and capability."""
    via = _u(float(viability) / 100.0)
    if homeostasis is not None:
        hy = _u(float(getattr(homeostasis, "hydration", viability)) / 100.0)
        en = _u(float(getattr(homeostasis, "energy", viability)) / 100.0)
        it = _u(float(getattr(homeostasis, "integrity", viability)) / 100.0)
    else:
        hy = en = it = via

    cap = 0.0
    n_parts = 0
    slots = getattr(working_memory, "slots", None)
    if isinstance(slots, dict) and slots:
        agencies = [
            float(getattr(s, "agency", 0.0))
            for s in slots.values()
            if getattr(s, "kind", "") == "self_part"
        ]
        n_parts = len(agencies)
        if agencies:
            cap = float(sum(agencies) / len(agencies))
    return RepresentedSelf(
        intero=(via, hy, en, it),
        affect=(_u(float(pain)), _u(float(pleasure)), _u(float(priority))),
        capability=_u(cap),
        n_parts=n_parts,
    )
