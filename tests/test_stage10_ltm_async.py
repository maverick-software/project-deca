from __future__ import annotations

from decadic.cycle.stages import stage_10
from decadic.cycle.types import CycleContext
from decadic.memory.episodic_store import EpisodicStore
from decadic.state.perceptual_state import PerceptualState
from decadic.state.state_bus import StateBus
from decadic.state.viability import ViabilityState
from decadic.state.working_memory import MemorySlot


class _AsyncOnlyGraph:
    def __init__(self) -> None:
        self.jobs: list[dict] = []

    def enqueue_consolidation_job(self, slots, **kwargs):
        self.jobs.append({"slots": list(slots), **kwargs})
        return {
            "status": "queued_consolidation",
            "queued": True,
            "accepted_ids": [],
            "semantic_update": {},
        }

    def consolidate(self, *args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("Stage 10 called consolidate synchronously")

    def bump_edge(self, *args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("Stage 10 called bump_edge synchronously")

    def record_semantic_evidence(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("Stage 10 called semantic evidence synchronously")

    def cached_belief_stats(self):
        return {"total_property_beliefs": 0, "avg_property_confidence": 0.0}


def test_stage10_enqueues_ltm_job_without_synchronous_graph_work():
    perceptual = PerceptualState(perception_mode="discovered")
    slot = MemorySlot(
        entity_id="obj",
        appearance=[1.0, 0.0],
        seen_count=3,
        confidence=1.0,
        precision=1.0,
        scene_entity_id="scene-obj",
        property_evidence={"compactness": 0.8},
    )
    perceptual.working_memory.slots[slot.entity_id] = slot
    perceptual.discovery_health = {"reason": "healthy"}
    perceptual.recent_events = [{"type": "contact", "intensity": 0.5}]
    graph = _AsyncOnlyGraph()
    ctx = CycleContext(
        state_bus=StateBus(cycle_index=12),
        perceptual=perceptual,
        viability=ViabilityState(),
        episodic=EpisodicStore(),
        ltm_graph=graph,  # type: ignore[arg-type]
        perception_mode="discovered",
        latents={"stage_traces": [], "action": "explore"},
    )

    tr = stage_10.run(ctx)

    assert tr.stage == 10
    assert len(graph.jobs) == 1
    assert graph.jobs[0]["cycle"] == 12
    assert perceptual.ltm_consolidation["status"] == "queued_consolidation"
    assert perceptual.discovery_health["ltm_write"] == "queued_consolidation"
