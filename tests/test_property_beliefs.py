from __future__ import annotations

from dataclasses import dataclass, field

from decadic.memory.ltm_write_behind import WriteBehindLongTermGraph
from decadic.memory.semantic_graph import LongTermGraph
from decadic.perception.object_files import object_files_from_proposals
from decadic.state.working_memory import MemorySlot, WorkingMemory


def _belief(node: dict, key: str) -> dict:
    for b in node.get("property_beliefs", []):
        if b["property_key"] == key:
            return b
    raise AssertionError(f"missing belief {key}")


def test_repeated_property_evidence_strengthens_one_belief():
    g = LongTermGraph()
    slot = MemorySlot(
        entity_id="obj",
        appearance=[1.0, 0.0],
        seen_count=3,
        confidence=1.0,
        property_evidence={"roundness": 0.8, "compactness": 0.7},
    )
    for cycle in range(6):
        g.consolidate([slot], cycle=cycle, min_seen=2)
    snap = g.snapshot()
    assert snap["total_nodes"] == 1
    assert snap["total_property_beliefs"] == 2
    b = _belief(snap["nodes"][0], "roundness")
    assert b["evidence_count"] == 6
    assert b["confidence"] > 0.25
    assert b["mean"] == 0.8


def test_contradictory_property_evidence_marks_unstable():
    g = LongTermGraph()
    slot = MemorySlot(
        entity_id="obj",
        appearance=[1.0, 0.0],
        seen_count=3,
        confidence=1.0,
        property_evidence={"roundness": 0.9},
    )
    for cycle in range(6):
        g.consolidate([slot], cycle=cycle, min_seen=2)
    slot.property_evidence = {"roundness": 0.0}
    g.consolidate([slot], cycle=7, min_seen=2)
    snap = g.snapshot()
    b = _belief(snap["nodes"][0], "roundness")
    assert b["unstable"] is True
    assert b["variance"] > 0.0
    assert b["confidence"] <= 0.5


def test_property_beliefs_persist_across_sqlite_reload(tmp_path):
    db = tmp_path / "ltm.sqlite"
    g = LongTermGraph(db)
    slot = MemorySlot(
        entity_id="obj",
        appearance=[0.0, 1.0],
        seen_count=3,
        confidence=1.0,
        property_evidence={"edge_strength": 0.4},
    )
    g.consolidate([slot], cycle=1, min_seen=2)
    reopened = LongTermGraph(db)
    snap = reopened.snapshot()
    assert snap["total_property_beliefs"] == 1
    assert _belief(snap["nodes"][0], "edge_strength")["mean"] == 0.4


@dataclass
class _Slot:
    entity_id: str
    kind: str
    appearance: list[float]
    seen_count: int = 3
    affective_weight: float = 0.0
    position: list[float] | None = None
    confidence: float = 1.0
    kind_hint: str = "object"
    property_evidence: dict = field(default_factory=dict)


def test_async_property_beliefs_match_sync_after_flush(tmp_path):
    slots = [
        _Slot("a", "unknown", [1.0, 0.0], property_evidence={"roundness": 0.8}),
        _Slot("b", "unknown", [0.0, 1.0], property_evidence={"roundness": 0.2}),
    ]
    sync = LongTermGraph(tmp_path / "sync.sqlite")
    async_g = WriteBehindLongTermGraph(tmp_path / "async.sqlite", enabled=True)
    try:
        for cycle in range(4):
            sync.consolidate(slots, cycle=cycle, min_seen=2)
            async_g.consolidate(slots, cycle=cycle, min_seen=2)
        async_g.flush()
        assert async_g.counts() == sync.counts()
        assert async_g.belief_stats() == sync.belief_stats()
    finally:
        async_g.close()


def test_object_files_preserve_anonymous_property_evidence_and_strip_labels():
    files = object_files_from_proposals(
        [
            {
                "idx": 0,
                "appearance": [1.0, 0.0],
                "presence": 0.9,
                "uv": [0.2, 0.3],
                "spread": 0.1,
                "property_evidence": {
                    "roundness": 0.8,
                    "semantic_label": "food",
                    "food_score": 1.0,
                },
            }
        ]
    )
    d = files[0].to_working_memory_proposal()
    assert d["property_evidence"]["roundness"] == 0.8
    assert "semantic_label" not in d["property_evidence"]
    assert "food_score" not in d["property_evidence"]


def test_collapsed_relationship_gate_still_updates_properties():
    g = LongTermGraph()
    a = MemorySlot(entity_id="a", appearance=[1.0, 0.0], seen_count=3, confidence=1.0)
    b = MemorySlot(entity_id="b", appearance=[0.0, 1.0], seen_count=3, confidence=1.0)
    a.property_evidence = {"roundness": 0.7}
    b.property_evidence = {"roundness": 0.3}
    ids = g.consolidate([a, b], cycle=1, min_seen=2, property_update=True, relationship_update=False)
    assert len(ids) == 2
    assert g.counts() == (2, 0)
    assert g.belief_stats()["total_property_beliefs"] == 2


def test_working_memory_event_updates_anonymous_consequence_evidence():
    wm = WorkingMemory()
    wm.integrate_discovered(
        [
            {
                "idx": 0,
                "appearance": [1.0, 0.0],
                "presence": 0.9,
                "confidence": 0.9,
                "uv": [0.5, 0.5],
                "property_evidence": {"roundness": 0.5},
            }
        ],
        events=[{"type": "water", "intensity": 1.0}],
    )
    slot = next(iter(wm.slots.values()))
    assert slot.property_evidence["predicts_hydration_relief"] == 1.0
    assert "water" not in " ".join(slot.property_evidence.keys())

