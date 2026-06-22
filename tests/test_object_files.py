from __future__ import annotations

import pytest

from decadic.memory.semantic_graph import LongTermGraph
from decadic.perception.object_files import (
    evaluate_discovery_health,
    object_files_from_proposals,
)
from decadic.state.working_memory import MemorySlot


def test_slot_centroids_separated_and_uniform():
    torch = pytest.importorskip("torch")
    from decadic.nn.slots import SlotAttention

    mod = SlotAttention(in_dim=8, n_patches=9, k=2, slot_dim=8)
    masks = torch.zeros(1, 2, 9)
    masks[0, 0, 0] = 1.0
    masks[0, 1, 8] = 1.0
    cents = mod.centroids(masks)
    assert cents[0, 0, 0] < 0.3 and cents[0, 0, 1] < 0.3
    assert cents[0, 1, 0] > 0.7 and cents[0, 1, 1] > 0.7

    uniform = torch.ones(1, 2, 9)
    cents_u = mod.centroids(uniform)
    assert torch.allclose(cents_u[..., :2], torch.full((1, 2, 2), 0.5), atol=1e-6)


def test_object_file_serialization_and_no_semantic_labels():
    files = object_files_from_proposals(
        [
            {
                "idx": 0,
                "appearance": [1.0, 0.0],
                "presence": 0.9,
                "uv": [0.2, 0.3],
                "relative": [1.0, 0.0, 0.0],
                "bearing": [0.0, 0.1],
                "spread": 0.1,
                "label": "food",
            }
        ]
    )
    d = files[0].to_dict()
    assert d["object_id"] is None
    assert d["kind_hint"] == "object"
    assert "label" not in d
    assert files[0].to_working_memory_proposal()["uv"] == [0.2, 0.3]


def test_discovery_health_detects_center_collapse():
    files = object_files_from_proposals(
        [
            {
                "idx": i,
                "appearance": [1.0, 0.01 * i, 0.0],
                "presence": 0.9,
                "uv": [0.5 + i * 0.001, 0.5],
                "relative": [1.0, 0.0, 0.0],
                "spread": 0.1,
            }
            for i in range(7)
        ]
    )
    health = evaluate_discovery_health(files, tracked_count=7, stable_tracked_objects=7)
    assert health.collapsed is True
    assert health.reason == "skipped_perception_collapsed"


def test_uniform_extended_region_is_provisional_entity():
    files = object_files_from_proposals(
        [
            {
                "idx": 0,
                "appearance": [1.0, 0.0],
                "presence": 0.95,
                "uv": [0.5, 0.5],
                "relative": [0.7, 0.0, 0.0],
                "spread": 0.42,
            }
        ]
    )
    assert files[0].kind_hint == "stuff"
    assert files[0].entity_role == "extended_entity"
    assert files[0].provisional is True
    assert files[0].confidence < 0.2
    health = evaluate_discovery_health(files)
    assert health.reason == "recorded_provisional_evidence"


def test_ltm_skips_unhealthy_slots_and_keeps_copresent_slots_distinct():
    g = LongTermGraph()
    bad = MemorySlot(
        entity_id="extended-a",
        appearance=[1.0, 0.0],
        seen_count=3,
        confidence=0.1,
        kind_hint="stuff",
        entity_role="extended_entity",
    )
    assert g.consolidate([bad], cycle=1, min_seen=2) == []
    assert g.counts() == (0, 0)

    support = MemorySlot(
        entity_id="extended-b",
        appearance=[0.0, 1.0],
        seen_count=3,
        confidence=0.12,
        kind_hint="stuff",
        entity_role="extended_entity",
        precision=0.35,
        provisional=False,
    )
    assert len(g.consolidate([support], cycle=2, min_seen=2)) == 1
    assert g.counts() == (1, 0)

    a = MemorySlot(entity_id="obj-a", appearance=[1.0, 0.0], seen_count=3, confidence=1.0)
    b = MemorySlot(entity_id="obj-b", appearance=[1.0, 0.0], seen_count=3, confidence=1.0)
    ids = g.consolidate([a, b], cycle=3, min_seen=2)
    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert g.counts() == (3, 1)
