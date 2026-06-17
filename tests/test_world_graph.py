"""Self-indexed graph: typed edges + decaying affective table (logical-layer Part A)."""

from decadic.state.world_graph import (
    edges_from_nodes,
    egocentric_graph_from_world_state,
    update_entity_affect,
)


def _ws(entities, **extra):
    ws = {"agent": {"id": "self", "position": [0.0, 0.0, 0.0]}, "entities": entities}
    ws.update(extra)
    return ws


def test_spatial_edges_and_closeness():
    g = egocentric_graph_from_world_state(
        _ws([{"id": "near", "kind": "box", "position": [1.0, 0.0, 0.0]},
             {"id": "far", "kind": "box", "position": [4.0, 0.0, 0.0]}])
    )
    spatial = {e["target"]: e for e in g["edges"] if e["kind"] == "spatial"}
    assert spatial["near"]["distance"] == 1.0
    assert spatial["far"]["distance"] == 4.0
    # closer entity has the stronger (larger) spatial weight
    assert spatial["near"]["weight"] > spatial["far"]["weight"]
    # all edges originate at the self-node (self-indexing)
    assert all(e["source"] == "self" for e in g["edges"] if e["kind"] in ("spatial", "context"))


def test_affective_edges_signed_by_valence():
    nodes = egocentric_graph_from_world_state(
        _ws([{"id": "bear", "kind": "bear", "position": [2.0, 0.0, 0.0]}])
    )["nodes"]
    edges = edges_from_nodes(nodes, affect={"bear": -1.5})
    aff = [e for e in edges if e["kind"] == "affective"]
    assert len(aff) == 1
    assert aff[0]["target"] == "bear"
    assert aff[0]["weight"] == -1.5
    # below the min threshold no affective edge appears
    assert not [e for e in edges_from_nodes(nodes, affect={"bear": 1e-6}) if e["kind"] == "affective"]


def test_proximity_edges_within_radius():
    g = egocentric_graph_from_world_state(
        _ws([{"id": "a", "kind": "box", "position": [1.0, 0.0, 0.0]},
             {"id": "b", "kind": "box", "position": [1.5, 0.0, 0.0]},
             {"id": "c", "kind": "box", "position": [20.0, 0.0, 0.0]}]),
        proximity_radius=2.0,
    )
    prox = [e for e in g["edges"] if e["kind"] == "proximity"]
    pairs = {frozenset((e["source"], e["target"])) for e in prox}
    assert frozenset(("a", "b")) in pairs
    # c is far from both, so no proximity edge touches it
    assert all("c" not in p for p in pairs)


def test_context_nodes_get_context_edges():
    g = egocentric_graph_from_world_state(
        _ws([], region={"id": "forest", "display_name": "Forest"})
    )
    ctx = [n for n in g["nodes"] if n["role"] == "context"]
    assert ctx and ctx[0]["id"] == "forest"
    assert any(e["kind"] == "context" and e["target"] == "forest" for e in g["edges"])


def test_affect_update_decays_and_clamps():
    affect: dict[str, float] = {}
    update_entity_affect(affect, [{"type": "threat_near", "intensity": 1.0, "source": "bear"}])
    assert affect["bear"] == -0.5  # threat = -0.5 * intensity
    update_entity_affect(affect, [{"type": "food", "intensity": 1.0, "source": "apple"}])
    assert affect["apple"] == 1.0
    # the untouched bear entry decays toward zero on the next update
    prev = affect["bear"]
    update_entity_affect(affect, [])
    assert abs(affect["bear"]) < abs(prev)

    # repeated harm is clamped to the configured magnitude
    big: dict[str, float] = {}
    for _ in range(50):
        update_entity_affect(big, [{"type": "collision", "intensity": 1.0, "source": "wall"}])
    assert big["wall"] >= -5.0


def test_affect_ignores_sensor_named_events():
    # body collisions carry a sensor name, not an entity id → no graph node, no edge
    affect: dict[str, float] = {}
    update_entity_affect(affect, [{"type": "collision", "intensity": 0.9, "source": "touch_right_foot"}])
    # it still records valence keyed by that source, but no entity node matches it
    g = egocentric_graph_from_world_state(
        _ws([{"id": "box", "kind": "box", "position": [1.0, 0.0, 0.0]}])
    )
    edges = edges_from_nodes(g["nodes"], affect=affect)
    assert not [e for e in edges if e["kind"] == "affective"]
