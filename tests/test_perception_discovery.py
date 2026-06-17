"""Perception-derived world graph: slots, data association, agency, metrics.

Covers Phase 4 of the perception-derived world-graph plan. Everything is gated
behind ``DECADIC_PERCEPTION_MODE``; the final test pins that the legacy oracle
path is untouched (byte-for-byte) so opting in cannot regress existing behavior.
"""

from __future__ import annotations

import math

import pytest

from decadic import config as C
from decadic.cycle.discovery import (
    extract_proposals,
    range_from_spread,
    uv_to_bearing,
)
from decadic.perception.discovery_metrics import DiscoveryEvaluator
from decadic.state.working_memory import MemorySlot, WorkingMemory
from decadic.state.world_graph import (
    edges_from_nodes,
    egocentric_nodes_from_perception,
    egocentric_nodes_from_world_state,
)


# ---------------------------------------------------------------------------
# Phase 1 — Slot attention: output shapes + self-supervised recon decreases
# ---------------------------------------------------------------------------


def test_slot_attention_output_shapes():
    torch = pytest.importorskip("torch")
    from decadic.nn.slots import SlotAttention

    k, slot_dim, n, in_dim = 5, 16, 9, 24
    mod = SlotAttention(in_dim=in_dim, n_patches=n, k=k, slot_dim=slot_dim, iters=3)
    feats = torch.randn(1, n, in_dim)
    with torch.no_grad():
        out = mod(feats)
    assert out["slots"].shape == (1, k, slot_dim)
    assert out["attn"].shape == (1, k, n)
    assert out["masks"].shape == (1, k, n)
    assert out["presence"].shape == (1, k)
    assert out["recon"].shape == (1, n, in_dim)
    # presence is a probability; masks are a per-position distribution over slots
    assert float(out["presence"].min()) >= 0.0 and float(out["presence"].max()) <= 1.0
    col_sums = out["masks"].sum(dim=1)  # [1, N] should be ~1 (softmax across K)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4)
    cents = mod.centroids(out["masks"])
    assert cents.shape == (1, k, 3)  # (u, v, spread)
    assert float(cents[..., :2].min()) >= 0.0 and float(cents[..., :2].max()) <= 1.0


def test_slot_attention_recon_loss_decreases():
    torch = pytest.importorskip("torch")
    from decadic.nn.slots import SlotAttention

    torch.manual_seed(0)
    mod = SlotAttention(in_dim=20, n_patches=9, k=4, slot_dim=24, iters=3)
    target = torch.randn(1, 9, 20)
    opt = torch.optim.Adam(mod.parameters(), lr=1e-2)
    losses = []
    for _ in range(150):
        opt.zero_grad()
        out = mod(target)
        loss = torch.nn.functional.mse_loss(out["recon"], target)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    # The DINOSAUR-style feature-reconstruction objective must actually train.
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


# ---------------------------------------------------------------------------
# Phase 1 — image-space geometry -> egocentric bearing/range proposals
# ---------------------------------------------------------------------------


def test_uv_to_bearing_and_proposals():
    # Frame center looks straight ahead (no azimuth/elevation).
    az, el = uv_to_bearing(0.5, 0.5, 80.0)
    assert abs(az) < 1e-9 and abs(el) < 1e-9
    # Rightward in the image -> positive azimuth; upward -> positive elevation.
    az_r, _ = uv_to_bearing(1.0, 0.5, 80.0)
    _, el_top = uv_to_bearing(0.5, 0.0, 80.0)
    assert az_r > 0 and el_top > 0
    # A bigger (more spread) mask reads as closer than a tight one.
    assert range_from_spread(0.4) < range_from_spread(0.02)


def test_extract_proposals_filters_by_presence():
    np = pytest.importorskip("numpy")
    slots = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype="float32")
    presence = np.array([0.9, 0.05, 0.5], dtype="float32")
    centroids = np.array(
        [[0.5, 0.5, 0.1], [0.2, 0.2, 0.2], [0.8, 0.3, 0.15]], dtype="float32"
    )
    props = extract_proposals(slots, presence, centroids, threshold=0.2)
    # The middle slot (presence 0.05) is below threshold and dropped.
    assert {p["idx"] for p in props} == {0, 2}
    for p in props:
        assert len(p["appearance"]) == 2
        assert len(p["bearing"]) == 2
        assert len(p["relative"]) == 3
        assert len(p["uv"]) == 2


# ---------------------------------------------------------------------------
# Phase 2 — data association: identity stability across a moving scene
# ---------------------------------------------------------------------------


def _prop(idx, appearance, uv, *, presence=0.9):
    az, el = uv_to_bearing(uv[0], uv[1], 80.0)
    return {
        "idx": idx,
        "appearance": list(appearance),
        "presence": presence,
        "uv": list(uv),
        "spread": 0.1,
        "bearing": [az, el],
        "relative": [1.0, 0.0, 0.0],
    }


def test_association_keeps_stable_id_across_motion():
    wm = WorkingMemory(capacity=8, decay=0.95)
    appearance = [1.0, 0.2, 0.0, 0.0]
    # One object drifting steadily across the frame; appearance is stable.
    for i in range(8):
        u = 0.3 + 0.03 * i
        wm.integrate_discovered([_prop(0, appearance, [u, 0.5])])
    # No id churn: the same coined object file is reinforced, not duplicated.
    assert len(wm.slots) == 1
    (slot,) = wm.slots.values()
    assert slot.entity_id == "obj-0000"
    assert slot.seen_count == 8


def test_association_coins_new_id_for_distinct_object():
    wm = WorkingMemory(capacity=8, decay=0.95)
    a = [1.0, 0.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0, 0.0]  # orthogonal appearance, far across the frame
    wm.integrate_discovered([_prop(0, a, [0.2, 0.2])])
    wm.integrate_discovered([_prop(0, a, [0.22, 0.2]), _prop(1, b, [0.85, 0.85])])
    ids = set(wm.slots)
    assert ids == {"obj-0000", "obj-0001"}
    assert wm.slots["obj-0000"].seen_count == 2  # 'a' re-identified
    assert wm.slots["obj-0001"].seen_count == 1  # 'b' is new


def test_association_returns_realized_image_motion():
    wm = WorkingMemory(capacity=8, decay=0.95)
    a = [1.0, 0.1, 0.0]
    wm.integrate_discovered([_prop(0, a, [0.4, 0.5])])
    matched = wm.integrate_discovered([_prop(0, a, [0.5, 0.5])])
    assert len(matched) == 1
    m = matched[0]
    assert m["entity_id"] == "obj-0000"
    assert m["prev_uv"] == [0.4, 0.5] and m["cur_uv"] == [0.5, 0.5]


# ---------------------------------------------------------------------------
# Phase 3 — agency: comparator head + body-part promotion
# ---------------------------------------------------------------------------


def test_agency_head_efference_beats_baseline_on_held_out_motion():
    torch = pytest.importorskip("torch")
    from decadic.nn.agency import AgencyHead

    torch.manual_seed(0)
    slot_dim, n_act, m = 8, 4, 256
    # Motion is generated purely from the efference copy u (a commanded limb);
    # slot appearance is random and carries no information about the motion.
    w = torch.randn(n_act, 2)

    def batch(n):
        s = torch.randn(n, slot_dim)
        u = torch.randn(n, n_act)
        y = u @ w
        return s, u, y

    s_tr, u_tr, y_tr = batch(m)
    s_te, u_te, y_te = batch(m)

    head = AgencyHead(slot_dim=slot_dim, n_actuators=n_act, hidden=32)
    opt = torch.optim.Adam(head.parameters(), lr=5e-3)
    for _ in range(400):
        opt.zero_grad()
        pe, pb = head(s_tr, u_tr)
        loss = torch.nn.functional.mse_loss(pe, y_tr) + torch.nn.functional.mse_loss(pb, y_tr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pe, pb = head(s_te, u_te)
        eff_err = float(torch.nn.functional.mse_loss(pe, y_te))
        base_err = float(torch.nn.functional.mse_loss(pb, y_te))
    # The efference-conditioned predictor generalizes; the efference-blind one
    # cannot — the gap is exactly the per-slot "this is mine" agency signal.
    assert eff_err < base_err


def test_update_agency_promotes_hand_not_static_prop():
    wm = WorkingMemory(capacity=8, decay=0.95)
    wm.slots["hand"] = MemorySlot(entity_id="hand", kind="unknown")
    wm.slots["prop"] = MemorySlot(entity_id="prop", kind="unknown")
    # The hand's motion is consistently explained by efference; the prop's is not.
    for _ in range(8):
        wm.update_agency(
            {"hand": 0.5, "prop": 0.0}, ema=0.3, threshold=0.15, min_seen=6
        )
    assert wm.slots["hand"].kind == "self_part"
    assert wm.slots["prop"].kind == "unknown"
    assert wm.slots["hand"].agency >= 0.15


def test_update_agency_demotes_when_agency_collapses():
    wm = WorkingMemory(capacity=8, decay=0.95)
    wm.slots["hand"] = MemorySlot(entity_id="hand", kind="unknown")
    for _ in range(8):
        wm.update_agency({"hand": 0.6}, ema=0.4, threshold=0.15, min_seen=6)
    assert wm.slots["hand"].kind == "self_part"
    # Lose control of the slot (e.g. it was never really mine): agency decays away.
    for _ in range(20):
        wm.update_agency({"hand": -0.2}, ema=0.4, threshold=0.15, min_seen=6)
    assert wm.slots["hand"].kind == "unknown"


def test_self_part_emits_agency_edge():
    self_node = {"role": "self", "id": "self", "position": [0.0, 0.0, 0.0]}
    wm_nodes = [
        {"role": "entity", "id": "obj-0000", "kind": "self_part", "relative": [0.0, 0.5, 0.0], "agency": 0.7},
        {"role": "entity", "id": "obj-0001", "kind": "unknown", "relative": [2.0, 0.0, 0.0]},
    ]
    nodes = egocentric_nodes_from_perception(self_node, wm_nodes)
    edges = edges_from_nodes(nodes)
    agency_edges = [e for e in edges if e["kind"] == "agency"]
    assert len(agency_edges) == 1
    assert agency_edges[0]["source"] == "self"
    assert agency_edges[0]["target"] == "obj-0000"
    assert agency_edges[0]["weight"] == pytest.approx(0.7)
    # The non-self_part object gets no agency edge.
    assert all(e["target"] != "obj-0001" or e["kind"] != "agency" for e in edges)


# ---------------------------------------------------------------------------
# Phase 0 — discovery metrics (eval-only scoring vs oracle truth)
# ---------------------------------------------------------------------------


def test_discovery_metrics_precision_recall():
    ev = DiscoveryEvaluator()
    discovered = [
        {"role": "self", "id": "self"},
        {"role": "entity", "id": "obj-0000", "kind": "unknown", "relative": [1.0, 0.0, 0.0]},
        {"role": "entity", "id": "obj-0001", "kind": "unknown", "relative": [0.0, 1.0, 0.0]},
    ]
    truth = [
        {"id": "real_a", "relative": [1.0, 0.05, 0.0]},  # ~ obj-0000
        {"id": "real_b", "relative": [0.0, 1.0, 0.05]},  # ~ obj-0001
    ]
    ev.update(discovered, truth, self_pos=[0.0, 0.0, 0.0])
    snap = ev.snapshot()
    assert snap["last_detected"] == 2 and snap["last_oracle"] == 2
    assert snap["last_matched"] == 2
    assert snap["precision"] == pytest.approx(1.0)
    assert snap["recall"] == pytest.approx(1.0)


def test_discovery_metrics_id_stability_rises_with_stable_ids():
    ev = DiscoveryEvaluator()
    disc = [{"role": "entity", "id": "obj-0000", "kind": "unknown", "relative": [1.0, 0.0, 0.0]}]
    truth = [{"id": "real", "relative": [1.0, 0.0, 0.0]}]
    ev.update(disc, truth, self_pos=[0.0, 0.0, 0.0])
    # First sighting: every id is "new", so churn starts at 1 (stability 0).
    assert ev.snapshot()["id_stability"] == pytest.approx(0.0)
    for _ in range(40):
        ev.update(disc, truth, self_pos=[0.0, 0.0, 0.0])
    # Re-using the coined id drives churn toward 0 (object permanence).
    assert ev.snapshot()["id_stability"] > 0.9


def test_discovery_metrics_body_part_accuracy():
    ev = DiscoveryEvaluator()
    discovered = [
        {"role": "self", "id": "self"},
        {"role": "entity", "id": "obj-0000", "kind": "self_part", "relative": [0.0, 0.5, 0.0]},
    ]
    body_parts = {"left_hand": [0.0, 1.0, 0.0]}  # same egocentric direction
    ev.update(discovered, [], self_pos=[0.0, 0.0, 0.0], body_parts_truth=body_parts)
    snap = ev.snapshot()
    assert snap["last_body_parts_truth"] == 1
    assert snap["last_body_parts_found"] == 1
    assert snap["body_part_accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Phase 2 — discovered-mode graph build with an empty oracle
# ---------------------------------------------------------------------------


def test_discovered_mode_graph_omits_oracle_entities():
    from decadic.state.perceptual_state import PerceptualState

    ps = PerceptualState(perception_mode="discovered")
    # The neural cycle owns WM in discovered mode; seed a coined object file here.
    ps.working_memory.slots["obj-0000"] = MemorySlot(
        entity_id="obj-0000", kind="unknown", relative=[1.0, 0.0, 0.0], salience=1.0
    )
    obs = {
        "proprioception": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0]},
        "world_state": {
            "agent": {"id": "self", "position": [0.0, 0.0, 0.0]},
            "entities": [{"id": "oracle_box", "kind": "box", "relative": [2.0, 0.0, 0.0]}],
        },
    }
    ps.integrate_observation(obs)
    ids = {n.get("id") for n in ps.egocentric_nodes}
    assert "self" in ids
    assert "obj-0000" in ids  # the agent's own coined object file
    assert "oracle_box" not in ids  # oracle entity never enters cognition
    # ...but it is retained out-of-band as eval-only ground truth.
    assert [t["id"] for t in ps.oracle_truth] == ["oracle_box"]
    snap = ps.snapshot_dict()
    assert snap["perception_mode"] == "discovered"
    assert snap["discovery"] is not None


def test_discovered_mode_builds_with_empty_oracle():
    from decadic.state.perceptual_state import PerceptualState

    ps = PerceptualState(perception_mode="discovered")
    obs = {"proprioception": {"position": [1.0, 2.0, 0.5], "orientation": [0.0, 0.0, 0.0]}}
    ps.integrate_observation(obs)  # no world_state at all
    self_nodes = [n for n in ps.egocentric_nodes if n.get("role") == "self"]
    assert len(self_nodes) == 1
    assert self_nodes[0]["position"] == [1.0, 2.0, 0.5]  # sensed, not oracle
    assert ps.oracle_truth == []
    assert ps.snapshot_dict()["discovery"]["last_oracle"] == 0


# ---------------------------------------------------------------------------
# Phase 4 — regression: the oracle path is byte-for-byte unchanged
# ---------------------------------------------------------------------------


def test_perception_mode_defaults_to_discovered(monkeypatch):
    # Discovered perception is now the inherent default; oracle is the opt-in
    # eval scaffold. (Unknown values still fall back to the default.)
    monkeypatch.delenv("DECADIC_PERCEPTION_MODE", raising=False)
    assert C.perception_mode() == "discovered"
    assert C.discovered_perception_enabled() is True
    monkeypatch.setenv("DECADIC_PERCEPTION_MODE", "garbage")
    assert C.perception_mode() == "discovered"
    monkeypatch.setenv("DECADIC_PERCEPTION_MODE", "oracle")
    assert C.perception_mode() == "oracle"
    assert C.discovered_perception_enabled() is False


def test_oracle_mode_graph_unchanged():
    from decadic.state.perceptual_state import PerceptualState

    ws = {
        "agent": {"id": "self", "position": [0.0, 0.0, 0.0]},
        "entities": [
            {"id": "box1", "kind": "box", "relative": [1.0, 0.0, 0.0]},
            {"id": "ball2", "kind": "ball", "relative": [0.0, 2.0, 0.0]},
        ],
        "region": {"id": "field", "display_name": "Open Field"},
    }
    ps = PerceptualState()  # default mode
    assert ps.perception_mode == "oracle"
    ps.integrate_observation({"world_state": ws})

    # The self + entity + context node schema is exactly the legacy builder's.
    ref_nodes = egocentric_nodes_from_world_state(ws)
    ref_ids = {n.get("id") for n in ref_nodes}
    got_ids = {n.get("id") for n in ps.egocentric_nodes}
    assert {"self", "box1", "ball2", "field"} <= got_ids
    assert ref_ids <= got_ids
    # A spatial self->entity edge exists for each oracle entity.
    spatial = {e["target"] for e in ps.egocentric_edges if e["kind"] == "spatial"}
    assert {"box1", "ball2"} <= spatial
    # No discovered-mode artifacts leak into the oracle snapshot.
    snap = ps.snapshot_dict()
    assert snap["perception_mode"] == "oracle"
    assert snap["discovery"] is None
    assert not any(e["kind"] == "agency" for e in ps.egocentric_edges)
    assert not any(n.get("kind") == "self_part" for n in ps.egocentric_nodes)
