"""WS5-M0: slot-tensor layout freeze + WorkingMemory.slot_tensor() adapter.

The layout test pins the frozen offsets (docs/ws5_m0_wm_inventory.md) exactly
as test_ws4_backends.py pins the episodic embedding layout: any drift is a
deliberate PRD amendment, never an accident.
"""

import numpy as np
import pytest

from decadic.state.working_memory import (
    SLOT_APPEARANCE_SLICE,
    SLOT_SCALAR_SLICE,
    SLOT_SPATIAL_SLICE,
    SLOT_TENSOR_APPEARANCE_DIM,
    SLOT_TENSOR_DIM,
    SLOT_TENSOR_SCALAR_DIM,
    SLOT_TENSOR_SPATIAL_DIM,
    WorkingMemory,
)


def _node(eid: str, pos=(1.0, 2.0, 3.0), kind="entity_kind"):
    return {"role": "entity", "id": eid, "kind": kind, "position": list(pos)}


def _wm(capacity: int = 6) -> WorkingMemory:
    return WorkingMemory(capacity=capacity)


# ------------------------------------------------------------------ layout


def test_slot_tensor_layout_frozen():
    assert SLOT_TENSOR_APPEARANCE_DIM == 16
    assert SLOT_TENSOR_SPATIAL_DIM == 11
    assert SLOT_TENSOR_SCALAR_DIM == 13
    assert SLOT_TENSOR_DIM == 40
    assert SLOT_APPEARANCE_SLICE == slice(0, 16)
    assert SLOT_SPATIAL_SLICE == slice(16, 27)
    assert SLOT_SCALAR_SLICE == slice(27, 40)


# ----------------------------------------------------------- slot_tensor


def test_slot_tensor_empty_wm_shapes():
    wm = _wm(capacity=5)
    t, m = wm.slot_tensor()
    assert t.shape == (5, SLOT_TENSOR_DIM) and t.dtype == np.float32
    assert m.shape == (5,) and m.dtype == bool
    assert not m.any() and float(np.abs(t).sum()) == 0.0
    t0, m0 = wm.slot_tensor(k_max=0)
    assert t0.shape == (0, SLOT_TENSOR_DIM) and m0.shape == (0,)


def test_slot_tensor_mask_order_and_determinism():
    wm = _wm(capacity=4)
    wm.integrate([_node("ent-b"), _node("ent-a"), _node("ent-c")])
    t1, m1 = wm.slot_tensor()
    assert m1.tolist() == [True, True, True, False]
    assert float(np.abs(t1[3]).sum()) == 0.0  # unfilled row stays zero

    # All three refreshed this cycle -> salience ties at 1.0 -> the
    # entity_id tie-break is load-bearing: order must be a, b, c.
    # in_view scalar (last position of the scalar block) is 1 for all.
    assert (t1[:3, SLOT_SCALAR_SLICE][:, -1] == 1.0).all()

    # Determinism: identical state -> identical tensor, twice.
    t2, m2 = wm.slot_tensor()
    assert np.array_equal(t1, t2) and np.array_equal(m1, m2)

    # Stable under dict-insertion order: a fresh WM fed in another order
    # yields the same tensor for the same logical state.
    wm2 = _wm(capacity=4)
    wm2.integrate([_node("ent-c"), _node("ent-a"), _node("ent-b")])
    t3, _ = wm2.slot_tensor()
    assert np.array_equal(t1, t3)


def test_slot_tensor_salience_ranking_and_staleness():
    wm = _wm(capacity=4)
    wm.integrate([_node("ent-old"), _node("ent-new")])
    wm.integrate([_node("ent-new")])  # old decays, new refreshed
    t, m = wm.slot_tensor()
    assert m.tolist() == [True, True, False, False]
    scalars = t[:, SLOT_SCALAR_SLICE]
    # Row 0 = highest salience = the refreshed entity: salience 1, in_view 1,
    # staleness 0. Row 1 = the decayed one: salience < 1, in_view 0,
    # staleness > 0.
    assert scalars[0, 0] == pytest.approx(1.0)
    assert scalars[0, -1] == 1.0 and scalars[0, -2] == 0.0
    assert scalars[1, 0] < 1.0
    assert scalars[1, -1] == 0.0 and scalars[1, -2] > 0.0


def test_slot_tensor_kmax_truncation_and_dslot_projection():
    wm = _wm(capacity=8)
    wm.integrate([_node(f"ent-{i}") for i in range(5)])
    t, m = wm.slot_tensor(k_max=3)
    assert t.shape == (3, SLOT_TENSOR_DIM) and m.all()

    # d_slot larger: zero-padded tail; smaller: truncated prefix of the
    # same frozen row (projection fixed, never re-laid-out).
    big, _ = wm.slot_tensor(k_max=3, d_slot=64)
    small, _ = wm.slot_tensor(k_max=3, d_slot=20)
    assert np.array_equal(big[:, :SLOT_TENSOR_DIM], t)
    assert float(np.abs(big[:, SLOT_TENSOR_DIM:]).sum()) == 0.0
    assert np.array_equal(small, t[:, :20])


# -------------------------------------------------- M0.3 retrieval tokens


def _episodic_pair(tmp_path):
    pytest.importorskip("lancedb")
    from decadic.memory.episodic_store import EpisodicRecord, EpisodicStore
    from decadic.memory.lancedb_store import LanceEpisodicStore

    rng = np.random.default_rng(19)
    embs = rng.normal(size=(8, 80)).astype(np.float32)
    sq = EpisodicStore(tmp_path / "tok.sqlite")
    lz = LanceEpisodicStore(tmp_path / "tok_lance")
    for store in (sq, lz):
        for i in range(8):
            store.append(
                EpisodicRecord(
                    cycle_index=i, summary={"i": i}, salience=0.9, embedding=embs[i]
                )
            )
    lz.flush()
    return sq, lz, embs


def test_retrieval_tokens_parity_and_k1_equivalence(tmp_path):
    sq, lz, embs = _episodic_pair(tmp_path)
    try:
        q = embs[3]
        tok_a, m_a = sq.retrieval_context_tokens(q, k=4)
        tok_b, m_b = lz.retrieval_context_tokens(q, k=4)
        assert tok_a.shape == (4, 80) and m_a.all()
        assert m_a.tolist() == m_b.tolist()
        assert np.allclose(tok_a, tok_b, atol=1e-5)  # backend parity
        assert np.allclose(tok_a[0], embs[3], atol=1e-5)  # top hit = itself

        # k=1 reproduces today's best-hit semantics EXACTLY: the token row
        # equals the mean-pooled context vector of top_k=1 (mean of one).
        for store in (sq, lz):
            tok, m = store.retrieval_context_tokens(q, k=1)
            rcv = store.retrieval_context_vector(q, 80, top_k=1)
            assert m.tolist() == [True]
            assert np.allclose(tok[0], rcv, atol=1e-6)
    finally:
        lz.close()


def test_retrieval_tokens_empty_and_partial(tmp_path):
    pytest.importorskip("lancedb")
    from decadic.memory.episodic_store import EpisodicRecord, EpisodicStore
    from decadic.memory.lancedb_store import LanceEpisodicStore

    q = np.ones(80, dtype=np.float32)
    empty_sq = EpisodicStore(tmp_path / "e.sqlite")
    empty_lz = LanceEpisodicStore(tmp_path / "e_lance")
    try:
        for store in (empty_sq, empty_lz):
            tok, m = store.retrieval_context_tokens(q, k=3)
            assert tok.shape == (3, 80) and not m.any()
            assert float(np.abs(tok).sum()) == 0.0
            tok0, m0 = store.retrieval_context_tokens(q, k=0)
            assert tok0.shape == (0, 80) and m0.shape == (0,)

        # Partial fill: 2 episodes, k=5 -> mask [T, T, F, F, F].
        embs = np.random.default_rng(4).normal(size=(2, 80)).astype(np.float32)
        for store in (empty_sq, empty_lz):
            for i in range(2):
                store.append(
                    EpisodicRecord(
                        cycle_index=i, summary={}, salience=0.9, embedding=embs[i]
                    )
                )
        empty_lz.flush()
        for store in (empty_sq, empty_lz):
            tok, m = store.retrieval_context_tokens(q, k=5)
            assert m.tolist() == [True, True, False, False, False]
            assert float(np.abs(tok[2:]).sum()) == 0.0
    finally:
        empty_lz.close()


# --------------------------------- M1: slot tensor into the stack (flagged)


def _tiny_stack(monkeypatch, wm_slot: bool):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_WM_SLOT_TENSOR", "1" if wm_slot else "0")
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack

    torch.manual_seed(21)
    stack = NeuralCognitiveStack(neural_config_from_env("tiny"))
    stack.eval()
    cfg = neural_config_from_env("tiny")
    return torch, stack, cfg


def _slot_batch(torch, fill: float, k: int = 4):
    slots = torch.zeros(k, SLOT_TENSOR_DIM)
    slots[:, :16] = fill
    slots[:, 27] = 1.0  # salience
    mask = torch.tensor([True, True, True, False])
    return slots, mask


def test_wm_slot_faculty_defaults(monkeypatch):
    from decadic.nn.faculties import CognitionFaculties

    assert CognitionFaculties().wm_slot_tensor is False
    monkeypatch.setenv("DECADIC_WM_SLOT_TENSOR", "1")
    assert CognitionFaculties.from_env().wm_slot_tensor is True
    from decadic.config import wm_slot_k

    monkeypatch.delenv("DECADIC_WM_SLOT_K", raising=False)
    assert wm_slot_k() == 6  # the neural WM window: cognitive, not scaled


def test_stack_flag_off_ignores_slots(monkeypatch):
    """M1.1 parity: an off-build has no slot modules and ignores slot args."""
    torch, stack, cfg = _tiny_stack(monkeypatch, wm_slot=False)
    assert stack.has_wm_slot_tensor is False
    assert not hasattr(stack, "slot_ingress")
    z0 = torch.randn(1, cfg.d_model)
    ep = torch.rand(1, 4)
    mem = torch.randn(1, cfg.memory_context_dim)
    slots, mask = _slot_batch(torch, 0.7)
    with torch.no_grad():
        stack.reset_recurrent_state()
        a = stack(z0, ep, mem)
        stack.reset_recurrent_state()
        b = stack(z0, ep, mem, wm_slots=slots, wm_slots_mask=mask)
    for key, v in a.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, b[key]), f"flag-off perturbed {key!r}"


def test_stack_flag_on_zero_init_parity_then_live(monkeypatch):
    """M1.2: zero-init ingress => slots are invisible at init; once the
    ingress moves, slot CONTENT changes cognition -- the boundary crossing."""
    torch, stack, cfg = _tiny_stack(monkeypatch, wm_slot=True)
    assert stack.has_wm_slot_tensor and hasattr(stack, "slot_ingress")
    z0 = torch.randn(1, cfg.d_model)
    ep = torch.rand(1, 4)
    mem = torch.randn(1, cfg.memory_context_dim)
    slots_a, mask = _slot_batch(torch, 0.7)
    slots_b, _ = _slot_batch(torch, -0.4)

    with torch.no_grad():
        stack.reset_recurrent_state()
        base = stack(z0, ep, mem)
        stack.reset_recurrent_state()
        at_init = stack(z0, ep, mem, wm_slots=slots_a, wm_slots_mask=mask)
    for key, v in base.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, at_init[key]), f"zero-init parity broke {key!r}"

    with torch.no_grad():
        stack.slot_ingress.weight.normal_(0.0, 0.2)
        stack.reset_recurrent_state()
        live_a = stack(z0, ep, mem, wm_slots=slots_a, wm_slots_mask=mask)
        stack.reset_recurrent_state()
        live_b = stack(z0, ep, mem, wm_slots=slots_b, wm_slots_mask=mask)
        stack.reset_recurrent_state()
        live_none = stack(z0, ep, mem)
    # Slots now reach cognition, and DIFFERENT slot content produces
    # DIFFERENT cognition (what pooling could never do).
    assert not torch.allclose(live_a["z5"], live_none["z5"])
    assert not torch.allclose(live_a["z5"], live_b["z5"])
    # All-masked slots are a strict no-op even with a live ingress.
    with torch.no_grad():
        stack.reset_recurrent_state()
        masked = stack(
            z0, ep, mem, wm_slots=slots_a, wm_slots_mask=torch.zeros(4, dtype=torch.bool)
        )
    assert torch.equal(masked["z5"], live_none["z5"])
    # Gradient isolation: the slot tensor is a per-cycle constant.
    assert slots_a.requires_grad is False and slots_a.grad is None


# --------------------------------- M2: memory tokens into the stack


def test_stack_memory_tokens_parity_then_live(monkeypatch):
    """M2.1: flag-off ignores tokens; flag-on is zero-init parity; once the
    ingress moves, DIFFERENT recalled episodes produce DIFFERENT cognition
    (mean-pooling made five memories indistinguishable from their average)."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_MEMORY_TOKENS", "0")
    _, stack_off, cfg = _tiny_stack(monkeypatch, wm_slot=False)
    assert stack_off.has_memory_tokens is False
    assert not hasattr(stack_off, "mem_tok_ingress")

    monkeypatch.setenv("DECADIC_MEMORY_TOKENS", "1")
    torch2, stack, cfg = _tiny_stack(monkeypatch, wm_slot=False)
    assert stack.has_memory_tokens and hasattr(stack, "mem_tok_ingress")

    z0 = torch.randn(1, cfg.d_model)
    ep = torch.rand(1, 4)
    mem = torch.randn(1, cfg.memory_context_dim)
    toks_a = torch.randn(5, 80) * 0.5
    toks_b = -toks_a
    mask = torch.tensor([True, True, True, False, False])

    # Flag-off build ignores token args entirely.
    with torch.no_grad():
        stack_off.reset_recurrent_state()
        a = stack_off(z0, ep, mem)
        stack_off.reset_recurrent_state()
        b = stack_off(z0, ep, mem, mem_tokens=toks_a, mem_tokens_mask=mask)
    assert torch.equal(a["z5"], b["z5"])

    # Flag-on at init: zero ingress => byte-identical with or without tokens.
    with torch.no_grad():
        stack.reset_recurrent_state()
        base = stack(z0, ep, mem)
        stack.reset_recurrent_state()
        at_init = stack(z0, ep, mem, mem_tokens=toks_a, mem_tokens_mask=mask)
    for key, v in base.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, at_init[key]), f"zero-init parity broke {key!r}"

    # Live ingress: tokens reach cognition, and content matters.
    with torch.no_grad():
        stack.mem_tok_ingress.weight.normal_(0.0, 0.2)
        stack.reset_recurrent_state()
        live_a = stack(z0, ep, mem, mem_tokens=toks_a, mem_tokens_mask=mask)
        stack.reset_recurrent_state()
        live_b = stack(z0, ep, mem, mem_tokens=toks_b, mem_tokens_mask=mask)
        stack.reset_recurrent_state()
        live_none = stack(z0, ep, mem)
        stack.reset_recurrent_state()
        masked = stack(
            z0,
            ep,
            mem,
            mem_tokens=toks_a,
            mem_tokens_mask=torch.zeros(5, dtype=torch.bool),
        )
    assert not torch.allclose(live_a["z5"], live_none["z5"])
    assert not torch.allclose(live_a["z5"], live_b["z5"])
    assert torch.equal(masked["z5"], live_none["z5"])


def test_memory_tokens_faculty_default(monkeypatch):
    from decadic.nn.faculties import CognitionFaculties

    assert CognitionFaculties().memory_tokens is False
    monkeypatch.setenv("DECADIC_MEMORY_TOKENS", "1")
    assert CognitionFaculties.from_env().memory_tokens is True


# --------------------------------- M3: relational core (stage 3->4)


def test_relational_core_parity_content_and_gate_economics(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_RELATIONAL_CORE", "1")
    _, stack, cfg = _tiny_stack(monkeypatch, wm_slot=False)
    assert stack.has_relational_core and hasattr(stack, "rel_ingress")

    z0 = torch.randn(1, cfg.d_model)
    ep = torch.rand(1, 4)
    mem = torch.randn(1, cfg.memory_context_dim)
    slots_a, mask = _slot_batch(torch, 0.7)
    slots_b, _ = _slot_batch(torch, 0.7)
    slots_b[:, 16:19] = -0.9  # same entities, DIFFERENT spatial relations
    mem_toks = torch.randn(3, 80) * 0.3
    mem_mask = torch.ones(3, dtype=torch.bool)

    # Zero-init ingress: relational compute runs but cannot yet move z4.
    with torch.no_grad():
        stack.reset_recurrent_state()
        base = stack(z0, ep, mem)
        stack.reset_recurrent_state()
        at_init = stack(
            z0, ep, mem,
            wm_slots=slots_a, wm_slots_mask=mask,
            mem_tokens=mem_toks, mem_tokens_mask=mem_mask,
        )
    for key, v in base.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, at_init[key]), f"zero-init parity broke {key!r}"

    # Live ingress: relations reach the RISK computation, and changing only
    # the spatial features (who is where, relative to whom) changes z4 --
    # the wolf/rock asymmetry becoming computable.
    with torch.no_grad():
        stack.rel_ingress.weight.normal_(0.0, 0.2)
        stack.reset_recurrent_state()
        live_a = stack(z0, ep, mem, wm_slots=slots_a, wm_slots_mask=mask,
                       mem_tokens=mem_toks, mem_tokens_mask=mem_mask)
        stack.reset_recurrent_state()
        live_b = stack(z0, ep, mem, wm_slots=slots_b, wm_slots_mask=mask,
                       mem_tokens=mem_toks, mem_tokens_mask=mem_mask)
        stack.reset_recurrent_state()
        live_a2 = stack(z0, ep, mem, wm_slots=slots_a, wm_slots_mask=mask,
                        mem_tokens=mem_toks, mem_tokens_mask=mem_mask)
    assert not torch.allclose(live_a["z4"], live_b["z4"])
    assert torch.equal(live_a["z4"], live_a2["z4"])  # deterministic

    # Empty-token degeneracy: intero token alone is a defined, finite path.
    with torch.no_grad():
        stack.reset_recurrent_state()
        lone = stack(z0, ep, mem)
    assert torch.isfinite(lone["z4"]).all()

    # Gate economics: a SKIPPED cycle (stage4_override) never runs the core.
    calls = []
    orig = stack.relational.forward

    def _counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    stack.relational.forward = _counting
    override = (live_a["z4"].detach() * 0.5, live_a["risk_logit"].detach() * 0.5)
    with torch.no_grad():
        stack.reset_recurrent_state()
        stack(z0, ep, mem, wm_slots=slots_a, wm_slots_mask=mask,
              stage4_override=override)
    assert calls == []  # skip path: relational deliberation not paid for
    with torch.no_grad():
        stack.reset_recurrent_state()
        stack(z0, ep, mem, wm_slots=slots_a, wm_slots_mask=mask)
    assert len(calls) == 1  # deliberative path: exactly one relational pass
    stack.relational.forward = orig


def test_relational_faculty_default(monkeypatch):
    from decadic.nn.faculties import CognitionFaculties

    assert CognitionFaculties().relational_core is False
    monkeypatch.setenv("DECADIC_RELATIONAL_CORE", "1")
    assert CognitionFaculties.from_env().relational_core is True


# --------------------------------- M4.1: graph-keyed slots (object permanence)


def test_graph_keyed_slots_object_permanence():
    """A re-encountered entity re-binds to the SAME key across an occlusion
    gap: the slot is evicted, the live appearance drifts, but the graph's
    stored appearance anchors identity -- the WBS M4.1 assertion."""
    stored = [0.9] + [0.1] * 15  # the graph's stored appearance for ent-7

    def lookup(sid):
        return stored if sid == "ent-7" else None

    def reid(_app):
        return "ent-7"

    wm = WorkingMemory(capacity=4, decay=0.5, min_salience=0.3)
    prop1 = {"appearance": [0.8] + [0.12] * 15, "uv": [0.5, 0.5], "confidence": 0.9}
    wm.integrate_discovered([prop1], reidentify=reid, key_lookup=lookup)
    assert "ent-7" in wm.slots
    assert wm.slots["ent-7"].key_appearance == stored
    t1, m1 = wm.slot_tensor()
    key_row_1 = t1[0][SLOT_APPEARANCE_SLICE].copy()
    assert m1[0] and np.allclose(key_row_1, np.asarray(stored, np.float32))

    # Occlusion: decay past eviction -- the slot is GONE from working memory.
    for _ in range(4):
        wm.integrate_discovered([], reidentify=reid, key_lookup=lookup)
    assert "ent-7" not in wm.slots

    # Re-encounter with a drifted live appearance: LTM reinstates the id and
    # the graph key re-binds IDENTICALLY -- object permanence at key level.
    prop2 = {"appearance": [0.5] + [0.3] * 15, "uv": [0.2, 0.7], "confidence": 0.9}
    wm.integrate_discovered([prop2], reidentify=reid, key_lookup=lookup)
    t2, m2 = wm.slot_tensor()
    assert m2[0]
    assert np.array_equal(t2[0][SLOT_APPEARANCE_SLICE], key_row_1)
    # The live appearance drifted; the key did not leak the drift.
    assert wm.slots["ent-7"].appearance != stored

    # Unmatched (anonymous) slots key on their own appearance -- no graph key.
    wm2 = WorkingMemory(capacity=4)
    wm2.integrate_discovered([prop1], reidentify=lambda _a: None, key_lookup=lookup)
    anon = next(iter(wm2.slots.values()))
    assert anon.key_appearance is None
    t3, _ = wm2.slot_tensor()
    assert np.allclose(
        t3[0][SLOT_APPEARANCE_SLICE], np.asarray(prop1["appearance"], np.float32)
    )


def test_ltm_graph_entity_appearance_getter(tmp_path):
    from decadic.memory.semantic_graph import LongTermGraph

    g = LongTermGraph(tmp_path / "g.sqlite")
    app = np.linspace(-0.5, 0.5, 16).astype(np.float32)
    nid = g.upsert_node(app, kind="npc", cycle=1)
    got = g.entity_appearance(nid)
    assert got is not None and np.allclose(got, app, atol=1e-5)
    assert g.entity_appearance("ent-nonexistent") is None


# ------------------------------------------- M0.4 oracle appearance seam


def test_oracle_seam_carries_appearance_into_slots():
    """world_state.entities[].appearance -> egocentric node -> WM slot ->
    slot_tensor appearance block. The binding probe's injection path."""
    from decadic.state.world_graph import egocentric_nodes_from_world_state

    app = [round(0.1 * i, 3) for i in range(16)]
    ws = {
        "agent": {"id": "self", "position": [0, 0, 0]},
        "entities": [
            {"id": "ent-A", "kind": "entity", "position": [3, 0, 1], "appearance": app},
            {"id": "ent-B", "kind": "entity", "position": [5, 0, 2]},  # none
        ],
    }
    nodes = egocentric_nodes_from_world_state(ws)
    ents = {n["id"]: n for n in nodes if n.get("role") == "entity"}
    assert ents["ent-A"]["appearance"] == app
    assert "appearance" not in ents["ent-B"]  # absent -> unchanged schema

    wm = _wm(capacity=4)
    wm.integrate(list(nodes))
    assert wm.slots["ent-A"].appearance == app
    assert wm.slots["ent-B"].appearance is None  # byte-identical legacy path

    t, m = wm.slot_tensor()
    # ent-A's appearance block reproduces the injected vector exactly.
    a_row = [r for r in t[m] if abs(float(r[1]) - app[1]) < 1e-6]
    assert a_row, "injected appearance must reach the slot tensor"
    assert np.allclose(a_row[0][SLOT_APPEARANCE_SLICE], np.asarray(app, np.float32))

    # Refresh path adopts an updated appearance too.
    app2 = [v + 1.0 for v in app]
    ws["entities"][0]["appearance"] = app2
    wm.integrate(egocentric_nodes_from_world_state(ws))
    assert wm.slots["ent-A"].appearance == app2


def test_binding_world_schedule_adjacency_and_events():
    """Client-side scenario engine: pair adjacency during phases, cadenced
    events sourced to the first pair member, distinct orbits otherwise."""
    import sys as _sys
    from pathlib import Path as _P

    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    import synthetic_ws_client as swc

    scenario = {
        "entities": [
            {"id": "ent-A", "appearance": [1.0] * 16, "home": [10, 0, 0]},
            {"id": "ent-B", "appearance": [0.5] * 16, "home": [-10, 0, 0]},
            {"id": "ent-C", "appearance": [0.2] * 16, "home": [0, 0, 10]},
        ],
        "schedule": [
            {
                "start": 100,
                "steps": 40,
                "pair": ["ent-A", "ent-B"],
                "gap": 1.5,
                "event": {"type": "threat_near", "intensity": 0.6, "every": 10},
            }
        ],
    }
    # Outside the phase: B orbits its own home, far from A.
    ents, evs = swc._binding_world(50, scenario)
    pos = {e["id"]: e["position"] for e in ents}
    d = sum((a - b) ** 2 for a, b in zip(pos["ent-A"], pos["ent-B"])) ** 0.5
    assert d > 10 and evs == []
    assert all(len(e["appearance"]) == 16 for e in ents)

    # Inside: adjacency at the configured gap; event on the cadence.
    ents, evs = swc._binding_world(100, scenario)
    pos = {e["id"]: e["position"] for e in ents}
    d = sum((a - b) ** 2 for a, b in zip(pos["ent-A"], pos["ent-B"])) ** 0.5
    assert d == pytest.approx(1.5, abs=1e-6)
    assert evs and evs[0]["type"] == "threat_near" and evs[0]["entity"] == "ent-A"
    _, evs_off = swc._binding_world(105, scenario)  # off-cadence step
    assert evs_off == []

    # Observation embeds the entities at the oracle key.
    obs = swc._observation(100, None, scenario)
    assert len(obs["world_state"]["entities"]) == 3


def test_slot_tensor_appearance_and_bounds():
    wm = _wm(capacity=2)
    wm.integrate([_node("ent-x", pos=(500.0, -3.0, 9999.0))])
    app = [0.5, -0.25] + [0.0] * 30  # longer than the 16-d appearance block
    wm.slots["ent-x"].appearance = list(app)
    wm.slots["ent-x"].affective_weight = 100.0  # must squash, not explode
    t, m = wm.slot_tensor()
    row = t[0]
    assert m[0]
    assert row[SLOT_APPEARANCE_SLICE][0] == pytest.approx(0.5)
    assert row[SLOT_APPEARANCE_SLICE][1] == pytest.approx(-0.25)
    # Everything bounded despite huge raw position/affect inputs.
    assert np.isfinite(row).all()
    assert float(np.abs(row).max()) <= 1.0 + 1e-6
    # Read-side purity: building the tensor mutated nothing.
    before = wm.snapshot()
    wm.slot_tensor()
    assert wm.snapshot() == before
