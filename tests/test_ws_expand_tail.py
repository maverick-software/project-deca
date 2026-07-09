"""WS-EXPAND E5.3/E6/E7/E8/E9/E10/E13 — the tail-batch milestones.

Pure tests (rest scheduling, other-agent adaptivity gate, threat resolution)
run anywhere; torch tests (veto, slot gate, interoceptive embed, FSQ) pin the
zero/identity-init parity contracts and that each pathway is real once weights
exist.
"""

import math

import pytest

from decadic.consolidation.rest import RestController
from decadic.state.other_agents import OtherAgentRegistry
from decadic.state.spatial_recall import resolve_threat_target


# ------------------------------------------------------------------- E7 rest


def _rc(**kw):
    defaults = dict(load_threshold=100.0, min_wake_cycles=50, rest_cycles=10, pc_load_scale=0.0)
    defaults.update(kw)
    return RestController(**defaults)


def test_rest_triggers_on_load_and_times_out():
    rc = _rc()
    resting = [rc.note_cycle(cycle=i, pc_loss=0.0, threat=False) for i in range(150)]
    assert resting[:99] == [False] * 99  # load below threshold
    assert resting[99] is True  # 100th active cycle crosses 100.0
    assert all(resting[99:110])  # entry cycle + rest_cycles=10 held
    assert resting[110] is False  # woke up
    assert rc.rests_entered == 1


def test_threat_aborts_rest_instantly():
    rc = _rc()
    for i in range(100):
        rc.note_cycle(cycle=i, pc_loss=0.0, threat=False)
    assert rc.in_rest is True
    assert rc.note_cycle(cycle=100, pc_loss=0.0, threat=True) is False  # startled awake
    assert rc.in_rest is False
    assert rc.rests_aborted == 1


def test_wake_time_bound_blocks_back_to_back_rests():
    rc = _rc(load_threshold=10.0, min_wake_cycles=100, rest_cycles=5)
    c = 0
    for _ in range(15):  # first rest
        rc.note_cycle(cycle=c, pc_loss=0.0, threat=False)
        c += 1
    assert rc.rests_entered == 1
    # Load re-crosses quickly, but the wake bound (value-drift guard) holds.
    for _ in range(60):
        rc.note_cycle(cycle=c, pc_loss=0.0, threat=False)
        c += 1
    assert rc.rests_entered == 1  # still only one
    for _ in range(60):
        rc.note_cycle(cycle=c, pc_loss=0.0, threat=False)
        c += 1
    assert rc.rests_entered == 2  # allowed once enough wake time passed


def test_prediction_error_accelerates_load():
    fast, slow = _rc(pc_load_scale=1.0), _rc(pc_load_scale=1.0)
    f = s = 0
    for i in range(100):
        if fast.note_cycle(cycle=i, pc_loss=3.0, threat=False) and not f:
            f = i
        if slow.note_cycle(cycle=i, pc_loss=0.0, threat=False) and not s:
            s = i
    assert f and (not s or f < s)  # a surprising life earns rest sooner


# ------------------------------------------------------------ E10 other agents


def _reg(**kw):
    defaults = dict(err_threshold=0.05, warmup_obs=5, ema_alpha=0.5, max_tracks=8)
    defaults.update(kw)
    return OtherAgentRegistry(**defaults)


def _ent(eid, x, y):
    return {"entity_id": eid, "position": [x, y, 0.0]}


def test_static_props_and_ballistic_movers_never_spawn_models():
    reg = _reg()
    for i in range(40):
        reg.ingest([
            _ent("rock", 3.0, 3.0),  # static prop
            _ent("ball", 0.1 * i, 0.0),  # perfect constant-velocity mover
        ])
    t = reg.telemetry()
    assert t["other_tracks"] == 2
    assert t["other_models_active"] == 0  # the solo/scripted guarantee


def test_adaptive_mover_defeats_the_prior_and_activates():
    reg = _reg()
    import random

    rng = random.Random(5)
    for i in range(40):
        # Erratic, chosen-looking movement: direction changes every step.
        reg.ingest([_ent("agentB", rng.uniform(0, 2), rng.uniform(0, 2))])
    assert reg.telemetry()["other_models_active"] == 1
    assert reg.adaptive_ids() == ["agentB"]
    assert reg.predicted_next("agentB") is not None


def test_track_budget_and_malformed_entities():
    reg = _reg(max_tracks=2)
    reg.ingest([_ent("a", 0, 0), _ent("b", 1, 1), _ent("c", 2, 2)])  # c over budget
    assert reg.telemetry()["other_tracks"] == 2
    reg.ingest([{"position": None}, {"entity_id": "x"}, "junk", None])  # never raises
    assert reg.telemetry()["other_tracks"] == 2


# ------------------------------------------------------- E5.1 threat resolution


class _FakeGraph:
    def __init__(self, beliefs, nodes):
        self._beliefs = beliefs
        self._nodes = nodes


def test_resolve_threat_target_picks_strongest_with_position():
    g = _FakeGraph(
        beliefs={
            ("fire", "predicts_pain"): {"confidence": 0.9, "mean": 0.8},
            ("wasp", "predicts_pain"): {"confidence": 0.4, "mean": 0.5},
            ("water", "predicts_hydration_relief"): {"confidence": 0.9, "mean": 0.9},
            ("ghost", "predicts_integrity_loss"): {"confidence": 0.95, "mean": 0.9},
        },
        nodes={
            "fire": {"position": [5.0, 1.0, 0.0]},
            "wasp": {"position": [1.0, 1.0, 0.0]},
            "water": {"position": [2.0, 2.0, 0.0]},
            "ghost": {},  # strongest belief but NO position -> skipped
        },
    )
    out = resolve_threat_target(g)
    assert out is not None
    tid, pos, strength = out
    assert tid == "fire" and pos[:2] == [5.0, 1.0]
    assert 0.0 < strength <= 1.0
    assert resolve_threat_target(None) is None
    assert resolve_threat_target(_FakeGraph({}, {})) is None


# ------------------------------------------------------------- torch: the heads


def _tiny_stack(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack
    from decadic.nn.plastic import PlasticityFlags

    stack = NeuralCognitiveStack(neural_config_from_env("tiny"), PlasticityFlags())
    stack.eval()
    return stack


def test_veto_zero_at_birth_then_attenuates(monkeypatch):
    torch = pytest.importorskip("torch")
    stack = _tiny_stack(monkeypatch)
    gen = torch.Generator().manual_seed(3)  # deterministic (unseeded weights
    # let w·h swamp the bias into the relu's dead side on some machines)
    z5 = torch.randn(1, stack.cfg.d_model, generator=gen)
    u = torch.randn(1, stack.cfg.n_actuators, generator=gen)
    with torch.no_grad():
        raw = stack.motor_veto_raw(z5, u)
    assert float(torch.tanh(raw).abs().max()) == 0.0  # zero-init -> no veto
    with torch.no_grad():
        # Weight stays zero; a positive bias alone predicts danger -> raw = 2
        # exactly, independent of inputs and platform RNG.
        stack.veto_l2.bias.fill_(2.0)
        att = 0.5 * float(torch.relu(torch.tanh(stack.motor_veto_raw(z5, u))).item())
    assert att == pytest.approx(0.5 * math.tanh(2.0), abs=1e-5)
    assert 0.0 < att <= 0.5  # capped: never a hard zero on the command


def test_intero_embed_parity_at_birth(monkeypatch):
    torch = pytest.importorskip("torch")
    import copy

    from decadic import config as C

    stack = _tiny_stack(monkeypatch)
    z0 = torch.zeros(1, stack.cfg.d_model)
    ep = torch.zeros(1, 4)
    bv = torch.rand(
        1, int(C.INTERO_PRED_DIM) + int(C.TACTILE_PRED_DIM) + int(C.EFFORT_PRED_DIM)
    )
    snap = copy.deepcopy(stack.state_dict())
    stack.load_state_dict(snap)
    with torch.no_grad():
        base = stack(z0, ep)["emotion"].clone()
    stack.load_state_dict(snap)
    with torch.no_grad():
        cond = stack(z0, ep, body_vec=bv)["emotion"].clone()
    assert torch.allclose(base, cond, atol=1e-7)  # zero-init ingress -> parity
    # Open the ingress: affect becomes body-conditioned (the pathway is real).
    with torch.no_grad():
        stack.intero_embed_ingress.weight.normal_(0.0, 0.3)
    snap2 = copy.deepcopy(stack.state_dict())
    stack.load_state_dict(snap2)
    with torch.no_grad():
        a = stack(z0, ep, body_vec=bv)["emotion"].clone()
    stack.load_state_dict(snap2)
    with torch.no_grad():
        b = stack(z0, ep, body_vec=bv * 0.0)["emotion"].clone()
    assert not torch.allclose(a, b, atol=1e-6)


def test_slot_gate_identity_at_birth_and_floored(monkeypatch):
    torch = pytest.importorskip("torch")
    from decadic import config as C
    from decadic.nn.goal_conditioning import GOAL_VEC_DIM

    stack = _tiny_stack(monkeypatch)
    k, sd = 4, int(C.slot_dim())
    slots = torch.randn(1, k, sd)
    gv = torch.rand(GOAL_VEC_DIM)
    with torch.no_grad():
        w = stack.slot_relevance(slots, gv, 0.1)
    assert w.shape == (1, k, 1)
    assert torch.allclose(w, torch.ones_like(w))  # EXACT identity at init
    with torch.no_grad():
        stack.slot_gate.bias.fill_(50.0)  # hostile: try to close everything
        w2 = stack.slot_relevance(slots, gv, 0.1)
    assert float(w2.min()) >= 0.1 - 1e-9  # floor holds: nothing fully silenced


def test_fsq_quantizer_properties():
    torch = pytest.importorskip("torch")
    from decadic.nn.symbol import FSQ_DIMS, FSQ_LEVELS, CodeUsage, fsq_quantize

    x = torch.randn(64, FSQ_DIMS) * 3
    q, idx = fsq_quantize(x)
    assert q.shape == x.shape
    assert float(q.abs().max()) <= 1.0 + 1e-6  # codes live on the bounded grid
    n_codes = 1
    for lv in FSQ_LEVELS:
        n_codes *= lv
    assert int(idx.min()) >= 0 and int(idx.max()) < n_codes
    # Deterministic: same input -> same code (it is a grid, not a codebook).
    q2, idx2 = fsq_quantize(x)
    assert torch.equal(idx, idx2)
    usage = CodeUsage(window=32)
    for c in idx.tolist():
        usage.note(int(c))
    t = usage.telemetry()
    assert t["symbol_codes_seen"] == 64 and 0.0 < t["symbol_utilization"] <= 1.0


def test_goal_ingress_padding_migration_math(monkeypatch):
    # The bundle pads narrower checkpoint goal_ingress weights with zero
    # columns; zero columns mean the new inputs contribute exactly 0.
    torch = pytest.importorskip("torch")
    old = torch.randn(6, 12)  # trained 12-wide ingress (pre-E5 layout)
    pad = torch.zeros(6, 4)
    new = torch.cat([old, pad], dim=1)  # 16-wide migrated weight
    v_old = torch.randn(12)
    v_new = torch.cat([v_old, torch.randn(4)])  # threat slots active
    out_old = old @ v_old
    # Function-preserving: with zero pad columns, even ACTIVE new slots
    # contribute exactly nothing until the columns train.
    assert torch.allclose(new @ v_new, out_old, atol=1e-6)
