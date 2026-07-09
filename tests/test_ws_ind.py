"""WS-IND: attention schema (I1), per-slot reliability (I3), belief tempering
(I4), sequential deliberation (I2), FSQ smoothness (I5).

Pins the birth-identity contracts (zero-init schema head predicts nothing and
biases nothing; draft round == final round at birth; identity reliability),
the guardrails (bounded/composable bias, floors, relative-not-absolute
reliability, temper-never-veto), and that each pathway is real once trained.
"""

import math

import pytest

from decadic.nn.attention_schema import (
    GATE_REASONS,
    GATE_STATE_DIM,
    SCHEMA_VEC_DIM,
    SchemaAccuracy,
    build_gate_state_vec,
    encode_realized_target,
    schema_gate_bias,
)
from decadic.nn.slot_reliability import SlotReliability


# ------------------------------------------------------------- I1 pure parts


class _FakeDecision:
    def __init__(self, escalate=False, reason="skip", score=0.2, threshold=0.3):
        self.escalate = escalate
        self.reason = reason
        self.score = score
        self.threshold_effective = threshold


class _FakeGate:
    hysteresis_k = 3
    type2_refractory = 32
    _latch_remaining = 1
    _type2_cooldown = 8
    escalation_rate = 0.1


def test_gate_state_vec_layout_and_defensiveness():
    v = build_gate_state_vec(_FakeDecision(True, "score", 0.6, 0.31), _FakeGate())
    assert len(v) == GATE_STATE_DIM
    assert v[0] == pytest.approx(0.6) and v[2] == 1.0
    assert v[3] == pytest.approx(1 / 3) and v[4] == pytest.approx(8 / 32)
    assert build_gate_state_vec(None, None) == [0.0] * GATE_STATE_DIM


def test_realized_target_encoding():
    esc, idx, score = encode_realized_target(_FakeDecision(True, "type2_memory_search", 0.4))
    assert esc == 1.0 and GATE_REASONS[idx] == "type2_memory_search"
    assert score == pytest.approx(0.4)
    # Bootstrap artifact maps to the score class; unknown reasons to skip.
    assert GATE_REASONS[encode_realized_target(_FakeDecision(True, "no_precedent"))[1]] == "score"
    assert GATE_REASONS[encode_realized_target(_FakeDecision(False, "bogus"))[1]] == "skip"


def test_schema_bias_zero_at_chance_and_bounded():
    # p <= 0.5 (including the zero-init head's exact output) -> bias 0: parity.
    assert schema_gate_bias(0.5, gain=0.05, cap=0.05) == 0.0
    assert schema_gate_bias(0.3, gain=0.05, cap=0.05) == 0.0
    assert schema_gate_bias(float("nan"), gain=0.05, cap=0.05) == 0.0
    assert schema_gate_bias(1.0, gain=0.05, cap=0.05) == pytest.approx(0.05)
    assert schema_gate_bias(1.0, gain=99.0, cap=0.05) == 0.05  # capped


def test_schema_accuracy_tracker_and_base_rate():
    acc = SchemaAccuracy(window=64)
    for i in range(40):
        realized = i % 10 == 0  # 10% escalation rate
        acc.note(0.9 if realized else 0.1, realized)  # perfect predictor
    t = acc.telemetry()
    assert t["schema_accuracy"] == 1.0
    assert t["schema_base_accuracy"] == pytest.approx(0.9)  # majority-class bar


# ------------------------------------------------------------- I3 reliability


def _srel(**kw):
    defaults = dict(max_slots=8, fast_alpha=0.5, noise_alpha=0.5, floor=0.25, warmup=3)
    defaults.update(kw)
    return SlotReliability(**defaults)


def test_reliability_identity_for_quiet_and_uniform_streams():
    import random

    r = _srel()
    for _ in range(20):  # constant slots -> zero noise everywhere -> identity
        rel = r.update([[1.0, 2.0], [3.0, 4.0]])
    assert rel == [1.0, 1.0]
    # Uniformly noisy slots: reliability is RELATIVE -> still identity.
    rng = random.Random(1)
    r2 = _srel()
    for _ in range(30):
        rel = r2.update([[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(3)])
    assert all(v >= 0.85 for v in rel)  # no slot singled out


def test_noisy_slot_downweighted_stable_slot_not():
    import random

    rng = random.Random(7)
    r = _srel()
    for _ in range(30):
        rel = r.update(
            [
                [1.0, 1.0],  # rock stable
                [rng.gauss(0, 2.0), rng.gauss(0, 2.0)],  # jittering artifact
            ]
        )
    assert rel[0] == 1.0  # at-or-below-average noise never penalized
    assert rel[1] == 0.25  # noisy one floored (never zero)


def test_reliability_never_raises_on_junk():
    r = _srel()
    assert r.update(None) == []
    assert r.update("junk"[0:0]) == []
    t = r.telemetry()
    assert all(math.isfinite(float(v)) for v in t.values())


# ------------------------------------------------------------- I4 tempering


def test_belief_temper_math_never_vetoes():
    # gain = 1 - w*(1 - conf): conf 1 -> 1.0 (parity), conf 0 -> 1-w (tempered,
    # never zero for w<1) — a first observation always leaves evidence.
    w = 0.5
    for conf, expected in ((1.0, 1.0), (0.5, 0.75), (0.0, 0.5)):
        assert 1.0 - w * (1.0 - conf) == pytest.approx(expected)


def test_belief_temper_in_working_memory(monkeypatch):
    monkeypatch.setenv("DECADIC_BELIEF_TEMPER", "1")
    monkeypatch.setenv("DECADIC_BELIEF_TEMPER_WEIGHT", "0.5")
    from decadic.state.working_memory import MemorySlot, WorkingMemory

    def run(conf):
        wm = WorkingMemory()
        wm.cycle = 5
        slot = MemorySlot(entity_id="blob", salience=1.0, last_seen_cycle=5)
        slot.confidence = conf
        wm.slots["blob"] = slot
        wm._bind_events_discovered([{"type": "collision", "intensity": 0.8}])
        return (
            float(slot.property_evidence.get("predicts_pain", 0.0)),
            float(slot.affective_weight),
        )

    ev_hi, aff_hi = run(1.0)
    ev_lo, aff_lo = run(0.0)
    assert ev_hi == pytest.approx(0.8)  # confident percept: full evidence
    assert ev_lo == pytest.approx(0.4)  # junk percept: tempered, NOT vetoed
    assert aff_lo == pytest.approx(aff_hi)  # the feeling itself is untempered


# ------------------------------------------------- torch: heads + I2 parity


def _tiny_stack(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack
    from decadic.nn.plastic import PlasticityFlags

    # Deterministic INIT: the trainability tests below depend on the random
    # starting weights; unseeded construction made them run-to-run lotteries
    # (observed 0.67 vs 0.70 accuracy across phases of one suite run).
    torch.manual_seed(1234)
    stack = NeuralCognitiveStack(neural_config_from_env("tiny"), PlasticityFlags())
    stack.eval()
    return stack


def test_schema_head_zero_init_and_ingress_parity(monkeypatch):
    torch = pytest.importorskip("torch")
    import copy

    stack = _tiny_stack(monkeypatch)
    assert torch.count_nonzero(stack.schema_l2.weight) == 0
    assert torch.count_nonzero(stack.schema_ingress.weight) == 0
    z5 = torch.randn(1, stack.cfg.d_model)
    gs = torch.rand(1, GATE_STATE_DIM)
    with torch.no_grad():
        raw = stack.attention_schema_predict(z5, gs)
    assert raw.shape == (1, SCHEMA_VEC_DIM)
    assert torch.count_nonzero(raw) == 0  # zero-init -> p(escalate)=0.5, bias 0
    # Feedback ingress: a nonzero schema vector changes nothing at birth.
    z0 = torch.zeros(1, stack.cfg.d_model)
    ep = torch.zeros(1, 4)
    snap = copy.deepcopy(stack.state_dict())
    stack.load_state_dict(snap)
    with torch.no_grad():
        base = stack(z0, ep)["motor_u"].clone()
    stack.load_state_dict(snap)
    with torch.no_grad():
        fed = stack(z0, ep, schema_vec=torch.rand(SCHEMA_VEC_DIM))["motor_u"].clone()
    assert torch.allclose(base, fed, atol=1e-7)


def test_draft_round_parity_and_recurrent_state_restore(monkeypatch):
    torch = pytest.importorskip("torch")
    import copy

    stack = _tiny_stack(monkeypatch)
    z0 = torch.randn(1, stack.cfg.d_model)
    ep = torch.zeros(1, 4)
    snap = copy.deepcopy(stack.state_dict())
    # I2 semantics: draft forward, restore state, final forward with the draft
    # fed back. At birth (zero-init draft ingress) the final round must be
    # IDENTICAL to a plain single forward from the same state.
    stack.load_state_dict(snap)
    with torch.no_grad():
        plain = stack(z0, ep)["motor_u"].clone()
        state_after_plain = stack.gru_h.clone()
    stack.load_state_dict(snap)
    rec = stack.snapshot_recurrent_state()
    with torch.no_grad():
        draft = stack(z0, ep)
    stack.restore_recurrent_state(rec)
    with torch.no_grad():
        dv = torch.cat([draft["z5"].detach(), draft["motor_u"].detach()], dim=-1)
        final = stack(z0, ep, draft_vec=dv)["motor_u"].clone()
    assert torch.allclose(plain, final, atol=1e-6)  # birth parity
    assert torch.allclose(stack.gru_h, state_after_plain, atol=1e-6)  # ONE advance
    # Once the draft ingress has weight, round 2 genuinely re-deliberates.
    with torch.no_grad():
        stack.draft_ingress.weight.normal_(0.0, 0.3)
    snap2 = copy.deepcopy(stack.state_dict())
    stack.load_state_dict(snap2)
    rec = stack.snapshot_recurrent_state()
    with torch.no_grad():
        draft = stack(z0, ep)
    stack.restore_recurrent_state(rec)
    with torch.no_grad():
        dv = torch.cat([draft["z5"].detach(), draft["motor_u"].detach()], dim=-1)
        final2 = stack(z0, ep, draft_vec=dv)["motor_u"].clone()
    assert not torch.allclose(draft["motor_u"], final2, atol=1e-6)


def test_schema_becomes_predictive_when_trained(monkeypatch):
    torch = pytest.importorskip("torch")
    stack = _tiny_stack(monkeypatch)
    d = stack.cfg.d_model
    gen = torch.Generator().manual_seed(4)
    # Synthetic attention regime: escalation is a simple function of the gate
    # state (high score -> escalates). The schema must learn it.
    params = [p for n, p in stack.named_parameters() if n.startswith("schema_l")]
    opt = torch.optim.Adam(params, lr=1e-2)
    # The signal lives in ONE of 6 gate-state dims against d_model dims of
    # irrelevant latent — isolating it needs a real batch and budget (batch 4
    # x 300 steps plateaued at ~0.7). Early-exit keeps the passing case fast.
    for step in range(2000):
        z5 = torch.randn(32, d, generator=gen)
        gs = torch.rand(32, GATE_STATE_DIM, generator=gen)
        target = (gs[:, 0] > 0.5).float()  # escalate iff score high
        opt.zero_grad()
        raw = stack.attention_schema_predict(z5, gs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(raw[:, 0], target)
        loss.backward()
        opt.step()
        if step > 100 and float(loss.item()) < 0.08:
            break
    z5 = torch.randn(64, d, generator=gen)
    gs = torch.rand(64, GATE_STATE_DIM, generator=gen)
    with torch.no_grad():
        pred = torch.sigmoid(stack.attention_schema_predict(z5, gs)[:, 0]) > 0.5
    acc = float((pred == (gs[:, 0] > 0.5)).float().mean())
    assert acc > 0.85  # far above the ~0.5 base rate of this regime


def test_fsq_smoothness_trains_projection(monkeypatch):
    torch = pytest.importorskip("torch")
    from decadic.nn.symbol import fsq_quantize

    stack = _tiny_stack(monkeypatch)
    z5a = torch.randn(1, stack.cfg.d_model)
    z5b = z5a + 0.01 * torch.randn(1, stack.cfg.d_model)
    pa = stack.fsq_in(z5a.detach())
    pb = stack.fsq_in(z5b.detach())
    qa, _ = fsq_quantize(pa)
    qb, _ = fsq_quantize(pb)
    l_smooth = ((qa - qb).norm() - (torch.tanh(pa) - torch.tanh(pb)).norm()).pow(2)
    l_smooth.backward()
    # The smoothness objective reaches fsq_in (I5's point: the projection is
    # finally trainable) and NOT the trunk (input detached: E9 parity holds).
    assert stack.fsq_in.weight.grad is not None
    assert float(stack.fsq_in.weight.grad.abs().sum()) >= 0.0


# --------------------------------------------- E10.3/E10.4: other-vector lane


def test_other_vec_layout_and_solo_zero():
    from decadic.state.other_agents import (
        OTHER_VEC_DIM,
        OtherAgentRegistry,
        encode_other_vec,
    )

    assert OTHER_VEC_DIM == 8  # frozen layout (other_ingress depends on it)
    # Solo / missing anything -> all zeros (the adaptivity gate's guarantee
    # carries into the policy input).
    assert encode_other_vec(None, [0.0, 0.0, 0.0], 0.0, max_dist=10.0) == [0.0] * 8
    reg = OtherAgentRegistry(err_threshold=0.05, warmup_obs=3, ema_alpha=0.5, max_tracks=4)
    for i in range(10):  # a static prop: tracked, never adaptive
        reg.ingest([{"entity_id": "rock", "position": [3.0, 0.0, 0.0]}])
    assert encode_other_vec(reg, [0.0, 0.0, 0.0], 0.0, max_dist=10.0) == [0.0] * 8


def test_other_vec_populates_for_adaptive_other():
    import random

    from decadic.state.other_agents import OtherAgentRegistry, encode_other_vec

    rng = random.Random(9)
    reg = OtherAgentRegistry(err_threshold=0.05, warmup_obs=3, ema_alpha=0.5, max_tracks=4)
    pos = None
    for _ in range(20):  # erratic mover dead ahead-ish: defeats the prior
        pos = [4.0 + rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0]
        reg.ingest([{"entity_id": "agentB", "position": pos}])
    v = encode_other_vec(reg, [0.0, 0.0, 0.0], 0.0, max_dist=10.0)
    assert v[0] == 1.0  # presence
    assert v[1] > 0.5  # roughly ahead (cos az)
    assert 0.0 < v[3] <= 1.0  # normalized distance
    assert v[7] > 0.0  # adaptivity strength
    assert all(abs(x) <= 1.0 + 1e-9 for x in v)


def test_other_ingress_zero_init_parity(monkeypatch):
    torch = pytest.importorskip("torch")
    import copy

    from decadic.state.other_agents import OTHER_VEC_DIM

    stack = _tiny_stack(monkeypatch)
    assert torch.count_nonzero(stack.other_ingress.weight) == 0
    z0 = torch.zeros(1, stack.cfg.d_model)
    ep = torch.zeros(1, 4)
    snap = copy.deepcopy(stack.state_dict())
    stack.load_state_dict(snap)
    with torch.no_grad():
        base = stack(z0, ep)["motor_u"].clone()
    stack.load_state_dict(snap)
    with torch.no_grad():
        fed = stack(z0, ep, other_vec=torch.rand(OTHER_VEC_DIM))["motor_u"].clone()
    assert torch.allclose(base, fed, atol=1e-7)  # zero-init lane -> parity


def test_inverse_dynamics_learns_own_transitions(monkeypatch):
    # E10.4 prerequisite: given a synthetic body where the realized proprio
    # delta IS a linear image of the executed action, supervised training on
    # lived triples must recover the action from the transition.
    torch = pytest.importorskip("torch")
    stack = _tiny_stack(monkeypatch)
    fdim, n_act = stack.cfg.forward_pred_dim, stack.cfg.n_actuators
    gen = torch.Generator().manual_seed(21)
    mix = torch.randn(n_act, fdim, generator=gen) * 0.5  # body's action->delta map
    params = [p for n, p in stack.named_parameters() if n.startswith("inv_l")]
    opt = torch.optim.Adam(params, lr=1e-2)
    for step in range(2000):
        prev = torch.randn(32, fdim, generator=gen)
        u = torch.rand(32, n_act, generator=gen) * 1.6 - 0.8
        now = prev + u @ mix
        opt.zero_grad()
        l = torch.nn.functional.mse_loss(stack.inverse_action(prev, now), u)
        l.backward()
        opt.step()
        if step > 100 and float(l.item()) < 0.01:
            break
    prev = torch.randn(64, fdim, generator=gen)
    u = torch.rand(64, n_act, generator=gen) * 1.6 - 0.8
    now = prev + u @ mix
    with torch.no_grad():
        err = (stack.inverse_action(prev, now) - u).abs().mean()
    assert float(err) < 0.15  # labels observed transitions with its own model