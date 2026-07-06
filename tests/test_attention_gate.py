"""WS3-G1: attention gate decision function, hysteresis, budget, pass-through."""

import math

from decadic.cycle.attention_gate import (
    AttentionGate,
    GateInputs,
    PrecedentPassThrough,
    extract_gate_inputs,
    gate_enabled,
    gate_novelty_source,
)


def _gate(**kw):
    defaults = dict(
        threshold=0.55,
        weights=(0.35, 0.30, 0.25, 0.10),
        target_rate=0.05,
        hysteresis_k=3,
        rate_window=100,
        budget_gain=0.5,
    )
    defaults.update(kw)
    return AttentionGate(**defaults)


# ------------------------------------------------- WS-FORAGE M5: Type-2 search


def test_type2_search_escalates_below_threshold():
    # Nothing novel in view (score far below threshold), but the agent needs a
    # remembered-but-not-here resource -> deliberate anyway (System-2).
    g = _gate()
    d = g.decide(GateInputs(novelty=0.0, prediction_error=0.0, affect=0.0, type2_search=True))
    assert d.escalate is True
    assert d.reason == "type2_memory_search"


def test_type2_search_yields_to_threat_fast_path():
    # A live threat outranks a memory search: fast_path wins the reason.
    g = _gate()
    d = g.decide(GateInputs(fast_path_threat=True, type2_search=True))
    assert d.escalate is True and d.reason == "fast_path"


def test_type2_search_latches_hysteresis():
    # Like the other escalation sources, a Type-2 escalation latches the gate
    # for a few cycles so a brief search doesn't flicker.
    g = _gate(hysteresis_k=2)
    assert g.decide(GateInputs(type2_search=True)).reason == "type2_memory_search"
    assert g.decide(GateInputs()).reason == "hysteresis"


def test_type2_trigger_truth_table():
    from decadic.cycle.attention_gate import type2_trigger

    def gv(deficit, dist, mask):
        # goal_conditioning layout: [0:3] one-hot, [3] deficit, [4:6] bearing,
        # [6] distance, [7] target mask.
        return [1.0, 0.0, 0.0, deficit, 1.0, 0.0, dist, mask]

    kw = dict(far_distance=0.15, min_deficit=0.05)
    assert type2_trigger(gv(0.3, 0.5, 1.0), **kw) is True  # needy + remembered + far
    assert type2_trigger(gv(0.3, 0.05, 1.0), **kw) is False  # target is HERE
    assert type2_trigger(gv(0.3, 0.5, 0.0), **kw) is False  # nothing remembered
    assert type2_trigger(gv(0.02, 0.5, 1.0), **kw) is False  # deficit below the graded bar
    assert type2_trigger(gv(0.05, 0.15, 1.0), **kw) is True  # boundary inclusive
    assert type2_trigger(None, **kw) is False
    assert type2_trigger([1.0, 0.0], **kw) is False  # malformed -> never raises
    assert type2_trigger(gv(float("nan"), 0.5, 1.0), **kw) is False


def test_extract_gate_inputs_threads_type2():
    gi = extract_gate_inputs(
        best_recall_similarity=1.0,
        pc_ema=0.0,
        pain_scalar=0.0,
        drive_pressure=0.0,
        priority_label="idle",
        observation_events=[],
        type2_search=True,
    )
    assert gi.type2_search is True
    # Default is off (byte-identical to pre-M5 callers that don't pass it).
    gi2 = extract_gate_inputs(
        best_recall_similarity=1.0,
        pc_ema=0.0,
        pain_scalar=0.0,
        drive_pressure=0.0,
        priority_label="idle",
        observation_events=[],
    )
    assert gi2.type2_search is False


def test_default_enabled(monkeypatch):
    # Default ON since 2026-07-04 (owner decision, post probe PASS); the
    # byte-identical baseline lives behind the conftest pin + env=0.
    monkeypatch.delenv("DECADIC_GATE_ENABLED", raising=False)
    assert gate_enabled() is True
    monkeypatch.setenv("DECADIC_GATE_ENABLED", "0")
    assert gate_enabled() is False  # baseline remains one env var away


def test_novelty_source_config(monkeypatch):
    monkeypatch.delenv("DECADIC_GATE_NOVELTY_SOURCE", raising=False)
    assert gate_novelty_source() == "percept"  # probe-validated default
    monkeypatch.setenv("DECADIC_GATE_NOVELTY_SOURCE", "full")
    assert gate_novelty_source() == "full"  # legacy signal stays selectable
    monkeypatch.setenv("DECADIC_GATE_NOVELTY_SOURCE", "bogus")
    assert gate_novelty_source() == "percept"  # unknown values fall back safely


def test_novelty_recency_horizon_config(monkeypatch):
    from decadic.cycle.attention_gate import (
        gate_novelty_peak_window,
        gate_novelty_recency_horizon,
    )

    monkeypatch.delenv("DECADIC_GATE_NOVELTY_RECENCY", raising=False)
    assert gate_novelty_recency_horizon() == 64  # below the 200-cycle patrol lap
    monkeypatch.setenv("DECADIC_GATE_NOVELTY_RECENCY", "0")
    assert gate_novelty_recency_horizon() == 0  # disabled
    monkeypatch.delenv("DECADIC_GATE_NOVELTY_PEAK_WINDOW", raising=False)
    assert gate_novelty_peak_window() == 32


def test_note_novelty_rolling_peak():
    """Telemetry peak holds a 1-cycle spike for the window, then releases."""
    g = _gate()
    g._novelty_peak_window = 10
    for c in range(5):
        assert g.note_novelty(0.01, c) <= 0.011
    assert g.note_novelty(0.85, 5) == 0.85  # the spike itself
    for c in range(6, 15):  # within window: peak persists for the sampler
        assert g.note_novelty(0.01, c) == 0.85
    assert g.note_novelty(0.01, 16) < 0.05  # spike evicted past the window
    # Never feeds decisions: decide() output is unaffected by peak state.
    d = g.decide(GateInputs(novelty=0.1, prediction_error=0.1, affect=0.0))
    assert d.escalate is False


def test_quiet_input_skips():
    g = _gate()
    d = g.decide(GateInputs(novelty=0.1, prediction_error=0.1, affect=0.0))
    assert d.escalate is False and d.reason == "skip"
    assert 0.0 <= d.score < 0.55


def test_high_novelty_escalates():
    g = _gate()
    d = g.decide(GateInputs(novelty=1.0, prediction_error=0.8, affect=0.5))
    assert d.escalate is True and d.reason == "score"


def test_fast_path_always_escalates_even_over_budget():
    g = _gate(target_rate=0.0, budget_gain=100.0)  # budget maximally hostile
    for _ in range(20):
        g.decide(GateInputs(novelty=1.0, prediction_error=1.0))  # saturate rate
    d = g.decide(GateInputs(fast_path_threat=True))
    assert d.escalate is True and d.reason == "fast_path"


def test_hysteresis_latches_k_cycles():
    g = _gate(hysteresis_k=3)
    first = g.decide(GateInputs(novelty=1.0, prediction_error=1.0, affect=1.0))
    assert first.escalate and first.reason == "score"
    reasons = [g.decide(GateInputs()).reason for _ in range(4)]
    assert reasons == ["hysteresis", "hysteresis", "hysteresis", "skip"]


def test_budget_pressure_raises_threshold():
    g = _gate(threshold=0.5, target_rate=0.05, budget_gain=1.0)
    borderline = GateInputs(novelty=0.6, prediction_error=0.6, affect=0.6, priority_investigate=0.6)
    assert g.decide(borderline).escalate is True
    # saturate the trailing rate far above target
    for _ in range(50):
        g.decide(borderline)
    d = g.decide(borderline)
    assert d.threshold_effective > 0.5
    # overshoot ~0.95 over a 0.05 target with gain 1.0 pushes the bar above 0.6
    assert d.escalate is False


def test_escalation_rate_and_streak():
    g = _gate()
    for _ in range(9):
        g.decide(GateInputs())
    g.decide(GateInputs(fast_path_threat=True))
    t = g.telemetry()
    assert t["gate_decisions"] == 10
    assert t["gate_escalations"] == 1
    assert abs(t["gate_escalation_rate"] - 0.1) < 1e-9
    assert t["gate_skip_streak"] == 0  # last decision escalated


def test_nan_and_out_of_range_inputs_are_clamped():
    g = _gate()
    d = g.decide(GateInputs(novelty=float("nan"), prediction_error=5.0, affect=-2.0))
    assert 0.0 <= d.score <= 1.0


def test_determinism():
    a, b = _gate(), _gate()
    seq = [GateInputs(novelty=i / 10.0, prediction_error=(10 - i) / 10.0) for i in range(11)]
    ra = [(a.decide(s).escalate, round(a.decide(s).score, 9)) for s in seq]
    rb = [(b.decide(s).escalate, round(b.decide(s).score, 9)) for s in seq]
    assert ra == rb


def test_extract_inputs_normalization():
    gi = extract_gate_inputs(
        best_recall_similarity=0.75,
        pc_ema=1.0,
        pain_scalar=0.3,
        drive_pressure=0.2,
        priority_label="investigate",
        observation_events=[{"type": "collision", "intensity": 0.9}],
    )
    assert abs(gi.novelty - 0.25) < 1e-9
    assert abs(gi.prediction_error - 0.5) < 1e-9  # 1/(1+1)
    assert abs(gi.affect - 0.5) < 1e-9
    assert gi.priority_investigate == 1.0
    assert gi.fast_path_threat is True


def test_extract_inputs_cold_store_reads_fully_novel():
    gi = extract_gate_inputs(
        best_recall_similarity=None,
        pc_ema=None,
        pain_scalar=-0.4,  # pleasure-side pain scalar must not raise affect
        drive_pressure=0.0,
        priority_label="explore",
        observation_events=[],
    )
    assert gi.novelty == 1.0
    assert gi.prediction_error == 0.0
    assert gi.affect == 0.0
    assert gi.priority_investigate == 0.0
    assert gi.fast_path_threat is False


def test_pass_through_decay_and_structure():
    p = PrecedentPassThrough(tau=10.0)
    assert p.emit() is None  # no precedent -> caller must escalate
    p.store({"risk": 0.8, "utility_vec": [1.0, -2.0], "label": "x", "flag": True})
    first = p.emit()
    assert first["gate_pass_through"] is True
    assert math.isclose(first["risk"], 0.8 * math.exp(-1 / 10.0), rel_tol=1e-9)
    assert first["label"] == "x" and first["flag"] is True
    for _ in range(50):
        out = p.emit()
    assert abs(out["risk"]) < 0.01  # decayed toward neutral
    assert out["gate_precedent_age"] == 51
    # a fresh escalation resets age and magnitude
    p.store({"risk": 0.5})
    assert math.isclose(p.emit()["risk"], 0.5 * math.exp(-1 / 10.0), rel_tol=1e-9)
