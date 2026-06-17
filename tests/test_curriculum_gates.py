"""Pure gate/window logic for the walking curriculum (no I/O, no cognition)."""

from decadic.curriculum.gates import (
    Criterion,
    delta_metric,
    evaluate_criterion,
    evaluate_gate,
    mean_metric,
)
from decadic.curriculum.phases import affective_phase, build_phases, default_phases


def _window(key: str, values: list[float]) -> list[dict]:
    return [{key: v} for v in values]


def test_mean_and_delta_metric():
    w = _window("x", [1.0, 2.0, 3.0])
    assert mean_metric(w, "x") == 2.0
    assert delta_metric(w, "x") == 2.0
    # Missing key -> safe defaults.
    assert mean_metric(w, "missing") == 0.0
    assert delta_metric([{"x": 1.0}], "x") == 0.0  # need >=2 samples


def test_below_criterion():
    c = Criterion("forward_model_error", "<=", 0.05, "PE low")
    res = evaluate_criterion(c, _window("forward_model_error", [0.02, 0.03]))
    assert res.satisfied is True
    assert res.progress == 1.0
    res2 = evaluate_criterion(c, _window("forward_model_error", [0.2, 0.2]))
    assert res2.satisfied is False
    assert res2.progress < 1.0


def test_above_criterion():
    c = Criterion("rom_mean", ">=", 0.25, "ROM")
    assert evaluate_criterion(c, _window("rom_mean", [0.3, 0.4])).satisfied is True
    low = evaluate_criterion(c, _window("rom_mean", [0.1, 0.1]))
    assert low.satisfied is False
    assert abs(low.progress - 0.4) < 1e-6  # 0.1 / 0.25


def test_trend_criterion():
    c = Criterion("distance_traveled", "trend>=", 1.0, "moves")
    rising = evaluate_criterion(c, _window("distance_traveled", [0.0, 0.5, 2.0]))
    assert rising.satisfied is True
    flat = evaluate_criterion(c, _window("distance_traveled", [5.0, 5.0]))
    assert flat.satisfied is False


def test_gate_requires_enough_samples():
    crits = [Criterion("rom_mean", ">=", 0.1, "rom")]
    w = _window("rom_mean", [0.5])  # only 1 sample
    gate = evaluate_gate(crits, w, min_samples=3)
    assert gate.enough_samples is False
    assert gate.satisfied is False
    # Enough samples + met -> open.
    gate2 = evaluate_gate(crits, _window("rom_mean", [0.5, 0.5, 0.5]), min_samples=3)
    assert gate2.satisfied is True


def test_gate_all_must_pass():
    crits = [
        Criterion("a", ">=", 1.0, "a"),
        Criterion("b", "<=", 1.0, "b"),
    ]
    window = [{"a": 2.0, "b": 0.5}, {"a": 2.0, "b": 0.5}]
    assert evaluate_gate(crits, window, min_samples=2).satisfied is True
    bad = [{"a": 2.0, "b": 5.0}, {"a": 2.0, "b": 5.0}]
    assert evaluate_gate(crits, bad, min_samples=2).satisfied is False


def test_gate_as_dict_serializes():
    crits = [Criterion("rom_mean", ">=", 0.1, "rom", "frac")]
    d = evaluate_gate(crits, _window("rom_mean", [0.2, 0.2]), min_samples=1).as_dict()
    assert d["satisfied"] is True
    assert d["criteria"][0]["label"] == "rom"
    assert d["criteria"][0]["unit"] == "frac"


def test_default_phase_table_shape():
    phases = default_phases()
    assert [p.index for p in phases] == [0, 1, 2, 3]
    assert phases[0].name == "Self-modeling"
    assert phases[0].config.viability_mode == "immortal"
    assert phases[-1].is_terminal is True
    # Only the locomotion phases place satisfiers.
    assert phases[0].satisfier.enabled is False
    assert phases[2].satisfier.enabled is True


def test_build_phases_affective_optin_and_overrides():
    assert len(build_phases()) == 4
    withbear = build_phases(include_affective=True)
    assert len(withbear) == 5
    assert withbear[3].is_terminal is False  # no longer terminal
    assert withbear[4].is_terminal is True
    tuned = build_phases(overrides={"Self-modeling": {"min_dwell_s": 7.5}})
    assert tuned[0].min_dwell_s == 7.5


def test_phase_config_to_kwargs_skips_none():
    phases = default_phases()
    kw = phases[1].config.to_configure_kwargs()
    assert kw["viability_mode"] == "metabolic"
    # Phase 1 doesn't set babble/drive, so they must be absent (env parity).
    assert "motor_babble_sigma" not in kw
    assert "drive_priority_gain" not in kw


def test_affective_phase_targets_threat_survival():
    p = affective_phase()
    assert p.index == 4
    assert p.satisfier.enabled is True
    keys = {c.key for c in p.promote_criteria}
    assert "viability" in keys and "distance_traveled" in keys
