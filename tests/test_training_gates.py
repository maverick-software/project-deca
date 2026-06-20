"""Pure gate/window logic and migrated Skill Dojo phase specs."""

from decadic.training.gates import (
    Criterion,
    delta_metric,
    evaluate_criterion,
    evaluate_gate,
    mean_metric,
)
from decadic.training.skills import AFFECTIVE_LOCOMOTION, DEVELOPMENTAL_LOCOMOTION


def _window(key: str, values: list[float]) -> list[dict]:
    return [{key: v} for v in values]


def test_mean_and_delta_metric():
    w = _window("x", [1.0, 2.0, 3.0])
    assert mean_metric(w, "x") == 2.0
    assert delta_metric(w, "x") == 2.0
    assert mean_metric(w, "missing") == 0.0
    assert delta_metric([{"x": 1.0}], "x") == 0.0


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
    assert abs(low.progress - 0.4) < 1e-6


def test_trend_criterion():
    c = Criterion("distance_traveled", "trend>=", 1.0, "moves")
    rising = evaluate_criterion(c, _window("distance_traveled", [0.0, 0.5, 2.0]))
    assert rising.satisfied is True
    flat = evaluate_criterion(c, _window("distance_traveled", [5.0, 5.0]))
    assert flat.satisfied is False


def test_gate_requires_enough_samples_and_all_criteria():
    crits = [Criterion("rom_mean", ">=", 0.1, "rom")]
    gate = evaluate_gate(crits, _window("rom_mean", [0.5]), min_samples=3)
    assert gate.enough_samples is False
    assert gate.satisfied is False
    assert evaluate_gate(crits, _window("rom_mean", [0.5, 0.5, 0.5]), min_samples=3).satisfied is True

    mixed = [
        Criterion("a", ">=", 1.0, "a"),
        Criterion("b", "<=", 1.0, "b"),
    ]
    assert evaluate_gate(mixed, [{"a": 2.0, "b": 0.5}], min_samples=1).satisfied is True
    assert evaluate_gate(mixed, [{"a": 2.0, "b": 5.0}], min_samples=1).satisfied is False


def test_gate_as_dict_serializes():
    crits = [Criterion("rom_mean", ">=", 0.1, "rom", "frac")]
    d = evaluate_gate(crits, _window("rom_mean", [0.2, 0.2]), min_samples=1).as_dict()
    assert d["satisfied"] is True
    assert d["criteria"][0]["label"] == "rom"
    assert d["criteria"][0]["unit"] == "frac"


def test_developmental_locomotion_migrates_legacy_phase_shape():
    phases = DEVELOPMENTAL_LOCOMOTION.phases
    assert [p.index for p in phases] == [0, 1, 2, 3]
    assert phases[0].name == "Self-modeling"
    assert phases[0].config["viability_mode"] == "immortal"
    assert phases[0].demote_on_death is False
    assert phases[-1].is_terminal is True
    assert {c.command for c in phases[2].periodic_body_commands} == {
        "give_food_near",
        "give_water_near",
    }
    assert {c.key for c in phases[-1].gate.criteria} >= {
        "consume_events",
        "distance_traveled",
        "gait_regularity",
    }


def test_affective_locomotion_adds_terminal_stretch_phase():
    phases = AFFECTIVE_LOCOMOTION.phases
    assert len(phases) == 5
    assert phases[3].is_terminal is False
    assert phases[4].name == "Affective gait"
    assert phases[4].is_terminal is True
    assert phases[4].config["drive_priority_gain"] == 4.0
