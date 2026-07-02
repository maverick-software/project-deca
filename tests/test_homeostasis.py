"""Human-like homeostasis: reservoirs, event routing, metabolism, modes."""

import asyncio

from decadic.config import (
    collision_damage_scale,
    fall_damage_scale,
    food_credit,
    max_integrity_damage_per_obs,
    medical_kit_credit,
    water_credit,
)
from decadic.cycle.pipeline import run_cycle
from decadic.cycle.types import CycleContext
from decadic.memory.episodic_store import EpisodicStore
from decadic.state.perceptual_state import PerceptualState
from decadic.state.state_bus import StateBus
from decadic.state.viability import (
    Homeostasis,
    ViabilityState,
    classify_events,
    passive_metabolism,
)

THRESH = 0.35


# --- Reservoir state --------------------------------------------------------


def test_viability_is_min_of_reservoirs():
    h = Homeostasis(hydration=80.0, energy=55.0, integrity=90.0)
    assert h.viability == 55.0
    h.apply_reservoir_deltas(energy=-100.0)  # clamps at 0
    assert h.energy == 0.0
    assert h.viability == 0.0
    assert h.is_dead()


def test_reservoir_deltas_clamp_to_bounds():
    h = Homeostasis()
    h.apply_reservoir_deltas(hydration=50.0)  # already full -> clamps at 100
    assert h.hydration == 100.0
    h.reset(42.0)
    assert h.hydration == h.energy == h.integrity == 42.0


# --- Event classification ---------------------------------------------------


def test_classify_events_routes_per_reservoir():
    events = [
        {"type": "food", "intensity": 1.0},
        {"type": "water", "intensity": 0.5},
        {"type": "medical_kit", "intensity": 0.8},
        {"type": "collision", "intensity": 0.6},
        {"type": "fall", "intensity": 0.5},
        {"type": "threat_near", "intensity": 0.8},
        {"type": "collision", "intensity": 0.2},  # below threshold -> ignored
    ]
    out = classify_events(events, THRESH)
    assert out["energy_gain"] == 1.0 * food_credit()
    assert out["hydration_gain"] == 0.5 * water_credit()
    assert out["integrity_gain"] == 0.8 * medical_kit_credit()
    # collision 0.6 -> impact-energy scale; fall 0.5 -> superficial scale; the
    # 0.2 collision is filtered out below threshold.
    expected = 0.6 * collision_damage_scale() + 0.5 * fall_damage_scale()
    assert abs(out["integrity_damage"] - expected) < 1e-6
    assert abs(out["stress"] - 0.8) < 1e-6


def test_classify_events_caps_integrity_damage():
    # No single observation may empty the reservoir, however many hits land.
    events = [{"type": "collision", "intensity": 1.0} for _ in range(10)]
    out = classify_events(events, THRESH)
    assert out["integrity_damage"] == max_integrity_damage_per_obs()


def test_fall_is_superficial_and_heals():
    # A normal fall costs only a few integrity points...
    out = classify_events([{"type": "fall", "intensity": 0.6}], THRESH)
    assert 0.0 < out["integrity_damage"] <= 5.0
    # ...and that scrape heals over time while fed and hydrated.
    h = Homeostasis(hydration=80.0, energy=80.0, integrity=100.0 - out["integrity_damage"])
    before = h.integrity
    passive_metabolism(h, 5000.0, 0.0, compression=1.0, **_KNOBS)
    assert h.integrity > before


# --- Passive metabolism -----------------------------------------------------


_KNOBS = dict(
    hydration_empty_s=3 * 24 * 3600,
    energy_empty_s=21 * 24 * 3600,
    integrity_heal_full_s=3 * 24 * 3600,
    heal_min_reserve=25.0,
    stress_gain=1.5,
)


def test_passive_metabolism_thirst_drains_faster_than_hunger():
    h = Homeostasis()
    passive_metabolism(h, 1000.0, 0.0, compression=1.0, **_KNOBS)
    hyd_loss = 100.0 - h.hydration
    eng_loss = 100.0 - h.energy
    assert hyd_loss > 0 and eng_loss > 0
    # 3 weeks / 3 days ~= 7x slower energy depletion
    assert abs(hyd_loss / eng_loss - 7.0) < 0.1


def test_integrity_heals_only_when_fed_and_hydrated():
    fed = Homeostasis(hydration=80.0, energy=80.0, integrity=50.0)
    passive_metabolism(fed, 5000.0, 0.0, compression=1.0, **_KNOBS)
    assert fed.integrity > 50.0  # heals while nourished

    starving = Homeostasis(hydration=10.0, energy=80.0, integrity=50.0)
    passive_metabolism(starving, 5000.0, 0.0, compression=1.0, **_KNOBS)
    assert starving.integrity == 50.0  # no healing while dehydrated


def test_stress_accelerates_depletion():
    calm = Homeostasis()
    stressed = Homeostasis()
    passive_metabolism(calm, 1000.0, 0.0, compression=1.0, **_KNOBS)
    passive_metabolism(stressed, 1000.0, 1.0, compression=1.0, **_KNOBS)
    assert (100.0 - stressed.hydration) > (100.0 - calm.hydration)


def test_compression_fast_forwards_clock():
    slow = Homeostasis()
    fast = Homeostasis()
    passive_metabolism(slow, 100.0, 0.0, compression=1.0, **_KNOBS)
    passive_metabolism(fast, 100.0, 0.0, compression=100.0, **_KNOBS)
    assert (100.0 - fast.hydration) > (100.0 - slow.hydration)


def test_compression_fast_forwards_healing():
    # Healing must scale with the metabolic clock, not crawl at real-time while
    # thirst/hunger are fast-forwarded.
    slow = Homeostasis(hydration=80.0, energy=80.0, integrity=50.0)
    fast = Homeostasis(hydration=80.0, energy=80.0, integrity=50.0)
    passive_metabolism(slow, 100.0, 0.0, compression=1.0, **_KNOBS)
    passive_metabolism(fast, 100.0, 0.0, compression=100.0, **_KNOBS)
    assert (fast.integrity - 50.0) > (slow.integrity - 50.0)


# --- Prediction error no longer drains viability ----------------------------


def test_prediction_error_does_not_drain_viability():
    via = ViabilityState()
    ctx = CycleContext(
        state_bus=StateBus(),
        perceptual=PerceptualState(),
        viability=via,
        episodic=EpisodicStore(None),
    )
    for _ in range(25):
        run_cycle(ctx)
    # The cognitive cycle still produces affect, but the survival scalar is
    # owned by the homeostatic reservoirs and is left untouched here.
    assert via.value == 100.0


# --- Runtime integration ----------------------------------------------------


def _runtime(tmp_path, monkeypatch, agent_id="homeo"):
    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path))
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    monkeypatch.setenv("DECADIC_METABOLIC_TICK_S", "0")
    monkeypatch.delenv("DECADIC_VIABILITY_MODE", raising=False)
    from decadic.agents.runtime import AgentRuntime

    return AgentRuntime(agent_id)


def _obs(events):
    return {
        "timestamp": "2026-06-13T00:00:00Z",
        "proprioception": {"position": [0, 0, 1.4]},
        "events": events,
        "world_state": {"nearby_entities": [], "agent_inventory": []},
    }


def test_runtime_routes_events_to_reservoirs(tmp_path, monkeypatch):
    async def go():
        rt = _runtime(tmp_path, monkeypatch)
        assert rt.viability_mode == "metabolic"
        rt.homeostasis.energy = 40.0
        rt.homeostasis.hydration = 40.0

        await rt.handle_observation_dict(_obs([{"type": "food", "intensity": 1.0}]))
        assert rt.homeostasis.energy > 40.0

        await rt.handle_observation_dict(_obs([{"type": "water", "intensity": 1.0}]))
        assert rt.homeostasis.hydration > 40.0

        await rt.handle_observation_dict(_obs([{"type": "collision", "intensity": 1.0}]))
        assert rt.homeostasis.integrity < 100.0
        assert rt.viability.value == rt.homeostasis.viability
        await rt.stop()

    asyncio.run(go())


def test_immortal_mode_pins_and_disables_death(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, monkeypatch)
    rt.homeostasis.integrity = 30.0
    cfg = rt.configure(viability_mode="immortal")
    assert cfg["viability_mode"] == "immortal"
    assert rt.homeostasis.integrity == 100.0
    assert rt.viability.value == 100.0

    # A reservoir forced to zero must not kill an immortal agent.
    rt.homeostasis.integrity = 0.0
    rt._check_death()
    assert rt.status == "alive"
    assert rt._time_to_death_s() is None


def test_immortal_mode_ignores_damage(tmp_path, monkeypatch):
    async def go():
        rt = _runtime(tmp_path, monkeypatch)
        rt.configure(viability_mode="immortal")
        await rt.handle_observation_dict(_obs([{"type": "collision", "intensity": 50.0}]))
        assert rt.status == "alive"
        assert rt.homeostasis.integrity == 100.0
        assert rt.viability.value == 100.0
        await rt.stop()

    asyncio.run(go())


def test_immortal_revives_dead_agent(tmp_path, monkeypatch):
    async def go():
        rt = _runtime(tmp_path, monkeypatch)
        # Kill the agent via a bottomed-out reservoir.
        rt.homeostasis.integrity = 0.0
        rt._check_death()
        assert rt.status == "dead"

        # Switching to immortal brings the same mind back, reservoirs pinned full
        # (configure runs inside the async API handler in production).
        rt.configure(viability_mode="immortal")
        assert rt.status == "alive"
        assert rt.homeostasis.integrity == 100.0
        assert rt.viability.value == 100.0
        await rt.stop()

    asyncio.run(go())


def test_time_to_death_is_hydration_bound(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, monkeypatch)
    rt.metabolic_compression = 1.0
    rt.stress = 0.0
    rt.homeostasis.hydration = 10.0
    rt.homeostasis.energy = 100.0
    ttd = rt._time_to_death_s()
    assert ttd is not None and ttd > 0
    # With hydration far lower (and draining ~7x faster), it bounds lifespan.
    assert ttd < 3 * 24 * 3600


def test_configure_compression_roundtrips(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, monkeypatch)
    cfg = rt.configure(metabolic_compression=250.0)
    assert cfg["metabolic_compression"] == 250.0
    assert rt.metabolic_compression == 250.0
