"""Viability and motivational scaffolding (pain / pleasure → element B)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ViabilityState:
    """Scalar viability on a bounded axis; changes drive pain/pleasure.

    In the homeostatic model this is the *derived* master scalar: it mirrors
    ``Homeostasis.viability`` (the min of the reservoirs). It is kept as its own
    object so existing readers/writers of ``agent.viability.value`` keep working.
    """

    value: float = 100.0
    min_value: float = 0.0
    max_value: float = 100.0

    def apply_delta(self, delta: float) -> float:
        prev = self.value
        self.value = float(np.clip(self.value + delta, self.min_value, self.max_value))
        return self.value - prev


@dataclass
class Homeostasis:
    """Three human-like reservoirs whose minimum is the agent's viability.

    - ``hydration``: drains fastest (thirst), refilled by water/drink events.
    - ``energy``: drains slowest (hunger), refilled by food/eat events.
    - ``integrity``: tissue health; cut by damage, healed slowly when fed.
    """

    hydration: float = 100.0
    energy: float = 100.0
    integrity: float = 100.0
    min_value: float = 0.0
    max_value: float = 100.0

    @property
    def viability(self) -> float:
        return float(min(self.hydration, self.energy, self.integrity))

    def _clamp(self, value: float) -> float:
        return float(np.clip(value, self.min_value, self.max_value))

    def apply_reservoir_deltas(
        self, *, hydration: float = 0.0, energy: float = 0.0, integrity: float = 0.0
    ) -> None:
        self.hydration = self._clamp(self.hydration + hydration)
        self.energy = self._clamp(self.energy + energy)
        self.integrity = self._clamp(self.integrity + integrity)

    def reset(self, value: float | None = None) -> None:
        v = self.max_value if value is None else self._clamp(value)
        self.hydration = v
        self.energy = v
        self.integrity = v

    def is_dead(self) -> bool:
        return self.viability <= self.min_value

    def snapshot(self) -> dict[str, float]:
        return {
            "hydration": round(self.hydration, 4),
            "energy": round(self.energy, 4),
            "integrity": round(self.integrity, 4),
            "viability": round(self.viability, 4),
        }


def classify_events(events: list[dict], threshold: float) -> dict[str, float]:
    """Split observation events into per-reservoir effects.

    Returns non-negative ``integrity_damage``, ``integrity_gain``,
    ``energy_gain``, ``hydration_gain`` and a ``stress`` term (anticipatory threat). This
    generalizes :func:`damage_from_events` / :func:`nourishment_from_events`.

    Injury is human-superficial: a ``collision`` carries an impact-energy
    intensity (0..1) and costs ``collision_damage_scale`` at most; a one-shot
    ``fall`` is a minor scrape; explicit game ``damage`` is moderate. The total
    integrity loss from a single observation is capped so contact can never be
    instantly lethal -- only sustained, untreated trauma (or starvation) kills.
    """
    from decadic.config import (
        collision_damage_scale,
        fall_damage_scale,
        food_credit,
        game_damage_scale,
        max_integrity_damage_per_obs,
        medical_kit_credit,
        water_credit,
    )

    fc = food_credit()
    wc = water_credit()
    mc = medical_kit_credit()
    cds = collision_damage_scale()
    fds = fall_damage_scale()
    gds = game_damage_scale()
    out = {
        "integrity_damage": 0.0,
        # The portion of integrity_damage from external threats (explicit damage /
        # combat / environmental hits - the "bear bite"). Tracked separately so the
        # runtime can exempt it from the learning-curriculum grace: a bite always
        # teaches at full strength, while a curriculum tumble (fall/collision while
        # the harness still holds the body) is discounted.
        "threat_damage": 0.0,
        "energy_gain": 0.0,
        "hydration_gain": 0.0,
        "integrity_gain": 0.0,
        "stress": 0.0,
    }
    for ev in events:
        et = str(ev.get("type", "")).lower()
        try:
            intensity = float(ev.get("intensity", 0.0))
        except (TypeError, ValueError):
            intensity = 0.0
        if intensity < threshold:
            continue
        if et == "collision":
            out["integrity_damage"] += intensity * cds
        elif et == "fall":
            out["integrity_damage"] += intensity * fds
        elif et in ("damage", "environment_damage", "combat_hit"):
            d = intensity * gds
            out["integrity_damage"] += d
            out["threat_damage"] += d
        elif et == "region_change":
            out["integrity_damage"] += intensity * 1.2
        elif et == "threat_near":
            out["stress"] += intensity
        elif et in ("food", "eat", "nourish"):
            out["energy_gain"] += intensity * fc
        elif et in ("water", "drink"):
            out["hydration_gain"] += intensity * wc
        elif et in ("medical", "medical_kit", "medkit", "heal", "care"):
            out["integrity_gain"] += intensity * mc
    out["integrity_damage"] = min(out["integrity_damage"], max_integrity_damage_per_obs())
    # Never let the exempt threat portion exceed the (capped) total.
    out["threat_damage"] = min(out["threat_damage"], out["integrity_damage"])
    return out


def passive_metabolism(
    res: Homeostasis,
    dt_s: float,
    stress: float,
    *,
    hydration_empty_s: float,
    energy_empty_s: float,
    integrity_heal_full_s: float,
    heal_min_reserve: float,
    stress_gain: float,
    compression: float = 1.0,
) -> None:
    """Advance the metabolic clock by ``dt_s`` real seconds (mutates ``res``).

    Hydration and energy drain proportionally to elapsed time over their
    survival horizons, accelerated by stress. Integrity heals toward full only
    when both hydration and energy are above ``heal_min_reserve`` (you cannot
    recover while starving or dehydrated), and healing slows under stress.
    ``compression`` fast-forwards the whole clock, so it accelerates draining
    and healing alike.
    """
    if dt_s <= 0.0:
        return
    stress = max(0.0, float(stress))
    span = res.max_value - res.min_value
    comp = max(0.0, compression)
    drain_mult = comp * (1.0 + stress_gain * stress)
    hydration_delta = -span * dt_s / hydration_empty_s * drain_mult
    energy_delta = -span * dt_s / energy_empty_s * drain_mult
    integrity_delta = 0.0
    if res.hydration > heal_min_reserve and res.energy > heal_min_reserve:
        # Healing fast-forwards with the metabolic clock just like draining, so
        # time compression speeds recovery too (stress still slows it).
        integrity_delta = (
            span * dt_s / integrity_heal_full_s * comp / (1.0 + stress_gain * stress)
        )
    res.apply_reservoir_deltas(
        hydration=hydration_delta, energy=energy_delta, integrity=integrity_delta
    )


def viability_delta_to_signals(delta: float) -> tuple[float, float]:
    """Map viability change to (pain, pleasure) non-negative scalars."""
    if delta < 0:
        return abs(delta), 0.0
    if delta > 0:
        return 0.0, delta
    return 0.0, 0.0


def motor_energy_cost(
    motor_command,
    *,
    scale: float,
    dt: float,
    compression: float = 1.0,
    mode: str = "l1",
) -> tuple[float, float]:
    """Energy spent on a motor-command SIGNAL. Returns (cost, activation).

    ``activation`` = Sum_j |u_j| (l1, per-joint) or Sum_j u_j^2 (l2). ``cost`` =
    activation x scale x dt x compression, so it is proportional to how hard the
    joints are driven AND to real elapsed time (dt-scaled -> respects the
    metabolic clock/compression and is independent of cycle rate). Pure and
    side-effect free for testing."""
    if not motor_command:
        return 0.0, 0.0
    try:
        if str(mode).lower() == "l2":
            activation = float(sum(float(x) * float(x) for x in motor_command))
        else:
            activation = float(sum(abs(float(x)) for x in motor_command))
    except (TypeError, ValueError):
        return 0.0, 0.0
    cost = max(
        0.0,
        activation * max(0.0, float(scale)) * max(0.0, float(dt)) * max(0.0, float(compression)),
    )
    return cost, activation


def ema_affect(prev: float, new: float, *, retain: float = 0.98) -> float:
    """Bounded EMA for a [0,1] affect scalar: retain*prev + (1-retain)*new, clamped.

    The phasic pain/pleasure scalars are felt intensities in [0,1], not running
    sums. Weighting the new term by ``1 - retain`` (a true EMA) and clamping keeps
    the value bounded; a leaky integrator like ``retain*prev + new`` instead
    settles at ``new / (1 - retain)`` and, when ``new`` itself scales with the
    fed-back scalar, diverges geometrically.
    """
    blended = retain * float(prev) + (1.0 - retain) * float(new)
    return float(min(1.0, max(0.0, blended)))


def interoceptive_drive_pain(
    res: "Homeostasis", *, comfort: float = 100.0, gain: float = 1.0, exponent: float = 2.0
) -> float:
    """Continuous, convex, compounding tonic deprivation pain.

    Drive theory: an unsatisfied homeostatic need is aversive in proportion to the
    *deprivation level*, not the per-tick change (which is near-zero). The response
    has **no dead zone** - any dip below ``comfort`` (full by default) registers a
    little pain - and is **convex** (``exponent`` > 1), so slight hunger is a faint,
    resistible nag while starvation is overwhelming. Per-reservoir deficits are
    **summed** so simultaneous needs **compound** (thirst AND hunger hurt more than
    either alone; the fastest drainer still leads because its deficit is largest).
    The setpoint and the aversiveness of deprivation are innate substrate; nothing
    here references an external satisfier. Returns a bounded pain scalar in [0, 1].
    """
    if comfort <= 0.0 or gain <= 0.0:
        return 0.0
    e = max(1.0, float(exponent))
    deficits = [
        max(0.0, (comfort - res.hydration) / comfort),
        max(0.0, (comfort - res.energy) / comfort),
        max(0.0, (comfort - res.integrity) / comfort),
    ]
    pressure = sum(d**e for d in deficits)
    return float(min(1.0, gain * pressure))


def drive_reduction_reward(
    prev_pressure: float, cur_pressure: float, *, gain: float = 1.0
) -> float:
    """Phasic homeostatic relief: reward = the per-cycle *reduction* in drive.

    The positive complement to :func:`interoceptive_drive_pain`. In homeostatic
    reinforcement learning (Keramati & Gutkin) the rewarding event is the
    *decrease* of drive - the agent moving its reservoirs back toward their
    innate setpoints. Only drops are rewarded (a rising drive is already felt as
    the tonic deprivation pain), and the result is bounded to ``[0, 1]`` so it
    composes with the other affect scalars. It references no external satisfier
    and no clock - only the agent's own drive pressure across two cycles - so it
    keeps the motivation wholly intrinsic.
    """
    drop = max(0.0, float(prev_pressure) - float(cur_pressure))
    return float(min(1.0, max(0.0, gain) * drop))


def apply_pain_pleasure_to_B(
    emotion_vec: np.ndarray, pain: float, pleasure: float, decay: float = 0.92
) -> np.ndarray:
    """Inject pain/pleasure into early channels of B with decay."""
    if emotion_vec.shape[0] < 2:
        return emotion_vec
    out = emotion_vec.astype(np.float32, copy=True)
    out[0] = out[0] * decay + float(np.clip(pain / 10.0, 0.0, 1.0))
    out[1] = out[1] * decay + float(np.clip(pleasure / 10.0, 0.0, 1.0))
    return out


def stub_prediction_error_penalty(perceptual_ticks: int, cycle_index: int) -> float:
    """Placeholder PE coupling for the non-neural numpy pipeline (Phase 1 wiring).

    A cycle-counter oscillation that gives the stub cycle non-zero affect dynamics
    without a trained network. The neural pipeline no longer relies on it: there
    the genuine surprise is the predictive-coding loss, and this term is blended in
    at ``config.pe_stub_weight()`` (default 0.0, i.e. removed).
    """
    # Mild oscillation so logs show non-zero dynamics without training
    phase = float((perceptual_ticks + cycle_index) % 17)
    return -0.02 * (phase / 17.0)


def damage_from_events(events: list[dict], threshold: float) -> float:
    """Sum harm / travel-stress from collision-like (and aligned game) events."""
    total = 0.0
    for ev in events:
        et = str(ev.get("type", "")).lower()
        try:
            intensity = float(ev.get("intensity", 0.0))
        except (TypeError, ValueError):
            intensity = 0.0
        if intensity < threshold:
            continue
        if et == "collision":
            total += intensity * 8.0
        elif et in ("damage", "environment_damage", "fall", "combat_hit"):
            total += intensity * 6.0
        elif et == "threat_near":
            # Anticipatory stress: a predator looming costs less than being hit.
            total += intensity * 0.5
        elif et == "region_change":
            total += intensity * 1.2
    return total


def nourishment_from_events(events: list[dict], threshold: float) -> float:
    """Sum viability credit from consumption-like events (food → pleasure)."""
    total = 0.0
    for ev in events:
        et = str(ev.get("type", "")).lower()
        try:
            intensity = float(ev.get("intensity", 0.0))
        except (TypeError, ValueError):
            intensity = 0.0
        if intensity < threshold:
            continue
        if et in ("food", "eat", "nourish"):
            total += intensity * 6.0
    return total


def reward_success_stub(cycle_index: int) -> float:
    """Periodic placeholder reward for the non-neural numpy pipeline (Phase 1).

    A tiny pulse every 50 cycles so the stub cycle shows reward dynamics without a
    trained network. The neural pipeline replaces this with the intrinsic
    :func:`drive_reduction_reward` (homeostatic relief) whenever
    ``config.drive_reward_enabled()`` is set; it is retained only for the legacy /
    disabled path and the byte-identical test baseline.
    """
    return 0.01 if cycle_index % 50 == 0 else 0.0
