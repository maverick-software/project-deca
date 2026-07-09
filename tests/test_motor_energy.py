"""WS-ENERGY — motor-signal energy cost.

Energy is spent on the motor SIGNAL the brain sends to its joints (Sum_j |u_j|),
integrated over real time (dt-scaled) so it's independent of cycle rate and
respects the metabolic compression clock. These pin the pure cost function; the
runtime wiring (debit + transition association + pain) is exercised on-box.
"""

from __future__ import annotations

import pytest

from decadic.state.viability import motor_energy_cost


def test_rest_zero_command_costs_nothing():
    cost, act = motor_energy_cost([0.0] * 21, scale=1e-3, dt=0.25)
    assert cost == 0.0 and act == 0.0


def test_empty_or_none_command_safe():
    assert motor_energy_cost([], scale=1e-3, dt=1.0) == (0.0, 0.0)
    assert motor_energy_cost(None, scale=1e-3, dt=1.0) == (0.0, 0.0)


def test_l1_is_sum_abs_per_joint():
    cost, act = motor_energy_cost([0.5, -0.3, 0.2], scale=1.0, dt=1.0, mode="l1")
    assert act == pytest.approx(1.0)  # 0.5 + 0.3 + 0.2 -- each joint contributes
    assert cost == pytest.approx(1.0)


def test_l2_is_sum_squares():
    cost, act = motor_energy_cost([0.5, -0.5], scale=1.0, dt=1.0, mode="l2")
    assert act == pytest.approx(0.5)  # 0.25 + 0.25
    assert cost == pytest.approx(0.5)


def test_dt_scaling_is_linear():
    c1, _ = motor_energy_cost([1.0], scale=1.0, dt=0.1)
    c2, _ = motor_energy_cost([1.0], scale=1.0, dt=0.2)
    assert c2 == pytest.approx(2.0 * c1)


def test_compression_accelerates_cost():
    c1, _ = motor_energy_cost([1.0], scale=1.0, dt=1.0, compression=1.0)
    c100, _ = motor_energy_cost([1.0], scale=1.0, dt=1.0, compression=100.0)
    assert c100 == pytest.approx(100.0 * c1)


def test_scale_zero_disables_but_reports_activation():
    cost, act = motor_energy_cost([1.0, 1.0], scale=0.0, dt=1.0)
    assert cost == 0.0 and act == pytest.approx(2.0)


def test_cycle_rate_independence():
    # Same real time (1.0 s) + same activation -> same TOTAL cost whether billed
    # as one dt=1.0 step or four dt=0.25 steps. This is the whole point of
    # dt-scaling: running at a higher cycle rate must not starve the agent faster.
    total_1 = motor_energy_cost([0.8] * 10, scale=1e-3, dt=1.0)[0]
    total_4 = sum(motor_energy_cost([0.8] * 10, scale=1e-3, dt=0.25)[0] for _ in range(4))
    assert total_4 == pytest.approx(total_1)


def test_negative_scale_or_dt_clamped():
    assert motor_energy_cost([1.0], scale=-1.0, dt=1.0)[0] == 0.0
    assert motor_energy_cost([1.0], scale=1.0, dt=-1.0)[0] == 0.0


def test_config_defaults_retire_system_a():
    from decadic import config as C

    assert C.motor_energy_enabled() is True
    assert C.motor_energy_mode() in ("l1", "l2")
    assert C.effort_energy_scale() == 0.0  # System A energy drain retired
    assert C.work_energy_scale() == 0.0
