"""Return-based credit assignment: lambda-returns, episode accumulation, HER, Transition fields."""

from decadic.consolidation.episodes import (
    EpisodeAccumulator,
    achieved_feature,
    build_hindsight_copies,
)
from decadic.consolidation.replay_buffer import ReplayBuffer, Transition
from decadic.consolidation.returns import lambda_returns, lambda_returns_vec


# --- returns math -----------------------------------------------------------


def test_lambda_one_is_full_discounted_return():
    r = [0.0, 0.0, 1.0]
    g = lambda_returns(r, gamma=0.9, lam=1.0)
    assert abs(g[0] - 0.81) < 1e-9
    assert abs(g[1] - 0.9) < 1e-9
    assert abs(g[2] - 1.0) < 1e-9


def test_lambda_less_than_one_smears_distal_reward_extra():
    r = [0.0, 0.0, 1.0]
    g = lambda_returns(r, gamma=0.9, lam=0.5)
    assert abs(g[0] - (0.9 * 0.5) ** 2) < 1e-9


def test_td0_uses_value_bootstrap():
    g = lambda_returns([1.0, 1.0], gamma=0.9, lam=0.0, values=[5.0, 7.0])
    assert abs(g[0] - (1.0 + 0.9 * 7.0)) < 1e-9


def test_vector_returns_match_scalar_for_1d():
    g = lambda_returns([0.0, 0.0, 1.0], gamma=0.9, lam=1.0)
    gv = lambda_returns_vec([[0.0], [0.0], [1.0]], gamma=0.9, lam=1.0)
    assert abs(gv[0][0] - g[0]) < 1e-9


def test_empty_returns():
    assert lambda_returns([], gamma=0.9, lam=1.0) == []
    assert lambda_returns_vec([], gamma=0.9, lam=1.0) == []


# --- Transition fields ------------------------------------------------------


def test_transition_defaults_are_inert():
    t = Transition(z0=None, ep=None, mem=None, prev_state=None, prev_motor=None,
                   proprio_target=None)
    assert t.feat is None and t.reward == 0.0
    assert t.episode_id == -1 and t.step_idx == 0 and t.goal_id == ""
    assert t.ret is None and t.sf_target is None


# --- episode accumulator ----------------------------------------------------


def _mk(feat, reward, salience=1.0):
    t = Transition(z0=None, ep=None, mem=None, prev_state=None, prev_motor=None,
                   proprio_target=None, drive_on=True, salience=salience)
    t.feat = list(feat)
    t.reward = reward
    return t


def test_accumulator_annotates_returns_in_place():
    acc = EpisodeAccumulator(gamma=0.9, lam=1.0)
    acc.on_open("hydration", 0)
    steps = [_mk([0.0, 0.0, 0.0], 0.0), _mk([0.0, 0.0, 0.0], 0.0), _mk([0.2, 0.0, 0.0], 0.2)]
    for s in steps:
        acc.add(s)
    closed = acc.on_close("achieved")
    assert [t.step_idx for t in closed] == [0, 1, 2]
    assert all(t.episode_id == 1 and t.goal_id == "hydration" for t in closed)
    assert abs(closed[0].ret - 0.81 * 0.2) < 1e-9
    assert closed[0].sf_target is not None and abs(closed[0].sf_target[0] - 0.81 * 0.2) < 1e-9
    assert acc.episodes_closed == 1 and acc.last_len == 3


def test_accumulator_skips_featureless_steps():
    acc = EpisodeAccumulator(gamma=0.9, lam=1.0)
    acc.on_open("energy", 0)
    no_feat = Transition(z0=None, ep=None, mem=None, prev_state=None, prev_motor=None,
                         proprio_target=None, drive_on=False)
    acc.add(no_feat)
    acc.add(_mk([0.0, 0.1, 0.0], 0.1))
    closed = acc.on_close("achieved")
    assert len(closed) == 1


def test_reset_drops_open_episode():
    acc = EpisodeAccumulator(gamma=0.9, lam=1.0)
    acc.on_open("hydration", 0)
    acc.add(_mk([0.1, 0.0, 0.0], 0.1))
    acc.reset()
    assert acc.on_close("died") == []


# --- hindsight relabeling ---------------------------------------------------


def test_achieved_feature_sums_phi():
    steps = [_mk([0.0, 0.0, 0.0], 0.0), _mk([0.0, 0.3, 0.0], 0.3)]
    assert achieved_feature(steps) == [0.0, 0.3, 0.0]


def test_hindsight_copies_are_independent_and_bootstrapped():
    steps = [_mk([0.0, 0.0, 0.0], 0.0), _mk([0.0, 0.3, 0.0], 0.3), _mk([0.0, 0.0, 0.0], 0.0)]
    ach = achieved_feature(steps)
    copies = build_hindsight_copies(steps, ach, gamma=0.9, lam=1.0, k=2)
    assert len(copies) == 2 * len(steps)
    assert all(c.goal_id == "hindsight" for c in copies)
    assert copies[0] is not steps[0]
    # Achieved-feature bootstrap lifts the energy-channel target above the plain close.
    plain = lambda_returns_vec([list(t.feat) for t in steps], gamma=0.9, lam=1.0)
    assert copies[0].sf_target[1] > plain[0][1]


def test_hindsight_copies_push_into_buffer():
    buf = ReplayBuffer(64)
    steps = [_mk([0.0, 0.2, 0.0], 0.2, salience=2.0)]
    copies = build_hindsight_copies(steps, achieved_feature(steps), gamma=0.9, lam=1.0, k=1)
    assert all(buf.push(c) for c in copies)
    assert len(buf) == 1
