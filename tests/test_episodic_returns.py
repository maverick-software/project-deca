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


# --- WS-FORAGE M1: (1-gamma) normalization ---------------------------------


def test_normalize_off_is_byte_identical():
    r = [0.1, 0.0, 1.0, 0.2]
    assert lambda_returns(r, gamma=0.97, lam=0.9) == lambda_returns(
        r, gamma=0.97, lam=0.9, normalize=False
    )
    f = [[0.1, 0.0], [0.0, 1.0]]
    assert lambda_returns_vec(f, gamma=0.97, lam=0.9) == lambda_returns_vec(
        f, gamma=0.97, lam=0.9, normalize=False
    )


def test_normalize_scales_by_one_minus_gamma():
    r = [0.0, 0.0, 1.0]
    raw = lambda_returns(r, gamma=0.9, lam=1.0)
    norm = lambda_returns(r, gamma=0.9, lam=1.0, normalize=True)
    assert all(abs(nx - rx * (1.0 - 0.9)) < 1e-12 for nx, rx in zip(norm, raw))


def test_normalize_bounds_magnitude_and_is_invariant_over_full_episode():
    # A steady per-step reward of 1. The raw discounted SUM grows ~1/(1-gamma)
    # as gamma->1 (unbounded target growth -> what breaks the zero-init SF head
    # at long horizons). The (1-gamma) normalization keeps the target BOUNDED in
    # [0, r_max] for any gamma, and -- over an episode long relative to the
    # horizon -- also ~invariant to gamma. That boundedness is the M1 safety
    # property that makes raising the horizon safe.
    n = 4000  # full-length episode (>= the horizon at gamma=0.999)
    r = [1.0] * n
    def g0(gamma, normalize):
        return lambda_returns(r, gamma=gamma, lam=1.0, normalize=normalize)[0]
    raw_097, raw_0999 = g0(0.97, False), g0(0.999, False)
    nrm_097, nrm_0999 = g0(0.97, True), g0(0.999, True)
    assert raw_0999 > 5.0 * raw_097  # raw target blows up with the horizon
    assert 0.0 < nrm_097 <= 1.0 and 0.0 < nrm_0999 <= 1.0  # normalized stays bounded
    assert abs(nrm_0999 - nrm_097) < 0.1  # and ~invariant over a full-length episode


def test_episode_accumulator_normalize_flag_threads_through():
    acc_off = EpisodeAccumulator(gamma=0.9, lam=1.0, normalize=False)
    acc_on = EpisodeAccumulator(gamma=0.9, lam=1.0, normalize=True)
    assert acc_off.normalize is False and acc_on.normalize is True


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
