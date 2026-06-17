"""C — lifetime neuron growth (dormant-neuron wake / sleep, function preserving)."""

import pytest


def _block(**kw):
    from decadic.nn.plastic import PlasticSparseGrowableMLP

    defaults = dict(
        in_features=12, out_features=6, hidden_active=8, hidden_ceiling=32, growth=True
    )
    defaults.update(kw)
    return PlasticSparseGrowableMLP(**defaults)


def test_awake_starts_at_preset_width():
    pytest.importorskip("torch")

    blk = _block()
    assert blk.awake_count() == 8
    assert blk.hidden_ceiling == 32


def test_growth_wakes_up_to_ceiling_never_beyond():
    pytest.importorskip("torch")

    blk = _block()
    blk.grow(10)
    assert blk.awake_count() == 18
    blk.grow(1000)  # cannot exceed the allocation ceiling
    assert blk.awake_count() == 32
    assert blk.grow(5) == []  # nothing left to wake


def test_set_ceiling_sleeps_neurons():
    pytest.importorskip("torch")

    blk = _block()
    blk.grow(20)  # -> 28 awake
    assert blk.awake_count() == 28
    blk.set_awake_ceiling(10)
    assert blk.awake_count() == 10


def test_external_dims_invariant_under_growth():
    pytest.importorskip("torch")
    import torch

    blk = _block()
    x = torch.randn(2, 12)
    assert blk(x).shape == (2, 6)
    blk.grow(16)
    assert blk(x).shape == (2, 6)  # out_features unchanged
    assert blk.l1_weight.shape[1] == 12  # in_features unchanged


def test_newly_woken_neuron_is_function_preserving():
    pytest.importorskip("torch")
    import torch

    blk = _block()
    blk.eval()
    x = torch.randn(4, 12)
    before = blk(x)
    blk.grow(4)  # outgoing weights start at 0 -> output unchanged on wake
    after = blk(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_optimizer_param_count_stable_across_growth():
    pytest.importorskip("torch")
    import torch

    blk = _block()
    n_before = sum(p.numel() for p in blk.parameters())
    blk.grow(16)
    n_after = sum(p.numel() for p in blk.parameters())
    assert n_before == n_after  # pre-allocated; no live resize


def test_checkpoint_preserves_awake_set():
    pytest.importorskip("torch")
    import torch

    blk = _block()
    blk.grow(12)
    awake = blk.awake.clone()
    sd = blk.state_dict()
    other = _block()
    other.load_state_dict(sd)
    assert torch.equal(other.awake, awake)
    assert other.awake_count() == blk.awake_count()


def test_brain_map_neuron_total_rises_with_growth():
    pytest.importorskip("torch")
    import os

    from decadic.nn.brain_map import brain_topology
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack
    from decadic.nn.plastic import PlasticityFlags

    os.environ.pop("DECADIC_MAX_NEURONS", None)
    flags = PlasticityFlags(growth=True, hidden_ceiling=128, max_neurons=128)
    stack = NeuralCognitiveStack(neural_config_from_env("tiny"), flags)
    t0 = brain_topology(stack)
    stack.grow_step(16, 128)
    t1 = brain_topology(stack)
    assert t1["totals"]["neurons"] > t0["totals"]["neurons"]
    assert t1["totals"]["awake_neurons"] > t0["totals"]["awake_neurons"]
