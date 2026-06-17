"""B — dynamic sparse training (masks, pruning, gradient-driven growth)."""

import pytest


def _block(**kw):
    from decadic.nn.plastic import PlasticSparseGrowableMLP

    defaults = dict(in_features=16, out_features=8, hidden_active=12)
    defaults.update(kw)
    return PlasticSparseGrowableMLP(**defaults)


def test_density_one_is_parity():
    pytest.importorskip("torch")
    import torch
    import torch.nn.functional as F

    blk = _block(sparse=True, density=1.0)
    assert torch.equal(blk.mask1, torch.ones_like(blk.mask1))
    assert torch.equal(blk.mask2, torch.ones_like(blk.mask2))
    blk.eval()
    x = torch.randn(2, 16)
    ref_h = F.gelu(F.linear(x, blk.l1_weight, blk.l1_bias))
    ref_y = F.linear(ref_h, blk.l2_weight, blk.l2_bias)
    assert torch.allclose(blk(x), ref_y, atol=1e-6)


def test_mask_enforces_target_density():
    pytest.importorskip("torch")

    blk = _block(sparse=True, density=0.4)
    frac1 = blk.mask1.mean().item()
    frac2 = blk.mask2.mean().item()
    # Random seeding → within a tolerance band of the target.
    assert 0.25 < frac1 < 0.55
    assert 0.25 < frac2 < 0.55


def test_pruned_weights_stay_zero_across_enforce():
    pytest.importorskip("torch")
    import torch

    blk = _block(sparse=True, density=0.4)
    with torch.no_grad():
        # Try to write into a pruned slot, then enforce.
        blk.l1_weight[blk.mask1 == 0] = 5.0
    blk.enforce_masks()
    assert torch.all(blk.l1_weight[blk.mask1 == 0] == 0.0)
    # And it stays zero across repeated enforcement.
    for _ in range(3):
        blk.enforce_masks()
    assert torch.all(blk.l1_weight[blk.mask1 == 0] == 0.0)


def test_rewire_keeps_connection_count_constant():
    pytest.importorskip("torch")
    import torch

    blk = _block(sparse=True, density=0.5)
    x = torch.randn(4, 16)
    y = blk(x)
    y.pow(2).mean().backward()  # populate .grad for growth scoring
    active_before = int(blk.mask1.sum().item() + blk.mask2.sum().item())
    blk.rewire(0.2)
    active_after = int(blk.mask1.sum().item() + blk.mask2.sum().item())
    assert active_after == active_before


def test_rewire_grows_highest_gradient_inactive_edge():
    pytest.importorskip("torch")
    import torch

    blk = _block(sparse=True, density=0.5)
    # Force a single inactive edge to carry a huge gradient; it should be grown.
    with torch.no_grad():
        blk.mask1.fill_(1.0)
        blk.mask1[0, 0] = 0.0  # exactly one inactive edge on layer 1
        blk.l1_weight[0, 0] = 0.0
    blk.l1_weight.grad = torch.zeros_like(blk.l1_weight)
    blk.l1_weight.grad[0, 0] = 100.0
    # Also give layer 2 a grad so its rewire is well-defined.
    blk.l2_weight.grad = torch.randn_like(blk.l2_weight)
    blk._rewire_matrix(blk.l1_weight, blk.mask1, 0.2)
    assert blk.mask1[0, 0] == 1.0  # the high-gradient inactive edge woke up
