"""A — neuromodulated Hebbian plasticity on the growable block."""

import pytest


def _block(**kw):
    from decadic.nn.plastic import PlasticSparseGrowableMLP

    defaults = dict(in_features=8, out_features=5, hidden_active=6)
    defaults.update(kw)
    return PlasticSparseGrowableMLP(**defaults)


def test_alpha_zero_is_identity_to_base_mlp():
    pytest.importorskip("torch")
    import torch
    import torch.nn.functional as F

    blk = _block(plastic=True, plastic_alpha=0.0)
    blk.eval()
    x = torch.randn(3, 8)
    ref_h = F.gelu(F.linear(x, blk.l1_weight, blk.l1_bias))
    ref_y = F.linear(ref_h, blk.l2_weight, blk.l2_bias)
    assert torch.allclose(blk(x), ref_y, atol=1e-6)


def test_hebbian_update_changes_effective_weights():
    pytest.importorskip("torch")
    import torch

    blk = _block(plastic=True, plastic_alpha=0.2)
    x = torch.randn(4, 8)
    blk(x)  # populate the activation cache
    before = blk.hebb1.clone()
    blk.hebbian_update(modulation=1.0, eta=0.5)
    assert not torch.equal(before, blk.hebb1)
    # Effective weight now differs from the raw parameter (alpha * hebb != 0).
    assert not torch.allclose(blk._eff_w1(), blk.l1_weight, atol=1e-7)


def test_effective_alpha_gate_is_separate_from_configured_alpha():
    pytest.importorskip("torch")

    blk = _block(plastic=True, plastic_alpha=0.2)
    assert abs(blk.configured_alpha_value() - 0.2) < 1e-6
    blk.set_effective_alpha(0.01)
    assert abs(blk.configured_alpha_value() - 0.2) < 1e-6
    assert abs(blk.effective_alpha_value() - 0.01) < 1e-6


def test_overlay_cap_bounds_effective_hebbian_weight_delta(monkeypatch):
    pytest.importorskip("torch")
    import torch

    monkeypatch.setenv("DECADIC_PLASTICITY_OVERLAY_MAX_FRAC", "0.05")
    blk = _block(plastic=True, plastic_alpha=0.2)
    with torch.no_grad():
        blk.hebb1.fill_(5.0)
    overlay = blk._eff_w1() - (blk.l1_weight * blk.mask1)
    cap = 0.05 * (blk.l1_weight.detach().abs() + 1e-6)
    assert torch.all(overlay.detach().abs() <= cap + 1e-7)


def test_modulation_sign_flips_potentiation():
    pytest.importorskip("torch")
    import torch

    pos = _block(plastic=True, plastic_alpha=0.2)
    x = torch.randn(4, 8)
    pos(x)
    neg_state = {k: v.clone() for k, v in pos.state_dict().items()}

    pos.hebbian_update(modulation=1.0, eta=0.5)
    hebb_pos = pos.hebb1.clone()

    neg = _block(plastic=True, plastic_alpha=0.2)
    neg.load_state_dict(neg_state)
    neg(x)
    neg.hebbian_update(modulation=-1.0, eta=0.5)
    # Opposite neuromodulation produces the opposite-sign trace update.
    assert torch.allclose(hebb_pos, -neg.hebb1, atol=1e-6)


def test_trace_decays_with_zero_modulation():
    pytest.importorskip("torch")
    import torch

    blk = _block(plastic=True, plastic_alpha=0.2)
    x = torch.randn(4, 8)
    blk(x)
    blk.hebbian_update(modulation=1.0, eta=0.5)
    mag0 = blk.hebb1.abs().sum().item()
    for _ in range(5):
        blk(x)
        blk.hebbian_update(modulation=0.0, eta=0.5)
    assert blk.hebb1.abs().sum().item() < mag0


def test_backprop_flows_through_weights_but_not_guardian_alpha_ceiling():
    pytest.importorskip("torch")
    import torch

    blk = _block(plastic=True, plastic_alpha=0.2)
    x = torch.randn(4, 8)
    blk(x)
    blk.hebbian_update(modulation=1.0, eta=0.5)
    out = blk(x)
    out.pow(2).mean().backward()
    assert blk.alpha.grad is None
    assert blk.l1_weight.grad is not None and torch.isfinite(blk.l1_weight.grad).all()


def test_save_load_roundtrips_trace():
    pytest.importorskip("torch")
    import torch

    blk = _block(plastic=True, plastic_alpha=0.2)
    x = torch.randn(4, 8)
    blk(x)
    blk.hebbian_update(modulation=1.0, eta=0.5)
    sd = blk.state_dict()
    other = _block(plastic=True, plastic_alpha=0.2)
    other.load_state_dict(sd)
    assert torch.equal(other.hebb1, blk.hebb1)
    assert torch.equal(other.hebb2, blk.hebb2)
