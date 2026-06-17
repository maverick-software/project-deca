"""Regression/parity: with every plasticity flag off the stack is the baseline.

These guard the "parity-by-default" invariant — the entire existing suite must
keep passing because a stack built with no A/B/C flags is structurally and
numerically identical to the dense, fixed-topology network it replaces.
"""

import pytest


def _cfg():
    from decadic.nn.config import neural_config_from_env

    return neural_config_from_env("tiny")


def test_default_stack_uses_plain_sequential():
    pytest.importorskip("torch")
    import torch.nn as nn

    from decadic.nn.neural_stack import NeuralCognitiveStack

    stack = NeuralCognitiveStack(_cfg())
    assert stack.has_plastic is False
    for name in ("stage1", "stage3", "risk_mlp", "motor"):
        assert isinstance(getattr(stack, name), nn.Sequential)
    assert stack.plastic_blocks() == []
    assert stack.awake_neurons() == 0
    assert stack.plastic_arch_meta() == {}


def test_flags_off_is_identical_to_no_flags():
    pytest.importorskip("torch")
    import torch

    from decadic.nn.neural_stack import NeuralCognitiveStack
    from decadic.nn.plastic import PlasticityFlags

    cfg = _cfg()
    torch.manual_seed(1234)
    a = NeuralCognitiveStack(cfg)
    torch.manual_seed(1234)
    b = NeuralCognitiveStack(cfg, PlasticityFlags())  # all-off flags

    sda, sdb = a.state_dict(), b.state_dict()
    assert sda.keys() == sdb.keys()
    for k in sda:
        if sda[k].dtype.is_floating_point:
            assert torch.equal(sda[k], sdb[k]), k

    a.eval()
    b.eval()
    from decadic.nn.frozen_encoders import CLIP_POOL_DIM, WHISPER_POOL_DIM

    fused_in = CLIP_POOL_DIM + WHISPER_POOL_DIM + cfg.proprio_emb
    torch.manual_seed(0)
    z0 = a.ingress(torch.randn(1, fused_in))
    epi = torch.randn(1, 4)
    oa = a(z0, epi)
    ob = b(z0, epi)
    assert torch.allclose(oa["motor_u"], ob["motor_u"], atol=1e-6)
    assert torch.allclose(oa["z5"], ob["z5"], atol=1e-6)


def test_plasticity_controllers_are_noops_when_off():
    pytest.importorskip("torch")

    from decadic.nn.neural_stack import NeuralCognitiveStack

    stack = NeuralCognitiveStack(_cfg())
    # All controller helpers must be safe no-ops on a non-plastic stack.
    stack.hebbian_update_all(0.5, 0.1)
    stack.enforce_masks_all()
    assert stack.rewire_all(0.1) == 0
    assert stack.grow_step(8, 256) == []
    assert stack.set_awake_ceiling_all(64) == []
    assert stack.allocated_neurons() == 0
    assert stack.active_connections() == 0
    assert stack.connection_density() == 1.0
    assert stack.plastic_alpha_mean() == 0.0
