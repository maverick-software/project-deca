"""WS3B-M0: gate decision log + shadow deliberation tap.

M0.1 -- GateDecisionLog: buffered JSONL, log-and-continue on IO failure,
        off-by-default config.
M0.2 -- stage4_shadow: fresh risk_mlp(z3) computed beside a stage4_override,
        no_grad diagnostics only; the forward's live outputs must be
        bit-identical with the tap on or off.
"""

import json

import pytest

from decadic.cycle.attention_gate import (
    GateDecisionLog,
    gate_log_enabled,
    gate_shadow_rate,
    shadow_sampled,
)

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------- M0.1


def test_gate_log_config_defaults(monkeypatch):
    monkeypatch.delenv("DECADIC_GATE_LOG", raising=False)
    assert gate_log_enabled() is False  # zero new IO on existing runs
    monkeypatch.setenv("DECADIC_GATE_LOG", "1")
    assert gate_log_enabled() is True

    monkeypatch.delenv("DECADIC_GATE_SHADOW_RATE", raising=False)
    assert gate_shadow_rate() == pytest.approx(0.05)
    monkeypatch.setenv("DECADIC_GATE_SHADOW_RATE", "7")  # clamped
    assert gate_shadow_rate() == 1.0
    monkeypatch.setenv("DECADIC_GATE_SHADOW_RATE", "junk")  # tolerated
    assert gate_shadow_rate() == pytest.approx(0.05)


def test_shadow_sampling_deterministic_and_near_rate():
    assert shadow_sampled(1234, 0.0) is False
    assert shadow_sampled(1234, 1.0) is True
    # Deterministic: same cycle, same answer, no RNG state involved.
    assert shadow_sampled(777, 0.05) == shadow_sampled(777, 0.05)
    # Over many cycles the hit rate lands near the configured rate.
    hits = sum(1 for c in range(20_000) if shadow_sampled(c, 0.05))
    assert 0.03 < hits / 20_000 < 0.07


def test_gate_decision_log_buffers_flushes_and_survives(tmp_path):
    path = tmp_path / "gate" / "decisions.jsonl"
    log = GateDecisionLog(path)
    for i in range(GateDecisionLog.FLUSH_EVERY + 5):
        log.log({"cycle": i, "escalate": i % 2})
    # Auto-flush fired at FLUSH_EVERY; the tail is still buffered.
    on_disk = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(on_disk) >= GateDecisionLog.FLUSH_EVERY
    log.close()
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(rows) == GateDecisionLog.FLUSH_EVERY + 5
    assert rows[3] == {"cycle": 3, "escalate": 1}

    # A malformed row is dropped, never raised.
    log2 = GateDecisionLog(tmp_path / "gate2.jsonl")
    log2.log({"bad": object()})
    log2.log({"ok": 1})
    log2.close()
    rows2 = (tmp_path / "gate2.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows2) == 1 and json.loads(rows2[0]) == {"ok": 1}


def test_gate_decision_log_io_failure_disables_quietly(tmp_path):
    # Parent "directory" is a FILE -> mkdir and appends fail -> sink disables
    # itself without raising (the cognitive loop never pays for telemetry).
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    log = GateDecisionLog(blocker / "sub" / "decisions.jsonl")
    for i in range(100):
        log.log({"cycle": i})
    log.flush()
    log.close()  # no exception = pass
    assert log._failed is True


# --------------------------------------------------------------------- M0.2


def _tiny_stack(monkeypatch):
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack

    return NeuralCognitiveStack(neural_config_from_env("tiny")), neural_config_from_env(
        "tiny"
    )


def test_stage4_shadow_is_pure_diagnostics(monkeypatch):
    """Live outputs bit-identical with the shadow tap on vs off; shadow keys
    carry a fresh deliberation that can disagree with the substituted one."""
    stack, cfg = _tiny_stack(monkeypatch)
    stack.eval()
    torch.manual_seed(11)
    z0 = torch.randn(1, cfg.d_model)
    ep = torch.rand(1, 4)
    mem = torch.randn(1, cfg.memory_context_dim)

    with torch.no_grad():
        # Reference deliberation supplies realistic override shapes.
        stack.reset_recurrent_state()
        base = stack(z0, ep, mem)
        override = (base["z4"].detach() * 0.5, base["risk_logit"].detach() * 0.5)

        stack.reset_recurrent_state()
        off = stack(z0, ep, mem, stage4_override=override, stage4_shadow=False)
        stack.reset_recurrent_state()
        on = stack(z0, ep, mem, stage4_override=override, stage4_shadow=True)

    assert "shadow_z4" not in off
    assert "shadow_z4" in on and "shadow_risk_logit" in on
    assert on["shadow_z4"].shape == on["z4"].shape

    # Bit-identical live path: every shared tensor output matches exactly.
    for k, v in off.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, on[k]), f"shadow tap perturbed output {k!r}"

    # The shadow is FRESH deliberation: with a halved override it must
    # disagree with the substituted z4 (that divergence IS the regret signal).
    assert not torch.allclose(on["shadow_z4"], on["z4"])
    # And it matches what un-overridden stage 4 would have produced.
    assert torch.allclose(on["shadow_z4"], base["z4"], atol=1e-5)


def _planted_log(path, n=400, seed=7):
    """Synthetic decision log with PLANTED structure: high novelty <=> high
    skip-regret. The M1.2 acceptance -- the pipeline must recover exactly the
    structure that was planted."""
    import random

    rng = random.Random(seed)
    lines = []
    for c in range(1, n + 1):
        nov = rng.random()
        row = {
            "cycle": c,
            "novelty": round(nov, 6),
            "pe": 0.3,
            "affect": 0.1,
            "priority": 0.0,
            "drive": 0.05,
            "esc_rate": 0.05,
            "latch": 0,
            "precedent_age": c % 20,
            "fast_path": 0,
            "escalate": 0,
            "reason": "skip",
            "score": round(nov * 0.5, 6),
            "threshold_eff": 0.3,
            "pain": 0.0,
            "viability": 90.0,
            "pc_ema": 0.4,
        }
        if c % 2 == 0:  # dense shadow sampling for the test
            row["shadow_kind"] = "skip"
            row["shadow_regret_z4"] = 0.5 if nov > 0.7 else 0.001
            row["shadow_regret_risk"] = row["shadow_regret_z4"]
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _builder():
    import sys
    from pathlib import Path as _P

    root = _P(__file__).resolve().parents[1]
    sp = str(root / "scripts")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    import build_gate_dataset

    return build_gate_dataset


def test_build_gate_dataset_recovers_planted_structure(tmp_path):
    np = pytest.importorskip("numpy")
    bgd = _builder()
    log = tmp_path / "gate_decisions_test.jsonl"
    _planted_log(log)

    arrays, manifest = bgd.build([log], horizon=20, alpha=100.0, cost=0.05, beta=0.5)
    assert manifest["totals"]["rows"] == 400
    lab = ~np.isnan(arrays["y"])
    assert manifest["totals"]["labeled"] == int(lab.sum()) > 100

    nov = arrays["X"][:, 0]
    y = arrays["y"]
    hi = lab & (nov > 0.7)
    lo = lab & (nov <= 0.7)
    # Planted: high-novelty rows carry regret 0.5 -> labels ~1; others ~0.
    assert float(y[hi].mean()) > 0.95
    assert float(y[lo].mean()) < 0.05

    # Hyperparameters actually steer labels: an impossible cost flattens all
    # labels toward zero (nothing is ever worth escalating).
    arrays2, _ = bgd.build([log], horizon=20, alpha=40.0, cost=1.0, beta=0.0)
    lab2 = ~np.isnan(arrays2["y"])
    assert float(arrays2["y"][lab2].mean()) < 0.01

    # Deterministic rebuild: identical inputs -> identical arrays.
    arrays3, _ = bgd.build([log], horizon=20, alpha=100.0, cost=0.05, beta=0.5)
    assert np.array_equal(arrays["y"], arrays3["y"], equal_nan=True)
    assert np.array_equal(arrays["X"], arrays3["X"])


def test_build_gate_dataset_outcome_sharpening(tmp_path):
    """A pain spike inside the horizon lifts a low-divergence label."""
    np = pytest.importorskip("numpy")
    bgd = _builder()
    rows = []
    for c in range(1, 61):
        rows.append(
            {
                "cycle": c,
                "novelty": 0.1,
                "pe": 0.1,
                "affect": 0.0,
                "priority": 0.0,
                "drive": 0.0,
                "esc_rate": 0.05,
                "latch": 0,
                "precedent_age": 3,
                "fast_path": 0,
                "escalate": 0,
                "reason": "skip",
                "score": 0.1,
                "threshold_eff": 0.3,
                # Pain spikes at cycles 25-30; the decision at cycle 20 with a
                # horizon of 15 sees it, the one at cycle 5 does not.
                "pain": 0.8 if 25 <= c <= 30 else 0.0,
                "viability": 90.0,
                "pc_ema": 0.4,
            }
        )
    for c in (5, 20):
        rows[c - 1]["shadow_kind"] = "skip"
        rows[c - 1]["shadow_regret_z4"] = 0.001  # divergence alone says "skip fine"
        rows[c - 1]["shadow_regret_risk"] = 0.001
    log = tmp_path / "gate_decisions_pain.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    arrays, _ = bgd.build([log], horizon=15, alpha=40.0, cost=0.05, beta=1.0)
    y_by_cycle = {int(c): float(v) for c, v in zip(arrays["cycle"], arrays["y"]) if v == v}
    assert y_by_cycle[5] < 0.2  # quiet horizon: divergence-only label, low
    assert y_by_cycle[20] > 0.7  # pain followed: attention was warranted


# --------------------------------------------------------------------- M2


def test_gate_net_shapes_determinism_and_roundtrip(tmp_path):
    from decadic.nn.gate_net import GateNet, featurize, normalize

    torch.manual_seed(3)
    net = GateNet(hidden=16)
    net.eval()
    x = torch.rand(5, 8)
    with torch.no_grad():
        a = net(x)
        b = net(x)
    assert a.shape == (5,)
    assert torch.equal(a, b)  # deterministic

    # featurize/normalize: bounded, age log-compressed, latch capped.
    row = {
        "novelty": 0.4,
        "pe": 0.2,
        "affect": 0.1,
        "priority": 1.0,
        "drive": 0.0,
        "esc_rate": 0.05,
        "latch": 99,
        "precedent_age": 3000,
    }
    f = featurize(row)
    assert f.shape == (8,)
    assert 0.0 <= f.max() <= 1.05
    assert f[6] == 1.0  # latch capped
    import numpy as np

    assert np.allclose(normalize(np.asarray([list(row.values())], dtype=float))[0], f)

    # Save/load round-trip: identical outputs; version guard rejects tampering.
    p = tmp_path / "gate_net.pt"
    net.save(p)
    net2 = GateNet.load(p)
    with torch.no_grad():
        assert torch.equal(net(x), net2(x))
    payload = net.to_payload()
    payload["version"] = 999
    with pytest.raises(ValueError):
        GateNet.from_payload(payload)


def test_gate_net_learns_planted_structure(tmp_path):
    """M2.2 acceptance: on the planted dataset (high novelty <=> high regret)
    the net must separate the classes to AUC > 0.95."""
    np = pytest.importorskip("numpy")
    bgd = _builder()
    import sys as _sys
    from pathlib import Path as _P

    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    import train_gate as tg

    from decadic.nn.gate_net import GateNet, normalize

    log = tmp_path / "gate_decisions_train.jsonl"
    _planted_log(log, n=600, seed=13)
    arrays, _ = bgd.build([log], horizon=20, alpha=100.0, cost=0.05, beta=0.0)
    lab = ~np.isnan(arrays["y"])
    X = normalize(arrays["X"][lab])
    y = arrays["y"][lab]

    torch.manual_seed(5)
    net = GateNet(hidden=16)
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    pos = float((y > 0.5).sum())
    neg = float((y <= 0.5).sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / max(1.0, pos)])
    )
    Xt = torch.as_tensor(X)
    yt = torch.as_tensor(y.astype("float32"))
    for _ in range(200):
        opt.zero_grad()
        loss_fn(net(Xt), yt).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(Xt)).numpy()
    auc = tg.auc_score(y, p)
    assert auc is not None and auc > 0.95


def test_stage4_shadow_requires_override(monkeypatch):
    """Shadow on an escalated (no-override) cycle is a no-op: fresh stage 4
    already ran; the counterfactual there is computed by the caller."""
    stack, cfg = _tiny_stack(monkeypatch)
    stack.eval()
    torch.manual_seed(12)
    with torch.no_grad():
        stack.reset_recurrent_state()
        out = stack(
            torch.randn(1, cfg.d_model),
            torch.rand(1, 4),
            torch.randn(1, cfg.memory_context_dim),
            stage4_shadow=True,
        )
    assert "shadow_z4" not in out
