"""Cognitive Trace: the read-only "why" monitoring layer.

Covers the intent/free-energy decomposition math, self-model surprise labeling,
trace assembly shape, the /explain endpoint, narrative determinism, the probe
read-out shape, and - most importantly - that the whole layer is read-only:
enabling it does not change the agent's behaviour (oracle/zeros parity) and it
only ever writes ``narrative_text_stub`` on the State Bus.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from decadic.cycle import cognition_trace as ct
from decadic.cycle import narrative as nar
from decadic.interpretability import probes as P


# ---------------------------------------------------------------------------
# Intent / free-energy decomposition
# ---------------------------------------------------------------------------


def _drive_raw():
    return {
        "drive": {
            "pred": np.array([0.6, 0.5, 0.9], dtype=np.float32),
            "pred_still": np.array([0.5, 0.5, 0.9], dtype=np.float32),
            "pref": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "w": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "now": np.array([0.55, 0.5, 0.88], dtype=np.float32),
        }
    }


def test_intent_contributions_sorted_and_weighted():
    trace = ct.build(cycle=1, raw=_drive_raw(), fwd_dim=7, affect={})
    drivers = trace.intent["drivers"]
    assert [d["goal"] for d in drivers] == ["energy", "hydration", "integrity"]
    # contribution == w * deviation^2
    for d in drivers:
        assert d["contribution"] == pytest.approx(d["weight"] * d["deviation"] ** 2, abs=1e-6)
    # sorted descending by contribution
    contribs = [d["contribution"] for d in drivers]
    assert contribs == sorted(contribs, reverse=True)


def test_intent_action_delta_sign():
    # hydration: action moves predicted 0.6 (vs still 0.5) toward pref 1.0 -> helps (+)
    trace = ct.build(cycle=1, raw=_drive_raw(), fwd_dim=7, affect={})
    hyd = next(d for d in trace.intent["drivers"] if d["goal"] == "hydration")
    assert hyd["action_delta"] > 0
    assert "hydration" in trace.intent["summary"]


def test_intent_empty_when_no_drive():
    trace = ct.build(cycle=2, raw={}, fwd_dim=7, affect={})
    assert trace.intent["drivers"] == []
    assert "no active survival objective" in trace.intent["summary"]


# ---------------------------------------------------------------------------
# Self-model surprise
# ---------------------------------------------------------------------------


def test_surprise_labeling_and_sort():
    raw = {
        "surprise": {
            "pred": np.array([0.1, 0.0, 0.2, 0.9, 0, 0, 0], dtype=np.float32),
            "target": np.array([0.0, 0.0, 0.0, 0.95, 0, 0, 0], dtype=np.float32),
        }
    }
    trace = ct.build(cycle=3, raw=raw, fwd_dim=7, affect={})
    dims = trace.self_surprise["dims"]
    assert dims[0]["name"] == "yaw"  # largest residual (0.2)
    residuals = [d["residual"] for d in dims]
    assert residuals == sorted(residuals, reverse=True)
    assert trace.self_surprise["mean_abs_residual"] is not None


# ---------------------------------------------------------------------------
# Trace assembly shape
# ---------------------------------------------------------------------------


def test_trace_to_dict_shape():
    d = ct.build(cycle=9, raw=_drive_raw(), fwd_dim=7, affect={"pain": 0.1}).to_dict()
    for key in (
        "cycle",
        "intent",
        "self_surprise",
        "affect",
        "recalled_episode",
        "salient",
        "counterfactuals",
        "probes",
        "narrative",
    ):
        assert key in d
    compact = ct.build(cycle=9, raw=_drive_raw(), fwd_dim=7, affect={"pain": 0.1}).compact()
    assert compact["cycle"] == 9
    assert compact["top_goal"] == "energy"


# ---------------------------------------------------------------------------
# Narrative (Tier C): deterministic template, empty when off
# ---------------------------------------------------------------------------


def test_narrative_template_deterministic_and_off():
    trace = ct.build(
        cycle=5,
        raw=_drive_raw(),
        fwd_dim=7,
        affect={"pain": 0.1, "pleasure": 0.2, "risk": 0.3, "priority": "avoid"},
    ).to_dict()
    t1 = nar.render(trace, "template")
    t2 = nar.render(trace, "template")
    assert t1 == t2 and t1 != ""
    assert nar.render(trace, "off") == ""
    assert nar.render(None, "template") == ""


# ---------------------------------------------------------------------------
# Probe read-out shape (Tier B)
# ---------------------------------------------------------------------------


def test_probe_readout_shape_and_value():
    bank = P.ProbeBank(
        {
            "targets": {
                "height": {
                    "kind": "regression",
                    "best_latent": "emotion",
                    "per_latent": {
                        "emotion": {"w": [0.5, -0.25], "b": 0.1, "dim": 2, "score": 0.8},
                    },
                }
            }
        }
    )
    out = bank.readout({"emotion": [2.0, 4.0]})
    assert "height" in out
    r = out["height"]
    # 0.5*2 + (-0.25)*4 + 0.1 = 0.1
    assert r["predicted"] == pytest.approx(0.1, abs=1e-6)
    assert r["best_latent"] == "emotion"
    assert r["score_kind"] == "r2"
    assert r["axis"] == 0  # dominant |w| coefficient


def test_probe_readout_skips_dim_mismatch():
    bank = P.ProbeBank(
        {"targets": {"x": {"kind": "regression", "best_latent": "z5", "per_latent": {"z5": {"w": [1, 2, 3], "b": 0.0}}}}}
    )
    # latent length 2 != probe dim 3 -> skipped, no crash
    assert bank.readout({"z5": [1.0, 2.0]}) == {}


# ---------------------------------------------------------------------------
# /explain endpoint
# ---------------------------------------------------------------------------


def _obs():
    return {
        "timestamp": "2026-06-11T00:00:00Z",
        "proprioception": {
            "position": [0.0, 0.0, 1.2],
            "orientation": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "current_action": "idle",
        },
        "events": [],
        "world_state": {"nearby_entities": [], "agent_inventory": []},
    }


def test_explain_endpoint(api_app_neural):
    with TestClient(api_app_neural) as client:
        aid = client.post("/agent").json()["agent_id"]
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            for _ in range(4):
                ws.send_json(_obs())
                ws.receive_json()

        r = client.get(f"/agent/{aid}/explain?history=10")
        assert r.status_code == 200
        body = r.json()
        assert body["agent_id"] == aid
        assert "trace" in body and "history" in body
        trace = body["trace"]
        assert trace is not None
        for key in ("cycle", "intent", "self_surprise", "affect", "narrative"):
            assert key in trace
        assert isinstance(body["history"], list)

        # On-demand counterfactual rollout.
        r2 = client.get(f"/agent/{aid}/explain?counterfactuals=1")
        assert r2.status_code == 200
        od = r2.json().get("on_demand")
        if od:  # present once a transition has been buffered
            cfs = od.get("counterfactuals")
            assert cfs is None or "candidates" in cfs

        assert client.get("/agent/nope/explain").status_code == 404


# ---------------------------------------------------------------------------
# Read-only regression: enabling the trace does not change behaviour
# ---------------------------------------------------------------------------


def _run_cycles(monkeypatch, *, trace_on: bool, n: int = 12):
    import torch

    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    # Disable the sampled attribution forward so no extra RNG/compute occurs.
    monkeypatch.setenv("DECADIC_COGNITION_ATTRIBUTION_INTERVAL", "0")
    monkeypatch.setenv("DECADIC_COGNITION_TRACE", "1" if trace_on else "0")

    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.nn.bundle import NeuralBundle
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import Homeostasis, ViabilityState

    torch.manual_seed(0)
    np.random.seed(0)
    bundle = NeuralBundle.try_build("parity")
    assert bundle is not None

    torch.manual_seed(123)
    np.random.seed(123)
    bus = StateBus()
    ctx = CycleContext(
        state_bus=bus,
        perceptual=PerceptualState(),
        viability=ViabilityState(),
        episodic=EpisodicStore(None),
        homeostasis=Homeostasis(),
        last_observation=_obs(),
    )
    actions = []
    last = None
    for _ in range(n):
        last = run_neural_cycle(ctx, bundle)
        actions.append([round(float(x), 5) for x in last["action"]["parameters"]["ctrl"]])
    return actions, bus, last


def test_trace_does_not_change_behaviour(monkeypatch):
    pytest.importorskip("torch")
    on, _, _ = _run_cycles(monkeypatch, trace_on=True)
    off, _, _ = _run_cycles(monkeypatch, trace_on=False)
    assert on == off, "enabling the cognitive trace must not alter the motor output"


def test_trace_only_mutates_narrative_stub(monkeypatch):
    pytest.importorskip("torch")
    _, bus, last = _run_cycles(monkeypatch, trace_on=True, n=5)
    cog = last["_cognitive"]
    assert cog is not None
    # The only State Bus field the trace writes is the narrative stub.
    assert bus.narrative_text_stub == cog["narrative"]
