"""Widened proprioception path: encoder caps + perceptual snapshots + body adapter."""

import importlib.util
import sys
from pathlib import Path

import torch

from decadic.nn.frozen_encoders import (
    CLIP_POOL_DIM,
    PROPRIO_BASE_DIM,
    WHISPER_POOL_DIM,
    FrozenSensoryEncoders,
    _capped_floats,
)
from decadic.state.perceptual_state import PerceptualState


def _load_adapter_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "mujoco_decadic_adapter.py"
    spec = importlib.util.spec_from_file_location("mujoco_decadic_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mujoco_decadic_adapter"] = mod
    spec.loader.exec_module(mod)
    return mod


def _encoders() -> FrozenSensoryEncoders:
    return FrozenSensoryEncoders(
        mode="zeros", device=torch.device("cpu"), proprio_dim_out=32
    )


def test_capped_floats_pad_and_truncate():
    assert _capped_floats([1, 2, 3], 5) == [1.0, 2.0, 3.0, 0.0, 0.0]
    assert _capped_floats(list(range(10)), 4) == [0.0, 1.0, 2.0, 3.0]
    assert _capped_floats(None, 3) == [0.0, 0.0, 0.0]
    assert _capped_floats(["bad", 2], 2) == [0.0, 2.0]


def test_encoder_forward_without_body_arrays():
    enc = _encoders()
    obs = {
        "proprioception": {
            "position": [1, 2, 3],
            "orientation": [0, 0, 0],
            "velocity": [0.1, 0, 0],
            "current_action": "walking_forward",
        }
    }
    out = enc(obs)
    assert out.shape == (1, CLIP_POOL_DIM + WHISPER_POOL_DIM + 32)


def test_encoder_forward_with_joints_and_contacts(monkeypatch):
    monkeypatch.setenv("DECADIC_PROPRIO_JOINT_CAP", "8")
    monkeypatch.setenv("DECADIC_PROPRIO_CONTACT_CAP", "4")
    enc = _encoders()
    assert enc.proprio_in_dim == PROPRIO_BASE_DIM + 8 + 4
    obs = {
        "proprioception": {
            "position": [0, 0, 1.4],
            "joints": [0.1] * 20,  # truncated to 8
            "contacts": [110.0, 110.0],  # padded to 4
        }
    }
    out = enc(obs)
    assert out.shape[-1] == CLIP_POOL_DIM + WHISPER_POOL_DIM + 32
    assert torch.isfinite(out).all()


def test_perceptual_state_stores_body_proprio():
    ps = PerceptualState()
    ps.integrate_observation(
        {
            "timestamp": "2026-06-11T00:00:00Z",
            "proprioception": {
                "position": [0, 0, 1.4],
                "joints": [0.1, -0.2],
                "contacts": [110.0, 0.0],
            },
        }
    )
    snap = ps.snapshot_dict()
    assert snap["proprio_joints"] == [0.1, -0.2]
    assert snap["proprio_contacts"] == [110.0, 0.0]


def test_body_events_impact_and_fall():
    mod = _load_adapter_module()

    # Steady weight-bearing (resting / lying down): same forces as last frame,
    # no body speed -> no spike -> NO collision. This is the bug fix.
    resting = {"touch_pelvis": 700.0, "touch_left_foot": 110.0}
    events, fallen = mod.body_events(
        resting,
        1.0,
        was_fallen=False,
        prev_contacts=dict(resting),
        velocity=[0.0, 0.0, 0.0],
        step=0,
        cooldown_until={},
    )
    assert all(e["type"] != "collision" for e in events)

    # A genuine impact: sudden force spike while moving downward -> one
    # collision, intensity scaled by impact energy, cooldown armed.
    cooldown: dict = {}
    impact = {"touch_right_hand": 900.0, "touch_left_foot": 110.0}
    events, _ = mod.body_events(
        impact,
        1.3,
        was_fallen=False,
        prev_contacts={"touch_right_hand": 0.0, "touch_left_foot": 100.0},
        velocity=[0.0, 0.0, -3.0],
        step=10,
        cooldown_until=cooldown,
    )
    collisions = [e for e in events if e["type"] == "collision"]
    assert len(collisions) == 1
    assert collisions[0]["source"] == "touch_right_hand"
    assert 0.0 < collisions[0]["intensity"] <= 1.0
    assert cooldown.get("impact", 0) > 10

    # Within the cooldown window, a repeat spike is suppressed (one tumble,
    # one event -- not damage every frame while it settles).
    events, _ = mod.body_events(
        impact,
        1.3,
        was_fallen=False,
        prev_contacts={"touch_right_hand": 0.0},
        velocity=[0.0, 0.0, -3.0],
        step=11,
        cooldown_until=cooldown,
    )
    assert all(e["type"] != "collision" for e in events)

    # Fall marker is one-shot when the root drops below the fall height.
    events, fallen = mod.body_events({}, 0.4, was_fallen=False)
    assert any(e["type"] == "fall" for e in events)
    assert fallen is True

    events, fallen = mod.body_events({}, 0.4, was_fallen=True)
    assert events == []
    assert fallen is True


def test_build_body_observation_shape():
    mod = _load_adapter_module()
    snap = mod.dry_snapshot(3)
    obs = mod.build_body_observation(snap, events=[{"type": "collision", "intensity": 0.5}])
    prop = obs["proprioception"]
    assert prop["current_action"] == "mujoco_humanoid:root_assist"
    assert len(prop["joints"]) == 42
    assert len(prop["contacts"]) == 4
    ws = obs["world_state"]
    assert ws["agent"]["id"] == "self"
    assert ws["body"]["id"] == "mujoco_humanoid"
    assert ws["body"]["standing"] is True
    assert ws["entities"][0]["id"] == "prop_box_red"
    assert len(ws["entities"][0]["relative"]) == 3
    assert obs["events"][0]["type"] == "collision"
    assert "vision" not in obs

    obs_v = mod.build_body_observation(snap, vision_b64="aGVsbG8=", vision_resolution=(224, 224))
    assert obs_v["vision"]["encoding"] == "base64_png"
    assert obs_v["vision"]["resolution"] == [224, 224]
    assert "debug_views" not in obs_v

    obs_dv = mod.build_body_observation(
        snap, vision_b64="aGVsbG8=", debug_views={"track": "YmFjaw==", "top": "dG9w"}
    )
    assert obs_dv["debug_views"] == {"track": "YmFjaw==", "top": "dG9w"}


def test_synth_audio_window_and_payload():
    import base64

    import numpy as np

    mod = _load_adapter_module()

    # Quiet: no events, no contacts → just the ambient floor
    quiet = mod.synth_audio_window([], {}, seed=7)
    assert quiet.shape == (int(mod.AUDIO_SR * mod.AUDIO_WINDOW_S),)
    assert float(np.abs(quiet).max()) < 0.05

    # Footstep + collision + fall → clearly louder than ambient
    loud = mod.synth_audio_window(
        [
            {"type": "collision", "intensity": 0.9, "source": "touch_right_hand"},
            {"type": "fall", "intensity": 0.6, "source": "root"},
            {"type": "threat_near", "intensity": 0.8, "source": "prop_bear"},
            {"type": "food", "intensity": 1.0, "source": "prop_food_1"},
        ],
        {"touch_right_foot": 600.0, "touch_left_foot": 20.0},
        seed=7,
    )
    assert float(np.abs(loud).max()) > 0.3
    assert float(np.abs(loud).max()) <= 0.95 + 1e-6
    assert float(np.sqrt((loud**2).mean())) > float(np.sqrt((quiet**2).mean()))

    payload = mod.audio_payload(loud)
    assert payload["encoding"] == "pcm16_base64"
    assert payload["sample_rate"] == mod.AUDIO_SR
    assert payload["duration_s"] == mod.AUDIO_WINDOW_S
    pcm = np.frombuffer(base64.b64decode(payload["data"]), dtype="<i2")
    assert pcm.shape == loud.shape

    obs = mod.build_body_observation(mod.dry_snapshot(1), audio=payload)
    assert obs["audio"]["sample_rate"] == mod.AUDIO_SR


def test_waveform_from_obs_decode():
    import base64

    import numpy as np

    from decadic.nn.frozen_encoders import _waveform_from_obs

    wav = (np.sin(np.linspace(0, 20, 1600)) * 0.5).astype(np.float32)
    pcm = (wav * 32767.0).astype("<i2").tobytes()
    obs = {
        "audio": {
            "encoding": "pcm16_base64",
            "sample_rate": 16000,
            "data": base64.b64encode(pcm).decode("ascii"),
        }
    }
    decoded = _waveform_from_obs(obs)
    assert decoded is not None
    assert decoded.shape == wav.shape
    assert float(np.abs(decoded).max()) <= 1.0
    assert float(np.abs(decoded - wav).max()) < 1e-3

    assert _waveform_from_obs({}) is None
    assert _waveform_from_obs({"audio": {"data": ""}}) is None
    assert _waveform_from_obs({"audio": {"encoding": "mp3", "data": "abcd"}}) is None
    assert _waveform_from_obs({"audio": {"data": "!!!notbase64!!!"}}) is None


def test_perceptual_state_audio_stats():
    import base64

    import numpy as np

    ps = PerceptualState()
    wav = (np.ones(8000, dtype=np.float32) * 0.5 * 32767.0).astype("<i2").tobytes()
    ps.integrate_observation(
        {
            "timestamp": "2026-06-11T00:00:00Z",
            "audio": {
                "encoding": "pcm16_base64",
                "sample_rate": 16000,
                "data": base64.b64encode(wav).decode("ascii"),
            },
        }
    )
    assert ps.audio_duration_s == 0.5
    assert ps.audio_rms is not None and abs(ps.audio_rms - 0.5) < 0.01
    snap = ps.snapshot_dict()
    assert snap["audio_duration_s"] == 0.5


def test_world_graph_body_context_node():
    from decadic.state.world_graph import egocentric_nodes_from_world_state

    ws = {
        "agent": {"id": "self", "position": [0, 0, 1.4]},
        "entities": [],
        "body": {
            "id": "mujoco_humanoid",
            "control_mode": "root_assist",
            "standing": True,
            "moving": False,
        },
    }
    nodes = egocentric_nodes_from_world_state(ws, cap=12)
    body_nodes = [n for n in nodes if n.get("kind") == "body"]
    assert len(body_nodes) == 1
    assert body_nodes[0]["role"] == "context"
    assert body_nodes[0]["id"] == "mujoco_humanoid"
    assert body_nodes[0]["standing"] is True
