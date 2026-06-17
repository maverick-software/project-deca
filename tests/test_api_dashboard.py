"""Dashboard endpoints: /agents, vision, and pause/resume/reset controls."""

import base64
import time

from fastapi.testclient import TestClient

# 1x1 red PNG
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/"
    "q842iQAAAABJRU5ErkJggg=="
)


def _obs_with_vision():
    return {
        "timestamp": "2026-06-11T00:00:00Z",
        "proprioception": {"position": [0, 0, 1.4]},
        "events": [],
        "world_state": {"nearby_entities": [], "agent_inventory": []},
        "vision": {"encoding": "base64_png", "data": _PNG_B64, "resolution": [1, 1]},
    }


def test_list_agents(api_app):
    with TestClient(api_app) as client:
        assert client.get("/agents").json() == {"agents": []}
        aid = client.post("/agent").json()["agent_id"]
        agents = client.get("/agents").json()["agents"]
        assert len(agents) == 1
        assert agents[0]["agent_id"] == aid
        assert "neural_enabled" in agents[0]
        assert "cycles_completed" in agents[0]
        client.delete(f"/agent/{aid}")
        assert client.get("/agents").json() == {"agents": []}


def test_vision_endpoint_404_then_png(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        assert client.get(f"/agent/{aid}/vision").status_code == 404
        assert client.get("/agent/nope/vision").status_code == 404

        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            ws.send_json(_obs_with_vision())
            ws.receive_json()

        r = client.get(f"/agent/{aid}/vision")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content == base64.b64decode(_PNG_B64)


def _obs_with_views():
    obs = _obs_with_vision()
    obs["debug_views"] = {"track": _PNG_B64, "top": _PNG_B64}
    return obs


def test_vision_camera_views(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            ws.send_json(_obs_with_views())
            ws.receive_json()

        # Default and explicit egocentric → brain's vision frame
        assert client.get(f"/agent/{aid}/vision").status_code == 200
        assert client.get(f"/agent/{aid}/vision?camera=egocentric").status_code == 200

        # Spectator cameras
        r = client.get(f"/agent/{aid}/vision?camera=track")
        assert r.status_code == 200
        assert r.content == base64.b64decode(_PNG_B64)
        assert client.get(f"/agent/{aid}/vision?camera=top").status_code == 200
        assert client.get(f"/agent/{aid}/vision?camera=nope").status_code == 404

        # Views advertised in state; debug_views stripped from cognition path
        st = client.get(f"/agent/{aid}/state").json()["payload"]
        assert st["vision_views"] == ["egocentric", "top", "track"]


def test_audio_endpoint(api_app):
    import struct

    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        assert client.get(f"/agent/{aid}/audio").status_code == 404
        assert client.get("/agent/nope/audio").status_code == 404

        # 0.1 s of pcm16 at 16 kHz
        n = 1600
        pcm = struct.pack(f"<{n}h", *([1000] * n))
        obs = _obs_with_vision()
        obs["audio"] = {
            "encoding": "pcm16_base64",
            "sample_rate": 16000,
            "data": base64.b64encode(pcm).decode("ascii"),
        }
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            ws.send_json(obs)
            ws.receive_json()

        r = client.get(f"/agent/{aid}/audio")
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/wav"
        assert r.content[:4] == b"RIFF"
        assert pcm in r.content

        st = client.get(f"/agent/{aid}/state").json()["payload"]
        assert st["perceptual"]["audio_duration_s"] == 0.1
        assert st["perceptual"]["audio_rms"] is not None

        agents = client.get("/agents").json()["agents"]
        assert "encoder_mode" in agents[0]


def test_body_recenter_queues_command(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        client.post(f"/agent/{aid}/pause")  # keep cycle actions out of the out_queue

        r = client.post(f"/agent/{aid}/body/recenter")
        assert r.status_code == 200
        assert r.json()["status"] == "recenter_queued"

        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            msg = ws.receive_json()
        assert msg["type"] == "body_command"
        assert msg["command"] == "recenter"

        assert client.post("/agent/nope/body/recenter").status_code == 404


def _cycles(client: TestClient, aid: str) -> int:
    return client.get(f"/agent/{aid}/metrics").json()["metrics"]["cycles_completed"]


def test_pause_resume(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        time.sleep(0.15)
        assert _cycles(client, aid) > 0

        r = client.post(f"/agent/{aid}/pause")
        assert r.json()["status"] == "paused"
        assert client.get("/agents").json()["agents"][0]["paused"] is True
        time.sleep(0.1)
        frozen = _cycles(client, aid)
        time.sleep(0.15)
        assert _cycles(client, aid) == frozen

        r = client.post(f"/agent/{aid}/resume")
        assert r.json()["status"] == "running"
        time.sleep(0.2)
        assert _cycles(client, aid) > frozen


def test_reset_clears_state_and_memory(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            ws.send_json(
                {
                    "timestamp": "2026-06-11T00:00:00Z",
                    "proprioception": {"position": [0, 0, 1.4]},
                    "events": [{"type": "collision", "intensity": 0.9, "source": "test"}],
                    "world_state": {"nearby_entities": [], "agent_inventory": []},
                }
            )
            ws.receive_json()

        m = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        assert m["viability"] < 100.0
        assert m["fast_path_hits"] >= 1

        r = client.post(f"/agent/{aid}/reset")
        assert r.json()["status"] == "reset"

        m2 = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        assert m2["viability"] == 100.0
        assert m2["fast_path_hits"] == 0
        st = client.get(f"/agent/{aid}/state").json()["payload"]
        assert st["perceptual"]["integration_ticks"] == 0
        mem = client.get(f"/agent/{aid}/memory").json()["episodes"]
        assert mem == []

        assert client.post("/agent/nope/reset").status_code == 404
        assert client.post("/agent/nope/pause").status_code == 404
        assert client.post("/agent/nope/resume").status_code == 404


def _obs_with_events(events):
    return {
        "timestamp": "2026-06-11T00:00:00Z",
        "proprioception": {"position": [0, 0, 1.4]},
        "events": events,
        "world_state": {"nearby_entities": [], "agent_inventory": []},
    }


def test_death_revive_and_status_surfaced(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        client.post(f"/agent/{aid}/pause")  # deterministic: fast-path still kills while paused

        a0 = client.get("/agents").json()["agents"][0]
        assert a0["status"] == "alive"
        assert a0["died_at_cycle"] is None

        # Contact can no longer be instantly lethal (impact damage is superficial
        # and capped), so bottom out a reservoir directly to drive death; a benign
        # observation then trips the fast-path death check while paused.
        api_app.state.registry.get(aid).homeostasis.hydration = 0.0
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            ws.send_json(_obs_with_events([]))
            ws.receive_json()  # death event (or a residual cycle frame)

        m = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        assert m["status"] == "dead"
        assert m["viability"] == 0.0
        a = client.get("/agents").json()["agents"][0]
        assert a["status"] == "dead"
        assert a["died_at_cycle"] is not None

        r = client.post(f"/agent/{aid}/revive")
        assert r.json()["status"] == "alive"
        m2 = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        assert m2["status"] == "alive"
        assert m2["viability"] > 0.0

        assert client.post("/agent/nope/revive").status_code == 404


def test_graph_persists_entity_out_of_view(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        seen = {
            "timestamp": "2026-06-11T00:00:00Z",
            "proprioception": {"position": [0, 0, 1.4]},
            "events": [{"type": "threat_near", "intensity": 0.8, "source": "prop_bear"}],
            "world_state": {
                "agent": {"id": "self", "position": [0, 0, 0]},
                "entities": [{"id": "prop_bear", "kind": "bear", "position": [2, 0, 0]}],
            },
        }
        gone = {
            "timestamp": "2026-06-11T00:00:01Z",
            "proprioception": {"position": [0, 0, 1.4]},
            "events": [],
            "world_state": {"agent": {"id": "self", "position": [0, 0, 0]}, "entities": []},
        }
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            ws.send_json(seen)
            ws.receive_json()
            for _ in range(3):  # bear leaves view
                ws.send_json(gone)
                ws.receive_json()

        graph = client.get(f"/agent/{aid}/state").json()["payload"]["perceptual"][
            "egocentric_graph"
        ]
        bear = next((n for n in graph["nodes"] if n.get("id") == "prop_bear"), None)
        assert bear is not None  # object permanence: still remembered out of view
        assert 0.0 < bear["salience"] < 1.0  # but decayed
        # the threat left a negative affective edge tied to the self
        aff = [e for e in graph["edges"] if e["kind"] == "affective" and e["target"] == "prop_bear"]
        assert aff and aff[0]["weight"] < 0


def test_configure_capacity(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        r = client.post(
            f"/agent/{aid}/config"
            "?parallel_sessions=4&working_memory_slots=20&working_memory_decay=0.8"
        )
        body = r.json()
        assert body["parallel_sessions"] == 4
        assert body["working_memory_slots"] == 20
        assert abs(body["working_memory_decay"] - 0.8) < 1e-6

        m = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        assert m["parallel_sessions"] == 4
        assert m["working_memory_slots"] == 20

        assert client.post("/agent/nope/config").status_code == 404


def test_configure_assist_override(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        # Default is Auto (None).
        body = client.post(f"/agent/{aid}/config").json()
        # Default is 0 (no training-wheel assist unless the operator opts in).
        assert abs(body["assist_override"] - 0.0) < 1e-6
        # Pin a manual level.
        body = client.post(f"/agent/{aid}/config?assist_override=2").json()
        assert abs(body["assist_override"] - 2.0) < 1e-6
        # Negative sentinel clears back to Auto.
        body = client.post(f"/agent/{aid}/config?assist_override=-1").json()
        assert body["assist_override"] is None


def test_configure_curriculum_mode(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        # Default (no env override) is the legacy training-wheels assist.
        body = client.post(f"/agent/{aid}/config").json()
        assert body["curriculum_mode"] == "legacy"
        # Switch to the guided assist-as-needed harness.
        body = client.post(f"/agent/{aid}/config?curriculum_mode=guided").json()
        assert body["curriculum_mode"] == "guided"
        # "standard" is a friendly alias back to legacy.
        body = client.post(f"/agent/{aid}/config?curriculum_mode=standard").json()
        assert body["curriculum_mode"] == "legacy"


def test_collision_damages_integrity_reservoir(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            ws.send_json(
                _obs_with_events([{"type": "collision", "intensity": 0.9, "source": "test"}])
            )
            ws.receive_json()
        m = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        # Damage hits integrity; hydration and energy are untouched.
        assert m["integrity"] < 100.0
        assert m["hydration"] == 100.0
        assert m["energy"] == 100.0
        assert m["viability"] == m["integrity"]  # viability = min of reservoirs


def test_food_and_water_credit_reservoirs(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        # Start hungry and thirsty so credits are not clamped at full.
        agent = api_app.state.registry.get(aid)
        agent.homeostasis.energy = 40.0
        agent.homeostasis.hydration = 40.0

        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            ws.send_json(
                _obs_with_events([{"type": "food", "intensity": 1.0, "source": "prop_food_1"}])
            )
            ws.receive_json()
            ws.send_json(
                _obs_with_events([{"type": "water", "intensity": 1.0, "source": "prop_water_w1"}])
            )
            ws.receive_json()

        m = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        assert m["energy"] > 40.0  # food refilled energy
        assert m["hydration"] > 40.0  # water refilled hydration
        state = client.get(f"/agent/{aid}/state").json()["payload"]
        assert state["state_bus"]["B_pleasure_scalar"] > 0.0


def test_plasticity_metrics_and_live_knobs(api_app_plastic):
    import time

    with TestClient(api_app_plastic) as client:
        aid = client.post("/agent").json()["agent_id"]

        m = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        assert m["plasticity_enabled"] is True
        assert m["sparse_enabled"] is True
        assert m["growth_enabled"] is True
        assert m["awake_neurons"] > 0
        assert m["allocated_neurons"] >= m["awake_neurons"]
        # capacity_config surfaces a plasticity block for the dashboard.
        cfg = client.post(f"/agent/{aid}/config").json()
        assert cfg["plasticity"]["available"] is True

        # A cycle still produces a motor action with all features on.
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            ws.send_json(
                {
                    "timestamp": "2026-06-11T00:00:00Z",
                    "proprioception": {"position": [0, 0, 1.4]},
                    "events": [],
                    "world_state": {"nearby_entities": [], "agent_inventory": []},
                }
            )
            msg = ws.receive_json()
        assert msg["action"]["type"] == "motor"

        # Live A knob: set plastic strength; metrics reflect it.
        client.post(f"/agent/{aid}/config?plasticity_alpha=0.25")
        m2 = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        assert abs(m2["plasticity_alpha"] - 0.25) < 1e-3

        # Live C knob: shrinking the cap sleeps neurons immediately.
        awake_before = client.get(f"/agent/{aid}/metrics").json()["metrics"]["awake_neurons"]
        client.post(f"/agent/{aid}/config?max_neurons=4")
        time.sleep(0.05)
        awake_after = client.get(f"/agent/{aid}/metrics").json()["metrics"]["awake_neurons"]
        assert awake_after < awake_before


def test_plasticity_brain_topology_growth(api_app_plastic):
    import time

    with TestClient(api_app_plastic) as client:
        aid = client.post("/agent").json()["agent_id"]
        topo = client.get(f"/agent/{aid}/brain/topology").json()
        assert "awake_neurons" in topo["totals"]
        assert topo["totals"]["allocated_neurons"] >= topo["totals"]["awake_neurons"]
        # Let the growth controller wake neurons over several cycles.
        time.sleep(0.4)
        topo2 = client.get(f"/agent/{aid}/brain/topology").json()
        assert topo2["totals"]["awake_neurons"] >= topo["totals"]["awake_neurons"]


def test_plasticity_reset_rebuilds_clean(api_app_plastic):
    with TestClient(api_app_plastic) as client:
        aid = client.post("/agent").json()["agent_id"]
        client.post(f"/agent/{aid}/config?plasticity_alpha=0.4")
        r = client.post(f"/agent/{aid}/reset")
        assert r.json()["status"] == "reset"
        m = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        # Reset rebuilds from env defaults; plasticity stays available.
        assert m["plasticity_enabled"] is True
        assert m["growth_events"] == 0
        assert m["plasticity_frozen"] is False


def test_agent_defaults_roundtrip(api_app_neural):
    with TestClient(api_app_neural) as client:
        # With no env flags set, defaults resolve to all-off.
        base = client.get("/settings/agent-defaults").json()
        assert base["plasticity_enabled"] is False
        assert base["sparse_enabled"] is False
        assert base["growth_enabled"] is False

        # Setting one flag leaves the others at their prior value (partial merge).
        r = client.post("/settings/agent-defaults?plasticity_enabled=1").json()
        assert r["plasticity_enabled"] is True
        assert r["sparse_enabled"] is False
        assert r["growth_enabled"] is False

        r2 = client.post("/settings/agent-defaults?max_neurons=120&growth_enabled=1").json()
        assert r2["plasticity_enabled"] is True  # sticky
        assert r2["growth_enabled"] is True
        assert r2["max_neurons"] == 120

        # GET round-trips the stored defaults.
        again = client.get("/settings/agent-defaults").json()
        assert again == r2


def test_agent_defaults_apply_to_new_agents_only(api_app_neural):
    with TestClient(api_app_neural) as client:
        # Agent created before any toggle: plasticity off.
        before = client.post("/agent").json()["agent_id"]
        m_before = client.get(f"/agent/{before}/metrics").json()["metrics"]
        assert m_before["plasticity_enabled"] is False
        assert m_before["growth_enabled"] is False

        # Toggle plasticity + growth on for new agents.
        client.post("/settings/agent-defaults?plasticity_enabled=1&growth_enabled=1")

        # New agent picks up the defaults at build time.
        after = client.post("/agent").json()["agent_id"]
        m_after = client.get(f"/agent/{after}/metrics").json()["metrics"]
        assert m_after["plasticity_enabled"] is True
        assert m_after["growth_enabled"] is True
        assert m_after["awake_neurons"] > 0

        # The pre-existing agent is untouched, even after a reset (which reuses
        # the flags it was created with, not the new registry defaults).
        m_before2 = client.get(f"/agent/{before}/metrics").json()["metrics"]
        assert m_before2["plasticity_enabled"] is False
        client.post(f"/agent/{before}/reset")
        m_before3 = client.get(f"/agent/{before}/metrics").json()["metrics"]
        assert m_before3["plasticity_enabled"] is False
        assert m_before3["growth_enabled"] is False


def test_configure_viability_mode(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        # Default is the metabolic model.
        body = client.post(f"/agent/{aid}/config").json()
        assert body["viability_mode"] == "metabolic"

        # Switch to immortal: reservoirs pin at full and damage cannot hurt.
        body = client.post(f"/agent/{aid}/config?viability_mode=immortal").json()
        assert body["viability_mode"] == "immortal"
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            ws.send_json(
                _obs_with_events([{"type": "collision", "intensity": 50.0, "source": "test"}])
            )
            ws.receive_json()
        m = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        assert m["status"] == "alive"
        assert m["viability"] == 100.0
        assert m["integrity"] == 100.0
        assert m["time_to_death_s"] is None

        # Compression knob round-trips.
        body = client.post(f"/agent/{aid}/config?metabolic_compression=120").json()
        assert abs(body["metabolic_compression"] - 120.0) < 1e-6


def test_configure_curriculum_live_knobs(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        # Default: the active-inference knobs follow the env (None override).
        body = client.post(f"/agent/{aid}/config").json()
        assert body["ai_intero_pref_weight"] is None
        assert body["drive_priority_gain"] is None
        assert body["motor_babble_sigma"] is None
        # Pin overrides.
        body = client.post(
            f"/agent/{aid}/config"
            "?ai_intero_pref_weight=0.5&drive_priority_gain=3&motor_babble_sigma=0.2"
        ).json()
        assert abs(body["ai_intero_pref_weight"] - 0.5) < 1e-6
        assert abs(body["drive_priority_gain"] - 3.0) < 1e-6
        assert abs(body["motor_babble_sigma"] - 0.2) < 1e-6
        # Negative sentinel clears back to env default (None).
        body = client.post(f"/agent/{aid}/config?motor_babble_sigma=-1").json()
        assert body["motor_babble_sigma"] is None


def test_locomotion_telemetry_surfaced(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        # New eval-only locomotion/gait metrics are present (default zero).
        m = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        for key in (
            "distance_traveled",
            "net_displacement",
            "fall_rate",
            "gait_regularity",
            "consume_events",
        ):
            assert key in m


def test_curriculum_status_default_and_unknown_agent(api_app):
    with TestClient(api_app) as client:
        st = client.get("/curriculum").json()
        assert st["state"] == "stopped"
        assert st["running"] is False
        # Starting against a non-existent agent is a 409.
        r = client.post("/curriculum/start", json={"agent_id": "nope"})
        assert r.status_code == 409


def test_curriculum_lifecycle_and_phase_override(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        # Start binds to the agent and applies phase 0 (immortal) immediately.
        st = client.post("/curriculum/start", json={"agent_id": aid}).json()
        assert st["state"] == "running"
        assert st["phase_index"] == 0
        assert st["phase_name"] == "Self-modeling"
        m = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        assert m["viability_mode"] == "immortal"

        # A second start while running is rejected.
        assert client.post("/curriculum/start", json={"agent_id": aid}).status_code == 409

        # Pause / resume round-trips.
        assert client.post("/curriculum/pause").json()["paused"] is True
        assert client.post("/curriculum/resume").json()["paused"] is False

        # Manual phase override jumps to phase 1 (metabolic) and applies its config.
        st1 = client.post("/curriculum/phase", json={"index": 1}).json()
        assert st1["phase_index"] == 1
        m1 = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        assert m1["viability_mode"] == "metabolic"

        stopped = client.post("/curriculum/stop").json()
        assert stopped["state"] == "stopped"
