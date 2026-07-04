"""WS4-M4: checkpoint/restore and save/load REST routes under the flipped
lancedb+kuzu defaults.

Store-level round-trips were proven in test_ws4_backends.py; this closes the
remaining M4 item -- ROUTE-level path handling. The pivotal difference vs the
legacy mode: `backup_to` targets named ``episodes.sqlite``/``graph.sqlite``
become DIRECTORIES under lance/kuzu (copytree snapshots), while the sqlite
backends write regular files at the same paths (test_saved_agents.py pins that
legacy layout explicitly). The route layer never inspects the paths, so both
layouts must round-trip through the identical endpoints.

Backends are pinned to lancedb+kuzu explicitly (they are the factory defaults,
but ambient env from launcher scripts must not silently downgrade this test).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("lancedb")
pytest.importorskip("kuzu")

from decadic.memory.episodic_store import EpisodicRecord


def _pin_new_backends(monkeypatch) -> None:
    monkeypatch.setenv("DECADIC_MEMORY_BACKEND", "lancedb")
    monkeypatch.setenv("DECADIC_GRAPH_BACKEND", "kuzu")


def _minimal_obs():
    return {
        "timestamp": "2026-07-03T12:00:00Z",
        "proprioception": {
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "current_action": "idle",
        },
        "events": [],
    }


def _drain_ws(ws, cap: int = 64) -> None:
    for _ in range(cap):
        try:
            ws.receive_json(timeout=0.05)
        except Exception:
            break


def _drive_cycles(client, agent_id: str, n: int) -> None:
    with client.websocket_connect(f"/agent/{agent_id}/cycle") as ws:
        _drain_ws(ws)
        for _ in range(n):
            ws.send_json(_minimal_obs())
            _ = ws.receive_json()


def test_checkpoint_restore_routes_lance_kuzu(api_app_neural, monkeypatch):
    """POST /checkpoint -> mutate live state -> POST /restore == checkpoint.

    Exercises the backups_dir route path under the new defaults: worker
    suspend/resume around a live lance store, brain save/load, and the
    checkpoint payload round-trip (PRD 5.4 acceptance at the route level).
    """
    _pin_new_backends(monkeypatch)
    with TestClient(api_app_neural) as client:
        aid = client.post("/agent").json()["agent_id"]
        _drive_cycles(client, aid, 4)
        client.post(f"/agent/{aid}/pause")

        agent = api_app_neural.state.registry.get(aid)
        assert agent is not None

        r = client.post(f"/agent/{aid}/checkpoint")
        assert r.status_code == 200, r.text
        ck = r.json()
        assert Path(ck["path"]).is_file()

        saved_via = float(agent.viability.value)
        saved_pain = float(agent.state_bus.pain_scalar)
        saved_cycle = int(agent.state_bus.cycle_index)

        # Mutate live state the checkpoint must win back (worker is paused, so
        # nothing else touches these).
        agent.viability.value = max(0.01, saved_via * 0.5 - 0.02)
        agent.state_bus.pain_scalar = 0.77
        assert float(agent.viability.value) != pytest.approx(saved_via)

        rr = client.post(f"/agent/{aid}/restore")
        assert rr.status_code == 200, rr.text
        payload = rr.json()["payload"]
        assert float(agent.viability.value) == pytest.approx(saved_via, abs=1e-6)
        assert float(agent.state_bus.pain_scalar) == pytest.approx(saved_pain, abs=1e-6)
        assert int(payload["state_bus"]["cycle_index"]) == saved_cycle


def test_saved_agent_roundtrip_lance_kuzu(api_app_neural, monkeypatch, tmp_path):
    """Full save -> load through the Saved Agents routes on lance+kuzu.

    Asserts the directory-shaped snapshot layout, memory fidelity into the NEW
    agent (exact episode set + working similarity search over the restored
    mirror), isolation from post-save mutation, and clean delete.
    """
    saved_dir = tmp_path / "saved_agents"
    monkeypatch.setenv("DECADIC_SAVED_DIR", str(saved_dir))
    _pin_new_backends(monkeypatch)

    with TestClient(api_app_neural) as client:
        aid = client.post("/agent").json()["agent_id"]
        _drive_cycles(client, aid, 6)
        client.post(f"/agent/{aid}/pause")

        # Guarantee searchable episodic content (cycle-driven rows are
        # config-dependent; the route's memory round-trip is what's on trial).
        agent = api_app_neural.state.registry.get(aid)
        rng = np.random.default_rng(41)
        embs = rng.normal(size=(5, 80)).astype(np.float32)
        for i in range(5):
            agent.episodic.append(
                EpisodicRecord(
                    cycle_index=9000 + i,
                    summary={"m4": i},
                    salience=0.9,
                    embedding=embs[i],
                )
            )
        agent.episodic.flush()

        r = client.post(f"/agent/{aid}/save", json={"name": "m4-lance", "notes": "ws4"})
        assert r.status_code == 200, r.text
        rec = r.json()
        save_id = rec["save_id"]
        assert rec["has_memory"] is True

        # New-backend layout: memory snapshots are DIRECTORIES at the legacy
        # file names; manifest/state/brain stay regular files.
        save_root = saved_dir / save_id
        assert (save_root / "episodes.sqlite").is_dir()
        graph_snap = save_root / "graph.sqlite"
        if graph_snap.exists():  # ltm graph is config-dependent in this fixture
            assert graph_snap.is_dir()
        for fname in ("manifest.json", "state.json", "brain.pt"):
            assert (save_root / fname).is_file(), fname

        # Source episode set as of the save (paused worker: stable).
        src_cycles = sorted(
            row["cycle_index"] for row in agent.episodic.recent(limit=1000)
        )
        assert set(range(9000, 9005)).issubset(src_cycles)

        # Mutate the source AFTER the save -- the snapshot must not see it.
        agent.episodic.append(
            EpisodicRecord(cycle_index=9999, summary={"post_save": True}, salience=0.9)
        )
        agent.episodic.flush()

        lr = client.post(f"/saved-agents/{save_id}/load")
        assert lr.status_code == 200, lr.text
        new_id = lr.json()["agent_id"]
        assert new_id != aid
        client.post(f"/agent/{new_id}/pause")

        restored = api_app_neural.state.registry.get(new_id)
        got_cycles = sorted(
            row["cycle_index"] for row in restored.episodic.recent(limit=1000)
        )
        assert got_cycles == src_cycles
        assert 9999 not in got_cycles

        # Similarity search works over the restored corpus (exercises the
        # mirror invalidate-and-rebuild path after restore_from).
        hits = restored.episodic.search_similar(embs[2], top_k=1)
        assert hits and hits[0]["cycle_index"] == 9002
        assert hits[0]["similarity"] == pytest.approx(1.0, abs=1e-5)

        # State restored into the new agent (worker may tick before pause).
        after = client.get(f"/agent/{new_id}/state").json()["payload"]
        assert after["state_bus"]["cycle_index"] >= int(rec["cycle_index"])

        # Delete removes the directory-shaped save cleanly.
        assert client.delete(f"/saved-agents/{save_id}").status_code == 200
        assert not save_root.exists()
