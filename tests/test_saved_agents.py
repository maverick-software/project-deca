"""Saved Agents library: episodic backup/restore, the filesystem store, and the
full save -> load round-trip through the API (durable, separate from backups/)."""

import pytest
from fastapi.testclient import TestClient


def _minimal_obs():
    return {
        "timestamp": "2026-06-16T12:00:00Z",
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


def test_episodic_backup_restore_roundtrip(tmp_path):
    """backup_to snapshots live rows; restore_from replaces the target's rows."""
    from decadic.memory.episodic_store import EpisodicRecord, EpisodicStore

    store = EpisodicStore(tmp_path / "src.sqlite")
    store.append(EpisodicRecord(cycle_index=1, summary={"a": 1}, salience=0.5))
    store.append(
        EpisodicRecord(cycle_index=2, summary={"b": 2}, salience=0.9, embedding=[0.1, 0.2])
    )

    snap = tmp_path / "snap.sqlite"
    store.backup_to(snap)
    assert snap.is_file()

    other = EpisodicStore(tmp_path / "dst.sqlite")
    # A pre-existing row must be overwritten by the restore, not merged.
    other.append(EpisodicRecord(cycle_index=99, summary={"junk": True}, salience=0.1))
    other.restore_from(snap)

    cycles = sorted(r["cycle_index"] for r in other.recent(limit=10))
    assert cycles == [1, 2]


def test_saved_agent_store_crud(tmp_path):
    """The filesystem store validates ids and round-trips manifests."""
    from decadic.api.saved_agents.store import SavedAgentStore, is_valid_save_id

    assert is_valid_save_id("abc123_-")
    assert not is_valid_save_id("../evil")
    assert not is_valid_save_id("a/b")
    assert not is_valid_save_id("")

    store = SavedAgentStore(tmp_path / "saved_agents")
    sid = store.new_save_id()
    store.write_manifest(
        sid, {"save_id": sid, "name": "x", "created_at": "2026-01-01T00:00:00Z"}
    )
    assert store.exists(sid)
    assert store.read_manifest(sid)["name"] == "x"
    assert any(m["save_id"] == sid for m in store.list())

    # Traversal-proof: bad ids never read or delete anything.
    assert store.read_manifest("../etc") is None
    assert store.delete("../etc") is False

    assert store.delete(sid)
    assert not store.exists(sid)


def test_saved_agent_roundtrip_api(api_app_neural, monkeypatch, tmp_path):
    """Save a live (neural) agent, then load it into a brand-new agent."""
    saved_dir = tmp_path / "saved_agents"
    monkeypatch.setenv("DECADIC_SAVED_DIR", str(saved_dir))

    with TestClient(api_app_neural) as client:
        aid = client.post("/agent").json()["agent_id"]
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            _drain_ws(ws)
            for _ in range(6):
                ws.send_json(_minimal_obs())
                _ = ws.receive_json()
        # The cycle worker free-runs on an interval; pause it so the save
        # captures a stable cycle index.
        client.post(f"/agent/{aid}/pause")

        # Save (memory always bundled).
        r = client.post(f"/agent/{aid}/save", json={"name": "test-walker", "notes": "n1"})
        assert r.status_code == 200, r.text
        rec = r.json()
        save_id = rec["save_id"]
        saved_cycle = rec["cycle_index"]
        assert rec["name"] == "test-walker"
        assert rec["has_memory"] is True
        assert rec["preset"] == "tiny"
        assert saved_cycle >= 1

        # Listed.
        listed = client.get("/saved-agents").json()["saves"]
        assert any(s["save_id"] == save_id for s in listed)

        # On-disk layout, in the dedicated dir (NOT backups/).
        save_root = saved_dir / save_id
        for fname in ("manifest.json", "state.json", "brain.pt", "episodes.sqlite"):
            assert (save_root / fname).is_file(), fname

        # Load -> a brand-new live agent with the restored mind.
        lr = client.post(f"/saved-agents/{save_id}/load")
        assert lr.status_code == 200, lr.text
        new_id = lr.json()["agent_id"]
        assert new_id != aid
        client.post(f"/agent/{new_id}/pause")

        after = client.get(f"/agent/{new_id}/state").json()["payload"]
        # Restored mind: the cycle index jumped to (at least) the saved value
        # rather than starting fresh at 0 (the worker may advance a tick before
        # the pause lands, hence >=).
        assert after["state_bus"]["cycle_index"] >= saved_cycle
        assert after["viability"]["value"] == pytest.approx(rec["viability"])

        # Delete removes the save.
        assert client.delete(f"/saved-agents/{save_id}").status_code == 200
        assert not save_root.exists()
        assert all(
            s["save_id"] != save_id
            for s in client.get("/saved-agents").json()["saves"]
        )
        assert client.delete(f"/saved-agents/{save_id}").status_code == 404
