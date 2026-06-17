"""Agent preset store + REST routes (seeded built-ins, CRUD, validation)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from decadic.api.presets.routes import register_preset_routes
from decadic.api.presets.store import (
    BUILTIN_PRESETS,
    PresetStore,
    is_valid_preset_id,
)


# --- Store -----------------------------------------------------------------


def test_store_seeds_builtins_on_first_init(tmp_path):
    store = PresetStore(tmp_path)
    records = store.list()
    assert len(records) == len(BUILTIN_PRESETS)
    ids = {r["id"] for r in records}
    assert {"calm", "village", "mind"} <= ids
    assert all(r["builtin"] for r in records)


def test_store_mind_only_has_no_elements(tmp_path):
    store = PresetStore(tmp_path)
    mind = store.get("mind")
    assert mind is not None
    assert mind["mind_only"] is True
    assert mind["elements"] == []


def test_store_create_get_delete_roundtrip(tmp_path):
    store = PresetStore(tmp_path)
    created = store.create(
        {
            "name": "My foraging scene",
            "elements": ["food", "water"],
            "vision": True,
            "audio": False,
            "braces": False,
            "mind_only": False,
        }
    )
    assert is_valid_preset_id(created["id"])
    assert created["builtin"] is False
    assert created["braces"] is False
    assert created["elements"] == ["food", "water"]

    fetched = store.get(created["id"])
    assert fetched is not None and fetched["name"] == "My foraging scene"

    assert store.delete(created["id"]) is True
    assert store.get(created["id"]) is None
    assert store.delete(created["id"]) is False
    assert store.delete("../etc") is False


def test_store_persists_across_instances(tmp_path):
    first = PresetStore(tmp_path)
    created = first.create({"name": "Persisted", "elements": ["house"]})
    # A fresh store reading the same dir sees the user preset (and does NOT
    # re-seed/duplicate the built-ins).
    second = PresetStore(tmp_path)
    ids = {r["id"] for r in second.list()}
    assert created["id"] in ids
    assert len([r for r in second.list() if r["id"] == "calm"]) == 1


def test_store_create_mind_only_clears_elements(tmp_path):
    store = PresetStore(tmp_path)
    created = store.create(
        {"name": "Pure mind", "elements": ["food"], "mind_only": True}
    )
    assert created["mind_only"] is True
    assert created["elements"] == []


# --- Routes ----------------------------------------------------------------


def _client(tmp_path) -> TestClient:
    app = FastAPI()
    app.state.presets_dir = tmp_path
    register_preset_routes(app)
    return TestClient(app)


def test_routes_list_returns_seeded_builtins(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/agent-presets")
    assert resp.status_code == 200
    presets = resp.json()["presets"]
    names = {p["id"] for p in presets}
    assert "calm" in names
    assert len(presets) == len(BUILTIN_PRESETS)


def test_routes_create_then_list_and_delete(tmp_path):
    client = _client(tmp_path)
    resp = client.post(
        "/agent-presets",
        json={
            "name": "Bear den",
            "elements": ["house", "bear"],
            "vision": True,
            "audio": False,
            "braces": True,
            "mind_only": False,
        },
    )
    assert resp.status_code == 200, resp.text
    created = resp.json()
    assert created["name"] == "Bear den"
    assert created["builtin"] is False

    listed = client.get("/agent-presets").json()["presets"]
    assert any(p["id"] == created["id"] for p in listed)

    deleted = client.delete(f"/agent-presets/{created['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] == created["id"]
    assert client.delete(f"/agent-presets/{created['id']}").status_code == 404


def test_routes_mind_only_preset_allows_empty_elements(tmp_path):
    client = _client(tmp_path)
    resp = client.post(
        "/agent-presets",
        json={"name": "Mind only", "elements": [], "mind_only": True},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["mind_only"] is True


def test_routes_reject_bodied_preset_with_no_valid_elements(tmp_path):
    client = _client(tmp_path)
    # Empty elements on a bodied preset is rejected...
    assert (
        client.post("/agent-presets", json={"name": "Empty", "elements": []}).status_code
        == 422
    )
    # ...and so is a list that filters down to nothing (unknown elements dropped).
    assert (
        client.post(
            "/agent-presets", json={"name": "Bad", "elements": ["volcano", "atlantis"]}
        ).status_code
        == 422
    )
