"""Agent preset store + REST routes (seeded built-ins, CRUD, validation)."""

from __future__ import annotations

import json

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
    assert all(r["braces"] is False for r in records)


def test_store_mind_only_has_no_elements(tmp_path):
    store = PresetStore(tmp_path)
    mind = store.get("mind")
    assert mind is not None
    assert mind["mind_only"] is True
    assert mind["elements"] == []
    # A mind has no world loop either.
    assert mind["curriculum"] == {"kind": "none"}


def test_store_forage_trainer_carries_curriculum(tmp_path):
    # WS-FORAGE: the trainer preset switches on the server-side curriculum;
    # scene presets default to no loop; parameter defaults are filled in.
    store = PresetStore(tmp_path)
    ft = store.get("forage_trainer")
    assert ft is not None
    assert ft["elements"] == ["food", "water"]  # store keeps elements verbatim
    cur = ft["curriculum"]
    assert cur["kind"] == "forage"
    assert cur["place_every_s"] == 8.0 and cur["contact_radius"] == 0.35
    # Plain scene presets stay curriculum-free (back-compat).
    assert store.get("forage")["curriculum"] == {"kind": "none"}
    assert store.get("calm")["curriculum"] == {"kind": "none"}


def test_store_user_curriculum_override_roundtrips(tmp_path):
    store = PresetStore(tmp_path)
    rec = store.create(
        {
            "name": "My tight forage",
            "elements": ["food", "water"],
            "vision": True,
            "audio": False,
            "braces": False,
            "mind_only": False,
            "curriculum": {"kind": "forage", "place_every_s": 5, "rescue_floor": 20},
        }
    )
    got = store.get(rec["id"])["curriculum"]
    assert got["kind"] == "forage"
    assert got["place_every_s"] == 5.0 and got["rescue_floor"] == 20.0
    assert got["reach_distance"] == 0.6  # unspecified -> default filled


def test_store_migrates_v2_file_preserving_user_presets(tmp_path):
    # An older v2 presets file (no curriculum field) must migrate: user presets
    # preserved (gaining curriculum=none), and the new forage_trainer builtin added.
    path = tmp_path / "agent_presets.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "presets": [
                    {"id": "calm", "name": "old", "elements": ["house"], "vision": True,
                     "audio": False, "braces": False, "mind_only": False, "builtin": True},
                    {"id": "myuser01", "name": "mine", "elements": ["food", "bear"], "vision": True,
                     "audio": True, "braces": False, "mind_only": False, "builtin": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    store = PresetStore(tmp_path)
    ids = {r["id"] for r in store.list()}
    assert "myuser01" in ids and "forage_trainer" in ids
    user = store.get("myuser01")
    assert user["elements"] == ["food", "bear"]
    assert user["curriculum"] == {"kind": "none"}


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


def test_store_migrates_builtin_braces_off_without_rewriting_user_presets(tmp_path):
    first = PresetStore(tmp_path)
    user = first.create(
        {
            "name": "Manual braces scene",
            "elements": ["house"],
            "braces": True,
        }
    )
    payload = {
        "schema_version": 1,
        "presets": [
            {**r, "braces": True} if r["builtin"] else r
            for r in first.list()
        ],
    }
    first.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    second = PresetStore(tmp_path)
    records = second.list()
    assert all(r["braces"] is False for r in records if r["builtin"])
    assert second.get(user["id"])["braces"] is True


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
