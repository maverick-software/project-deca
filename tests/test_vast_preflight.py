"""GET /vast/preflight + POST /vast/ssh/setup (vast_connect M0/M1.4).

Route-level: the preflight aggregates named checks with actionable reasons and
a single ``ready`` gate; the setup route refuses to run without an API key and
never leaks private key material.

Isolation matters here: ``create_app()`` builds its VastSettingsStore at the
DEFAULT config path (~/.decadic/vast.json) -- the operator's REAL credentials.
These tests therefore assemble a minimal FastAPI app with a tmp_path-backed
store, so they can never read or clobber a real key. No network: the CLI seam
is faked wherever a route would shell out.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from decadic.api.vast.controller import VastController
from decadic.api.vast.routes import register_vast_routes
from decadic.api.vast.settings_store import VastSettingsStore


@pytest.fixture
def vast_app(tmp_path):
    app = FastAPI()
    store = VastSettingsStore(tmp_path / "vast.json")
    app.state.vast_settings = store
    app.state.vast_controller = VastController(store, log_dir=tmp_path / "logs")
    register_vast_routes(app)
    return app


def test_preflight_reports_named_checks_and_reasons(vast_app):
    with TestClient(vast_app) as client:
        r = client.get("/vast/preflight")
        assert r.status_code == 200
        pf = r.json()
        for k in (
            "api_key",
            "cli",
            "ssh_binary",
            "ssh_key_present",
            "ssh_key_registered",
            "ready",
            "reasons",
        ):
            assert k in pf, k
        # Isolated store: no API key -> not ready, with the named reason.
        assert pf["api_key"] is False
        assert pf["ready"] is False
        assert any("API key" in reason for reason in pf["reasons"])


def test_preflight_ready_requires_all_gates(vast_app):
    with TestClient(vast_app) as client:
        pf = client.get("/vast/preflight").json()
        gates = ("api_key", "cli", "ssh_binary", "ssh_key_present", "ssh_key_registered")
        assert pf["ready"] == all(pf[g] for g in gates)


def test_preflight_all_green_when_every_gate_passes(vast_app, tmp_path, monkeypatch):
    store = vast_app.state.vast_settings
    store.set_api_key("test-key")
    priv = tmp_path / "id_ed25519"
    priv.write_text("PRIVATE", encoding="utf-8")
    (tmp_path / "id_ed25519.pub").write_text(
        "ssh-ed25519 AAAATESTBODY decadic-vast\n", encoding="utf-8"
    )
    store.set_ssh_key_path(str(priv))

    ctrl = vast_app.state.vast_controller
    monkeypatch.setattr(ctrl.cli, "available", lambda: True)

    async def fake_show_ssh_keys():
        # Vast echoes the key with a different trailing comment; must still match.
        return [{"id": 1, "public_key": "ssh-ed25519 AAAATESTBODY other"}]

    monkeypatch.setattr(ctrl.cli, "show_ssh_keys", fake_show_ssh_keys)
    monkeypatch.setattr("decadic.api.vast.ssh_keys.which_ssh", lambda: "/usr/bin/ssh")
    monkeypatch.setattr(
        "decadic.api.vast.ssh_keys.which_keygen", lambda: "/usr/bin/ssh-keygen"
    )

    with TestClient(vast_app) as client:
        pf = client.get("/vast/preflight").json()
        assert pf["ready"] is True
        assert pf["reasons"] == []


def test_ssh_setup_requires_api_key(vast_app):
    with TestClient(vast_app) as client:
        r = client.post("/vast/ssh/setup")
        assert r.status_code == 400
        assert "API key" in r.json()["detail"]


def test_ssh_setup_never_returns_private_material(vast_app, tmp_path, monkeypatch):
    """Even on success, the response carries fingerprint + masked path only."""
    store = vast_app.state.vast_settings
    store.set_api_key("test-key")
    priv = tmp_path / "id_ed25519"
    priv.write_text("PRIVATE-KEY-MATERIAL", encoding="utf-8")
    (tmp_path / "id_ed25519.pub").write_text(
        "ssh-ed25519 AAAATESTBODY decadic-vast\n", encoding="utf-8"
    )
    store.set_ssh_key_path(str(priv))

    ctrl = vast_app.state.vast_controller
    monkeypatch.setattr(ctrl.cli, "available", lambda: True)

    async def fake_show_ssh_keys():
        return [{"id": 1, "public_key": "ssh-ed25519 AAAATESTBODY other-comment"}]

    monkeypatch.setattr(ctrl.cli, "show_ssh_keys", fake_show_ssh_keys)

    with TestClient(vast_app) as client:
        r = client.post("/vast/ssh/setup")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["registered"] is True
        assert "PRIVATE-KEY-MATERIAL" not in r.text
        assert str(priv) not in r.text  # raw path (with username) never leaves
        assert body["ssh_key_path_masked"].startswith(".../")
