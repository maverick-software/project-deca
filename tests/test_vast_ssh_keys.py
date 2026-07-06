"""SSH key lifecycle (vast_connect M1-M2): generate, register idempotently,
attach per-instance, and the deploy step order that makes SSH the FIRST act.

All Vast CLI calls are faked at the VastCli method seam (no network, no real
``vastai``); key generation uses a fake ``ssh-keygen`` writer so the suite
never touches the operator's real ~/.decadic. The one real-tool test
(``test_real_keygen_roundtrip``) is skipped when OpenSSH isn't installed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from decadic.api.vast.cli import VastCliError
from decadic.api.vast.settings_store import VastSettingsStore
from decadic.api.vast.ssh_keys import SshKeyManager, _normalize_pub

PUB = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleKeyBody decadic-vast"


class FakeCli:
    """Records calls; behavior driven by the `registered` list contents."""

    def __init__(self, registered: list[str] | None = None) -> None:
        self.registered = list(registered or [])
        self.create_calls: list[str] = []
        self.attach_calls: list[tuple[str, str]] = []
        self.fail_attach = False

    async def show_ssh_keys(self):
        return [{"id": i, "public_key": k} for i, k in enumerate(self.registered)]

    async def create_ssh_key(self, public_key: str) -> None:
        self.create_calls.append(public_key)
        self.registered.append(public_key)

    async def attach_ssh(self, instance_id, public_key: str) -> None:
        if self.fail_attach:
            raise VastCliError("duplicate key")
        self.attach_calls.append((str(instance_id), public_key))


def _mgr(tmp_path, *, key: bool = True, registered: bool = False):
    store = VastSettingsStore(tmp_path / "vast.json")
    if key:
        priv = tmp_path / "id_ed25519"
        priv.write_text("PRIVATE", encoding="utf-8")
        Path(str(priv) + ".pub").write_text(PUB + "\n", encoding="utf-8")
        store.set_ssh_key_path(str(priv))
    cli = FakeCli([PUB] if registered else [])
    return SshKeyManager(store, cli), cli, store


def test_normalize_pub_ignores_comment_and_whitespace():
    a = "ssh-ed25519 KEYBODY user@desktop"
    b = "ssh-ed25519  KEYBODY   vast-echoed-comment"
    assert _normalize_pub(a) == _normalize_pub(b) == "ssh-ed25519 KEYBODY"


def test_is_registered_matches_on_key_body_not_comment(tmp_path):
    mgr, cli, _ = _mgr(tmp_path)
    # Vast echoes the key back with a DIFFERENT comment; must still match.
    cli.registered = [PUB.rsplit(" ", 1)[0] + " some-other-comment"]
    assert asyncio.run(mgr.is_registered()) is True


def test_ensure_registered_registers_once_then_noop(tmp_path):
    mgr, cli, _ = _mgr(tmp_path, registered=False)
    r1 = asyncio.run(mgr.ensure_registered())
    assert r1["registered"] is True
    assert len(cli.create_calls) == 1  # registered exactly once
    r2 = asyncio.run(mgr.ensure_registered())
    assert r2["registered"] is True
    assert len(cli.create_calls) == 1  # second call is a no-op


def test_ensure_local_key_reuses_existing(tmp_path):
    mgr, _, store = _mgr(tmp_path)
    before = store.get_ssh_key_path()
    p = asyncio.run(mgr.ensure_local_key())
    assert str(p) == before  # no regeneration, no path churn


def test_ensure_local_key_missing_pub_is_actionable(tmp_path):
    mgr, _, store = _mgr(tmp_path)
    Path(store.get_ssh_key_path() + ".pub").unlink()
    with pytest.raises(VastCliError, match="no matching .pub"):
        asyncio.run(mgr.ensure_local_key())


def test_attach_failure_is_swallowed(tmp_path):
    """Per-instance attach is best-effort: a duplicate-key rejection must not
    fail a deploy (account registration already covers auth)."""
    mgr, cli, _ = _mgr(tmp_path, registered=True)
    cli.fail_attach = True
    asyncio.run(mgr.attach_to_instance(123))  # must not raise


def test_attach_passes_id_and_pub(tmp_path):
    mgr, cli, _ = _mgr(tmp_path, registered=True)
    asyncio.run(mgr.attach_to_instance(456))
    assert cli.attach_calls == [("456", PUB)]


def test_preflight_reports_and_reasons(tmp_path, monkeypatch):
    mgr, cli, _ = _mgr(tmp_path, registered=False)
    monkeypatch.setattr("decadic.api.vast.ssh_keys.which_ssh", lambda: "/usr/bin/ssh")
    monkeypatch.setattr("decadic.api.vast.ssh_keys.which_keygen", lambda: "/usr/bin/ssh-keygen")
    pf = asyncio.run(mgr.preflight())
    assert pf["ssh_key_present"] is True
    assert pf["ssh_key_registered"] is False
    assert any("not registered" in r for r in pf["reasons"])
    cli.registered = [PUB]
    pf2 = asyncio.run(mgr.preflight())
    assert pf2["ssh_key_registered"] is True
    assert pf2["reasons"] == []


def test_preflight_no_key(tmp_path, monkeypatch):
    mgr, _, _ = _mgr(tmp_path, key=False)
    monkeypatch.setattr("decadic.api.vast.ssh_keys.which_ssh", lambda: None)
    monkeypatch.setattr("decadic.api.vast.ssh_keys.which_keygen", lambda: None)
    pf = asyncio.run(mgr.preflight())
    assert pf["ssh_binary"] is False
    assert pf["ssh_key_present"] is False
    assert pf["ssh_key_registered"] is False
    assert len(pf["reasons"]) >= 2  # missing tools AND missing key, each named


# --- M2: deploy wiring --------------------------------------------------------


def _deploy_req():
    from decadic.api.vast.controller import DeployRequest

    return DeployRequest(
        offer_id=1,
        image="pytorch/pytorch:test",
        disk=25,
        preset="tiny",
        encoder="zeros",
        whisper_model="w",
        scene="none",
    )


def test_provision_runs_ensure_ssh_before_create(tmp_path):
    """The step ORDER is the contract: SSH must be green before money moves."""
    from decadic.api.vast.controller import VastController

    store = VastSettingsStore(tmp_path / "vast.json")
    store.set_api_key("k")
    ctrl = VastController(store, log_dir=tmp_path / "logs")
    calls: list[str] = []

    async def fake_ensure_ssh():
        calls.append("ensure_ssh")

    async def fake_create(req):
        calls.append("create")
        raise VastCliError("stop here")  # halt the pipeline after the two steps

    ctrl._step_ensure_ssh = fake_ensure_ssh
    ctrl._step_create = fake_create
    asyncio.run(ctrl._provision(_deploy_req()))
    assert calls == ["ensure_ssh", "create"]
    assert ctrl._state.phase == "error"  # surfaced, not swallowed


def test_step_create_attaches_key_to_new_instance(tmp_path):
    from decadic.api.vast.controller import VastController

    store = VastSettingsStore(tmp_path / "vast.json")
    store.set_api_key("k")
    ctrl = VastController(store, log_dir=tmp_path / "logs")
    ctrl._ssh_pub = PUB
    attached: list[tuple[str, str]] = []

    async def fake_create_instance(offer_id, *, image, disk, ssh, direct):
        return 777

    async def fake_attach(iid, pub):
        attached.append((str(iid), pub))

    ctrl._cli.create_instance = fake_create_instance
    ctrl._cli.attach_ssh = fake_attach
    asyncio.run(ctrl._step_create(_deploy_req()))
    assert ctrl._state.instance_id == 777
    assert attached == [("777", PUB)]


def test_step_create_attach_failure_does_not_fail_deploy(tmp_path):
    from decadic.api.vast.controller import VastController

    store = VastSettingsStore(tmp_path / "vast.json")
    store.set_api_key("k")
    ctrl = VastController(store, log_dir=tmp_path / "logs")
    ctrl._ssh_pub = PUB

    async def fake_create_instance(offer_id, *, image, disk, ssh, direct):
        return 888

    async def fake_attach(iid, pub):
        raise VastCliError("already attached")

    ctrl._cli.create_instance = fake_create_instance
    ctrl._cli.attach_ssh = fake_attach
    asyncio.run(ctrl._step_create(_deploy_req()))  # must not raise
    assert ctrl._state.instance_id == 888


# --- optional real-tool roundtrip ---------------------------------------------


def test_real_keygen_roundtrip(tmp_path, monkeypatch):
    """With real OpenSSH present: generate into an isolated dir, verify both
    halves + store path + fingerprint. Skipped where OpenSSH is absent."""
    import shutil

    if shutil.which("ssh-keygen") is None:
        pytest.skip("OpenSSH not installed")
    import decadic.api.vast.ssh_keys as sk

    monkeypatch.setattr(sk, "DEFAULT_KEY_DIR", tmp_path / "keys")
    store = VastSettingsStore(tmp_path / "vast.json")
    mgr = SshKeyManager(store, FakeCli())
    p = asyncio.run(mgr.ensure_local_key())
    assert p.is_file() and Path(str(p) + ".pub").is_file()
    assert store.get_ssh_key_path() == str(p)
    assert (asyncio.run(mgr.fingerprint()) or "").startswith("SHA256:")
    # Reuse on second call (no regeneration).
    assert asyncio.run(mgr.ensure_local_key()) == p
