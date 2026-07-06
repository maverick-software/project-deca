"""SSH key lifecycle for Vast deployments (gap G1, vast_connect_gap_analysis).

The deploy pipeline authenticates every provisioning step (upload, install,
serve, tunnel) over SSH, and Vast only authorizes keys that are registered on
the account (or attached to the instance). The settings store has always held a
*private key path*; this module supplies the missing half of the lifecycle:

  * ``ensure_local_key()``   — generate an ed25519 keypair if none is configured
    (``~/.decadic/ssh/id_ed25519``, 0600 where the OS supports it) and persist
    the path in the settings store. Idempotent: an existing key is reused.
  * ``ensure_registered()``  — register the public key with the Vast account
    (``vastai create ssh-key``), reconciling against ``show ssh-keys`` first so
    repeat calls are no-ops.
  * ``attach_to_instance()`` — per-instance ``vastai attach ssh`` right after
    create, the reliable guarantee independent of account-key propagation lag.

Security: the private key never leaves the machine, is never logged, and no
public API in this module returns it — callers get paths and fingerprints only.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from decadic.api.vast.cli import VastCli, VastCliError
from decadic.api.vast.settings_store import VastSettingsStore

logger = logging.getLogger(__name__)

DEFAULT_KEY_DIR = Path.home() / ".decadic" / "ssh"
DEFAULT_KEY_NAME = "id_ed25519"


def which_ssh() -> str | None:
    """Path to the ``ssh`` client, or None (Windows: the OpenSSH optional
    feature provides it; absence is a preflight failure, not an exception)."""
    return shutil.which("ssh")


def which_keygen() -> str | None:
    return shutil.which("ssh-keygen")


def _normalize_pub(key: str) -> str:
    """Comparable form of an OpenSSH public key: ``type base64body`` only.

    The trailing comment (user@host) differs between where a key was generated
    and how Vast echoes it back, so reconciliation must ignore it — matching on
    the raw string would re-register the same key forever."""
    parts = (key or "").strip().split()
    return " ".join(parts[:2]) if len(parts) >= 2 else (key or "").strip()


class SshKeyManager:
    """Generate / register / attach the deploy SSH key. All methods idempotent."""

    def __init__(self, store: VastSettingsStore, cli: VastCli) -> None:
        self._store = store
        self._cli = cli

    # -- local key ------------------------------------------------------------
    def key_path(self) -> Path | None:
        raw = self._store.get_ssh_key_path()
        return Path(raw).expanduser() if raw else None

    def key_exists(self) -> bool:
        p = self.key_path()
        return bool(p and p.is_file())

    def public_key(self) -> str | None:
        p = self.key_path()
        if not p:
            return None
        pub = Path(str(p) + ".pub")
        try:
            return pub.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    async def fingerprint(self) -> str | None:
        """``ssh-keygen -lf <pub>`` fingerprint (safe for UI display)."""
        p = self.key_path()
        keygen = which_keygen()
        if not p or not keygen:
            return None
        pub = Path(str(p) + ".pub")
        if not pub.is_file():
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                keygen, "-lf", str(pub),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            if proc.returncode == 0:
                # "256 SHA256:xxxx comment (ED25519)" -> the SHA256 token
                for tok in out_b.decode("utf-8", "replace").split():
                    if tok.startswith("SHA256:"):
                        return tok
        except (OSError, asyncio.TimeoutError):
            pass
        return None

    async def ensure_local_key(self) -> Path:
        """Reuse the configured key, or generate a fresh ed25519 pair.

        Never overwrites an existing file; never returns without both halves
        present."""
        p = self.key_path()
        if p and p.is_file():
            if not Path(str(p) + ".pub").is_file():
                raise VastCliError(
                    f"configured SSH key {p.name} has no matching .pub next to it; "
                    "restore the public half or clear the key path and re-run setup"
                )
            return p
        keygen = which_keygen()
        if not keygen:
            raise VastCliError(
                "ssh-keygen not found. Install the OpenSSH client "
                "(Windows: Settings > Optional features > OpenSSH Client)."
            )
        DEFAULT_KEY_DIR.mkdir(parents=True, exist_ok=True)
        target = DEFAULT_KEY_DIR / DEFAULT_KEY_NAME
        if not target.is_file():
            proc = await asyncio.create_subprocess_exec(
                keygen, "-t", "ed25519", "-N", "", "-C", "decadic-vast",
                "-f", str(target),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            if proc.returncode != 0 or not target.is_file():
                raise VastCliError(
                    "ssh-keygen failed: "
                    + (err_b or out_b).decode("utf-8", "replace").strip()[-300:]
                )
            try:
                os.chmod(target, 0o600)  # POSIX; effectively a no-op on Windows
            except OSError:
                pass
            logger.info("generated deploy SSH key at %s", target)
        self._store.set_ssh_key_path(str(target))
        return target

    # -- Vast registration ------------------------------------------------------
    async def is_registered(self) -> bool:
        pub = self.public_key()
        if not pub:
            return False
        mine = _normalize_pub(pub)
        try:
            keys = await self._cli.show_ssh_keys()
        except VastCliError:
            return False
        return any(
            _normalize_pub(str(k.get("public_key", ""))) == mine for k in keys
        )

    async def ensure_registered(self) -> dict[str, Any]:
        """Idempotent: register the public key with the Vast account."""
        await self.ensure_local_key()
        pub = self.public_key()
        if not pub:
            raise VastCliError("no public key available after ensure_local_key")
        if not await self.is_registered():
            await self._cli.create_ssh_key(pub)
            logger.info("registered deploy SSH key with vast account")
        return {"registered": True, "fingerprint": await self.fingerprint()}

    async def attach_to_instance(self, instance_id: int | str) -> None:
        """Best-effort per-instance attach (guarantees auth even if the
        account-level registration hasn't propagated to the new box)."""
        pub = self.public_key()
        if not pub:
            return
        try:
            await self._cli.attach_ssh(instance_id, pub)
        except VastCliError as exc:
            # Vast rejects duplicates on some builds; account-level registration
            # still covers us, so log and continue rather than failing a deploy.
            logger.warning("attach ssh to instance %s failed: %s", instance_id, exc)

    # -- preflight ---------------------------------------------------------------
    async def preflight(self) -> dict[str, Any]:
        """The SSH portion of GET /vast/preflight (pure reads, no mutation)."""
        checks: dict[str, bool] = {
            "ssh_binary": which_ssh() is not None,
            "keygen_binary": which_keygen() is not None,
            "ssh_key_present": self.key_exists(),
        }
        checks["ssh_key_registered"] = (
            await self.is_registered() if checks["ssh_key_present"] else False
        )
        reasons: list[str] = []
        if not checks["ssh_binary"]:
            reasons.append(
                "ssh client not found — install the OpenSSH Client "
                "(Windows: Settings > Optional features)"
            )
        if not checks["keygen_binary"]:
            reasons.append("ssh-keygen not found — comes with the OpenSSH Client")
        if not checks["ssh_key_present"]:
            reasons.append("no SSH key configured — run SSH setup (POST /vast/ssh/setup)")
        elif not checks["ssh_key_registered"]:
            reasons.append(
                "SSH key not registered with your Vast account — run SSH setup "
                "(POST /vast/ssh/setup)"
            )
        return {**checks, "reasons": reasons}
