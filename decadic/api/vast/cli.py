"""Async wrappers around the ``vastai`` CLI + ssh exec/tunnel helpers.

All external process invocation for the Vast control plane lives here so the
rest of the package never shells out directly (and so a future swap to the REST
API is localized). The API key is injected per call via ``--api-key`` and is
never written to the logs (argv is redacted before logging).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import shutil
import sys
import sysconfig
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Run the vastai console-script entry point through the server's own
# interpreter. PATH-independent: works whenever the ``vastai`` package is
# importable (e.g. a user-site install whose Scripts dir is not on PATH).
_VASTAI_ENTRYPOINT_SHIM = "import sys; from vastai.cli.main import main; sys.exit(main())"


class VastCliError(RuntimeError):
    """A ``vastai``/``ssh`` invocation failed or returned unparseable output."""


def _redact(argv: list[str]) -> list[str]:
    """Copy of argv with any value following --api-key masked."""
    out: list[str] = []
    skip = False
    for tok in argv:
        if skip:
            out.append("***")
            skip = False
            continue
        out.append(tok)
        if tok == "--api-key":
            skip = True
    return out


def _vastai_module_available() -> bool:
    try:
        return importlib.util.find_spec("vastai") is not None
    except (ImportError, ValueError):
        return False


def _script_dirs() -> list[str]:
    """Interpreter-adjacent dirs that may hold the vastai console script."""
    candidates = [
        os.path.dirname(sys.executable),
        os.path.join(os.path.dirname(sys.executable), "Scripts"),
    ]
    for key in ("scripts", None):
        try:
            candidates.append(
                sysconfig.get_path("scripts")
                if key
                else sysconfig.get_path("scripts", os.name + "_user")
            )
        except (KeyError, OSError):
            pass
    seen: set[str] = set()
    out: list[str] = []
    for d in candidates:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _find_vastai_script() -> str | None:
    names = ("vastai.exe", "vastai") if os.name == "nt" else ("vastai",)
    for d in _script_dirs():
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    return None


def resolve_vastai_bin() -> list[str]:
    """Locate the vastai CLI as an argv prefix, independent of PATH.

    Order: explicit ``DECADIC_VASTAI_BIN`` override -> ``vastai`` on PATH -> the
    console script in any interpreter-adjacent scripts dir -> the entry point run
    via the server's own interpreter (the robust fallback; the package ships as a
    project dependency, so it is importable even when its Scripts dir is off
    PATH, as with a user-site install). ``python -m vastai`` is NOT used: the
    package has no ``__main__``.
    """
    override = os.environ.get("DECADIC_VASTAI_BIN", "").strip()
    if override:
        return [override]
    found = shutil.which("vastai")
    if found:
        return [found]
    if _vastai_module_available():
        return [sys.executable, "-c", _VASTAI_ENTRYPOINT_SHIM]
    script = _find_vastai_script()
    if script:
        return [script]
    return ["vastai"]


class VastCli:
    """Thin async wrapper; holds a getter for the (live) API key."""

    def __init__(self, get_api_key: Callable[[], str]) -> None:
        self._get_api_key = get_api_key
        self._bin = resolve_vastai_bin()

    # --- availability ------------------------------------------------------
    def available(self) -> bool:
        override = os.environ.get("DECADIC_VASTAI_BIN", "").strip()
        if override:
            return bool(shutil.which(override) or os.path.isfile(override))
        if shutil.which("vastai") is not None:
            return True
        if _vastai_module_available():
            return True
        return _find_vastai_script() is not None

    # --- low-level runner --------------------------------------------------
    async def _run(self, args: list[str], *, timeout: float = 120.0) -> tuple[int, str, str]:
        argv = [*self._bin, *args]
        logger.info("vast_cli_run argv=%s", _redact(argv))
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise VastCliError(
                "vastai CLI not found. Install it with: pip install vastai"
            ) from exc
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            raise VastCliError(f"vastai timed out after {timeout:.0f}s: {args[:2]}") from exc
        return proc.returncode or 0, out_b.decode("utf-8", "replace"), err_b.decode("utf-8", "replace")

    async def _run_json(self, args: list[str], *, timeout: float = 120.0) -> Any:
        code, out, err = await self._run([*args, "--raw"], timeout=timeout)
        if code != 0:
            raise VastCliError(f"vastai {args[:2]} failed (exit {code}): {err.strip() or out.strip()}")
        text = out.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Some commands print a non-JSON line then JSON; try the last brace block.
            start = text.find("[")
            sb = text.find("{")
            if sb >= 0 and (start < 0 or sb < start):
                start = sb
            if start >= 0:
                try:
                    return json.loads(text[start:])
                except json.JSONDecodeError:
                    pass
            raise VastCliError(f"vastai {args[:2]} returned non-JSON output")

    def _key_args(self) -> list[str]:
        key = (self._get_api_key() or "").strip()
        return ["--api-key", key] if key else []

    # --- account -----------------------------------------------------------
    async def show_user(self) -> dict[str, Any]:
        data = await self._run_json(["show", "user", *self._key_args()], timeout=30.0)
        return data if isinstance(data, dict) else {}

    # --- offers ------------------------------------------------------------
    async def search_offers(
        self,
        query: str,
        order: str = "dlperf_usd-",
        *,
        no_default: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        args = ["search", "offers", query, "-o", order]
        if no_default:
            args.append("-n")  # ignore the default external=false rentable/verified filters
        if limit:
            args += ["--limit", str(int(limit))]
        args += self._key_args()
        data = await self._run_json(args, timeout=90.0)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict) and isinstance(data.get("offers"), list):
            return [d for d in data["offers"] if isinstance(d, dict)]
        return []

    # --- instance lifecycle ------------------------------------------------
    async def create_instance(
        self,
        offer_id: int | str,
        *,
        image: str,
        disk: int,
        onstart_cmd: str | None = None,
        ssh: bool = True,
        direct: bool = True,
    ) -> int:
        args = ["create", "instance", str(offer_id), "--image", image, "--disk", str(disk)]
        if ssh:
            args.append("--ssh")
        if direct:
            args.append("--direct")
        if onstart_cmd:
            args += ["--onstart-cmd", onstart_cmd]
        args += self._key_args()
        data = await self._run_json(args, timeout=120.0)
        if not isinstance(data, dict) or not data.get("success"):
            raise VastCliError(f"create instance failed: {data}")
        new_id = data.get("new_contract")
        if new_id is None:
            raise VastCliError(f"create instance returned no contract id: {data}")
        return int(new_id)

    async def show_instance(self, instance_id: int | str) -> dict[str, Any]:
        data = await self._run_json(
            ["show", "instance", str(instance_id), *self._key_args()], timeout=45.0
        )
        if isinstance(data, list):
            return data[0] if data and isinstance(data[0], dict) else {}
        return data if isinstance(data, dict) else {}

    async def ssh_url(self, instance_id: int | str) -> str:
        code, out, err = await self._run(
            ["ssh-url", str(instance_id), *self._key_args()], timeout=45.0
        )
        if code != 0:
            raise VastCliError(f"ssh-url failed (exit {code}): {err.strip() or out.strip()}")
        url = out.strip().splitlines()[-1].strip() if out.strip() else ""
        if not url.startswith("ssh://"):
            raise VastCliError(f"ssh-url returned unexpected value: {url!r}")
        return url

    async def copy(self, src: str, dst: str, *, timeout: float = 600.0) -> None:
        code, out, err = await self._run(["copy", src, dst, *self._key_args()], timeout=timeout)
        if code != 0:
            raise VastCliError(f"copy {src} -> {dst} failed (exit {code}): {err.strip() or out.strip()}")

    async def destroy_instance(self, instance_id: int | str) -> None:
        code, out, err = await self._run(
            ["destroy", "instance", str(instance_id), *self._key_args()], timeout=60.0
        )
        if code != 0:
            raise VastCliError(f"destroy failed (exit {code}): {err.strip() or out.strip()}")

    async def stop_instance(self, instance_id: int | str) -> None:
        code, out, err = await self._run(
            ["stop", "instance", str(instance_id), *self._key_args()], timeout=60.0
        )
        if code != 0:
            raise VastCliError(f"stop failed (exit {code}): {err.strip() or out.strip()}")


# --- ssh / scp helpers (data plane) ---------------------------------------

def parse_ssh_url(url: str) -> tuple[str, int]:
    """``ssh://root@host:port`` -> (host, port)."""
    rest = url[len("ssh://"):] if url.startswith("ssh://") else url
    if "@" in rest:
        rest = rest.split("@", 1)[1]
    host, _, port = rest.partition(":")
    return host, int(port or "22")


def _ssh_base(host: str, port: int, key_path: str | None) -> list[str]:
    args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=20",
        "-p", str(port),
    ]
    if key_path:
        args += ["-i", os.path.expanduser(key_path)]
    args.append(f"root@{host}")
    return args


async def ssh_exec(
    host: str,
    port: int,
    command: str,
    *,
    key_path: str | None = None,
    timeout: float = 900.0,
) -> tuple[int, str, str]:
    """Run a remote shell command; returns (exit_code, stdout, stderr)."""
    argv = [*_ssh_base(host, port, key_path), command]
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        raise VastCliError(f"ssh command timed out after {timeout:.0f}s") from exc
    return proc.returncode or 0, out_b.decode("utf-8", "replace"), err_b.decode("utf-8", "replace")


async def open_tunnel(
    host: str,
    port: int,
    local_port: int,
    remote_port: int,
    *,
    key_path: str | None = None,
) -> asyncio.subprocess.Process:
    """Start a background ``ssh -N -L`` local-forward tunnel; returns the process.

    The caller owns the returned process and must terminate it on teardown.
    """
    argv = [
        "ssh",
        "-N",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-p", str(port),
        "-L", f"{local_port}:localhost:{remote_port}",
    ]
    if key_path:
        argv += ["-i", os.path.expanduser(key_path)]
    argv.append(f"root@{host}")
    logger.info("vast_open_tunnel local=%s remote=%s host=%s", local_port, remote_port, host)
    return await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
