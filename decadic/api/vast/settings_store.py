"""Persistent, permission-restricted store for the Vast.ai API key + defaults.

The key is written to a JSON file outside the repo (``DECADIC_CONFIG_DIR`` or
``~/.decadic/vast.json``) with 0600 permissions, is never logged, and is masked
on read (only a short suffix + a "set" flag are returned to the UI). Deploy
defaults (GPU filter, disk, image, preset, scene, encoder) live alongside it so
the dashboard remembers the operator's last choices.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Defaults the dashboard seeds its Deploy form with. Mirrors the plan + the
# launcher env (full preset, hf encoders) and the docs' search example.
DEFAULT_DEFAULTS: dict[str, Any] = {
    "gpu_name": "RTX_4090",
    "num_gpus": 1,
    "max_dph": 1.0,           # max $/hr to surface in the offer search
    "min_gpu_ram": 16,        # GB
    "verified": True,
    "disk": 40,               # GB
    "image": "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
    "preset": "full",         # tiny | medium | full
    "encoder": "hf",          # hf | zeros
    "whisper_model": "openai/whisper-small",
    "scene": "bear",          # none | bear | food | <element csv>
}


def _config_dir() -> Path:
    raw = os.environ.get("DECADIC_CONFIG_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".decadic"


def _config_path() -> Path:
    return _config_dir() / "vast.json"


def _mask(key: str) -> str:
    """Return a non-reversible hint of the key: ``...`` + last 4 chars."""
    k = (key or "").strip()
    if not k:
        return ""
    if len(k) <= 4:
        return "*" * len(k)
    return f"...{k[-4:]}"


class VastSettingsStore:
    """Thread-safe JSON-backed store for the Vast key + deploy defaults."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _config_path()
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"api_key": "", "ssh_key_path": "", "defaults": {}}
        self._load()

    # --- persistence -------------------------------------------------------
    def _load(self) -> None:
        try:
            if self._path.is_file():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data.update(
                        {
                            "api_key": str(raw.get("api_key", "") or ""),
                            "ssh_key_path": str(raw.get("ssh_key_path", "") or ""),
                            "defaults": dict(raw.get("defaults", {}) or {}),
                        }
                    )
        except Exception:
            # Never let a corrupt config file crash the server; start fresh.
            logger.exception("vast_settings_load_failed path=%s", self._path)

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                # Best-effort on platforms without POSIX perms (Windows).
                pass
        except Exception:
            logger.exception("vast_settings_persist_failed path=%s", self._path)
            raise

    # --- key ---------------------------------------------------------------
    def get_api_key(self) -> str:
        with self._lock:
            return self._data.get("api_key", "") or ""

    def set_api_key(self, key: str) -> None:
        with self._lock:
            self._data["api_key"] = (key or "").strip()
            self._persist()

    def clear_api_key(self) -> None:
        with self._lock:
            self._data["api_key"] = ""
            self._persist()

    def has_api_key(self) -> bool:
        return bool(self.get_api_key())

    # --- ssh key path ------------------------------------------------------
    def get_ssh_key_path(self) -> str:
        with self._lock:
            return self._data.get("ssh_key_path", "") or ""

    def set_ssh_key_path(self, path: str) -> None:
        with self._lock:
            self._data["ssh_key_path"] = (path or "").strip()
            self._persist()

    # --- defaults ----------------------------------------------------------
    def get_defaults(self) -> dict[str, Any]:
        with self._lock:
            merged = dict(DEFAULT_DEFAULTS)
            merged.update(self._data.get("defaults", {}) or {})
            return merged

    def set_defaults(self, partial: dict[str, Any]) -> dict[str, Any]:
        """Merge ``partial`` (only known keys) into the stored defaults."""
        with self._lock:
            current = dict(self._data.get("defaults", {}) or {})
            for k, v in (partial or {}).items():
                if k in DEFAULT_DEFAULTS and v is not None:
                    current[k] = v
            self._data["defaults"] = current
            self._persist()
            merged = dict(DEFAULT_DEFAULTS)
            merged.update(current)
            return merged

    # --- UI-safe view ------------------------------------------------------
    def public_view(self) -> dict[str, Any]:
        """Masked snapshot safe to return over the API (no raw key)."""
        with self._lock:
            key = self._data.get("api_key", "") or ""
            return {
                "has_api_key": bool(key),
                "api_key_masked": _mask(key),
                "ssh_key_path": self._data.get("ssh_key_path", "") or "",
                "defaults": {**DEFAULT_DEFAULTS, **(self._data.get("defaults", {}) or {})},
                "config_path": str(self._path),
            }
