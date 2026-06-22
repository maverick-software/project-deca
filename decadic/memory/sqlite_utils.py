"""SQLite persistence helpers shared by memory stores."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from decadic import config as C


def configure_connection(conn: sqlite3.Connection) -> None:
    """Apply production-safe SQLite durability pragmas to a connection."""
    conn.execute(f"PRAGMA busy_timeout={5000}")
    if C.sqlite_wal_enabled():
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA wal_autocheckpoint={C.sqlite_wal_autocheckpoint()}")
    conn.execute(f"PRAGMA synchronous={C.sqlite_synchronous()}")


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    configure_connection(conn)
    return conn


def encode_vector_blob(values: Any) -> bytes | None:
    if values is None or not C.sqlite_vector_blob_enabled():
        return None
    arr = np.asarray(values, dtype="<f4").reshape(-1)
    if arr.size == 0:
        return None
    return arr.tobytes(order="C")


def decode_vector_blob(blob: bytes | memoryview | None) -> np.ndarray | None:
    if blob is None:
        return None
    raw = bytes(blob)
    if not raw or len(raw) % 4 != 0:
        return None
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)


def wal_checkpoint_truncate(conn: sqlite3.Connection) -> float:
    started = time.perf_counter()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return (time.perf_counter() - started) * 1000.0


def db_file_sizes(path: Path | None) -> dict[str, int]:
    if path is None:
        return {"memory_db_bytes": 0, "memory_wal_bytes": 0}
    db = Path(path)
    wal = Path(str(db) + "-wal")
    return {
        "memory_db_bytes": db.stat().st_size if db.exists() else 0,
        "memory_wal_bytes": wal.stat().st_size if wal.exists() else 0,
    }
