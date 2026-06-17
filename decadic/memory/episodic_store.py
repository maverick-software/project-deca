"""Episodic memory: SQLite-backed cycle log + embedding similarity search."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from decadic.memory.embeddings import EMBEDDING_DIM


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class EpisodicRecord:
    cycle_index: int
    summary: dict[str, Any]
    salience: float
    embedding: list[float] | None = None


class EpisodicStore:
    """Thread-safe SQLite store for per-cycle summaries and optional embeddings."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._memory_rows: list[dict[str, Any]] = []
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_index INTEGER NOT NULL,
                    salience REAL NOT NULL,
                    summary_json TEXT NOT NULL,
                    embedding_json TEXT
                )
                """
            )
            self._conn.commit()
            self._ensure_embedding_column()

    def _ensure_embedding_column(self) -> None:
        if self._conn is None:
            return
        cur = self._conn.execute("PRAGMA table_info(episodes)")
        cols = {row[1] for row in cur.fetchall()}
        if "embedding_json" not in cols:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN embedding_json TEXT")
            self._conn.commit()

    def append(self, record: EpisodicRecord) -> None:
        emb_json: str | None
        if record.embedding is not None:
            emb_json = json.dumps(record.embedding, default=float)
        else:
            emb_json = None

        if self._conn is None:
            row = {
                "cycle_index": record.cycle_index,
                "salience": record.salience,
                "summary": record.summary,
            }
            if record.embedding is not None:
                row["embedding"] = list(record.embedding)
            with self._lock:
                self._memory_rows.append(row)
            return

        payload = json.dumps(record.summary, default=str)
        with self._lock:
            self._conn.execute(
                "INSERT INTO episodes (cycle_index, salience, summary_json, embedding_json) "
                "VALUES (?, ?, ?, ?)",
                (record.cycle_index, record.salience, payload, emb_json),
            )
            self._conn.commit()

    def clear(self) -> None:
        """Wipe all stored episodes (agent reset)."""
        with self._lock:
            if self._conn is None:
                self._memory_rows.clear()
            else:
                self._conn.execute("DELETE FROM episodes")
                self._conn.commit()

    def backup_to(self, path: Path) -> None:
        """Copy the live episode DB to ``path`` via SQLite's online-backup API.

        Safe to call on a running agent: ``Connection.backup`` takes a
        consistent snapshot without a long writer stall (and works on Windows
        where copying an open DB file would fail). An in-memory store (no
        sqlite file) is materialized into a fresh on-disk DB so saves still
        capture its rows.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        dest = sqlite3.connect(str(path))
        try:
            with self._lock:
                if self._conn is not None:
                    self._conn.backup(dest)
                else:
                    self._dump_memory_rows(dest)
            dest.commit()
        finally:
            dest.close()

    def restore_from(self, path: Path) -> None:
        """Replace live episodes with the SQLite file at ``path`` (no-op if missing).

        Used when loading a saved agent: the freshly reset store is overwritten
        by the saved memory snapshot.
        """
        src_path = Path(path)
        if not src_path.is_file():
            return
        src = sqlite3.connect(str(src_path))
        try:
            with self._lock:
                if self._conn is not None:
                    src.backup(self._conn)
                    self._conn.commit()
                else:
                    self._load_memory_rows(src)
        finally:
            src.close()

    def _dump_memory_rows(self, dest: sqlite3.Connection) -> None:
        """Write the in-memory row buffer into ``dest`` (in-memory store only)."""
        dest.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_index INTEGER NOT NULL,
                salience REAL NOT NULL,
                summary_json TEXT NOT NULL,
                embedding_json TEXT
            )
            """
        )
        for r in self._memory_rows:
            emb = r.get("embedding")
            dest.execute(
                "INSERT INTO episodes (cycle_index, salience, summary_json, embedding_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    int(r["cycle_index"]),
                    float(r["salience"]),
                    json.dumps(r["summary"], default=str),
                    json.dumps(list(emb), default=float) if emb is not None else None,
                ),
            )

    def _load_memory_rows(self, src: sqlite3.Connection) -> None:
        """Load rows from ``src`` into the in-memory buffer (in-memory store only)."""
        self._memory_rows.clear()
        cur = src.execute(
            "SELECT cycle_index, salience, summary_json, embedding_json FROM episodes ORDER BY id"
        )
        for cyc, sal, js, emb_js in cur.fetchall():
            try:
                summary = json.loads(js)
            except json.JSONDecodeError:
                summary = {"raw": js}
            row: dict[str, Any] = {"cycle_index": cyc, "salience": sal, "summary": summary}
            if emb_js:
                try:
                    row["embedding"] = json.loads(emb_js)
                except json.JSONDecodeError:
                    pass
            self._memory_rows.append(row)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if self._conn is None:
            with self._lock:
                tail = self._memory_rows[-limit:] if limit else list(self._memory_rows)
                rows_raw = list(reversed(tail))
            out: list[dict[str, Any]] = []
            for r in rows_raw:
                item = {
                    "cycle_index": r["cycle_index"],
                    "salience": r["salience"],
                    "summary": r["summary"],
                }
                if "embedding" in r and r["embedding"] is not None:
                    item["embedding"] = list(r["embedding"])
                out.append(item)
            return out

        with self._lock:
            cur = self._conn.execute(
                "SELECT cycle_index, salience, summary_json, embedding_json FROM episodes "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        out = []
        for cyc, sal, js, emb_js in rows:
            try:
                summary = json.loads(js)
            except json.JSONDecodeError:
                summary = {"raw": js}
            item = {"cycle_index": cyc, "salience": sal, "summary": summary}
            if emb_js:
                try:
                    item["embedding"] = json.loads(emb_js)
                except json.JSONDecodeError:
                    pass
            out.append(item)
        return out

    def _iter_scored_rows(
        self,
        *,
        min_salience: float,
        exclude_cycle: int | None,
    ) -> list[tuple[float, dict[str, Any]]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        if self._conn is None:
            with self._lock:
                snapshot = list(self._memory_rows)
            for r in snapshot:
                if r["salience"] < min_salience:
                    continue
                if exclude_cycle is not None and r["cycle_index"] == exclude_cycle:
                    continue
                emb = r.get("embedding")
                if emb is None:
                    continue
                vec = np.asarray(emb, dtype=np.float32).reshape(-1)
                if vec.size != EMBEDDING_DIM:
                    continue
                item = {
                    "cycle_index": r["cycle_index"],
                    "salience": r["salience"],
                    "summary": r["summary"],
                    "embedding": list(map(float, emb)),
                }
                scored.append((float(r["salience"]), item))
            return scored

        with self._lock:
            cur = self._conn.execute(
                "SELECT cycle_index, salience, summary_json, embedding_json FROM episodes "
                "WHERE embedding_json IS NOT NULL AND embedding_json != ''"
            )
            raw_rows = cur.fetchall()
        for cyc, sal, js, emb_js in raw_rows:
            if sal < min_salience:
                continue
            if exclude_cycle is not None and cyc == exclude_cycle:
                continue
            if not emb_js:
                continue
            try:
                emb_list = json.loads(emb_js)
            except json.JSONDecodeError:
                continue
            vec = np.asarray(emb_list, dtype=np.float32).reshape(-1)
            if vec.size != EMBEDDING_DIM:
                continue
            try:
                summary = json.loads(js)
            except json.JSONDecodeError:
                summary = {"raw": js}
            item = {
                "cycle_index": cyc,
                "salience": sal,
                "summary": summary,
                "embedding": list(map(float, emb_list)),
            }
            scored.append((sal, item))
        return scored

    def search_similar(
        self,
        query: np.ndarray,
        *,
        top_k: int = 5,
        min_salience: float = 0.0,
        exclude_cycle: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to ``top_k`` episodes ranked by cosine similarity to ``query``."""
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.size != EMBEDDING_DIM:
            q = np.pad(q, (0, max(0, EMBEDDING_DIM - q.size)))[:EMBEDDING_DIM]

        ranked: list[tuple[float, dict[str, Any]]] = []
        for _sal, item in self._iter_scored_rows(
            min_salience=min_salience, exclude_cycle=exclude_cycle
        ):
            emb = np.asarray(item["embedding"], dtype=np.float32).reshape(-1)
            sim = _cosine_similarity(q, emb)
            row = {**item, "similarity": sim}
            ranked.append((sim, row))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in ranked[: max(0, top_k)]]

    def retrieval_context_vector(
        self,
        query: np.ndarray,
        out_dim: int,
        *,
        top_k: int = 5,
        min_salience: float = 0.0,
        exclude_cycle: int | None = None,
    ) -> np.ndarray:
        """Mean-pool top similar episode embeddings, resized to ``out_dim`` (zeros if none)."""
        hits = self.search_similar(
            query,
            top_k=top_k,
            min_salience=min_salience,
            exclude_cycle=exclude_cycle,
        )
        if not hits:
            return np.zeros(out_dim, dtype=np.float32)
        mats = np.stack(
            [np.asarray(h["embedding"], dtype=np.float32) for h in hits],
            axis=0,
        )
        v = mats.mean(axis=0).astype(np.float32, copy=False)
        if v.shape[0] >= out_dim:
            return v[:out_dim].copy()
        out = np.zeros(out_dim, dtype=np.float32)
        out[: v.shape[0]] = v
        return out
