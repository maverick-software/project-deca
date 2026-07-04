"""Episodic memory: SQLite-backed cycle log + embedding similarity search."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from decadic import config as C
from decadic.memory.embeddings import EMBEDDING_DIM, PERCEPT_KEY_SLICE
from decadic.memory.sqlite_utils import (
    connect,
    configure_connection,
    db_file_sizes,
    decode_vector_blob,
    encode_vector_blob,
    wal_checkpoint_truncate,
)


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
        self._lock = threading.RLock()
        # Best cosine similarity of the most recent search_similar call; None
        # until the first search (or when the store is empty). Read by the
        # stage 3->4 attention gate as its novelty signal (WS3-G2).
        self.last_best_similarity: float | None = None
        # Best percept-key cosine of the most recent search_similar_percept
        # call; None until the first search (or when the store is empty).
        # WS4-M3.1: percept-only novelty channel (WS3 Phase B fix #1) -- the
        # full-vector similarity above is swamped by internal-state drift.
        self.last_best_percept_similarity: float | None = None
        self._conn: sqlite3.Connection | None = None
        self._memory_rows: list[dict[str, Any]] = []
        self._recall_cache_enabled = C.episodic_recall_cache_enabled()
        self._recall_items: list[dict[str, Any]] = []
        self._recall_dirty = True
        self._recall_norm: np.ndarray = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._recall_raw: np.ndarray = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._recall_salience: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._recall_cycles: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._recall_meta: list[dict[str, Any]] = []
        self._recall_hits = 0
        self._recall_misses = 0
        self._sqlite_commit_count = 0
        self._sqlite_batch_commit_count = 0
        self._sqlite_last_commit_ms = 0.0
        self._sqlite_wal_checkpoint_count = 0
        self._write_count_since_prune = 0
        self._db_pruned_rows = 0
        self._write_batch_size_last = 0
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = connect(db_path)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_index INTEGER NOT NULL,
                    salience REAL NOT NULL,
                    summary_json TEXT NOT NULL,
                    embedding_json TEXT,
                    embedding_blob BLOB
                )
                """
            )
            self._commit_locked()
            self._ensure_embedding_columns()
            self._load_recall_cache_from_db_locked()

    def _commit_locked(self, *, batch: bool = False) -> None:
        if self._conn is None:
            return
        started = time.perf_counter()
        self._conn.commit()
        self._sqlite_last_commit_ms = (time.perf_counter() - started) * 1000.0
        self._sqlite_commit_count += 1
        if batch:
            self._sqlite_batch_commit_count += 1

    def _ensure_embedding_columns(self) -> None:
        if self._conn is None:
            return
        cur = self._conn.execute("PRAGMA table_info(episodes)")
        cols = {row[1] for row in cur.fetchall()}
        if "embedding_json" not in cols:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN embedding_json TEXT")
            self._commit_locked()
        if "embedding_blob" not in cols:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN embedding_blob BLOB")
            self._commit_locked()

    def append(self, record: EpisodicRecord) -> None:
        self._append_persist(record, update_cache=True)

    def _embedding_payloads(self, embedding: list[float] | None) -> tuple[str | None, bytes | None]:
        if embedding is None:
            return None, None
        blob = encode_vector_blob(embedding)
        emb_json = (
            json.dumps(embedding, default=float)
            if (not C.sqlite_vector_blob_enabled() or C.sqlite_write_legacy_json_vectors())
            else None
        )
        return emb_json, blob

    def _append_persist(self, record: EpisodicRecord, *, update_cache: bool = True) -> None:
        if update_cache:
            self._cache_record(record)
        emb_json, emb_blob = self._embedding_payloads(record.embedding)

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
                "INSERT INTO episodes (cycle_index, salience, summary_json, embedding_json, embedding_blob) "
                "VALUES (?, ?, ?, ?, ?)",
                (record.cycle_index, record.salience, payload, emb_json, emb_blob),
            )
            self._write_batch_size_last = 1
            self._write_count_since_prune += 1
            self._maybe_prune_db_locked()
            self._commit_locked()

    def _append_many_persist(self, records: list[EpisodicRecord], *, update_cache: bool = False) -> None:
        if not records:
            return
        if update_cache:
            for record in records:
                self._cache_record(record)
        if self._conn is None:
            for record in records:
                self._append_persist(record, update_cache=False)
            return
        rows = []
        for record in records:
            emb_json, emb_blob = self._embedding_payloads(record.embedding)
            rows.append(
                (
                    int(record.cycle_index),
                    float(record.salience),
                    json.dumps(record.summary, default=str),
                    emb_json,
                    emb_blob,
                )
            )
        with self._lock:
            self._conn.executemany(
                "INSERT INTO episodes (cycle_index, salience, summary_json, embedding_json, embedding_blob) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self._write_batch_size_last = len(rows)
            self._write_count_since_prune += len(rows)
            self._maybe_prune_db_locked()
            self._commit_locked(batch=len(rows) > 1)

    def _cache_record(self, record: EpisodicRecord) -> None:
        if not self._recall_cache_enabled or record.embedding is None:
            return
        vec = np.asarray(record.embedding, dtype=np.float32).reshape(-1)
        if vec.size != EMBEDDING_DIM:
            return
        with self._lock:
            self._recall_items.append(
                {
                    "cycle_index": int(record.cycle_index),
                    "salience": float(record.salience),
                    "summary": dict(record.summary),
                    "embedding": vec,
                }
            )
            self._prune_recall_items_locked()
            self._recall_dirty = True

    def _prune_recall_items_locked(self) -> None:
        recent_cap = C.episodic_recall_recent_cap()
        salient_cap = C.episodic_recall_salient_cap()
        if len(self._recall_items) <= recent_cap + salient_cap:
            return
        recent = self._recall_items[-recent_cap:]
        recent_ids = {id(x) for x in recent}
        older = [x for x in self._recall_items if id(x) not in recent_ids]
        salient = sorted(older, key=lambda x: (float(x["salience"]), int(x["cycle_index"])), reverse=True)[
            :salient_cap
        ]
        keep = {id(x) for x in recent + salient}
        self._recall_items = [x for x in self._recall_items if id(x) in keep]

    def _embedding_from_storage(self, emb_json: Any, emb_blob: Any = None) -> np.ndarray | None:
        emb = decode_vector_blob(emb_blob)
        if emb is not None:
            return emb.reshape(-1)
        if emb_json:
            try:
                return np.asarray(json.loads(emb_json), dtype=np.float32).reshape(-1)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return None

    def _rebuild_recall_cache_locked(self) -> None:
        if not self._recall_cache_enabled:
            return
        self._prune_recall_items_locked()
        if not self._recall_items:
            self._recall_norm = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            self._recall_raw = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            self._recall_salience = np.zeros((0,), dtype=np.float32)
            self._recall_cycles = np.zeros((0,), dtype=np.int64)
            self._recall_meta = []
            self._recall_dirty = False
            return
        raw = np.stack([np.asarray(x["embedding"], dtype=np.float32) for x in self._recall_items], axis=0)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norm = raw / np.maximum(norms, 1e-8)
        self._recall_raw = raw.astype(np.float32, copy=False)
        self._recall_norm = norm.astype(np.float32, copy=False)
        self._recall_salience = np.asarray([float(x["salience"]) for x in self._recall_items], dtype=np.float32)
        self._recall_cycles = np.asarray([int(x["cycle_index"]) for x in self._recall_items], dtype=np.int64)
        self._recall_meta = [
            {
                "cycle_index": int(x["cycle_index"]),
                "salience": float(x["salience"]),
                "summary": dict(x["summary"]),
            }
            for x in self._recall_items
        ]
        self._recall_dirty = False

    def _load_recall_cache_from_db_locked(self) -> None:
        if not self._recall_cache_enabled or self._conn is None:
            return
        recent_cap = C.episodic_recall_recent_cap()
        salient_cap = C.episodic_recall_salient_cap()
        rows: dict[tuple[int, float], tuple[int, float, str, str | None, bytes | None]] = {}
        for query, cap in (
            (
                "SELECT cycle_index, salience, summary_json, embedding_json, embedding_blob FROM episodes "
                "WHERE (embedding_blob IS NOT NULL AND length(embedding_blob) > 0) "
                "OR (embedding_json IS NOT NULL AND embedding_json != '') ORDER BY id DESC LIMIT ?",
                recent_cap,
            ),
            (
                "SELECT cycle_index, salience, summary_json, embedding_json, embedding_blob FROM episodes "
                "WHERE (embedding_blob IS NOT NULL AND length(embedding_blob) > 0) "
                "OR (embedding_json IS NOT NULL AND embedding_json != '') ORDER BY salience DESC LIMIT ?",
                salient_cap,
            ),
        ):
            for cyc, sal, js, emb_js, emb_blob in self._conn.execute(query, (cap,)).fetchall():
                rows[(int(cyc), float(sal))] = (
                    int(cyc),
                    float(sal),
                    str(js),
                    str(emb_js) if emb_js else None,
                    emb_blob,
                )
        self._recall_items.clear()
        for cyc, sal, js, emb_js, emb_blob in sorted(rows.values(), key=lambda r: r[0]):
            emb = self._embedding_from_storage(emb_js, emb_blob)
            if emb is None:
                continue
            if emb.size != EMBEDDING_DIM:
                continue
            try:
                summary = json.loads(js)
            except json.JSONDecodeError:
                summary = {"raw": js}
            self._recall_items.append(
                {"cycle_index": cyc, "salience": sal, "summary": summary, "embedding": emb}
            )
        self._recall_dirty = True

    def _maybe_prune_db_locked(self) -> None:
        if (
            self._conn is None
            or not C.episodic_db_retention_enabled()
            or self._write_count_since_prune < C.episodic_db_prune_interval_writes()
        ):
            return
        self._write_count_since_prune = 0
        recent_cap = C.episodic_db_recent_cap()
        salient_cap = C.episodic_db_salient_cap()
        total = int(self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] or 0)
        if total <= recent_cap + salient_cap:
            return
        ids = [
            int(row[0])
            for row in self._conn.execute(
                """
                SELECT id FROM episodes
                WHERE id NOT IN (SELECT id FROM episodes ORDER BY id DESC LIMIT ?)
                  AND id NOT IN (SELECT id FROM episodes ORDER BY salience DESC, cycle_index DESC LIMIT ?)
                ORDER BY salience ASC, cycle_index ASC
                LIMIT ?
                """,
                (recent_cap, salient_cap, C.episodic_db_prune_batch()),
            ).fetchall()
        ]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(f"DELETE FROM episodes WHERE id IN ({placeholders})", ids)
        self._db_pruned_rows += len(ids)

    def recall_cache_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": bool(self._recall_cache_enabled),
                "size": int(len(self._recall_items)),
                "hits": int(self._recall_hits),
                "misses": int(self._recall_misses),
            }

    def clear(self) -> None:
        """Wipe all stored episodes (agent reset)."""
        with self._lock:
            if self._conn is None:
                self._memory_rows.clear()
            else:
                self._conn.execute("DELETE FROM episodes")
                self._commit_locked()
            self._recall_items.clear()
            self._recall_dirty = True

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
        configure_connection(dest)
        try:
            with self._lock:
                if self._conn is not None:
                    wal_checkpoint_truncate(self._conn)
                    self._sqlite_wal_checkpoint_count += 1
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
        configure_connection(src)
        try:
            with self._lock:
                if self._conn is not None:
                    src.backup(self._conn)
                    self._commit_locked()
                    self._ensure_embedding_columns()
                    self._load_recall_cache_from_db_locked()
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
                embedding_json TEXT,
                embedding_blob BLOB
            )
            """
        )
        for r in self._memory_rows:
            emb = r.get("embedding")
            emb_json, emb_blob = self._embedding_payloads(list(emb) if emb is not None else None)
            dest.execute(
                "INSERT INTO episodes (cycle_index, salience, summary_json, embedding_json, embedding_blob) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    int(r["cycle_index"]),
                    float(r["salience"]),
                    json.dumps(r["summary"], default=str),
                    emb_json,
                    emb_blob,
                ),
            )

    def _load_memory_rows(self, src: sqlite3.Connection) -> None:
        """Load rows from ``src`` into the in-memory buffer (in-memory store only)."""
        self._memory_rows.clear()
        cur = src.execute(
            "SELECT cycle_index, salience, summary_json, embedding_json, embedding_blob FROM episodes ORDER BY id"
        )
        for cyc, sal, js, emb_js, emb_blob in cur.fetchall():
            try:
                summary = json.loads(js)
            except json.JSONDecodeError:
                summary = {"raw": js}
            row: dict[str, Any] = {"cycle_index": cyc, "salience": sal, "summary": summary}
            emb = self._embedding_from_storage(emb_js, emb_blob)
            if emb is not None:
                row["embedding"] = emb.astype(float).tolist()
            self._memory_rows.append(row)
            if "embedding" in row:
                emb = np.asarray(row["embedding"], dtype=np.float32).reshape(-1)
                if emb.size == EMBEDDING_DIM:
                    self._recall_items.append(
                        {"cycle_index": int(cyc), "salience": float(sal), "summary": summary, "embedding": emb}
                    )
        self._recall_dirty = True

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
                "SELECT cycle_index, salience, summary_json, embedding_json, embedding_blob FROM episodes "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        out = []
        for cyc, sal, js, emb_js, emb_blob in rows:
            try:
                summary = json.loads(js)
            except json.JSONDecodeError:
                summary = {"raw": js}
            item = {"cycle_index": cyc, "salience": sal, "summary": summary}
            emb = self._embedding_from_storage(emb_js, emb_blob)
            if emb is not None:
                item["embedding"] = emb.astype(float).tolist()
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
                "SELECT cycle_index, salience, summary_json, embedding_json, embedding_blob FROM episodes "
                "WHERE (embedding_blob IS NOT NULL AND length(embedding_blob) > 0) "
                "OR (embedding_json IS NOT NULL AND embedding_json != '') "
                "ORDER BY salience DESC LIMIT ?",
                (C.episodic_recall_sql_fallback_cap(),),
            )
            raw_rows = cur.fetchall()
        for cyc, sal, js, emb_js, emb_blob in raw_rows:
            if sal < min_salience:
                continue
            if exclude_cycle is not None and cyc == exclude_cycle:
                continue
            emb = self._embedding_from_storage(emb_js, emb_blob)
            if emb is None:
                continue
            if emb.size != EMBEDDING_DIM:
                continue
            try:
                summary = json.loads(js)
            except json.JSONDecodeError:
                summary = {"raw": js}
            item = {
                "cycle_index": cyc,
                "salience": sal,
                "summary": summary,
                "embedding": emb.astype(float).tolist(),
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

        cached = self._search_similar_cache(
            q,
            top_k=top_k,
            min_salience=min_salience,
            exclude_cycle=exclude_cycle,
        )
        if cached is not None:
            # Side effect for the stage 3->4 attention gate (WS3-G2): expose
            # the best similarity of the most recent search without a second
            # scan. None means "no episodes yet" (reads as fully novel).
            self.last_best_similarity = (
                float(cached[0]["similarity"]) if cached else None
            )
            return cached

        ranked: list[tuple[float, dict[str, Any]]] = []
        for _sal, item in self._iter_scored_rows(
            min_salience=min_salience, exclude_cycle=exclude_cycle
        ):
            emb = np.asarray(item["embedding"], dtype=np.float32).reshape(-1)
            sim = _cosine_similarity(q, emb)
            row = {**item, "similarity": sim}
            ranked.append((sim, row))

        ranked.sort(key=lambda x: x[0], reverse=True)
        self.last_best_similarity = ranked[0][0] if ranked else None
        return [r for _, r in ranked[: max(0, top_k)]]

    def _search_similar_cache(
        self,
        q: np.ndarray,
        *,
        top_k: int,
        min_salience: float,
        exclude_cycle: int | None,
    ) -> list[dict[str, Any]] | None:
        if not self._recall_cache_enabled:
            return None
        with self._lock:
            if not self._recall_items and self._conn is not None:
                self._load_recall_cache_from_db_locked()
            if self._recall_dirty:
                self._rebuild_recall_cache_locked()
            if self._recall_norm.shape[0] == 0:
                self._recall_misses += 1
                return []
            qn = q.astype(np.float32, copy=False)
            qn = qn / max(1e-8, float(np.linalg.norm(qn)))
            mask = self._recall_salience >= float(min_salience)
            if exclude_cycle is not None:
                mask &= self._recall_cycles != int(exclude_cycle)
            idxs = np.flatnonzero(mask)
            if idxs.size == 0:
                self._recall_misses += 1
                return []
            sims_all = self._recall_norm[idxs] @ qn
            k = max(0, min(int(top_k), int(idxs.size)))
            if k <= 0:
                return []
            if idxs.size > k:
                part = np.argpartition(-sims_all, k - 1)[:k]
                order = part[np.argsort(-sims_all[part])]
            else:
                order = np.argsort(-sims_all)
            self._recall_hits += 1
            out: list[dict[str, Any]] = []
            for local in order:
                global_idx = int(idxs[int(local)])
                meta = dict(self._recall_meta[global_idx])
                emb = self._recall_raw[global_idx].astype(float).tolist()
                meta["embedding"] = emb
                meta["similarity"] = float(sims_all[int(local)])
                out.append(meta)
            return out

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

    def search_similar_percept(
        self,
        key: np.ndarray,
        top_k: int = 5,
        exclude_cycle_after: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to ``top_k`` episodes ranked by cosine over the percept-key
        sub-vector alone (WS4-M3.1).

        The full 80-d embedding mixes internal state (narrative/emotion/metacog/z5)
        with the 16-d perceptual key; internal drift swamps external familiarity,
        which is why the gate's full-vector novelty signal has almost no dynamic
        range (the WS3 finding). Ranking on ``embedding[PERCEPT_KEY_SLICE]`` alone
        recovers "have I *seen* this before?". Side channel:
        ``last_best_percept_similarity`` (None when the store is empty).

        ``exclude_cycle_after`` excludes rows with ``cycle_index >=`` it (the
        WS3 novelty recency horizon): without it the best match is always the
        just-written previous frame (similarity ~1.0) and novelty collapses to
        zero everywhere, including during injected events (probe 2026-07-04).
        """
        key_dim = PERCEPT_KEY_SLICE.stop - PERCEPT_KEY_SLICE.start
        k = np.asarray(key, dtype=np.float32).reshape(-1)
        if k.size != key_dim:
            k = np.pad(k, (0, max(0, key_dim - k.size)))[:key_dim]

        cached = self._search_percept_cache(
            k, top_k=top_k, exclude_cycle_after=exclude_cycle_after
        )
        if cached is not None:
            self.last_best_percept_similarity = (
                float(cached[0]["similarity"]) if cached else None
            )
            return cached

        ranked: list[tuple[float, dict[str, Any]]] = []
        for _sal, item in self._iter_scored_rows(min_salience=0.0, exclude_cycle=None):
            if (
                exclude_cycle_after is not None
                and int(item["cycle_index"]) >= int(exclude_cycle_after)
            ):
                continue
            emb = np.asarray(item["embedding"], dtype=np.float32).reshape(-1)
            sim = _cosine_similarity(k, emb[PERCEPT_KEY_SLICE])
            ranked.append((sim, {**item, "similarity": sim}))

        ranked.sort(key=lambda x: x[0], reverse=True)
        self.last_best_percept_similarity = ranked[0][0] if ranked else None
        return [r for _, r in ranked[: max(0, top_k)]]

    def retrieval_context_tokens(
        self,
        query: np.ndarray,
        k: int = 5,
        *,
        min_salience: float = 0.0,
        exclude_cycle: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """WS5-M0.3: top-k recalled episodes as a (k, EMBEDDING_DIM) token
        matrix + validity mask -- the un-pooled counterpart of
        :meth:`retrieval_context_vector` (which mean-pools the same hits into
        mush; binding chokepoint #2 in docs/ws5_m0_wm_inventory.md).

        Rows are the ranked hits in order; unfilled rows are zero with mask
        False. With ``k=1``, row 0 is exactly today's best-hit embedding --
        the equivalence the WBS requires.
        """
        kk = max(0, int(k))
        tokens = np.zeros((kk, EMBEDDING_DIM), dtype=np.float32)
        mask = np.zeros((kk,), dtype=bool)
        if kk == 0:
            return tokens, mask
        hits = self.search_similar(
            query, top_k=kk, min_salience=min_salience, exclude_cycle=exclude_cycle
        )
        for i, h in enumerate(hits[:kk]):
            emb = h.get("embedding")
            if emb is None:
                continue
            v = np.asarray(emb, dtype=np.float32).reshape(-1)
            n = min(v.size, EMBEDDING_DIM)
            tokens[i, :n] = v[:n]
            mask[i] = True
        return tokens, mask

    def _search_percept_cache(
        self,
        k: np.ndarray,
        *,
        top_k: int,
        exclude_cycle_after: int | None = None,
    ) -> list[dict[str, Any]] | None:
        if not self._recall_cache_enabled:
            return None
        with self._lock:
            if not self._recall_items and self._conn is not None:
                self._load_recall_cache_from_db_locked()
            if self._recall_dirty:
                self._rebuild_recall_cache_locked()
            if self._recall_raw.shape[0] == 0:
                self._recall_misses += 1
                return []
            keys = self._recall_raw[:, PERCEPT_KEY_SLICE]
            key_norms = np.linalg.norm(keys, axis=1)
            kn = float(np.linalg.norm(k))
            if kn < 1e-8:
                sims_all = np.zeros(keys.shape[0], dtype=np.float32)
            else:
                sims_all = (keys @ k.astype(np.float32, copy=False)) / np.maximum(
                    key_norms * kn, 1e-8
                )
                # Match _cosine_similarity: a degenerate stored key reads as 0.
                sims_all = np.where(key_norms < 1e-8, 0.0, sims_all)
            n = int(sims_all.shape[0])
            eligible = n
            if exclude_cycle_after is not None:
                cycles = np.fromiter(
                    (int(m["cycle_index"]) for m in self._recall_meta),
                    dtype=np.int64,
                    count=n,
                )
                elig_mask = cycles < int(exclude_cycle_after)
                eligible = int(np.count_nonzero(elig_mask))
                if eligible == 0:
                    self._recall_hits += 1
                    return []
                sims_all = np.asarray(sims_all, dtype=np.float32).copy()
                sims_all[~elig_mask] = -np.inf
            kk = max(0, min(int(top_k), eligible))
            if kk <= 0:
                return []
            if n > kk:
                part = np.argpartition(-sims_all, kk - 1)[:kk]
                order = part[np.argsort(-sims_all[part])]
            else:
                order = np.argsort(-sims_all)
            self._recall_hits += 1
            out: list[dict[str, Any]] = []
            for idx in order:
                global_idx = int(idx)
                meta = dict(self._recall_meta[global_idx])
                meta["embedding"] = self._recall_raw[global_idx].astype(float).tolist()
                meta["similarity"] = float(sims_all[global_idx])
                out.append(meta)
            return out

    def persistence_metrics(self) -> dict[str, Any]:
        with self._lock:
            rows = 0
            if self._conn is not None:
                try:
                    rows = int(self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] or 0)
                except sqlite3.Error:
                    rows = 0
            else:
                rows = len(self._memory_rows)
            return {
                "sqlite_commit_count": int(self._sqlite_commit_count),
                "sqlite_batch_commit_count": int(self._sqlite_batch_commit_count),
                "sqlite_last_commit_ms": float(self._sqlite_last_commit_ms),
                "sqlite_wal_checkpoint_count": int(self._sqlite_wal_checkpoint_count),
                "episodic_write_batch_size_last": int(self._write_batch_size_last),
                "episodic_db_rows": rows,
                "episodic_db_pruned_rows": int(self._db_pruned_rows),
                **db_file_sizes(self._db_path),
            }
