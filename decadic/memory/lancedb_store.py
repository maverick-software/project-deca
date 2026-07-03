"""LanceDB-backed episodic store (WS4-M1.1).

Drop-in replacement for :class:`decadic.memory.episodic_store.EpisodicStore`
selected via ``DECADIC_MEMORY_BACKEND=lancedb`` (see
``decadic.memory.factory``). Same public surface, same hit-dict shapes, same
``last_best_similarity`` / ``last_best_percept_similarity`` side channels, and
the same thread-safety contract (an ``RLock`` serializes all access, so the
write-behind worker threads and the cycle can share one instance).

Storage layout (PRD 5.2): one Lance table ``episodes`` per store with the
SQLite columns mapped 1:1 (``cycle_index``/``salience``/``summary_json`` plus
``ts``) and two vector columns -- the full 80-d ``embedding`` and a dedicated
16-d ``percept_key`` column holding ``embedding[PERCEPT_KEY_SLICE]`` so
percept-only novelty search (WS3 Phase B fix #1) is a native indexed query.

Design notes:
- ``lancedb``/``pyarrow`` are imported lazily, so importing this module never
  fails when the extra is absent (lightweight tooling stays torch- and
  lancedb-free).
- ``db_path=None`` uses a lazily created temp directory, removed on
  ``close()``/``__del__`` (the ephemeral test mode). A real ``db_path`` maps to
  a sibling directory ``<db_path>.lance`` (never touches the sqlite file).
- Appends are buffered in RAM (bounded) and flushed as batched ``table.add``
  calls; every read path merges the pending tail, so reads are always
  consistent with writes ("read your writes" holds without a fragment per
  cycle). ``flush()``/``close()``/``backup_to`` drain the buffer.
- Search is brute-force ``table.search(...)`` (exact below the ANN threshold,
  WS4-M1.3 adds the index); similarities are recomputed with the exact numpy
  cosine used by the SQLite backend so parity holds even for unnormalized or
  degenerate vectors.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from decadic import config as C
from decadic.memory.embeddings import EMBEDDING_DIM, PERCEPT_KEY_SLICE
from decadic.memory.episodic_store import EpisodicRecord, _cosine_similarity

logger = logging.getLogger(__name__)

PERCEPT_KEY_DIM = PERCEPT_KEY_SLICE.stop - PERCEPT_KEY_SLICE.start
_TABLE_NAME = "episodes"
# Appends accumulate in RAM up to this many rows before a batched table.add;
# reads always merge the pending tail so the buffer is invisible to callers.
_WRITE_BUFFER_MAX = 256
# Extra candidates fetched per vector search so the exact-cosine re-rank can
# never lose a borderline row to Lance's own distance ordering.
_FETCH_MARGIN = 8


def _storage_dir_for(db_path: Path) -> Path:
    """Directory for a caller-supplied db path (kept distinct from the sqlite file)."""
    p = Path(db_path)
    if p.suffix == ".lance":
        return p
    return Path(str(p) + ".lance")


def _dir_size_bytes(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    total = 0
    try:
        for f in path.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


class LanceEpisodicStore:
    """Thread-safe LanceDB store for per-cycle summaries and embeddings."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else None
        self._lock = threading.RLock()
        # Side channels read by the stage 3->4 attention gate (WS3-G2 / WS4-M3.1).
        self.last_best_similarity: float | None = None
        self.last_best_percept_similarity: float | None = None
        self._dir: Path | None = (
            _storage_dir_for(self._db_path) if self._db_path is not None else None
        )
        self._owns_tmp = False
        self._db: Any = None
        self._table: Any = None
        self._schema: Any = None
        self._pending: list[dict[str, Any]] = []
        self._next_id = 0
        # When async is off every append flushes immediately (durability parity
        # with the synchronous sqlite store); when on, appends micro-batch.
        self._async_enabled = True
        # Metric counters (persistence_metrics keeps the sqlite_* key names so
        # dashboards/telemetry stay wired; here a "commit" is a table.add call).
        self._commit_count = 0
        self._batch_commit_count = 0
        self._last_commit_ms = 0.0
        self._write_batch_size_last = 0
        self._write_count_since_prune = 0
        self._db_pruned_rows = 0
        self._search_hits = 0
        self._search_misses = 0

    # ------------------------------------------------------------------ setup

    def _storage_dir_locked(self) -> Path:
        if self._dir is None:
            self._dir = Path(tempfile.mkdtemp(prefix="decadic_lance_"))
            self._owns_tmp = True
        return self._dir

    def _arrow_schema(self) -> Any:
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("cycle_index", pa.int64()),
                pa.field("ts", pa.float64()),
                pa.field("salience", pa.float64()),
                pa.field("summary_json", pa.string()),
                pa.field("has_embedding", pa.bool_()),
                pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
                pa.field("percept_key", pa.list_(pa.float32(), PERCEPT_KEY_DIM)),
            ]
        )

    def _ensure_table_locked(self) -> Any:
        if self._table is not None:
            return self._table
        import lancedb  # lazy: module import must not require the extra

        d = self._storage_dir_locked()
        d.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(d))
        self._schema = self._arrow_schema()
        # Open-then-create: table enumeration APIs are version-unstable
        # (table_names() deprecated; list_tables() return shape varies across
        # lancedb releases - on 8.x iterating it yields lists, which broke a
        # set comprehension here). Opening and falling back to create is
        # version-proof and race-free under self._lock.
        try:
            self._table = self._db.open_table(_TABLE_NAME)
            self._next_id = self._max_id_locked() + 1
        except Exception:
            self._table = self._db.create_table(_TABLE_NAME, schema=self._schema)
            self._next_id = 0
        return self._table

    def _close_handles_locked(self) -> None:
        self._table = None
        self._db = None
        import gc

        gc.collect()  # best effort: release Lance file handles (Windows rmtree)

    # ------------------------------------------------------------ scan helpers

    def _query_rows(self, qb: Any) -> list[dict[str, Any]]:
        try:
            return list(qb.to_list())
        except AttributeError:
            return list(qb.to_arrow().to_pylist())

    def _scan_locked(self, columns: list[str]) -> list[dict[str, Any]]:
        """Fetch ``columns`` for every committed row (no pending rows)."""
        tbl = self._ensure_table_locked()
        total = int(tbl.count_rows())
        if total <= 0:
            return []
        try:
            qb = tbl.search().select(list(columns)).limit(total)
            return self._query_rows(qb)
        except Exception:
            data = tbl.to_arrow()
            keep = [c for c in columns if c in data.column_names]
            return list(data.select(keep).to_pylist())

    def _max_id_locked(self) -> int:
        rows = self._scan_locked(["id"])
        if not rows:
            return -1
        return max(int(r["id"]) for r in rows)

    # ---------------------------------------------------------------- writes

    def _row_from_record(self, record: EpisodicRecord) -> dict[str, Any]:
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        has = False
        if record.embedding is not None:
            arr = np.asarray(record.embedding, dtype=np.float32).reshape(-1)
            if arr.size == EMBEDDING_DIM:
                vec = arr
                has = True
            else:
                # Wrong-length embeddings are stored (padded/truncated) but
                # excluded from search -- the sqlite backend also skips them.
                n = min(arr.size, EMBEDDING_DIM)
                vec[:n] = arr[:n]
        row = {
            "id": int(self._next_id),
            "cycle_index": int(record.cycle_index),
            "ts": float(time.time()),
            "salience": float(record.salience),
            "summary_json": json.dumps(record.summary, default=str),
            "has_embedding": bool(has),
            "embedding": [float(x) for x in vec],
            "percept_key": [float(x) for x in vec[PERCEPT_KEY_SLICE]],
        }
        self._next_id += 1
        return row

    def append(self, record: EpisodicRecord) -> None:
        with self._lock:
            self._pending.append(self._row_from_record(record))
            self._write_count_since_prune += 1
            if not self._async_enabled or len(self._pending) >= _WRITE_BUFFER_MAX:
                self._flush_pending_locked()
            # Prune check per appended record -- the exact cadence the sqlite
            # backend uses (its INSERT path checks after every write), so both
            # backends evaluate the keep-set at identical write counts.
            self._maybe_prune_locked()

    def flush(self) -> None:
        """Persist any buffered appends (no-op when the buffer is empty)."""
        with self._lock:
            self._flush_pending_locked()

    def _flush_pending_locked(self) -> None:
        if not self._pending:
            return
        import pyarrow as pa

        tbl = self._ensure_table_locked()
        batch = self._pending
        self._pending = []
        started = time.perf_counter()
        tbl.add(pa.Table.from_pylist(batch, schema=self._schema))
        self._last_commit_ms = (time.perf_counter() - started) * 1000.0
        self._commit_count += 1
        if len(batch) > 1:
            self._batch_commit_count += 1
        self._write_batch_size_last = len(batch)

    # ---------------------------------------------------------------- pruning

    def _maybe_prune_locked(self) -> None:
        """Salience/age retention via delete-where; mirrors the sqlite policy.

        Keep-set = newest ``recent_cap`` rows by insertion order UNION the top
        ``salient_cap`` rows by (salience desc, cycle desc); at most
        ``prune_batch`` victims are deleted per pass, lowest salience first.

        Parity note: sqlite prunes incrementally as it writes, so early
        low-salience rows can "ratchet" into permanence by making a keep-set
        before higher-salience rows arrive. Matching that requires (a) the
        per-append check cadence (see :meth:`append`) and (b) evaluating the
        keep-set over ALL rows visible at that moment -- hence the flush below,
        which makes the table the single source of truth at prune time.
        """
        if (
            not C.episodic_db_retention_enabled()
            or self._write_count_since_prune < C.episodic_db_prune_interval_writes()
        ):
            return
        self._write_count_since_prune = 0
        recent_cap = C.episodic_db_recent_cap()
        salient_cap = C.episodic_db_salient_cap()
        self._flush_pending_locked()  # drain the pending tail before evaluating
        tbl = self._ensure_table_locked()
        total = int(tbl.count_rows())
        if total <= recent_cap + salient_cap:
            return
        rows = self._scan_locked(["id", "cycle_index", "salience"])
        by_id = sorted(rows, key=lambda r: int(r["id"]), reverse=True)
        keep = {int(r["id"]) for r in by_id[:recent_cap]}
        by_salience = sorted(
            rows,
            key=lambda r: (float(r["salience"]), int(r["cycle_index"])),
            reverse=True,
        )
        keep |= {int(r["id"]) for r in by_salience[:salient_cap]}
        victims = [r for r in rows if int(r["id"]) not in keep]
        victims.sort(key=lambda r: (float(r["salience"]), int(r["cycle_index"])))
        victims = victims[: C.episodic_db_prune_batch()]
        if not victims:
            return
        ids = ",".join(str(int(r["id"])) for r in victims)
        tbl.delete(f"id IN ({ids})")
        self._db_pruned_rows += len(victims)

    # ----------------------------------------------------------------- reads

    def _hit_from_row(self, row: dict[str, Any]) -> dict[str, Any]:
        js = row.get("summary_json") or ""
        try:
            summary = json.loads(js)
        except (json.JSONDecodeError, TypeError):
            summary = {"raw": js}
        return {
            "cycle_index": int(row["cycle_index"]),
            "salience": float(row["salience"]),
            "summary": summary,
        }

    def _candidate_rows_locked(
        self,
        query: np.ndarray,
        *,
        column: str,
        top_k: int,
        where: str | None,
    ) -> list[dict[str, Any]]:
        """Vector-search committed rows + merge the pending tail (post-filtered)."""
        candidates: list[dict[str, Any]] = []
        fetch = max(1, int(top_k)) + _FETCH_MARGIN
        tbl = None
        if self._table is not None or (self._dir is not None and self._dir.exists()):
            tbl = self._ensure_table_locked()
        if tbl is not None and int(tbl.count_rows()) > 0:
            qb = tbl.search(
                [float(x) for x in query], vector_column_name=column
            )
            try:
                qb = qb.metric("cosine")
            except AttributeError:
                qb = qb.distance_type("cosine")
            if where:
                try:
                    qb = qb.where(where, prefilter=True)
                except TypeError:
                    qb = qb.where(where)
            candidates.extend(self._query_rows(qb.limit(fetch)))
        return candidates

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
        clauses = ["has_embedding = true", f"salience >= {float(min_salience)!r}"]
        if exclude_cycle is not None:
            clauses.append(f"cycle_index <> {int(exclude_cycle)}")
        where = " AND ".join(clauses)

        with self._lock:
            candidates = self._candidate_rows_locked(
                q, column="embedding", top_k=top_k, where=where
            )
            for row in self._pending:
                if not row["has_embedding"]:
                    continue
                if float(row["salience"]) < float(min_salience):
                    continue
                if exclude_cycle is not None and int(row["cycle_index"]) == int(exclude_cycle):
                    continue
                candidates.append(row)

            ranked: list[tuple[float, dict[str, Any]]] = []
            for row in candidates:
                emb = np.asarray(row["embedding"], dtype=np.float32).reshape(-1)
                sim = _cosine_similarity(q, emb)
                item = self._hit_from_row(row)
                item["embedding"] = emb.astype(float).tolist()
                item["similarity"] = sim
                ranked.append((sim, item))
            ranked.sort(key=lambda x: x[0], reverse=True)
            if ranked:
                self._search_hits += 1
            else:
                self._search_misses += 1
            self.last_best_similarity = ranked[0][0] if ranked else None
            return [r for _, r in ranked[: max(0, top_k)]]

    def search_similar_percept(
        self,
        key: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rank by cosine over the 16-d percept-key sub-vector only (WS4-M3.1).

        Same hit-dict shape as :meth:`search_similar`; ``similarity`` is the
        percept-key cosine. Side channel: ``last_best_percept_similarity``
        (None when the store is empty).
        """
        k = np.asarray(key, dtype=np.float32).reshape(-1)
        if k.size != PERCEPT_KEY_DIM:
            k = np.pad(k, (0, max(0, PERCEPT_KEY_DIM - k.size)))[:PERCEPT_KEY_DIM]

        with self._lock:
            candidates = self._candidate_rows_locked(
                k, column="percept_key", top_k=top_k, where="has_embedding = true"
            )
            candidates.extend(r for r in self._pending if r["has_embedding"])

            ranked: list[tuple[float, dict[str, Any]]] = []
            for row in candidates:
                emb = np.asarray(row["embedding"], dtype=np.float32).reshape(-1)
                sim = _cosine_similarity(k, emb[PERCEPT_KEY_SLICE])
                item = self._hit_from_row(row)
                item["embedding"] = emb.astype(float).tolist()
                item["similarity"] = sim
                ranked.append((sim, item))
            ranked.sort(key=lambda x: x[0], reverse=True)
            self.last_best_percept_similarity = ranked[0][0] if ranked else None
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

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows: list[dict[str, Any]] = list(self._pending)
            if self._table is not None or (self._dir is not None and self._dir.exists()):
                tbl = self._ensure_table_locked()
                if int(tbl.count_rows()) > 0:
                    ids = sorted(
                        (int(r["id"]) for r in self._scan_locked(["id"])), reverse=True
                    )
                    lim = int(limit) if limit else len(ids) + len(self._pending)
                    take = ids[: max(0, lim)]
                    if take:
                        id_list = ",".join(str(i) for i in take)
                        try:
                            qb = (
                                tbl.search()
                                .where(f"id IN ({id_list})")
                                .limit(len(take))
                            )
                            fetched = self._query_rows(qb)
                        except Exception:
                            data = tbl.to_arrow().to_pylist()
                            wanted = set(take)
                            fetched = [r for r in data if int(r["id"]) in wanted]
                        rows.extend(fetched)
            rows.sort(key=lambda r: int(r["id"]), reverse=True)
            if limit:
                rows = rows[: int(limit)]
        out: list[dict[str, Any]] = []
        for row in rows:
            item = self._hit_from_row(row)
            if row.get("has_embedding"):
                emb = np.asarray(row["embedding"], dtype=np.float32)
                item["embedding"] = emb.astype(float).tolist()
            out.append(item)
        return out

    # ------------------------------------------------------------ maintenance

    def clear(self) -> None:
        """Wipe all stored episodes (agent reset)."""
        with self._lock:
            self._pending.clear()
            if self._table is not None or (self._dir is not None and self._dir.exists()):
                tbl = self._ensure_table_locked()
                if int(tbl.count_rows()) > 0:
                    tbl.delete("id >= 0")

    def backup_to(self, path: Path) -> None:
        """Quiesced snapshot: drain buffered writes, then copy the Lance directory.

        ``path`` becomes a directory (the sqlite backend writes a file there
        instead); ``restore_from`` accepts the same directory back.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._ensure_table_locked()  # materialize schema so empty stores round-trip
            self._flush_pending_locked()
            src = self._storage_dir_locked()
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.copytree(src, target)

    def restore_from(self, path: Path) -> None:
        """Replace live episodes with the Lance directory at ``path`` (no-op if missing)."""
        src = Path(path)
        if not src.is_dir():
            return
        with self._lock:
            self._pending.clear()
            self._close_handles_locked()
            d = self._storage_dir_locked()
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(src, d)
            self._ensure_table_locked()  # reopen + resume the id counter

    # ------------------------------------------------------------- lifecycle

    @property
    def async_enabled(self) -> bool:
        return self._async_enabled

    def set_async(self, enabled: bool) -> None:
        """Toggle write micro-batching (off = flush on every append)."""
        with self._lock:
            self._async_enabled = bool(enabled)
            if not self._async_enabled:
                self._flush_pending_locked()

    def close(self) -> None:
        """Drain buffered writes and release the Lance handle (temp dirs removed)."""
        with self._lock:
            try:
                if self._pending and (
                    self._table is not None
                    or (self._dir is not None and (self._dir.exists() or not self._owns_tmp))
                ):
                    self._flush_pending_locked()
            except Exception:  # pragma: no cover - close must never raise
                logger.exception("lance episodic close: flush failed")
            self._close_handles_locked()
            if self._owns_tmp and self._dir is not None:
                shutil.rmtree(self._dir, ignore_errors=True)
                self._dir = None
                self._owns_tmp = False

    def __del__(self) -> None:  # pragma: no cover - GC timing dependent
        try:
            self.close()
        except Exception:
            pass

    # --------------------------------------------------------------- metrics

    def recall_cache_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": False,
                "size": 0,
                "hits": int(self._search_hits),
                "misses": int(self._search_misses),
            }

    def persistence_metrics(self) -> dict[str, Any]:
        with self._lock:
            rows = len(self._pending)
            if self._table is not None or (self._dir is not None and self._dir.exists()):
                try:
                    rows += int(self._ensure_table_locked().count_rows())
                except Exception:
                    pass
            return {
                "backend": "lancedb",
                "sqlite_commit_count": int(self._commit_count),
                "sqlite_batch_commit_count": int(self._batch_commit_count),
                "sqlite_last_commit_ms": float(self._last_commit_ms),
                "sqlite_wal_checkpoint_count": 0,
                "episodic_write_batch_size_last": int(self._write_batch_size_last),
                "episodic_db_rows": rows,
                "episodic_db_pruned_rows": int(self._db_pruned_rows),
                "memory_db_bytes": _dir_size_bytes(self._dir),
                "memory_wal_bytes": 0,
            }
