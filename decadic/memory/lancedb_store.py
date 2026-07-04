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
- Search is served by an in-process **full-mirror L1 cache** (WS4-M5 cutover):
  every live row's embedding sits in one contiguous float32 ``(n, 80)`` matrix
  (grown by doubling) with parallel id/cycle/salience/has-embedding arrays, so
  ``search_similar``/``search_similar_percept`` are a single vectorized
  normalized-dot-product over the entire corpus (~sub-ms at 100k rows) with
  Lance as durability/source-of-truth. The percept-key matrix is the
  ``[:, PERCEPT_KEY_SLICE]`` view of the embedding matrix (no copy). The
  mirror is write-through on ``append`` (read-your-writes holds), prune
  deletions mask rows out (compacted periodically), and on open with existing
  data the full corpus is lazily bulk-loaded from Lance (one columnar read) on
  first search. Scoring matches the SQLite backend's ``_cosine_similarity``
  semantics: degenerate (near-zero-norm) vectors score exactly 0.0.
- Memory guard: ``DECADIC_LANCE_CACHE_MAX_ROWS`` (default 2,000,000) caps the
  mirror (80-d float32 x 2M rows = 640 MB). Past the cap the mirror stops
  growing and queries fall back to brute-force ``table.search(...)`` with the
  exact numpy cosine re-rank (a one-time warning is logged); ANN indexing for
  that regime stays available via ``DECADIC_LANCE_INDEX_THRESHOLD``
  (default off, see :meth:`_maybe_create_indexes_locked`).
"""

from __future__ import annotations

import json
import logging
import os
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
# L1 full-mirror cache (WS4-M5): row cap before queries fall back to Lance.
_MIRROR_CAP_ENV = "DECADIC_LANCE_CACHE_MAX_ROWS"
_MIRROR_CAP_DEFAULT = 2_000_000
# Initial mirror capacity; grows by doubling (amortized O(1) appends).
_MIRROR_MIN_CAPACITY = 1024


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
        # WS4-M1.3 ANN index state (created lazily past the row threshold).
        self._indexed_at_rows: int = 0
        self._index_builds: int = 0
        self._last_index_ms: float = 0.0
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
        # WS4-M5 cutover: full-mirror L1 recall cache (see module docstring).
        # Contiguous column-parallel arrays over [0:_mirror_n); dead (pruned)
        # rows are masked via _mirror_alive and compacted periodically.
        self._mirror_loaded = False
        self._mirror_disabled = False
        self._mirror_warned = False
        self._mirror_n = 0
        self._mirror_dead = 0
        self._mirror_hits = 0
        self._mirror_misses = 0
        self._mirror_emb = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._mirror_ids = np.zeros((0,), dtype=np.int64)
        self._mirror_cycles = np.zeros((0,), dtype=np.int64)
        self._mirror_salience = np.zeros((0,), dtype=np.float32)
        self._mirror_has = np.zeros((0,), dtype=bool)
        self._mirror_alive = np.zeros((0,), dtype=bool)
        self._mirror_norm_full = np.zeros((0,), dtype=np.float32)
        self._mirror_norm_key = np.zeros((0,), dtype=np.float32)
        self._mirror_summary: list[str] = []

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
        # Sanitize non-finite values at the store boundary. The sqlite backend
        # silently persisted NaN/inf embeddings (raw blob) and its cosine path
        # simply never ranked them; lance VALIDATES vectors on add() and raises
        # "Vector column contains NaN values", which turns one bad episode into
        # a failed flush on the caller's thread (observed: stub-pipeline rows in
        # test_api_dashboard under the flipped defaults). Zeroing them preserves
        # the sqlite-observable behavior: the row is stored, and a zero vector's
        # norm-guard gives similarity 0.0 so it never wins a search.
        if not np.all(np.isfinite(vec)):
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
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
            row = self._row_from_record(record)
            self._pending.append(row)
            # Write-through BEFORE any flush/prune so read-your-writes holds
            # regardless of buffering state (no-op until the lazy bulk load).
            self._mirror_write_through_locked(row)
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
        self._maybe_create_indexes_locked(tbl)

    # ---------------------------------------------------------------- ANN (M1.3)

    def _maybe_create_indexes_locked(self, tbl: Any) -> None:
        """WS4-M1.3: create ANN indexes once the corpus crosses the threshold.

        Brute-force cosine over 80-d is ~30 ms p50 at 100k rows on the dev box
        (measured 2026-07-03) - too slow for a 70-90 ms cycle budget, and 100k
        episodes is only ~3 h of agent life at 10 Hz. IVF-PQ indexes on both
        vector columns restore single-digit-ms search at full corpus fidelity.
        Lance uses the index transparently at query time; rows added after
        index creation are still searched (lance merges unindexed fragments),
        so no staleness handling is needed - we simply re-create periodically
        to keep the indexed fraction high.

        Env: DECADIC_LANCE_INDEX_THRESHOLD (default 0 = DISABLED),
             DECADIC_LANCE_INDEX_REBUILD_ROWS (rows between re-creations).

        Default OFF (measured 2026-07-03, 100k rows, dev box): our
        search_similar applies where-clause filters, and without scalar
        indexes on the filter columns lance's filter path dominates - the
        ANN index gave zero query speedup while synchronous rebuilds cut
        ingest from ~10k to ~1.5k rows/s. Brute force measures 20-30 ms
        full-corpus at 100k, inside the cycle budget. Re-enable after
        filter-aware tuning (scalar indexes on has_embedding/salience or a
        post-filter search path) - tracked as a WS4 follow-up.
        """
        threshold = int(os.environ.get("DECADIC_LANCE_INDEX_THRESHOLD", "0"))
        if threshold <= 0:
            return
        try:
            n = int(tbl.count_rows())
        except Exception:
            return
        if n < threshold:
            return
        # Rows added after an index build sit in unindexed fragments that
        # lance brute-forces at query time (measured: ~50k-row tail costs
        # ~15 ms/query). Keep the tail small by rebuilding often - index
        # creation at these scales is fast relative to the query time it buys.
        rebuild_every = int(os.environ.get("DECADIC_LANCE_INDEX_REBUILD_ROWS", "20000"))
        if self._indexed_at_rows and n - self._indexed_at_rows < max(1000, rebuild_every):
            return
        # Dimension-aware IVF-PQ parameters: lance's defaults assume large
        # dims; num_sub_vectors must divide the vector dim (80 -> 10 chunks of
        # 8; 16 -> 2 chunks of 8) and num_partitions should scale ~sqrt(n).
        from decadic.memory.embeddings import EMBEDDING_DIM as _EDIM
        from decadic.memory.embeddings import PERCEPT_KEY_DIM as _KDIM

        partitions = max(16, min(1024, int(n**0.5)))
        built = 0
        started = time.perf_counter()
        for column, dim in (("embedding", _EDIM), ("percept_key", _KDIM)):
            try:
                tbl.create_index(
                    metric="cosine",
                    vector_column_name=column,
                    num_partitions=partitions,
                    num_sub_vectors=max(1, dim // 8),
                    replace=True,
                )
                built += 1
            except Exception:
                logger.warning(
                    "lance ANN index creation failed for %s at %d rows",
                    column,
                    n,
                    exc_info=True,
                )
        # Remember the attempt either way so we do not retry every flush.
        self._indexed_at_rows = n
        if built:
            self._index_builds += built
            self._last_index_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "lance ANN indexes built (%d columns) at %d rows in %.0f ms",
                built,
                n,
                self._last_index_ms,
            )

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
        self._mirror_delete_ids_locked({int(r["id"]) for r in victims})

    # ------------------------------------------------------- L1 mirror (M5)

    @staticmethod
    def _mirror_cap_rows() -> int:
        try:
            return int(os.environ.get(_MIRROR_CAP_ENV, str(_MIRROR_CAP_DEFAULT)))
        except ValueError:
            return _MIRROR_CAP_DEFAULT

    def _mirror_live_rows_locked(self) -> int:
        return int(self._mirror_n - self._mirror_dead)

    def _mirror_reset_arrays_locked(self) -> None:
        self._mirror_n = 0
        self._mirror_dead = 0
        self._mirror_emb = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._mirror_ids = np.zeros((0,), dtype=np.int64)
        self._mirror_cycles = np.zeros((0,), dtype=np.int64)
        self._mirror_salience = np.zeros((0,), dtype=np.float32)
        self._mirror_has = np.zeros((0,), dtype=bool)
        self._mirror_alive = np.zeros((0,), dtype=bool)
        self._mirror_norm_full = np.zeros((0,), dtype=np.float32)
        self._mirror_norm_key = np.zeros((0,), dtype=np.float32)
        self._mirror_summary = []

    def _mirror_disable_locked(self, total: int) -> None:
        """Cap exceeded: stop mirroring, free the arrays, fall back to Lance."""
        if not self._mirror_warned:
            logger.warning(
                "lance L1 mirror disabled: corpus (%d rows) exceeds %s=%d; "
                "queries fall back to lance table search",
                total,
                _MIRROR_CAP_ENV,
                self._mirror_cap_rows(),
            )
            self._mirror_warned = True
        self._mirror_disabled = True
        self._mirror_loaded = False
        self._mirror_reset_arrays_locked()

    def _mirror_grow_to_locked(self, need: int) -> None:
        capacity = int(self._mirror_emb.shape[0])
        if need <= capacity:
            return
        new_cap = max(_MIRROR_MIN_CAPACITY, capacity * 2, int(need))
        n = self._mirror_n

        def _grown(old: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            out = np.zeros(shape, dtype=old.dtype)
            out[:n] = old[:n]
            return out

        self._mirror_emb = _grown(self._mirror_emb, (new_cap, EMBEDDING_DIM))
        self._mirror_ids = _grown(self._mirror_ids, (new_cap,))
        self._mirror_cycles = _grown(self._mirror_cycles, (new_cap,))
        self._mirror_salience = _grown(self._mirror_salience, (new_cap,))
        self._mirror_has = _grown(self._mirror_has, (new_cap,))
        self._mirror_alive = _grown(self._mirror_alive, (new_cap,))
        self._mirror_norm_full = _grown(self._mirror_norm_full, (new_cap,))
        self._mirror_norm_key = _grown(self._mirror_norm_key, (new_cap,))

    def _mirror_append_row_locked(self, row: dict[str, Any]) -> None:
        """Write-through a single storage row (dict from _row_from_record)."""
        n = self._mirror_n
        self._mirror_grow_to_locked(n + 1)
        vec = np.asarray(row["embedding"], dtype=np.float32).reshape(-1)
        self._mirror_emb[n] = vec
        self._mirror_ids[n] = int(row["id"])
        self._mirror_cycles[n] = int(row["cycle_index"])
        self._mirror_salience[n] = float(row["salience"])
        self._mirror_has[n] = bool(row["has_embedding"])
        self._mirror_alive[n] = True
        self._mirror_norm_full[n] = float(np.linalg.norm(vec))
        self._mirror_norm_key[n] = float(np.linalg.norm(vec[PERCEPT_KEY_SLICE]))
        self._mirror_summary.append(str(row.get("summary_json") or ""))
        self._mirror_n = n + 1

    def _mirror_write_through_locked(self, row: dict[str, Any]) -> None:
        """append() hook: mirror the new row immediately (read-your-writes).

        Rows written before the lazy bulk load are NOT mirrored here -- the
        load itself picks them up (from the table or the pending buffer), which
        keeps the mirror duplicate-free without an id set.
        """
        if self._mirror_disabled or not self._mirror_loaded:
            return
        live = self._mirror_live_rows_locked()
        if live >= self._mirror_cap_rows():
            self._mirror_disable_locked(live + 1)
            return
        self._mirror_append_row_locked(row)

    def _ensure_mirror_locked(self) -> None:
        """Lazy bulk load: mirror the committed corpus + the pending tail."""
        if self._mirror_disabled or self._mirror_loaded:
            return
        tbl = None
        committed = 0
        if self._table is not None or (self._dir is not None and self._dir.exists()):
            tbl = self._ensure_table_locked()
            committed = int(tbl.count_rows())
        total = committed + len(self._pending)
        if total > self._mirror_cap_rows():
            self._mirror_disable_locked(total)
            return
        self._mirror_reset_arrays_locked()
        if tbl is not None and committed > 0:
            self._mirror_bulk_load_locked(tbl)
        for row in self._pending:
            self._mirror_append_row_locked(row)
        self._mirror_loaded = True

    def _mirror_bulk_load_locked(self, tbl: Any) -> None:
        """One columnar read of every committed row into the mirror arrays."""
        data = tbl.to_arrow()
        try:
            ids = np.asarray(
                data.column("id").to_numpy(zero_copy_only=False), dtype=np.int64
            )
            cycles = np.asarray(
                data.column("cycle_index").to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            salience = np.asarray(
                data.column("salience").to_numpy(zero_copy_only=False),
                dtype=np.float32,
            )
            has = np.asarray(
                data.column("has_embedding").to_numpy(zero_copy_only=False),
                dtype=bool,
            )
            emb_col = data.column("embedding")
            if hasattr(emb_col, "combine_chunks"):
                emb_col = emb_col.combine_chunks()
            flat = getattr(emb_col, "values", None)
            if flat is None:
                flat = emb_col.flatten()
            emb = np.asarray(flat, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
            summaries = [str(s or "") for s in data.column("summary_json").to_pylist()]
        except Exception:
            # Version-proof fallback: row-wise load (arrow API shape drifted).
            for row in data.to_pylist():
                self._mirror_append_row_locked(row)
            return
        m = int(ids.shape[0])
        if m == 0:
            return
        n = self._mirror_n
        self._mirror_grow_to_locked(n + m)
        self._mirror_emb[n : n + m] = emb
        self._mirror_ids[n : n + m] = ids
        self._mirror_cycles[n : n + m] = cycles
        self._mirror_salience[n : n + m] = salience
        self._mirror_has[n : n + m] = has
        self._mirror_alive[n : n + m] = True
        self._mirror_norm_full[n : n + m] = np.linalg.norm(emb, axis=1)
        self._mirror_norm_key[n : n + m] = np.linalg.norm(
            emb[:, PERCEPT_KEY_SLICE], axis=1
        )
        self._mirror_summary.extend(summaries)
        self._mirror_n = n + m

    def _mirror_delete_ids_locked(self, ids: set[int]) -> None:
        """Prune hook: mask victim rows out; compact when half the rows are dead."""
        if self._mirror_disabled or not self._mirror_loaded or not ids:
            return
        n = self._mirror_n
        if n == 0:
            return
        victims = np.isin(
            self._mirror_ids[:n], np.fromiter(ids, dtype=np.int64, count=len(ids))
        )
        victims &= self._mirror_alive[:n]
        count = int(victims.sum())
        if count == 0:
            return
        self._mirror_alive[:n][victims] = False
        self._mirror_dead += count
        if self._mirror_dead >= _MIRROR_MIN_CAPACITY and self._mirror_dead * 2 >= n:
            self._mirror_compact_locked()

    def _mirror_compact_locked(self) -> None:
        n = self._mirror_n
        keep = np.flatnonzero(self._mirror_alive[:n])
        m = int(keep.size)
        self._mirror_emb[:m] = self._mirror_emb[keep]
        self._mirror_ids[:m] = self._mirror_ids[keep]
        self._mirror_cycles[:m] = self._mirror_cycles[keep]
        self._mirror_salience[:m] = self._mirror_salience[keep]
        self._mirror_has[:m] = self._mirror_has[keep]
        self._mirror_norm_full[:m] = self._mirror_norm_full[keep]
        self._mirror_norm_key[:m] = self._mirror_norm_key[keep]
        self._mirror_alive[:m] = True
        self._mirror_alive[m:n] = False
        self._mirror_summary = [self._mirror_summary[int(i)] for i in keep]
        self._mirror_n = m
        self._mirror_dead = 0

    def _mirror_search_locked(
        self,
        qvec: np.ndarray,
        *,
        percept: bool,
        top_k: int,
        min_salience: float | None = None,
        exclude_cycle: int | None = None,
        exclude_cycle_after: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Vectorized cosine ranking over the mirror.

        Returns ``None`` when the mirror is unavailable (over the row cap) so
        the caller falls back to the Lance table path; otherwise the ranked
        hit list (possibly empty). Scoring matches the sqlite backend's
        ``_cosine_similarity``: any vector with norm < 1e-8 scores exactly 0.0.
        ``exclude_cycle_after`` drops rows with ``cycle_index >=`` it (the
        WS3 novelty recency horizon: "familiar" must mean seen BEFORE the
        recent past, not one write-through frame ago).
        """
        self._ensure_mirror_locked()
        if self._mirror_disabled or not self._mirror_loaded:
            return None
        n = self._mirror_n
        mask = self._mirror_alive[:n] & self._mirror_has[:n]
        if min_salience is not None:
            mask &= self._mirror_salience[:n] >= float(min_salience)
        if exclude_cycle is not None:
            mask &= self._mirror_cycles[:n] != int(exclude_cycle)
        if exclude_cycle_after is not None:
            mask &= self._mirror_cycles[:n] < int(exclude_cycle_after)
        eligible = int(np.count_nonzero(mask))
        # Mirror-served counts as a hit even when the filters match nothing --
        # the mirror answered authoritatively (miss = lance-scan fallback).
        self._mirror_hits += 1
        if eligible == 0:
            return []
        # One contiguous BLAS matvec over the WHOLE mirror -- never gather.
        # Fancy-indexing `emb[idxs]` copies every eligible row; with permissive
        # filters that is a full-matrix alloc+copy per query (measured 8.5 ms
        # p50 at 100k rows vs ~1 ms for the straight matvec). Filtered-out rows
        # are excluded by -inf masking of the similarity vector instead.
        # Percept queries are zero-padded to 80-d: q is zero outside
        # PERCEPT_KEY_SLICE, so the full-row dot product equals the key-slice
        # dot product and rides the same contiguous matvec (a strided
        # column-slice view would fall off the fast BLAS path).
        if percept:
            q = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            q[PERCEPT_KEY_SLICE] = np.asarray(qvec, dtype=np.float32).reshape(-1)
            norms = self._mirror_norm_key[:n]
        else:
            q = np.asarray(qvec, dtype=np.float32).reshape(-1)
            norms = self._mirror_norm_full[:n]
        qn = float(np.linalg.norm(q))
        if qn < 1e-8:
            sims = np.zeros(n, dtype=np.float32)
        else:
            sims = (self._mirror_emb[:n] @ q) / np.maximum(norms * qn, 1e-8)
            # Match _cosine_similarity: a degenerate stored vector reads as 0.
            sims = np.where(norms < 1e-8, 0.0, sims).astype(np.float32, copy=False)
        sims[~mask] = -np.inf  # in-place: keeps float32, no per-row gather
        k = max(0, min(int(top_k), eligible))
        if k <= 0:
            return []
        # k <= eligible guarantees no -inf (masked) row can enter the top-k.
        if n > k:
            part = np.argpartition(-sims, k - 1)[:k]
            order = part[np.argsort(-sims[part], kind="stable")]
        else:
            order = np.argsort(-sims, kind="stable")
        out: list[dict[str, Any]] = []
        for gi_ in order:
            gi = int(gi_)
            item = self._hit_from_row(
                {
                    "cycle_index": int(self._mirror_cycles[gi]),
                    "salience": float(self._mirror_salience[gi]),
                    "summary_json": self._mirror_summary[gi],
                }
            )
            item["embedding"] = self._mirror_emb[gi].astype(float).tolist()
            item["similarity"] = float(sims[gi])
            out.append(item)
        return out

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
        with self._lock:
            # Rank at least 1 so the side channel matches sqlite exactly even
            # for top_k <= 0 (sqlite sets last_best from the best candidate,
            # then returns the trimmed-empty list).
            hits = self._mirror_search_locked(
                q,
                percept=False,
                top_k=max(1, int(top_k)),
                min_salience=min_salience,
                exclude_cycle=exclude_cycle,
            )
            if hits is not None:
                self.last_best_similarity = (
                    float(hits[0]["similarity"]) if hits else None
                )
                return hits[: max(0, int(top_k))]

            # Over the mirror cap: brute-force Lance search + pending merge.
            self._mirror_misses += 1
            clauses = [
                "has_embedding = true",
                f"salience >= {float(min_salience)!r}",
            ]
            if exclude_cycle is not None:
                clauses.append(f"cycle_index <> {int(exclude_cycle)}")
            where = " AND ".join(clauses)
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
            self.last_best_similarity = ranked[0][0] if ranked else None
            return [r for _, r in ranked[: max(0, top_k)]]

    def search_similar_percept(
        self,
        key: np.ndarray,
        top_k: int = 5,
        exclude_cycle_after: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rank by cosine over the 16-d percept-key sub-vector only (WS4-M3.1).

        Same hit-dict shape as :meth:`search_similar`; ``similarity`` is the
        percept-key cosine. Side channel: ``last_best_percept_similarity``
        (None when the store is empty). ``exclude_cycle_after`` excludes rows
        with ``cycle_index >=`` it -- the WS3 novelty recency horizon: without
        it the best match is always the previous write-through frame
        (similarity ~1.0) and novelty is identically zero (probe 2026-07-04).
        """
        k = np.asarray(key, dtype=np.float32).reshape(-1)
        if k.size != PERCEPT_KEY_DIM:
            k = np.pad(k, (0, max(0, PERCEPT_KEY_DIM - k.size)))[:PERCEPT_KEY_DIM]

        with self._lock:
            hits = self._mirror_search_locked(
                k,
                percept=True,
                top_k=max(1, int(top_k)),
                exclude_cycle_after=exclude_cycle_after,
            )
            if hits is not None:
                self.last_best_percept_similarity = (
                    float(hits[0]["similarity"]) if hits else None
                )
                return hits[: max(0, int(top_k))]

            # Over the mirror cap: brute-force Lance search + pending merge.
            self._mirror_misses += 1
            where = "has_embedding = true"
            if exclude_cycle_after is not None:
                where += f" AND cycle_index < {int(exclude_cycle_after)}"
            candidates = self._candidate_rows_locked(
                k, column="percept_key", top_k=top_k, where=where
            )
            candidates.extend(
                r
                for r in self._pending
                if r["has_embedding"]
                and (
                    exclude_cycle_after is None
                    or int(r["cycle_index"]) < int(exclude_cycle_after)
                )
            )

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
            # Empty corpus: the mirror is trivially complete (and re-enabled).
            self._mirror_reset_arrays_locked()
            self._mirror_disabled = False
            self._mirror_loaded = True

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
            # Invalidate the mirror: the next search bulk-loads the restored corpus.
            self._mirror_reset_arrays_locked()
            self._mirror_disabled = False
            self._mirror_loaded = False

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
            self._mirror_reset_arrays_locked()
            self._mirror_loaded = False
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
        """The L1 full mirror IS the recall cache (WS4-M5 cutover).

        ``size`` is live mirror rows (0 until the lazy bulk load or when the
        row cap disabled it); a ``hit`` is a query answered by the mirror
        (even when the filters match nothing -- the mirror is authoritative),
        a ``miss`` is a query that fell back to the Lance table scan.
        """
        with self._lock:
            return {
                "enabled": not self._mirror_disabled,
                "size": self._mirror_live_rows_locked(),
                "hits": int(self._mirror_hits),
                "misses": int(self._mirror_misses),
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
                "lance_index_builds": int(self._index_builds),
                "lance_indexed_at_rows": int(self._indexed_at_rows),
                "lance_last_index_ms": float(self._last_index_ms),
                "lance_mirror_rows": self._mirror_live_rows_locked(),
                "lance_mirror_enabled": not self._mirror_disabled,
            }
