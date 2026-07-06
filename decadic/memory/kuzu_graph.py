"""Kuzu-backed long-term graph (WS4-M2).

Drop-in replacement for :class:`decadic.memory.semantic_graph.LongTermGraph`
selected via ``DECADIC_GRAPH_BACKEND=kuzu`` (see ``decadic.memory.factory``).

Design: :class:`KuzuLongTermGraph` *subclasses* ``LongTermGraph`` and keeps its
entire in-memory model (nodes/edges/beliefs/semantic dicts, the recent+salient
match cache, the Bayesian belief math, ``consolidate``/``snapshot``/stats) --
so identity decisions, belief trajectories and snapshot shapes are identical to
the sqlite backend *by construction*. Only the persistence layer is swapped:
instead of an sqlite connection, dirty records are queued and flushed to an
embedded Kuzu database. Kuzu (like sqlite here) is the durability layer; the
in-memory dicts remain the read model.

Schema mapping (sqlite -> kuzu):

- ``nodes``            -> node table ``Entity(id STRING PK, kind, appearance
  FLOAT[dim], appearance_json STRING, salience, seen_count, first_cycle,
  last_cycle, position_json, affect)``. ``appearance_json`` is the exact,
  dim-agnostic source of truth (mirrors sqlite's json/blob column);
  ``appearance`` is a fixed-dim copy (zero-padded/truncated to
  ``DECADIC_KUZU_APPEARANCE_DIM``, default 16 -- the production appearance
  fingerprint dim) that exists solely for the native vector index.
- ``edges``            -> ONE rel table ``RELATES(FROM Entity TO Entity, kind,
  weight, count, last_cycle)``. The sqlite table is a single table with an
  open-ended ``kind`` column (``co_occurrence``, dynamic ``scene_<kind>``...),
  so one rel table with a ``kind`` property is the minimal-diff mapping; the
  PRD's fixed CO_PRESENT/SPATIAL/TEMPORAL tables cannot hold dynamic kinds.
- ``property_beliefs`` -> node table ``PropertyBelief`` keyed by a synthesized
  ``pk = node_id + "\\x1f" + property_key`` (kuzu has no composite PKs).
  Separate table (not JSON-on-Entity) because the sqlite backend persists
  belief rows individually; row-level upsert/delete mirrors that exactly while
  the belief *semantics* run in the inherited in-memory code.
- ``semantic_records`` -> node table ``SemanticRecord`` keyed by
  ``pk = category + "\\x1f" + id`` (same rationale).

Identity matching: the in-memory match cache stays in front (identical hit-rate
logic, inherited). The Kuzu native vector index replaces the *linear scan*,
i.e. it is used only when the cache is disabled
(``DECADIC_LTM_MATCH_CACHE_ENABLED=0``), exactly as PRD 5.3 specifies. Index
staleness (kuzu vector indexes do not reflect rows written after index
creation, and EMA appearance updates stale-ify indexed vectors): every
persisted node id joins a "tail" set; a match query = index top-k UNION the
tail, re-ranked with the exact numpy cosine used by the sqlite scan; the index
is dropped+recreated once the tail reaches
``DECADIC_KUZU_VECTOR_REBUILD_INTERVAL`` (default 128) new/updated nodes. If
the vector extension fails to install/load/build at first use, the instance
remembers that and falls back to the inherited linear scan permanently.

Ephemeral mode (``db_path=None``): a lazily created temp directory removed on
``close()`` -- the same pattern as ``LanceEpisodicStore`` (chosen over
``kuzu.Database(':memory:')``, which is not verified on the pinned version).
``backup_to``/``restore_from`` follow the LanceEpisodicStore pattern too:
flush -> checkpoint -> close handles -> copytree -> reopen (kuzu requires an
explicit close before same-process reopen or the file lock is still held).

Write-behind wiring: :class:`WriteBehindLongTermGraph` is a *subclass* of the
sqlite graph (not a wrapper around an instance), and all of its logic is
written against the public graph surface via ``super()`` calls. So
:class:`WriteBehindKuzuLongTermGraph` is the diamond
``(WriteBehindLongTermGraph, KuzuLongTermGraph)`` -- C3 linearization puts
``KuzuLongTermGraph`` between the wrapper and ``LongTermGraph``, so every
``super()`` call inside the wrapper resolves to the kuzu overrides. Zero
changes to ``ltm_write_behind.py``; full async/enqueue/retention/runtime-
metrics parity for the runtime.

``kuzu`` is imported lazily; this module imports cleanly (and an ephemeral
graph can even be constructed) without the extra installed. No torch anywhere.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from decadic.memory.ltm_write_behind import WriteBehindLongTermGraph
from decadic.memory.semantic_graph import (
    DEFAULT_APPEARANCE_EMA,
    DEFAULT_MATCH_THRESHOLD,
    LongTermGraph,
    _cosine,
)

logger = logging.getLogger(__name__)

_DB_FILE = "graph.kuzu"
_SEP = "\x1f"
_VECTOR_INDEX = "app_idx"
# Extra candidates fetched from the vector index so the exact-cosine re-rank
# never loses a borderline row to the index's own (approximate) ordering.
_INDEX_TOP_K = 8

# --- WS4B: persistence statements (resolve/execute split) --------------------
_Q_NODE_SET = (
    "MATCH (e:Entity) WHERE e.id = $id "
    "SET e.kind = $kind, e.appearance = $appearance, "
    "e.appearance_json = $appearance_json, e.salience = $salience, "
    "e.seen_count = $seen_count, e.first_cycle = $first_cycle, "
    "e.last_cycle = $last_cycle, e.position_json = $position_json, "
    "e.affect = $affect"
)
_Q_NODE_CREATE = (
    "CREATE (:Entity {id: $id, kind: $kind, appearance: $appearance, "
    "appearance_json: $appearance_json, salience: $salience, "
    "seen_count: $seen_count, first_cycle: $first_cycle, "
    "last_cycle: $last_cycle, position_json: $position_json, "
    "affect: $affect})"
)
_Q_EDGE_SET = (
    "MATCH (a:Entity)-[r:RELATES]->(b:Entity) "
    "WHERE a.id = $src AND b.id = $dst AND r.kind = $kind "
    "SET r.weight = $weight, r.count = $count, r.last_cycle = $last_cycle"
)
_Q_EDGE_CREATE = (
    "MATCH (a:Entity), (b:Entity) WHERE a.id = $src AND b.id = $dst "
    "CREATE (a)-[:RELATES {kind: $kind, weight: $weight, count: $count, "
    "last_cycle: $last_cycle}]->(b)"
)
_Q_BELIEF_SET = (
    "MATCH (b:PropertyBelief) WHERE b.pk = $pk "
    "SET b.node_id = $node_id, b.property_key = $property_key, "
    "b.value_json = $value_json, b.mean = $mean, b.variance = $variance, "
    "b.confidence = $confidence, b.evidence_count = $evidence_count, "
    "b.first_cycle = $first_cycle, b.last_cycle = $last_cycle, "
    "b.source = $source, b.unstable = $unstable"
)
_Q_BELIEF_CREATE = (
    "CREATE (:PropertyBelief {pk: $pk, node_id: $node_id, "
    "property_key: $property_key, value_json: $value_json, mean: $mean, "
    "variance: $variance, confidence: $confidence, "
    "evidence_count: $evidence_count, first_cycle: $first_cycle, "
    "last_cycle: $last_cycle, source: $source, unstable: $unstable})"
)
_Q_SEM_SET = (
    "MATCH (s:SemanticRecord) WHERE s.pk = $pk "
    "SET s.category = $category, s.id = $id, "
    "s.payload_json = $payload_json, s.evidence_count = $evidence_count, "
    "s.confidence = $confidence, s.first_cycle = $first_cycle, "
    "s.last_cycle = $last_cycle, s.promoted = $promoted"
)
_Q_SEM_CREATE = (
    "CREATE (:SemanticRecord {pk: $pk, category: $category, id: $id, "
    "payload_json: $payload_json, evidence_count: $evidence_count, "
    "confidence: $confidence, first_cycle: $first_cycle, "
    "last_cycle: $last_cycle, promoted: $promoted})"
)
_Q_NODE_DEL = "MATCH (e:Entity) WHERE e.id = $id DETACH DELETE e"
_Q_SEM_DEL = "MATCH (s:SemanticRecord) WHERE s.pk = $pk DELETE s"


# Unwind alias MUST NOT collide with any variable used in the templates:
# `e` (Entity), `a`/`b` (Entity endpoints AND PropertyBelief `b`), `r`
# (the RELATES relationship in the edge SET/MATCH), `s` (SemanticRecord).
# `__u` is safe against all of them. (Using `r` silently broke edge-SET flush
# with "r has data type STRUCT but REL was expected" -- the unwound row
# shadowed the relationship variable; caught by the embodied soak, now pinned
# by a dedicated edge test.)
_UNWIND_ALIAS = "__u"


def _multirow(q: str) -> str:
    """UNWIND variant of a single-row statement: every ``$name`` parameter
    becomes a field of the unwound row. Verified on the pinned kuzu (0.11):
    list-of-dict params bind as structs, NULL fields and FLOAT[dim] vector
    columns round-trip, heterogeneous int/float/None rows unify, and MATCH
    inside UNWIND works for node SET/CREATE, edge SET, and edge CREATE."""
    return f"UNWIND $rows AS {_UNWIND_ALIAS} " + re.sub(
        r"\$(\w+)", rf"{_UNWIND_ALIAS}.\1", q
    )


# WS4B M3.4: multi-row templates. DELETEs stay single-row (rare; DETACH DELETE
# ordering is easiest to reason about one at a time).
_Q_MULTIROW = {
    q: _multirow(q)
    for q in (
        _Q_NODE_SET,
        _Q_NODE_CREATE,
        _Q_EDGE_SET,
        _Q_EDGE_CREATE,
        _Q_BELIEF_SET,
        _Q_BELIEF_CREATE,
        _Q_SEM_SET,
        _Q_SEM_CREATE,
    )
}


def _multirow_enabled() -> bool:
    """WS4B M3.4 (2026-07-05): coalesce same-template statement runs into one
    ``UNWIND $rows`` statement each. The 1-h soak showed insert-heavy streams
    (unique-pk semantic records) defeat per-key dedupe and saturate the
    flusher on per-statement overhead -- multi-row execution is the fix."""
    return os.environ.get("DECADIC_KUZU_MULTIROW", "1").strip().lower() not in {
        "0",
        "false",
        "off",
    }


def _group_stmts(
    stmts: list[tuple[str, dict | None]],
) -> list[tuple[str, dict | None]]:
    """Coalesce consecutive RUNS of the same multirow-capable template into a
    single UNWIND statement. Runs only -- global statement order is preserved
    exactly, so create-before-reference and delete/recreate orderings survive
    untouched. On insert-heavy batches this turns N statements into ~1 per
    table; mixed batches degrade gracefully toward the original shape."""
    out: list[tuple[str, dict | None]] = []
    i, n = 0, len(stmts)
    while i < n:
        q, p = stmts[i]
        mq = _Q_MULTIROW.get(q) if p is not None else None
        if mq is None:
            out.append((q, p))
            i += 1
            continue
        j = i + 1
        while j < n and stmts[j][0] is q and stmts[j][1] is not None:
            j += 1
        if j - i == 1:
            out.append((q, p))
        else:
            out.append((mq, {"rows": [pp for _, pp in stmts[i:j]]}))
        i = j
    return out


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _appearance_dim() -> int:
    """Fixed dim of the indexed ``appearance`` column (production fingerprint = 16)."""
    return _env_int("DECADIC_KUZU_APPEARANCE_DIM", 16)


def _flush_max_ops() -> int:
    """Unbatched mutations accumulate until this many distinct keys are
    pending. Sized UP 64->512 (2026-07-05): the pending dict dedupes per key,
    and an embodied frame re-touches the same ~6 entities every cycle -- so a
    LONGER window collapses ~5x more repeat updates into one statement each.
    Kuzu absorbs ~2.5 statement-batches/s; the window is the statement-rate
    lever, not just a latency knob."""
    return max(1, int(os.environ.get("DECADIC_KUZU_FLUSH_OPS", "512")))


def _flush_max_age_s() -> float:
    """...or until the oldest pending mutation is this stale (durability
    window; memory remains the live source of truth and episodic memory is
    the experiential ground truth -- a crash loses at most this many seconds
    of GRAPH deltas, which re-derive from experience)."""
    return max(0.1, float(os.environ.get("DECADIC_KUZU_FLUSH_S", "10.0")))


def _flush_queue_cap() -> int:
    """Max queued batches before new batches COALESCE into the tail batch
    (2026-07-05: a drowning flusher grew the queue unboundedly at depth 39;
    coalescing bounds memory and lets same-key statements land last-wins)."""
    return max(1, int(os.environ.get("DECADIC_KUZU_FLUSH_QUEUE_CAP", "4")))


def _offlock_flush_enabled() -> bool:
    """WS4B: execute flush transactions on a background flusher thread so the
    cognitive loop never waits on kuzu (measured 2026-07-05: 293 ms per batch
    under the graph lock = the whole 2.27-vs-4.89 cycles/s gap). OFF restores
    the 2026-07-04 inline behavior for A/B."""
    return os.environ.get("DECADIC_KUZU_OFFLOCK_FLUSH", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _drain_timeout_s() -> float:
    return max(1.0, float(os.environ.get("DECADIC_KUZU_DRAIN_TIMEOUT_S", "30")))


def _write_conn_enabled() -> bool:
    """WS4B upgrade (M0.1 probe PASSED on the dev box): the flusher executes
    on a DEDICATED write connection, so readers of the shared connection never
    wait behind a ~300 ms batch. OFF -> shared-connection + io-lock fallback
    (the behavior the probe's skip branch would have mandated)."""
    return os.environ.get("DECADIC_KUZU_WRITE_CONN", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _rebuild_min_interval_s() -> float:
    """Vector-index rebuilds are full DROP+CREATE passes; a busy embodied
    world stales entities every frame, so the count trigger alone caused
    rebuild storms. Wall-clock floor between rebuilds."""
    return max(0.0, float(os.environ.get("DECADIC_KUZU_REBUILD_MIN_S", "60")))


def _rebuild_interval() -> int:
    return _env_int("DECADIC_KUZU_VECTOR_REBUILD_INTERVAL", 128)


def _vector_index_enabled() -> bool:
    return str(os.environ.get("DECADIC_KUZU_VECTOR_ENABLED", "1")).strip().lower() not in (
        "0",
        "false",
        "off",
    )


def _storage_root_for(db_path: Path) -> Path:
    """Directory for a caller-supplied db path (kept distinct from the sqlite file)."""
    p = Path(db_path)
    if p.suffix == ".kuzu":
        return p
    return Path(str(p) + ".kuzu")


class KuzuLongTermGraph(LongTermGraph):
    """Thread-safe long-term graph persisted to an embedded Kuzu database."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        appearance_ema: float = DEFAULT_APPEARANCE_EMA,
    ) -> None:
        # Base ctor with db_path=None: full in-memory init, no sqlite connection
        # (self._conn stays None forever; every base persistence hook is overridden).
        super().__init__(None, match_threshold=match_threshold, appearance_ema=appearance_ema)
        self._db_path = Path(db_path) if db_path is not None else None
        self._root: Path | None = (
            _storage_root_for(self._db_path) if self._db_path is not None else None
        )
        self._owns_tmp = False
        self._kz_db: Any = None
        self._kz_conn: Any = None
        # Dirty-record buffer flushed at commit points (dedupes within a batch;
        # dict preserves insertion order so nodes flush before edges that need them).
        self._pending: dict[tuple, tuple[str, dict[str, Any]]] = {}
        # Which records exist in kuzu (memory is the source of truth, so we know
        # create-vs-update without MERGE); populated on load, updated on flush.
        self._kuzu_node_ids: set[str] = set()
        self._kuzu_edge_keys: set[tuple[str, str, str]] = set()
        self._kuzu_belief_pks: set[str] = set()
        self._kuzu_semantic_pks: set[str] = set()
        # Vector-index state (see module docstring).
        self._vector_dim = _appearance_dim()
        self._vector_state = "untried" if _vector_index_enabled() else "disabled"
        self._vector_loaded = False
        self._index_built = False
        self._index_rebuilds = 0
        self._rebuild_interval = _rebuild_interval()
        self._last_flush_s = time.perf_counter()
        self._last_rebuild_s = 0.0
        # --- WS4B off-lock flusher state -------------------------------------
        # Lock ORDER (deadlock discipline): self._lock -> {_kz_io_lock |
        # _kz_write_lock}; the io and write locks are never nested in each
        # other. READS use the shared connection under _kz_io_lock; the
        # flusher WRITES on a dedicated connection under _kz_write_lock
        # (M0.1 probe passed), so readers never queue behind a ~300 ms batch.
        # Fallback (probe-skip machines / DECADIC_KUZU_WRITE_CONN=0): writes
        # share the read connection under _kz_io_lock.
        self._kz_io_lock = threading.Lock()
        # Dedicated write connection (lazy; falls back to the shared conn +
        # io-lock when creation fails or DECADIC_KUZU_WRITE_CONN=0).
        self._kz_write_conn: Any | None = None
        self._kz_write_lock = threading.Lock()
        self._flush_mu = threading.Lock()  # guards queue + counters
        self._flush_cv = threading.Condition(self._flush_mu)
        self._flush_queue: deque[tuple[list[tuple[str, dict | None]], list[tuple[str, dict]]]] = deque()
        self._flush_thread: threading.Thread | None = None
        self._flush_stop = False
        self._batches_enqueued = 0
        self._batches_done = 0
        self._flush_error_batches = 0
        self._graph_flush_ms = 0.0  # off-lock execution time of the last batch
        self._graph_flush_lock_ms = 0.0  # resolve time under self._lock (~0 target)
        self._graph_flush_rows = 0  # single-row statements in the last batch (pre-grouping)
        self._graph_flush_stmts = 0  # statements actually executed (post M3.4 grouping)
        self._test_fail_next_batch = False  # failure-drill hook (tests only)
        self._ids_since_index: set[str] = set()
        if self._root is not None:
            with self._lock:
                self._open_locked()
                self._load_kuzu_locked()

    # ---- storage lifecycle ----------------------------------------------
    def _storage_root_locked(self) -> Path:
        if self._root is None:
            self._root = Path(tempfile.mkdtemp(prefix="decadic_kuzu_"))
            self._owns_tmp = True
        return self._root

    def _open_locked(self) -> Any:
        if self._kz_conn is not None:
            return self._kz_conn
        import kuzu  # lazy: module import must not require the extra

        root = self._storage_root_locked()
        root.mkdir(parents=True, exist_ok=True)
        self._kz_db = kuzu.Database(str(root / _DB_FILE))
        self._kz_conn = kuzu.Connection(self._kz_db)
        self._ensure_kuzu_schema_locked()
        # A fresh connection has no extension loaded and (conservatively) no
        # trusted index: everything currently in memory is "tail" until rebuilt.
        self._vector_loaded = False
        self._index_built = False
        self._ids_since_index = set(self._nodes)
        return self._kz_conn

    def _ensure_kuzu_schema_locked(self) -> None:
        dim = int(self._vector_dim)
        statements = (
            "CREATE NODE TABLE IF NOT EXISTS Entity("
            "id STRING, kind STRING, "
            f"appearance FLOAT[{dim}], appearance_json STRING, "
            "salience DOUBLE, seen_count INT64, first_cycle INT64, "
            "last_cycle INT64, position_json STRING, affect DOUBLE, "
            "PRIMARY KEY(id))",
            "CREATE REL TABLE IF NOT EXISTS RELATES("
            "FROM Entity TO Entity, kind STRING, weight DOUBLE, "
            "count INT64, last_cycle INT64)",
            "CREATE NODE TABLE IF NOT EXISTS PropertyBelief("
            "pk STRING, node_id STRING, property_key STRING, value_json STRING, "
            "mean DOUBLE, variance DOUBLE, confidence DOUBLE, evidence_count DOUBLE, "
            "first_cycle INT64, last_cycle INT64, source STRING, unstable BOOLEAN, "
            "PRIMARY KEY(pk))",
            "CREATE NODE TABLE IF NOT EXISTS SemanticRecord("
            "pk STRING, category STRING, id STRING, payload_json STRING, "
            "evidence_count DOUBLE, confidence DOUBLE, first_cycle INT64, "
            "last_cycle INT64, promoted BOOLEAN, PRIMARY KEY(pk))",
        )
        for stmt in statements:
            try:
                self._kz_conn.execute(stmt)
            except Exception:
                # Older kuzu without IF NOT EXISTS: plain create, tolerate "exists".
                try:
                    self._kz_conn.execute(stmt.replace(" IF NOT EXISTS", ""))
                except Exception:
                    logger.debug("kuzu schema statement skipped (table exists?)", exc_info=True)

    def _checkpoint_locked(self) -> None:
        if self._kz_conn is None:
            return
        try:
            with self._kz_io_lock:
                self._kz_conn.execute("CHECKPOINT;")
            self._sqlite_wal_checkpoint_count += 1
        except Exception:
            logger.debug("kuzu checkpoint unsupported/failed (harmless)", exc_info=True)

    def _close_handles_locked(self) -> None:
        # Explicit, hasattr-guarded close BEFORE any same-process reopen or file
        # copy: on some kuzu versions `del` alone leaves the file lock held
        # ("Could not set lock on file").
        with self._kz_write_lock:
            for h in (self._kz_write_conn, self._kz_conn, self._kz_db):
                close = getattr(h, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            self._kz_write_conn = None
        self._kz_conn = None
        self._kz_db = None
        self._vector_loaded = False
        self._index_built = False
        gc.collect()  # best effort: release file handles (Windows rmtree/copy)

    def close(self) -> None:
        """Flush, checkpoint and release kuzu handles (temp dirs removed)."""
        with self._lock:
            try:
                if self._pending and (
                    self._kz_conn is not None
                    or (self._root is not None and not self._owns_tmp)
                ):
                    self._drain_locked()
                # Stop + join the flusher BEFORE handle release (WS4B drain
                # barrier: the Windows file-lock discipline).
                if self._flush_thread is not None:
                    self._quiesce_flusher_locked()
            except Exception:  # pragma: no cover - close must never raise
                logger.exception("kuzu graph close: drain failed")
            self._checkpoint_locked()
            self._close_handles_locked()
            if self._owns_tmp and self._root is not None:
                shutil.rmtree(self._root, ignore_errors=True)
                self._root = None
                self._owns_tmp = False

    def __del__(self) -> None:  # pragma: no cover - GC timing dependent
        try:
            self.close()
        except Exception:
            pass

    # ---- persistence hooks (override the sqlite writes) -----------------
    def _persist_node(self, node: dict[str, Any]) -> None:
        app = node.get("appearance")
        app_list = (
            [float(x) for x in np.asarray(app, dtype=np.float32).reshape(-1).tolist()]
            if app is not None
            else []
        )
        vec = [0.0] * self._vector_dim
        n = min(len(app_list), self._vector_dim)
        vec[:n] = app_list[:n]
        nid = str(node["id"])
        params = {
            "id": nid,
            "kind": str(node["kind"]),
            "appearance": [float(x) for x in vec],
            "appearance_json": json.dumps(app_list),
            "salience": float(node["salience"]),
            "seen_count": int(node["seen_count"]),
            "first_cycle": int(node["first_cycle"]),
            "last_cycle": int(node["last_cycle"]),
            "position_json": json.dumps(node["position"]) if node["position"] is not None else None,
            "affect": float(node["affect"]),
        }
        self._pending[("node", nid)] = ("node", params)
        self._ids_since_index.add(nid)  # new or EMA-updated: index entry is stale
        self._commit_locked()

    def _persist_edge(self, e: dict[str, Any]) -> None:
        params = {
            "src": str(e["src"]),
            "dst": str(e["dst"]),
            "kind": str(e["kind"]),
            "weight": float(e["weight"]),
            "count": int(e["count"]),
            "last_cycle": int(e["last_cycle"]),
        }
        self._pending[("edge", params["src"], params["dst"], params["kind"])] = ("edge", params)
        self._commit_locked()

    def _persist_belief(self, b: dict[str, Any]) -> None:
        pk = f"{b['node_id']}{_SEP}{b['property_key']}"
        params = {
            "pk": pk,
            "node_id": str(b["node_id"]),
            "property_key": str(b["property_key"]),
            "value_json": json.dumps(b.get("value")),
            "mean": float(b["mean"]),
            "variance": float(b["variance"]),
            "confidence": float(b["confidence"]),
            "evidence_count": float(b["evidence_count"]),
            "first_cycle": int(b["first_cycle"]),
            "last_cycle": int(b["last_cycle"]),
            "source": str(b.get("source", "perception")),
            "unstable": bool(b.get("unstable", False)),
        }
        self._pending[("belief", pk)] = ("belief", params)
        self._commit_locked()

    def _persist_semantic_record(self, category: str, rec: dict[str, Any]) -> None:
        pk = f"{category}{_SEP}{rec['id']}"
        params = {
            "pk": pk,
            "category": str(category),
            "id": str(rec["id"]),
            "payload_json": json.dumps(rec.get("payload", {})),
            "evidence_count": float(rec.get("evidence_count", 0.0)),
            "confidence": float(rec.get("confidence", 0.0)),
            "first_cycle": int(rec.get("first_cycle", 0)),
            "last_cycle": int(rec.get("last_cycle", 0)),
            "promoted": bool(rec.get("promoted", False)),
        }
        self._pending[("sem", pk)] = ("sem", params)
        self._commit_locked()

    def _commit_locked(self, *, batch: bool = False) -> None:
        if self._persist_batch_depth > 0:
            return
        if not self._pending and self._kz_conn is None:
            return  # nothing to do; never create storage for a no-op
        # Deferred-flush policy (2026-07-04 MuJoCo disk-storm fix): the sqlite
        # backend's protection rules were write-behind + BATCHED commits; this
        # port flushed the whole pending buffer after every unbatched mutation
        # with per-op autocommit fsyncs -- 100% disk active time the first
        # time discovered perception populated the graph. Memory is the source
        # of truth (reads never touch kuzu on the hot path); kuzu is
        # durability. Accumulate, flush when big or stale.
        # 2026-07-05 finding: EVERY production flush arrived via write_batch()
        # (batch_commit_count == commit_count in all diag arms), whose exit
        # bypassed this deferral -- one forced flush per cycle, saturating the
        # flusher regardless of window size. Under WS4B, write_batch marks a
        # CONSISTENCY GROUPING, not urgency: memory is the source of truth and
        # the pending dict is per-key last-wins, so the deferral window applies
        # to batched and unbatched mutations alike. Only drain/close/backup
        # force durability (they call _flush_pending_locked directly).
        if self._pending:
            age = time.perf_counter() - self._last_flush_s
            if len(self._pending) < _flush_max_ops() and age < _flush_max_age_s():
                return
        started = time.perf_counter()
        self._flush_pending_locked()
        self._last_flush_s = time.perf_counter()
        self._graph_flush_lock_ms = (time.perf_counter() - started) * 1000.0
        self._sqlite_last_commit_ms = self._graph_flush_lock_ms
        self._sqlite_commit_count += 1
        if batch:
            self._sqlite_batch_commit_count += 1

    @contextmanager
    def write_batch(self):
        with self._lock:
            self._persist_batch_depth += 1
            try:
                yield
            finally:
                self._persist_batch_depth = max(0, self._persist_batch_depth - 1)
                if self._persist_batch_depth == 0:
                    self._commit_locked(batch=True)

    def _flush_pending_locked(self) -> None:
        """WS4B two-phase flush: RESOLVE under self._lock, EXECUTE off it.

        Resolve is pure Python (microseconds): pending ops become fully-formed
        (cypher, params) statements, membership sets are consulted AND updated
        here. Execution -- one explicit transaction, one WAL fsync -- happens
        on the flusher thread (or inline when DECADIC_KUZU_OFFLOCK_FLUSH=0),
        so the cognitive loop never waits the ~300 ms a batch costs.
        """
        if not self._pending:
            return
        self._open_locked()  # connection must exist before the flusher needs it
        ops = list(self._pending.values())
        self._pending = {}
        stmts = self._resolve_ops_locked(ops)
        if not stmts:
            return
        if _offlock_flush_enabled():
            with self._flush_mu:
                if len(self._flush_queue) >= _flush_queue_cap():
                    # Backpressure by coalescing: extend the tail batch
                    # instead of growing the queue (same-key statements
                    # execute in order, last wins -- correct, just merged).
                    tail_stmts, tail_ops = self._flush_queue[-1]
                    tail_stmts.extend(stmts)
                    tail_ops.extend(ops)
                else:
                    self._flush_queue.append((stmts, ops))
                    self._batches_enqueued += 1
            self._ensure_flusher()
        else:
            self._execute_batch(stmts, ops)

    def _resolve_ops_locked(
        self, ops: list[tuple[str, dict]]
    ) -> list[tuple[str, dict | None]]:
        """Pure resolve: ops -> statements; sets updated; NO kuzu calls.

        Set-consistency contract: sets reflect resolve-time intent. If the
        batch later fails, the replay path re-probes existence per op, so a
        successful replay converges kuzu to what the sets already claim. Only
        a double failure leaves a divergence (logged loudly; memory unharmed).
        """
        stmts: list[tuple[str, dict | None]] = []
        for kind, p in ops:
            if kind == "node":
                if p["id"] in self._kuzu_node_ids:
                    stmts.append((_Q_NODE_SET, p))
                else:
                    stmts.append((_Q_NODE_CREATE, p))
                    self._kuzu_node_ids.add(p["id"])
            elif kind == "edge":
                key = (p["src"], p["dst"], p["kind"])
                if key in self._kuzu_edge_keys:
                    stmts.append((_Q_EDGE_SET, p))
                else:
                    stmts.append((_Q_EDGE_CREATE, p))
                    self._kuzu_edge_keys.add(key)
            elif kind == "belief":
                if p["pk"] in self._kuzu_belief_pks:
                    stmts.append((_Q_BELIEF_SET, p))
                else:
                    stmts.append((_Q_BELIEF_CREATE, p))
                    self._kuzu_belief_pks.add(p["pk"])
            elif kind == "sem":
                if p["pk"] in self._kuzu_semantic_pks:
                    stmts.append((_Q_SEM_SET, p))
                else:
                    stmts.append((_Q_SEM_CREATE, p))
                    self._kuzu_semantic_pks.add(p["pk"])
            elif kind == "del_node":
                stmts.append((_Q_NODE_DEL, p))
                self._kuzu_node_ids.discard(p["id"])
                self._ids_since_index.discard(p["id"])
            elif kind == "del_sem":
                stmts.append((_Q_SEM_DEL, p))
                self._kuzu_semantic_pks.discard(p["pk"])
        return stmts

    # -- execution side (flusher thread or inline; NEVER takes self._lock) ----
    def _ensure_flusher(self) -> None:
        t = self._flush_thread
        if t is not None and t.is_alive():
            with self._flush_cv:
                self._flush_cv.notify_all()
            return
        self._flush_stop = False
        t = threading.Thread(
            target=self._flusher_loop, name="kuzu-flusher", daemon=True
        )
        self._flush_thread = t
        t.start()

    def _flusher_loop(self) -> None:
        while True:
            with self._flush_cv:
                while not self._flush_queue and not self._flush_stop:
                    self._flush_cv.wait(timeout=0.5)
                if self._flush_stop and not self._flush_queue:
                    return
                batch = self._flush_queue.popleft() if self._flush_queue else None
            if batch is None:
                continue
            stmts, ops = batch
            try:
                self._execute_batch(stmts, ops)
            except Exception:  # pragma: no cover - flusher must never die
                logger.exception("kuzu flusher: batch execution crashed")
            with self._flush_cv:
                self._batches_done += 1
                self._flush_cv.notify_all()

    def _acquire_write_channel(self) -> tuple[Any, Any]:
        """(connection, lock) for batch execution.

        Dedicated write connection when available (M0.1 probe passed:
        cross-connection write+read is safe on this kuzu build), so readers
        of the shared connection never queue behind a batch. Falls back to
        the shared connection + io-lock on any failure -- the probe's skip
        branch, always available."""
        if _write_conn_enabled():
            with self._kz_write_lock:
                if self._kz_write_conn is None and self._kz_db is not None:
                    try:
                        import kuzu  # lazy, mirrors _open_locked

                        self._kz_write_conn = kuzu.Connection(self._kz_db)
                    except Exception:
                        logger.warning(
                            "kuzu dedicated write connection unavailable; "
                            "sharing the read connection",
                            exc_info=True,
                        )
            if self._kz_write_conn is not None:
                return self._kz_write_conn, self._kz_write_lock
        return self._kz_conn, self._kz_io_lock

    def _execute_batch(
        self, stmts: list[tuple[str, dict | None]], ops: list[tuple[str, dict]]
    ) -> None:
        started = time.perf_counter()
        # M3.4: run-coalesced multi-row statements. Grouping happens off-lock
        # too (pure list walk); the failure path replays from OPS, which is
        # statement-shape agnostic, so grouping never weakens durability.
        exec_stmts = _group_stmts(stmts) if _multirow_enabled() else stmts
        self._graph_flush_rows = len(stmts)
        self._graph_flush_stmts = len(exec_stmts)
        conn, channel_lock = self._acquire_write_channel()
        with channel_lock:
            if conn is None:
                logger.warning("kuzu flush: connection gone; batch dropped (%d ops)", len(ops))
                return
            try:
                if self._test_fail_next_batch:
                    self._test_fail_next_batch = False
                    raise RuntimeError("test-injected batch failure")
                conn.execute("BEGIN TRANSACTION;")
                for q, p in exec_stmts:
                    conn.execute(q, p) if p is not None else conn.execute(q)
                conn.execute("COMMIT;")
                self._graph_flush_ms = (time.perf_counter() - started) * 1000.0
                return
            except Exception:
                self._flush_error_batches += 1
                logger.exception("kuzu batched flush failed; replaying per-op")
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
            for kind, p in ops:
                try:
                    self._replay_op(conn, kind, p)
                except Exception:  # pragma: no cover - durability only
                    logger.exception("kuzu replay op failed (%s)", kind)
        self._graph_flush_ms = (time.perf_counter() - started) * 1000.0

    def _replay_op(self, conn: Any, kind: str, p: dict) -> None:
        """Set-agnostic single-op replay: probe existence, then write.

        Used only after a rolled-back batch; converges kuzu to the state the
        membership sets already claim (PRD ws4b 3.2)."""

        def _exists(query: str, params: dict) -> bool:
            res = conn.execute(query, params)
            return bool(res.has_next())

        if kind == "node":
            hit = _exists("MATCH (e:Entity) WHERE e.id = $id RETURN e.id", {"id": p["id"]})
            conn.execute(_Q_NODE_SET if hit else _Q_NODE_CREATE, p)
        elif kind == "edge":
            hit = _exists(
                "MATCH (a:Entity)-[r:RELATES]->(b:Entity) "
                "WHERE a.id = $src AND b.id = $dst AND r.kind = $kind RETURN r.kind",
                {"src": p["src"], "dst": p["dst"], "kind": p["kind"]},
            )
            conn.execute(_Q_EDGE_SET if hit else _Q_EDGE_CREATE, p)
        elif kind == "belief":
            hit = _exists("MATCH (b:PropertyBelief) WHERE b.pk = $pk RETURN b.pk", {"pk": p["pk"]})
            conn.execute(_Q_BELIEF_SET if hit else _Q_BELIEF_CREATE, p)
        elif kind == "sem":
            hit = _exists("MATCH (s:SemanticRecord) WHERE s.pk = $pk RETURN s.pk", {"pk": p["pk"]})
            conn.execute(_Q_SEM_SET if hit else _Q_SEM_CREATE, p)
        elif kind == "del_node":
            conn.execute(_Q_NODE_DEL, p)
        elif kind == "del_sem":
            conn.execute(_Q_SEM_DEL, p)

    def _drain_locked(self, timeout: float | None = None) -> bool:
        """Flush pending and block until every enqueued batch has executed.

        The barrier for callers that need COMMITTED state (backup, restore,
        close, clear, index build). Safe under self._lock: the flusher only
        needs _kz_io_lock, so it makes progress while we wait."""
        if self._pending:
            started = time.perf_counter()
            self._flush_pending_locked()
            self._last_flush_s = time.perf_counter()
            self._graph_flush_lock_ms = (time.perf_counter() - started) * 1000.0
            self._sqlite_commit_count += 1
        if not _offlock_flush_enabled():
            return True
        deadline = time.perf_counter() + (timeout if timeout is not None else _drain_timeout_s())
        with self._flush_cv:
            while self._batches_done < self._batches_enqueued:
                remain = deadline - time.perf_counter()
                if remain <= 0:
                    logger.warning(
                        "kuzu drain timed out (%d/%d batches)",
                        self._batches_done,
                        self._batches_enqueued,
                    )
                    return False
                self._flush_cv.wait(timeout=min(0.5, remain))
        return True

    def drain(self, timeout: float | None = None) -> bool:
        """Public drain barrier (durability checkpoint for callers/tests)."""
        with self._lock:
            return self._drain_locked(timeout)

    def _quiesce_flusher_locked(self) -> None:
        """Drain, then stop and join the flusher (restore/close handle swaps)."""
        self._drain_locked()
        t = self._flush_thread
        if t is not None and t.is_alive():
            with self._flush_cv:
                self._flush_stop = True
                self._flush_cv.notify_all()
            t.join(timeout=5.0)
        self._flush_thread = None
        self._flush_stop = False

    def _apply_op_locked(self, conn: Any, kind: str, p: dict[str, Any]) -> None:
        """Retired by WS4B (resolve/execute split) -- delegates to the
        set-agnostic replay path for any residual caller. The statement text
        lives in the module-level _Q_* constants; the dead branch below is
        retained only until the next cleanup pass."""
        self._replay_op(conn, kind, p)
        return
        if True:  # pragma: no cover - unreachable (retired inline path)
            if True:
                if kind == "node":
                    if p["id"] in self._kuzu_node_ids:
                        conn.execute(
                            "MATCH (e:Entity) WHERE e.id = $id "
                            "SET e.kind = $kind, e.appearance = $appearance, "
                            "e.appearance_json = $appearance_json, e.salience = $salience, "
                            "e.seen_count = $seen_count, e.first_cycle = $first_cycle, "
                            "e.last_cycle = $last_cycle, e.position_json = $position_json, "
                            "e.affect = $affect",
                            p,
                        )
                    else:
                        conn.execute(
                            "CREATE (:Entity {id: $id, kind: $kind, appearance: $appearance, "
                            "appearance_json: $appearance_json, salience: $salience, "
                            "seen_count: $seen_count, first_cycle: $first_cycle, "
                            "last_cycle: $last_cycle, position_json: $position_json, "
                            "affect: $affect})",
                            p,
                        )
                        self._kuzu_node_ids.add(p["id"])
                elif kind == "edge":
                    key = (p["src"], p["dst"], p["kind"])
                    if key in self._kuzu_edge_keys:
                        conn.execute(
                            "MATCH (a:Entity)-[r:RELATES]->(b:Entity) "
                            "WHERE a.id = $src AND b.id = $dst AND r.kind = $kind "
                            "SET r.weight = $weight, r.count = $count, r.last_cycle = $last_cycle",
                            p,
                        )
                    else:
                        conn.execute(
                            "MATCH (a:Entity), (b:Entity) WHERE a.id = $src AND b.id = $dst "
                            "CREATE (a)-[:RELATES {kind: $kind, weight: $weight, count: $count, "
                            "last_cycle: $last_cycle}]->(b)",
                            p,
                        )
                        self._kuzu_edge_keys.add(key)
                elif kind == "belief":
                    if p["pk"] in self._kuzu_belief_pks:
                        conn.execute(
                            "MATCH (b:PropertyBelief) WHERE b.pk = $pk "
                            "SET b.node_id = $node_id, b.property_key = $property_key, "
                            "b.value_json = $value_json, b.mean = $mean, b.variance = $variance, "
                            "b.confidence = $confidence, b.evidence_count = $evidence_count, "
                            "b.first_cycle = $first_cycle, b.last_cycle = $last_cycle, "
                            "b.source = $source, b.unstable = $unstable",
                            p,
                        )
                    else:
                        conn.execute(
                            "CREATE (:PropertyBelief {pk: $pk, node_id: $node_id, "
                            "property_key: $property_key, value_json: $value_json, mean: $mean, "
                            "variance: $variance, confidence: $confidence, "
                            "evidence_count: $evidence_count, first_cycle: $first_cycle, "
                            "last_cycle: $last_cycle, source: $source, unstable: $unstable})",
                            p,
                        )
                        self._kuzu_belief_pks.add(p["pk"])
                elif kind == "sem":
                    if p["pk"] in self._kuzu_semantic_pks:
                        conn.execute(
                            "MATCH (s:SemanticRecord) WHERE s.pk = $pk "
                            "SET s.category = $category, s.id = $id, "
                            "s.payload_json = $payload_json, s.evidence_count = $evidence_count, "
                            "s.confidence = $confidence, s.first_cycle = $first_cycle, "
                            "s.last_cycle = $last_cycle, s.promoted = $promoted",
                            p,
                        )
                    else:
                        conn.execute(
                            "CREATE (:SemanticRecord {pk: $pk, category: $category, id: $id, "
                            "payload_json: $payload_json, evidence_count: $evidence_count, "
                            "confidence: $confidence, first_cycle: $first_cycle, "
                            "last_cycle: $last_cycle, promoted: $promoted})",
                            p,
                        )
                        self._kuzu_semantic_pks.add(p["pk"])
                elif kind == "del_node":
                    conn.execute("MATCH (e:Entity) WHERE e.id = $id DETACH DELETE e", p)
                    self._kuzu_node_ids.discard(p["id"])
                    self._ids_since_index.discard(p["id"])
                elif kind == "del_sem":
                    conn.execute("MATCH (s:SemanticRecord) WHERE s.pk = $pk DELETE s", p)
                    self._kuzu_semantic_pks.discard(p["pk"])

    # ---- load (kuzu -> memory) ------------------------------------------
    def _rows_locked(self, query: str) -> list[Any]:
        conn = self._open_locked()
        with self._kz_io_lock:
            res = conn.execute(query)
            out: list[Any] = []
            while res.has_next():
                out.append(res.get_next())
        return out

    def _load_kuzu_locked(self) -> None:
        """Rebuild the in-memory model from kuzu (mirror of the sqlite ``_load_all``)."""
        self._nodes.clear()
        self._edges.clear()
        self._beliefs.clear()
        for bucket in self._semantic.values():
            bucket.clear()
        self._kuzu_node_ids.clear()
        self._kuzu_edge_keys.clear()
        self._kuzu_belief_pks.clear()
        self._kuzu_semantic_pks.clear()
        max_id = 0
        for row in self._rows_locked(
            "MATCH (e:Entity) RETURN e.id, e.kind, e.appearance_json, e.salience, "
            "e.seen_count, e.first_cycle, e.last_cycle, e.position_json, e.affect"
        ):
            nid = str(row[0])
            app = (
                np.asarray(json.loads(row[2]), dtype=np.float32)
                if row[2]
                else np.zeros(0, dtype=np.float32)
            )
            pos = json.loads(row[7]) if row[7] else None
            self._nodes[nid] = {
                "id": nid,
                "kind": row[1] or "unknown",
                "appearance": app,
                "salience": float(row[3] or 0.0),
                "seen_count": int(row[4] or 0),
                "first_cycle": int(row[5] or 0),
                "last_cycle": int(row[6] or 0),
                "position": pos,
                "affect": float(row[8] or 0.0),
            }
            self._kuzu_node_ids.add(nid)
            try:
                max_id = max(max_id, int(nid.split("-")[-1]))
            except ValueError:
                pass
        for row in self._rows_locked(
            "MATCH (a:Entity)-[r:RELATES]->(b:Entity) "
            "RETURN a.id, b.id, r.kind, r.weight, r.count, r.last_cycle"
        ):
            key = (str(row[0]), str(row[1]), str(row[2]))
            self._edges[key] = {
                "src": key[0],
                "dst": key[1],
                "kind": key[2],
                "weight": float(row[3] or 0.0),
                "count": int(row[4] or 0),
                "last_cycle": int(row[5] or 0),
            }
            self._kuzu_edge_keys.add(key)
        for row in self._rows_locked(
            "MATCH (b:PropertyBelief) RETURN b.node_id, b.property_key, b.value_json, "
            "b.mean, b.variance, b.confidence, b.evidence_count, b.first_cycle, "
            "b.last_cycle, b.source, b.unstable"
        ):
            node_id = str(row[0])
            key = str(row[1])
            self._beliefs[(node_id, key)] = {
                "node_id": node_id,
                "property_key": key,
                "value": json.loads(row[2]) if row[2] else None,
                "mean": float(row[3] or 0.0),
                "variance": float(row[4] or 0.0),
                "confidence": float(row[5] or 0.0),
                "evidence_count": float(row[6] or 0.0),
                "first_cycle": int(row[7] or 0),
                "last_cycle": int(row[8] or 0),
                "source": str(row[9] or "perception"),
                "unstable": bool(row[10]),
            }
            self._kuzu_belief_pks.add(f"{node_id}{_SEP}{key}")
        for row in self._rows_locked(
            "MATCH (s:SemanticRecord) RETURN s.category, s.id, s.payload_json, "
            "s.evidence_count, s.confidence, s.first_cycle, s.last_cycle, s.promoted"
        ):
            cat = str(row[0])
            if cat not in self._semantic:
                continue
            rid = str(row[1])
            self._semantic[cat][rid] = {
                "id": rid,
                "payload": json.loads(row[2]) if row[2] else {},
                "evidence_count": float(row[3] or 0.0),
                "confidence": float(row[4] or 0.0),
                "first_cycle": int(row[5] or 0),
                "last_cycle": int(row[6] or 0),
                "promoted": bool(row[7]),
            }
            self._kuzu_semantic_pks.add(f"{cat}{_SEP}{rid}")
            try:
                self._next_semantic_id = max(self._next_semantic_id, int(rid.split("-")[-1]) + 1)
            except ValueError:
                pass
        self._next_id = max_id + 1
        self._ids_since_index = set(self._nodes)
        self._index_built = False
        self._mark_match_cache_dirty()
        self._cached_belief_stats = self._compute_belief_stats_locked()

    # ---- identity match: native vector index path -----------------------
    def _vector_ready_locked(self) -> bool:
        if self._vector_state in ("unavailable", "disabled"):
            return False
        conn = self._open_locked()
        if self._vector_loaded:
            return True
        try:
            conn.execute("INSTALL vector; LOAD vector;")
            self._vector_loaded = True
            self._vector_state = "ready"
            return True
        except Exception as exc:
            logger.warning("kuzu vector extension unavailable; linear-scan fallback: %s", exc)
            self._vector_state = "unavailable"
            return False

    def _ensure_index_locked(self) -> bool:
        if not self._vector_ready_locked():
            return False
        if self._index_built and len(self._ids_since_index) < self._rebuild_interval:
            return True
        # Wall-clock floor between full DROP+CREATE rebuilds (2026-07-04): an
        # embodied world stales entities every frame, so the count trigger
        # alone fired a rebuild storm. Between rebuilds the tail-scan over
        # `_ids_since_index` covers freshly-changed rows, so recall stays
        # correct -- just approximate on the newest handful until the next
        # rebuild. A built index is never dropped early just for staleness.
        if self._index_built and _rebuild_min_interval_s() > 0.0:
            if (time.perf_counter() - self._last_rebuild_s) < _rebuild_min_interval_s():
                return True
        self._drain_locked()  # the index only sees committed rows (WS4B barrier)
        if not self._kuzu_node_ids:
            return False  # empty table: nothing to index (tail scan covers it)
        conn = self._kz_conn
        try:
            with self._kz_io_lock:
                try:
                    conn.execute(f"CALL DROP_VECTOR_INDEX('Entity', '{_VECTOR_INDEX}')")
                except Exception:
                    pass  # first build: no index to drop
                conn.execute(
                    f"CALL CREATE_VECTOR_INDEX('Entity', '{_VECTOR_INDEX}', 'appearance', "
                    "metric := 'cosine')"
                )
        except Exception as exc:
            logger.warning("kuzu vector index build failed; linear-scan fallback: %s", exc)
            self._vector_state = "unavailable"
            return False
        self._index_built = True
        self._index_rebuilds += 1
        self._last_rebuild_s = time.perf_counter()
        self._ids_since_index = set()
        return True

    @staticmethod
    def _id_order(nid: str) -> tuple[int, int, str]:
        try:
            return (0, int(str(nid).rsplit("-", 1)[-1]), str(nid))
        except ValueError:
            return (1, 0, str(nid))

    def _match(
        self,
        appearance: np.ndarray | None,
        threshold: float,
        *,
        exclude: set[str] | None = None,
    ) -> str | None:
        # The in-memory match cache stays in front (inherited, exact parity).
        # The vector index replaces only the linear scan (cache disabled).
        if self._match_cache_enabled:
            return super()._match(appearance, threshold, exclude=exclude)
        if (
            appearance is None
            or int(getattr(appearance, "size", 0)) != self._vector_dim
            or self._vector_state in ("unavailable", "disabled")
        ):
            return super()._match(appearance, threshold, exclude=exclude)
        started = time.perf_counter()
        exclude = exclude or set()
        candidates: set[str] = set(self._ids_since_index)  # post-index tail (incl. pending)
        indexed = False
        if self._kuzu_node_ids or self._index_built:
            try:
                if self._ensure_index_locked():
                    k = min(32, _INDEX_TOP_K + len(exclude))
                    q = [float(x) for x in np.asarray(appearance, dtype=np.float32).reshape(-1)]
                    with self._kz_io_lock:
                        res = self._kz_conn.execute(
                            f"CALL QUERY_VECTOR_INDEX('Entity', '{_VECTOR_INDEX}', $q, {int(k)}) "
                            "RETURN node.id, distance",
                            {"q": q},
                        )
                        while res.has_next():
                            candidates.add(str(res.get_next()[0]))
                    indexed = True
            except Exception as exc:
                logger.warning("kuzu vector query failed; linear-scan fallback: %s", exc)
                self._vector_state = "unavailable"
        if not indexed and self._vector_state == "unavailable":
            return super()._match(appearance, threshold, exclude=exclude)
        if not indexed:
            candidates = set(self._nodes)  # nothing committed yet: scan everything
        # Exact re-rank with the sqlite scan's scoring rule (>= keeps the last
        # max in creation order), restricted to index-top-k + tail candidates.
        best_id: str | None = None
        best = threshold
        for nid in sorted(candidates, key=self._id_order):
            if nid in exclude:
                continue
            node = self._nodes.get(nid)
            if node is None:
                continue  # index may return ids pruned from memory
            b = node["appearance"]
            if b.size != appearance.size or b.size == 0:
                continue
            score = _cosine(appearance, b)
            if score >= best:
                best, best_id = score, nid
        self._match_cache_misses += 1
        self._match_last_ms = (time.perf_counter() - started) * 1000.0
        return best_id

    # ---- maintenance ------------------------------------------------------
    def clear(self) -> None:
        with self._lock:
            super().clear()  # wipes the in-memory model (base sqlite branch is a no-op)
            self._pending.clear()
            self._drain_locked()  # WS4B: no queued batch may land post-wipe
            if self._kz_conn is not None:
                with self._kz_io_lock:
                    for q in (
                        "MATCH (n:Entity) DETACH DELETE n",
                        "MATCH (b:PropertyBelief) DELETE b",
                        "MATCH (s:SemanticRecord) DELETE s",
                    ):
                        try:
                            self._kz_conn.execute(q)
                        except Exception:
                            logger.exception("kuzu graph clear failed (%s)", q)
                self._sqlite_commit_count += 1
            self._kuzu_node_ids.clear()
            self._kuzu_edge_keys.clear()
            self._kuzu_belief_pks.clear()
            self._kuzu_semantic_pks.clear()
            self._ids_since_index.clear()
            self._index_built = False

    def prune_retention(self, *, cycle: int = 0) -> dict[str, int]:
        with self._lock:
            nodes_before = set(self._nodes)
            sem_before = {cat: set(bucket) for cat, bucket in self._semantic.items()}
            out = super().prune_retention(cycle=cycle)
            # The base prune mutates memory only (its sqlite DELETEs are behind
            # self._conn, which is None here); diff the id sets to mirror the
            # deletions into kuzu.
            for nid in nodes_before - set(self._nodes):
                self._pending[("node", nid)] = ("del_node", {"id": nid})
            for cat, before in sem_before.items():
                for rid in before - set(self._semantic[cat]):
                    pk = f"{cat}{_SEP}{rid}"
                    self._pending[("sem", pk)] = ("del_sem", {"pk": pk})
            if out.get("nodes") or out.get("semantic_records"):
                self._commit_locked()
            return out

    # ---- checkpoint/restore (close -> copytree -> reopen) ----------------
    def backup_to(self, path: Path) -> None:
        """Quiesced snapshot: flush + checkpoint, close handles, copy the kuzu dir."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._open_locked()  # materialize schema so empty graphs round-trip
            self._drain_locked()
            self._quiesce_flusher_locked()  # no writer may touch the dir mid-copy
            self._checkpoint_locked()
            src = self._storage_root_locked()
            self._close_handles_locked()
            try:
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.copytree(src, target)
            finally:
                self._open_locked()  # reopen for continued use

    def restore_from(self, path: Path) -> None:
        """Replace the live graph with the kuzu directory at ``path`` (no-op if missing)."""
        src = Path(path)
        if not src.is_dir():
            return
        with self._lock:
            self._pending.clear()  # pending writes describe pre-restore state
            # WS4B: drop queued pre-restore batches and stop the flusher
            # before the directory swap (Windows file-lock discipline).
            with self._flush_mu:
                dropped = len(self._flush_queue)
                self._flush_queue.clear()
                self._batches_enqueued -= dropped  # exact barrier accounting
            if dropped:
                logger.info("kuzu restore: dropped %d queued pre-restore batches", dropped)
            self._quiesce_flusher_locked()
            self._close_handles_locked()
            root = self._storage_root_locked()
            if root.exists():
                shutil.rmtree(root)
            shutil.copytree(src, root)
            self._open_locked()
            self._load_kuzu_locked()

    # ---- metrics ----------------------------------------------------------
    def persistence_metrics(self) -> dict[str, Any]:
        with self._lock:
            db_file = (self._root / _DB_FILE) if self._root is not None else None
            wal_file = Path(str(db_file) + ".wal") if db_file is not None else None
            db_bytes = 0
            wal_bytes = 0
            try:
                if db_file is not None and db_file.exists():
                    if db_file.is_dir():  # directory-layout kuzu versions
                        db_bytes = sum(
                            f.stat().st_size for f in db_file.rglob("*") if f.is_file()
                        )
                    else:
                        db_bytes = db_file.stat().st_size
                if wal_file is not None and wal_file.exists():
                    wal_bytes = wal_file.stat().st_size
            except OSError:
                pass
            with self._flush_mu:
                queue_depth = len(self._flush_queue)
            return {
                "backend": "kuzu",
                # WS4B off-lock flusher telemetry (PRD G5):
                "graph_flush_ms": float(self._graph_flush_ms),
                "graph_flush_lock_ms": float(self._graph_flush_lock_ms),
                "graph_flush_queue_depth": int(queue_depth),
                "graph_flush_rows": int(self._graph_flush_rows),
                "graph_flush_stmts": int(self._graph_flush_stmts),
                "graph_flush_error_batches": int(self._flush_error_batches),
                "graph_dedicated_write_conn": bool(self._kz_write_conn is not None),
                "sqlite_commit_count": int(self._sqlite_commit_count),
                "sqlite_batch_commit_count": int(self._sqlite_batch_commit_count),
                "sqlite_last_commit_ms": float(self._sqlite_last_commit_ms),
                "sqlite_wal_checkpoint_count": int(self._sqlite_wal_checkpoint_count),
                "ltm_pruned_nodes": int(self._ltm_pruned_nodes),
                "ltm_pruned_edges": int(self._ltm_pruned_edges),
                "ltm_pruned_semantic_records": int(self._ltm_pruned_semantic_records),
                "memory_db_bytes": int(db_bytes),
                "memory_wal_bytes": int(wal_bytes),
                # Additive kuzu-only diagnostics (dashboards ignore unknown keys).
                "kuzu_vector_index_state": str(self._vector_state),
                "kuzu_vector_index_rebuilds": int(self._index_rebuilds),
            }


class WriteBehindKuzuLongTermGraph(WriteBehindLongTermGraph, KuzuLongTermGraph):
    """Write-behind kuzu graph for the runtime (async consolidation worker).

    Diamond MRO ``[self, WriteBehindLongTermGraph, KuzuLongTermGraph,
    LongTermGraph]``: every ``super()`` call inside the write-behind layer
    (consolidate/prune/backup/metrics) resolves to the kuzu overrides, so the
    wrapper's queueing, backpressure and flush semantics apply unchanged.
    """

    def close(self) -> None:
        """Drain + stop the worker, then release kuzu handles."""
        WriteBehindLongTermGraph.close(self)
        KuzuLongTermGraph.close(self)
