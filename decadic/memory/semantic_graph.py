"""Long-term knowledge graph: persistent, unbounded relational memory.

The hippocampal *index* half of the memory system. Working memory (bounded,
decaying) consolidates its stable object files into this store as permanent
nodes keyed by their learned appearance embedding; co-present nodes accrue
relational edges. Unlike episodic memory (a per-cycle vector log), this is a
growing relational graph the agent can be re-identified against (reinstatement)
and that the dashboard renders as it expands.

SQLite-backed for lifelong persistence, mirroring the patterns in
``episodic_store.py`` (thread lock, lazy connection, in-memory fallback when no
``db_path`` is given). The graph itself is unbounded - nodes and edges accumulate
without a capacity cap; only the dashboard read-out (:meth:`snapshot`) is windowed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import math
import time
from contextlib import contextmanager
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from decadic import config as C
from decadic.memory.sqlite_utils import (
    connect,
    configure_connection,
    db_file_sizes,
    decode_vector_blob,
    encode_vector_blob,
    wal_checkpoint_truncate,
)
from decadic.state.body_map import BODY_PARTS

DEFAULT_MATCH_THRESHOLD = 0.6  # cosine over appearance embeddings for re-identification
DEFAULT_APPEARANCE_EMA = 0.2  # how fast a node's stored appearance tracks new sightings
DEFAULT_SNAPSHOT_LIMIT = 64  # nodes returned to the dashboard (graph itself is unbounded)
CONFIDENCE_EVIDENCE_SCALE = 20.0
CONTRADICTION_MIN_EVIDENCE = 5.0
CONTRADICTION_DELTA = 0.55
FORBIDDEN_PROPERTY_TOKENS = (
    "label",
    "class",
    "kind_name",
    "food",
    "water",
    "floor",
    "hand",
    "wall",
    "building",
    "ball",
    "bear",
)
SEMANTIC_CATEGORIES = ("entity", "event", "relationship", "correlation", "conclusion", "value")
FORBIDDEN_SEMANTIC_TOKENS = FORBIDDEN_PROPERTY_TOKENS + (
    "semantic_label",
    "oracle",
    "sim_kind",
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _appearance_hue(vec: Iterable[float]) -> int:
    """Deterministic 0-359 hue from an appearance vector (frontend node color).

    A fixed pseudo-random projection so the same learned appearance always maps
    to the same color, across cycles and process restarts (no PYTHONHASHSEED
    dependence) - distinct discovered objects look distinct without any label.
    """
    h = 0.0
    for i, v in enumerate(list(vec)[:32]):
        h += float(v) * float(((i * 2654435761) % 1009) + 1)
    return int(abs(h)) % 360


def _clean_property_key(key: Any) -> str | None:
    k = str(key).strip()
    low = k.lower()
    if low.startswith("predicts_") and low.endswith("_pain"):
        middle = low[len("predicts_") : -len("_pain")]
        if middle in BODY_PARTS:
            return k
    if not k or any(tok in low for tok in FORBIDDEN_PROPERTY_TOKENS):
        return None
    return k


def _numeric_property_value(value: Any) -> tuple[Any, float] | None:
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        x = float(value)
        return x, x
    if isinstance(value, list) and value:
        vals: list[float] = []
        for v in value[:32]:
            try:
                x = float(v)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(x):
                return None
            vals.append(x)
        if vals:
            return vals, float(sum(vals) / len(vals))
    return None


def _blend_value(old: Any, new: Any, alpha: float) -> Any:
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [(1.0 - alpha) * float(a) + alpha * float(b) for a, b in zip(old, new)]
    return new


def _belief_confidence(evidence_count: float, variance: float) -> float:
    support = min(1.0, max(0.0, evidence_count / CONFIDENCE_EVIDENCE_SCALE))
    stability = max(0.0, min(1.0, 1.0 - variance))
    return float(support * stability)


def _belief_snapshot(b: dict[str, Any]) -> dict[str, Any]:
    return {
        "property_key": b["property_key"],
        "value": b.get("value"),
        "mean": round(float(b["mean"]), 6),
        "variance": round(float(b["variance"]), 6),
        "confidence": round(float(b["confidence"]), 6),
        "evidence_count": round(float(b["evidence_count"]), 3),
        "first_cycle": int(b["first_cycle"]),
        "last_cycle": int(b["last_cycle"]),
        "source": str(b.get("source", "perception")),
        "unstable": bool(b.get("unstable", False)),
    }


def _clean_semantic_text(value: Any, default: str = "unknown") -> str:
    text = str(value or default).strip()
    low = text.lower()
    if not text or any(tok in low for tok in FORBIDDEN_SEMANTIC_TOKENS):
        return default
    return text


def _clean_semantic_payload(raw: Any) -> Any:
    if isinstance(raw, dict):
        out: dict[str, Any] = {}
        for k, v in raw.items():
            key = _clean_semantic_text(k, "")
            if not key:
                continue
            out[key] = _clean_semantic_payload(v)
        return out
    if isinstance(raw, list):
        return [_clean_semantic_payload(v) for v in raw[:64]]
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        return float(raw)
    if isinstance(raw, bool) or raw is None:
        return raw
    if isinstance(raw, str):
        return _clean_semantic_text(raw, "anonymous")
    return str(raw)


def _anonymous_event_class(ev: dict[str, Any]) -> str:
    et = str((ev or {}).get("type", "")).lower()
    if et in ("collision", "damage", "environment_damage", "fall", "combat_hit", "threat_near"):
        return "aversive_state_change"
    if et in ("food", "eat", "nourish", "water", "drink", "hydrate", "offer"):
        return "interoceptive_relief"
    if et in ("contact", "touch", "support"):
        return "contact_state_change"
    if et:
        return "sensory_state_change"
    return "state_change"


def _value_context_for_event_class(event_class: str) -> str:
    if event_class == "aversive_state_change":
        return "risk_context"
    if event_class == "interoceptive_relief":
        return "need_relief_context"
    if event_class == "contact_state_change":
        return "support_context"
    return "exploration_context"


class LongTermGraph:
    """Thread-safe, persistent, unbounded relational object graph (the LTM index)."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        appearance_ema: float = DEFAULT_APPEARANCE_EMA,
    ) -> None:
        self._db_path = Path(db_path) if db_path is not None else None
        self._threshold = float(match_threshold)
        self._ema = float(appearance_ema)
        self._lock = threading.RLock()
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._beliefs: dict[tuple[str, str], dict[str, Any]] = {}
        self._semantic: dict[str, dict[str, dict[str, Any]]] = {cat: {} for cat in SEMANTIC_CATEGORIES}
        self._next_id = 1
        self._next_semantic_id = 1
        self._conn: sqlite3.Connection | None = None
        self._match_cache_enabled = C.ltm_match_cache_enabled()
        self._match_cache_dirty = True
        self._match_cache_dim: int | None = None
        self._match_cache_ids: list[str] = []
        self._match_cache_matrix = np.zeros((0, 0), dtype=np.float32)
        self._match_cache_hits = 0
        self._match_cache_misses = 0
        self._match_last_ms = 0.0
        self._persist_batch_depth = 0
        self._sqlite_commit_count = 0
        self._sqlite_batch_commit_count = 0
        self._sqlite_last_commit_ms = 0.0
        self._sqlite_wal_checkpoint_count = 0
        self._ltm_pruned_nodes = 0
        self._ltm_pruned_edges = 0
        self._ltm_pruned_semantic_records = 0
        self._cached_belief_stats: dict[str, Any] = {
            "total_property_beliefs": 0,
            "unstable_property_count": 0,
            "avg_property_confidence": 0.0,
            **{f"semantic_{k}": 0 for k in SEMANTIC_CATEGORIES},
        }
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = connect(self._db_path)
            self._ensure_schema(self._conn)
            self._load_all(self._conn)

    # ---- schema / persistence ------------------------------------------
    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                kind TEXT,
                appearance_json TEXT,
                appearance_blob BLOB,
                salience REAL,
                seen_count INTEGER,
                first_cycle INTEGER,
                last_cycle INTEGER,
                position_json TEXT,
                affect REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                src TEXT,
                dst TEXT,
                kind TEXT,
                weight REAL,
                count INTEGER,
                last_cycle INTEGER,
                PRIMARY KEY (src, dst, kind)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS property_beliefs (
                node_id TEXT,
                property_key TEXT,
                value_json TEXT,
                mean REAL,
                variance REAL,
                confidence REAL,
                evidence_count REAL,
                first_cycle INTEGER,
                last_cycle INTEGER,
                source TEXT,
                unstable INTEGER DEFAULT 0,
                PRIMARY KEY (node_id, property_key)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_records (
                category TEXT,
                id TEXT,
                payload_json TEXT,
                evidence_count REAL,
                confidence REAL,
                first_cycle INTEGER,
                last_cycle INTEGER,
                promoted INTEGER DEFAULT 0,
                PRIMARY KEY (category, id)
            )
            """
        )
        conn.commit()

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        LongTermGraph._create_schema(conn)
        cur = conn.execute("PRAGMA table_info(nodes)")
        cols = {row[1] for row in cur.fetchall()}
        if "appearance_blob" not in cols:
            conn.execute("ALTER TABLE nodes ADD COLUMN appearance_blob BLOB")
            conn.commit()

    @staticmethod
    def _write_node(conn: sqlite3.Connection, node: dict[str, Any]) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO nodes "
            "(id, kind, appearance_json, appearance_blob, salience, seen_count, first_cycle, "
            "last_cycle, position_json, affect) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                node["id"],
                node["kind"],
                json.dumps([float(x) for x in node["appearance"].tolist()])
                if (not C.sqlite_vector_blob_enabled() or C.sqlite_write_legacy_json_vectors())
                else None,
                encode_vector_blob(node["appearance"]),
                float(node["salience"]),
                int(node["seen_count"]),
                int(node["first_cycle"]),
                int(node["last_cycle"]),
                json.dumps(node["position"]) if node["position"] is not None else None,
                float(node["affect"]),
            ),
        )

    @staticmethod
    def _write_edge(conn: sqlite3.Connection, e: dict[str, Any]) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO edges (src, dst, kind, weight, count, last_cycle) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (e["src"], e["dst"], e["kind"], float(e["weight"]), int(e["count"]), int(e["last_cycle"])),
        )

    @staticmethod
    def _write_belief(conn: sqlite3.Connection, b: dict[str, Any]) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO property_beliefs "
            "(node_id, property_key, value_json, mean, variance, confidence, "
            "evidence_count, first_cycle, last_cycle, source, unstable) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                b["node_id"],
                b["property_key"],
                json.dumps(b.get("value")),
                float(b["mean"]),
                float(b["variance"]),
                float(b["confidence"]),
                float(b["evidence_count"]),
                int(b["first_cycle"]),
                int(b["last_cycle"]),
                str(b.get("source", "perception")),
                1 if b.get("unstable") else 0,
            ),
        )

    @staticmethod
    def _write_semantic_record(conn: sqlite3.Connection, category: str, rec: dict[str, Any]) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO semantic_records "
            "(category, id, payload_json, evidence_count, confidence, first_cycle, last_cycle, promoted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                category,
                rec["id"],
                json.dumps(rec.get("payload", {})),
                float(rec.get("evidence_count", 0.0)),
                float(rec.get("confidence", 0.0)),
                int(rec.get("first_cycle", 0)),
                int(rec.get("last_cycle", 0)),
                1 if rec.get("promoted") else 0,
            ),
        )

    def _load_all(self, conn: sqlite3.Connection) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._beliefs.clear()
        max_id = 0
        for row in conn.execute(
            "SELECT id, kind, appearance_json, appearance_blob, salience, seen_count, first_cycle, "
            "last_cycle, position_json, affect FROM nodes"
        ):
            nid = str(row[0])
            app_blob = decode_vector_blob(row[3])
            app = (
                app_blob
                if app_blob is not None
                else (
                    np.asarray(json.loads(row[2]), dtype=np.float32)
                    if row[2]
                    else np.zeros(0, dtype=np.float32)
                )
            )
            pos = json.loads(row[8]) if row[8] else None
            self._nodes[nid] = {
                "id": nid,
                "kind": row[1] or "unknown",
                "appearance": app,
                "salience": float(row[4] or 0.0),
                "seen_count": int(row[5] or 0),
                "first_cycle": int(row[6] or 0),
                "last_cycle": int(row[7] or 0),
                "position": pos,
                "affect": float(row[9] or 0.0),
            }
            try:
                max_id = max(max_id, int(nid.split("-")[-1]))
            except ValueError:
                pass
        for row in conn.execute("SELECT src, dst, kind, weight, count, last_cycle FROM edges"):
            self._edges[(str(row[0]), str(row[1]), str(row[2]))] = {
                "src": str(row[0]),
                "dst": str(row[1]),
                "kind": str(row[2]),
                "weight": float(row[3] or 0.0),
                "count": int(row[4] or 0),
                "last_cycle": int(row[5] or 0),
            }
        try:
            belief_rows = conn.execute(
                "SELECT node_id, property_key, value_json, mean, variance, confidence, "
                "evidence_count, first_cycle, last_cycle, source, unstable FROM property_beliefs"
            )
        except sqlite3.OperationalError:
            belief_rows = []
        for row in belief_rows:
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
        try:
            semantic_rows = conn.execute(
                "SELECT category, id, payload_json, evidence_count, confidence, "
                "first_cycle, last_cycle, promoted FROM semantic_records"
            )
        except sqlite3.OperationalError:
            semantic_rows = []
        for row in semantic_rows:
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
            try:
                self._next_semantic_id = max(self._next_semantic_id, int(rid.split("-")[-1]) + 1)
            except ValueError:
                pass
        self._next_id = max_id + 1
        self._mark_match_cache_dirty()
        self._cached_belief_stats = self._compute_belief_stats_locked()

    def _persist_node(self, node: dict[str, Any]) -> None:
        if self._conn is None:
            return
        self._write_node(self._conn, node)
        self._commit_locked()

    def _persist_edge(self, e: dict[str, Any]) -> None:
        if self._conn is None:
            return
        self._write_edge(self._conn, e)
        self._commit_locked()

    def _persist_belief(self, b: dict[str, Any]) -> None:
        if self._conn is None:
            return
        self._write_belief(self._conn, b)
        self._commit_locked()

    def _persist_semantic_record(self, category: str, rec: dict[str, Any]) -> None:
        if self._conn is None:
            return
        self._write_semantic_record(self._conn, category, rec)
        self._commit_locked()

    def _commit_locked(self, *, batch: bool = False) -> None:
        if self._conn is None or self._persist_batch_depth > 0:
            return
        started = time.perf_counter()
        self._conn.commit()
        self._sqlite_last_commit_ms = (time.perf_counter() - started) * 1000.0
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
                if self._conn is not None and self._persist_batch_depth == 0:
                    started = time.perf_counter()
                    self._conn.commit()
                    self._sqlite_last_commit_ms = (time.perf_counter() - started) * 1000.0
                    self._sqlite_commit_count += 1
                    self._sqlite_batch_commit_count += 1

    # ---- internal graph ops (assume lock held) -------------------------
    def _coin_id(self) -> str:
        nid = f"ent-{self._next_id:05d}"
        self._next_id += 1
        return nid

    def _coin_semantic_id(self, prefix: str) -> str:
        rid = f"{prefix}-{self._next_semantic_id:05d}"
        self._next_semantic_id += 1
        return rid

    def _mark_match_cache_dirty(self) -> None:
        self._match_cache_dirty = True

    def _compute_belief_stats_locked(self) -> dict[str, Any]:
        total = len(self._beliefs)
        unstable = sum(1 for b in self._beliefs.values() if b.get("unstable"))
        avg = (
            sum(float(b.get("confidence", 0.0) or 0.0) for b in self._beliefs.values()) / total
            if total
            else 0.0
        )
        return {
            "total_property_beliefs": total,
            "unstable_property_count": unstable,
            "avg_property_confidence": round(avg, 6),
            **{f"semantic_{k}": v for k, v in self.semantic_stats().items()},
        }

    def _refresh_cached_belief_stats_locked(self) -> None:
        self._cached_belief_stats = self._compute_belief_stats_locked()

    def _rebuild_match_cache_locked(self, dim: int) -> None:
        recent_cap = C.ltm_match_recent_cap()
        salient_cap = C.ltm_match_salient_cap()
        candidates = [
            node
            for node in self._nodes.values()
            if isinstance(node.get("appearance"), np.ndarray)
            and node["appearance"].size == dim
            and dim > 0
        ]
        recent = sorted(
            candidates,
            key=lambda n: (int(n.get("last_cycle", 0)), int(n.get("seen_count", 0))),
            reverse=True,
        )[:recent_cap]
        recent_ids = {str(n["id"]) for n in recent}
        salient = sorted(
            (n for n in candidates if str(n["id"]) not in recent_ids),
            key=lambda n: (
                float(n.get("salience", 0.0) or 0.0),
                int(n.get("seen_count", 0) or 0),
                int(n.get("last_cycle", 0) or 0),
            ),
            reverse=True,
        )[:salient_cap]
        selected = recent + salient
        self._match_cache_ids = [str(n["id"]) for n in selected]
        if selected:
            raw = np.stack([np.asarray(n["appearance"], dtype=np.float32) for n in selected]).astype(
                np.float32,
                copy=False,
            )
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            self._match_cache_matrix = raw / np.maximum(norms, 1e-8)
        else:
            self._match_cache_matrix = np.zeros((0, dim), dtype=np.float32)
        self._match_cache_dim = dim
        self._match_cache_dirty = False

    def _upsert_semantic(
        self,
        category: str,
        key: str,
        payload: dict[str, Any],
        *,
        cycle: int,
        evidence_weight: float = 1.0,
        confidence: float = 0.0,
        promoted: bool = False,
    ) -> str:
        if category not in self._semantic:
            raise ValueError(f"unknown semantic category {category!r}")
        rid = key or self._coin_semantic_id(category[:3])
        bucket = self._semantic[category]
        rec = bucket.get(rid)
        weight = max(0.0, float(evidence_weight))
        if rec is None:
            rec = {
                "id": rid,
                "payload": _clean_semantic_payload(payload),
                "evidence_count": weight,
                "confidence": max(0.0, min(1.0, float(confidence))),
                "first_cycle": int(cycle),
                "last_cycle": int(cycle),
                "promoted": bool(promoted),
            }
            bucket[rid] = rec
        else:
            rec["payload"] = _clean_semantic_payload({**dict(rec.get("payload", {})), **payload})
            rec["evidence_count"] = float(rec.get("evidence_count", 0.0)) + weight
            rec["confidence"] = max(float(rec.get("confidence", 0.0)), max(0.0, min(1.0, float(confidence))))
            rec["last_cycle"] = int(cycle)
            rec["promoted"] = bool(rec.get("promoted", False) or promoted)
        self._persist_semantic_record(category, rec)
        return rid

    def _match(
        self,
        appearance: np.ndarray | None,
        threshold: float,
        *,
        exclude: set[str] | None = None,
    ) -> str | None:
        started = time.perf_counter()
        if appearance is None or appearance.size == 0:
            self._match_last_ms = (time.perf_counter() - started) * 1000.0
            return None
        exclude = exclude or set()
        if self._match_cache_enabled:
            dim = int(appearance.size)
            if self._match_cache_dirty or self._match_cache_dim != dim:
                self._rebuild_match_cache_locked(dim)
            if self._match_cache_matrix.size:
                q = appearance.astype(np.float32, copy=False)
                qn = float(np.linalg.norm(q))
                if qn >= 1e-8:
                    q = q / qn
                    if exclude:
                        idxs = [
                            i for i, nid in enumerate(self._match_cache_ids) if nid not in exclude
                        ]
                        if not idxs:
                            self._match_cache_misses += 1
                            self._match_last_ms = (time.perf_counter() - started) * 1000.0
                            return None
                        sims = self._match_cache_matrix[idxs] @ q
                        best_local = int(np.argmax(sims))
                        best_idx = idxs[best_local]
                        best = float(sims[best_local])
                    else:
                        sims = self._match_cache_matrix @ q
                        best_idx = int(np.argmax(sims))
                        best = float(sims[best_idx])
                    self._match_cache_hits += 1
                    self._match_last_ms = (time.perf_counter() - started) * 1000.0
                    return self._match_cache_ids[best_idx] if best >= threshold else None
            self._match_cache_misses += 1
            self._match_last_ms = (time.perf_counter() - started) * 1000.0
            return None
        best_id: str | None = None
        best = threshold
        for nid, node in self._nodes.items():
            if nid in exclude:
                continue
            b = node["appearance"]
            if b.size != appearance.size or b.size == 0:
                continue
            score = _cosine(appearance, b)
            if score >= best:
                best, best_id = score, nid
        self._match_cache_misses += 1
        self._match_last_ms = (time.perf_counter() - started) * 1000.0
        return best_id

    def _upsert(
        self,
        appearance: Any,
        kind: str,
        position: Any,
        affect: float,
        cycle: int,
        exclude_matches: set[str] | None = None,
    ) -> str:
        a = None if appearance is None else np.asarray(appearance, dtype=np.float32).reshape(-1)
        nid = (
            self._match(a, self._threshold, exclude=exclude_matches)
            if a is not None and a.size
            else None
        )
        if nid is None:
            nid = self._coin_id()
            node = {
                "id": nid,
                "kind": kind or "unknown",
                "appearance": a if a is not None else np.zeros(0, dtype=np.float32),
                "salience": 1.0,
                "seen_count": 1,
                "first_cycle": int(cycle),
                "last_cycle": int(cycle),
                "position": list(position) if position is not None else None,
                "affect": float(affect),
            }
            self._nodes[nid] = node
        else:
            node = self._nodes[nid]
            if a is not None and a.size:
                if node["appearance"].size == a.size:
                    node["appearance"] = (1.0 - self._ema) * node["appearance"] + self._ema * a
                elif node["appearance"].size == 0:
                    node["appearance"] = a
            node["seen_count"] += 1
            node["salience"] = 1.0
            node["last_cycle"] = int(cycle)
            if position is not None:
                node["position"] = list(position)
            node["affect"] = float(affect)
            if kind and kind != "unknown":
                node["kind"] = kind
        self._persist_node(node)
        self._mark_match_cache_dirty()
        return nid

    def _bump(self, src: str, dst: str, kind: str, weight: float, cycle: int) -> None:
        if src == dst:
            return
        if kind == "co_occurrence" and src > dst:
            src, dst = dst, src
        key = (src, dst, kind)
        e = self._edges.get(key)
        if e is None:
            e = {"src": src, "dst": dst, "kind": kind, "weight": float(weight), "count": 1, "last_cycle": int(cycle)}
            self._edges[key] = e
        else:
            e["count"] += 1
            e["weight"] = min(1.0, e["weight"] + 0.05)
            e["last_cycle"] = int(cycle)
        self._persist_edge(e)

    def _upsert_property_beliefs(
        self,
        node_id: str,
        evidence: dict[str, Any] | None,
        *,
        cycle: int,
        evidence_weight: float = 1.0,
        source: str = "perception",
    ) -> int:
        if node_id not in self._nodes or not isinstance(evidence, dict):
            return 0
        weight = max(0.0, min(4.0, float(evidence_weight)))
        if weight <= 0.0:
            return 0
        updated = 0
        for raw_key, raw_val in evidence.items():
            key = _clean_property_key(raw_key)
            parsed = _numeric_property_value(raw_val)
            if key is None or parsed is None:
                continue
            value, mean = parsed
            bk = (node_id, key)
            b = self._beliefs.get(bk)
            if b is None:
                b = {
                    "node_id": node_id,
                    "property_key": key,
                    "value": value,
                    "mean": float(mean),
                    "variance": 0.0,
                    "confidence": _belief_confidence(weight, 0.0),
                    "evidence_count": weight,
                    "first_cycle": int(cycle),
                    "last_cycle": int(cycle),
                    "source": source,
                    "unstable": False,
                }
                self._beliefs[bk] = b
            else:
                prev_mean = float(b["mean"])
                prev_count = float(b["evidence_count"])
                alpha = weight / max(weight, prev_count + weight)
                residual = abs(float(mean) - prev_mean)
                variance = (1.0 - alpha) * float(b["variance"]) + alpha * min(1.0, residual * residual)
                unstable = bool(b.get("unstable", False))
                if prev_count >= CONTRADICTION_MIN_EVIDENCE and residual >= CONTRADICTION_DELTA:
                    unstable = True
                    variance = min(1.0, max(variance, residual))
                new_count = prev_count + weight
                b["value"] = _blend_value(b.get("value"), value, alpha)
                b["mean"] = prev_mean + alpha * (float(mean) - prev_mean)
                b["variance"] = variance
                b["evidence_count"] = new_count
                b["confidence"] = _belief_confidence(new_count, variance)
                if unstable:
                    b["confidence"] = min(float(b["confidence"]), 0.5)
                b["last_cycle"] = int(cycle)
                b["source"] = source
                b["unstable"] = unstable
            self._persist_belief(b)
            updated += 1
        if updated:
            self._refresh_cached_belief_stats_locked()
        return updated

    # ---- public API ----------------------------------------------------
    def match(self, appearance: Any, threshold: float | None = None) -> str | None:
        """Re-identify ``appearance`` against existing nodes; returns a node id or None."""
        a = None if appearance is None else np.asarray(appearance, dtype=np.float32).reshape(-1)
        with self._lock:
            return self._match(a, self._threshold if threshold is None else float(threshold))

    def entity_appearance(self, node_id: str) -> list[float] | None:
        """WS5-M4.1: the STORED appearance of a known entity, or None.

        The stable identity anchor for graph-keyed WM slots: a slot's live
        appearance EMA-drifts with every sighting, but the graph's stored
        embedding moves on its own slow EMA -- so a re-encountered entity
        re-binds to (nearly) the same key across an occlusion gap. This is
        what makes object permanence visible to the network.
        """
        with self._lock:
            node = self._nodes.get(str(node_id))
            if node is None:
                return None
            app = node.get("appearance")
            if app is None:
                return None
            try:
                a = np.asarray(app, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                return None
            return [float(x) for x in a] if a.size else None

    def upsert_node(
        self,
        appearance: Any,
        *,
        kind: str = "unknown",
        position: Any = None,
        affect: float = 0.0,
        cycle: int = 0,
    ) -> str:
        """Create a new node or EMA-update the matched one; returns its stable id."""
        with self._lock:
            return self._upsert(appearance, kind, position, affect, cycle)

    def bump_edge(
        self, src: str, dst: str, *, kind: str = "co_occurrence", weight: float = 1.0, cycle: int = 0
    ) -> None:
        with self._lock:
            self._bump(src, dst, kind, weight, cycle)

    def upsert_property_beliefs(
        self,
        node_id: str,
        evidence: dict[str, Any] | None,
        *,
        cycle: int = 0,
        evidence_weight: float = 1.0,
        source: str = "perception",
    ) -> int:
        """Strengthen anonymous property beliefs for an existing node."""
        with self._lock:
            return self._upsert_property_beliefs(
                node_id,
                evidence,
                cycle=cycle,
                evidence_weight=evidence_weight,
                source=source,
            )

    def consolidate(
        self,
        slots: Iterable[Any],
        affect: dict[str, float] | None = None,
        *,
        cycle: int = 0,
        min_seen: int = 2,
        property_update: bool = True,
        relationship_update: bool = True,
    ) -> list[str]:
        """Commit stable working-memory slots as nodes; link co-present ones.

        A slot is stable when it carries an appearance fingerprint and has been
        seen at least ``min_seen`` cycles. Oracle-mode slots have no appearance,
        so this is naturally a no-op there (parity).
        """
        affect = affect or {}
        with self._lock:
            ids: list[str] = []
            claimed: set[str] = set()
            for s in slots:
                app = getattr(s, "appearance", None)
                if not app or int(getattr(s, "seen_count", 0)) < min_seen:
                    continue
                try:
                    confidence = float(getattr(s, "confidence", 1.0) or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                try:
                    precision = float(getattr(s, "precision", confidence) or confidence)
                except (TypeError, ValueError):
                    precision = confidence
                if precision < 0.2:
                    continue
                a_val = float(getattr(s, "affective_weight", 0.0) or affect.get(getattr(s, "entity_id", ""), 0.0))
                nid = self._upsert(
                    app,
                    str(getattr(s, "entity_role", getattr(s, "kind", "unknown"))),
                    getattr(s, "position", None),
                    a_val,
                    cycle,
                    exclude_matches=claimed,
                )
                if property_update:
                    try:
                        evidence_weight = max(0.0, min(1.0, confidence))
                    except (TypeError, ValueError):
                        evidence_weight = 0.0
                    self._upsert_property_beliefs(
                        nid,
                        getattr(s, "property_evidence", None),
                        cycle=cycle,
                        evidence_weight=evidence_weight,
                        source="perception",
                    )
                ids.append(nid)
                claimed.add(nid)
            if relationship_update:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        self._bump(ids[i], ids[j], "co_occurrence", 1.0, cycle)
            return ids

    def record_semantic_evidence(
        self,
        slots: Iterable[Any],
        *,
        events: list[dict[str, Any]] | None = None,
        scene_relationships: list[dict[str, Any]] | None = None,
        cycle: int = 0,
        promoted_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record Framework-style semantic evidence without requiring promotion."""
        promoted_ids = promoted_ids or []
        with self._lock:
            counts = {cat: 0 for cat in SEMANTIC_CATEGORIES}
            slot_ids: list[str] = []
            for s in slots:
                sid = str(getattr(s, "entity_id", ""))
                if not sid:
                    continue
                try:
                    conf = float(getattr(s, "confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
                try:
                    precision = float(getattr(s, "precision", conf) or conf)
                except (TypeError, ValueError):
                    precision = conf
                role = _clean_semantic_text(getattr(s, "entity_role", getattr(s, "kind_hint", "entity")), "entity")
                promoted = bool(not getattr(s, "provisional", True) or sid in promoted_ids)
                self._upsert_semantic(
                    "entity",
                    f"entity:{sid}",
                    {
                        "wm_entity_id": sid,
                        "entity_role": role,
                        "seen_count": int(getattr(s, "seen_count", 0) or 0),
                        "provisional": bool(getattr(s, "provisional", True)),
                        "property_keys": sorted((getattr(s, "property_evidence", {}) or {}).keys())[:24],
                    },
                    cycle=cycle,
                    evidence_weight=max(0.05, conf),
                    confidence=precision,
                    promoted=promoted,
                )
                counts["entity"] += 1
                slot_ids.append(sid)
                if float(getattr(s, "affective_weight", 0.0) or 0.0) != 0.0:
                    rel_id = f"relationship:self:affective:{sid}"
                    self._upsert_semantic(
                        "relationship",
                        rel_id,
                        {
                            "src": "self",
                            "dst": sid,
                            "kind": "affective",
                            "weight": float(getattr(s, "affective_weight", 0.0) or 0.0),
                        },
                        cycle=cycle,
                        evidence_weight=max(0.05, abs(float(getattr(s, "affective_weight", 0.0) or 0.0))),
                        confidence=precision,
                    )
                    counts["relationship"] += 1
            for ev in events or []:
                if not isinstance(ev, dict):
                    continue
                event_class = _anonymous_event_class(ev)
                try:
                    intensity = max(0.0, min(1.0, float(ev.get("intensity", 0.0) or 0.0)))
                except (TypeError, ValueError):
                    intensity = 0.0
                eid = self._coin_semantic_id("evt")
                self._upsert_semantic(
                    "event",
                    eid,
                    {"event_class": event_class, "intensity": intensity},
                    cycle=cycle,
                    evidence_weight=max(0.05, intensity),
                    confidence=intensity,
                )
                counts["event"] += 1
                for sid in slot_ids[:3]:
                    rel_key = f"relationship:{sid}:involves:{event_class}"
                    self._upsert_semantic(
                        "relationship",
                        rel_key,
                        {"src": sid, "dst": eid, "kind": "event_involves_entity", "event_class": event_class},
                        cycle=cycle,
                        evidence_weight=max(0.05, intensity),
                        confidence=intensity,
                    )
                    counts["relationship"] += 1
                    corr_key = f"correlation:{sid}:predicts:{event_class}"
                    corr_id = self._upsert_semantic(
                        "correlation",
                        corr_key,
                        {"entity": sid, "pattern": "entity_predicts_event", "event_class": event_class},
                        cycle=cycle,
                        evidence_weight=max(0.05, intensity),
                        confidence=intensity,
                    )
                    counts["correlation"] += 1
                    corr = self._semantic["correlation"][corr_id]
                    if float(corr.get("evidence_count", 0.0)) >= 2.0:
                        conclusion_key = f"conclusion:{sid}:{event_class}"
                        conc_id = self._upsert_semantic(
                            "conclusion",
                            conclusion_key,
                            {"from_correlation": corr_id, "entity": sid, "event_class": event_class},
                            cycle=cycle,
                            evidence_weight=1.0,
                            confidence=min(1.0, float(corr.get("evidence_count", 0.0)) / 8.0),
                            promoted=True,
                        )
                        counts["conclusion"] += 1
                        value_context = _value_context_for_event_class(event_class)
                        sign = -1.0 if event_class == "aversive_state_change" else 1.0
                        self._upsert_semantic(
                            "value",
                            f"value:{sid}:{value_context}",
                            {
                                "from_conclusion": conc_id,
                                "entity": sid,
                                "context": value_context,
                                "valence": sign * max(0.05, intensity),
                            },
                            cycle=cycle,
                            evidence_weight=max(0.05, intensity),
                            confidence=min(1.0, float(corr.get("evidence_count", 0.0)) / 8.0),
                            promoted=True,
                        )
                        counts["value"] += 1
            for rel in scene_relationships or []:
                if not isinstance(rel, dict):
                    continue
                src = _clean_semantic_text(rel.get("src"), "")
                dst = _clean_semantic_text(rel.get("dst"), "")
                kind = _clean_semantic_text(rel.get("kind"), "scene_relation")
                if not src or not dst:
                    continue
                try:
                    conf = max(0.0, min(1.0, float(rel.get("confidence", 1.0) or 1.0)))
                except (TypeError, ValueError):
                    conf = 1.0
                self._upsert_semantic(
                    "relationship",
                    f"relationship:{src}:{kind}:{dst}",
                    {"src": src, "dst": dst, "kind": kind},
                    cycle=cycle,
                    evidence_weight=conf,
                    confidence=conf,
                )
                counts["relationship"] += 1
            self._refresh_cached_belief_stats_locked()
            return {
                "entities": counts["entity"],
                "events": counts["event"],
                "relationships": counts["relationship"],
                "correlations": counts["correlation"],
                "conclusions": counts["conclusion"],
                "values": counts["value"],
            }

    def snapshot(self, limit: int = DEFAULT_SNAPSHOT_LIMIT) -> dict[str, Any]:
        """Read-out for the dashboard with explicit render/cap metadata.

        The graph itself is unbounded, but the live API may request a bounded
        node window to keep polling cheap. Totals always describe the full LTM;
        ``rendered_*`` fields describe the included read-out so the UI cannot
        mistake a capped view for the entire persistent graph.
        """
        with self._lock:
            total_nodes = len(self._nodes)
            total_edges = len(self._edges)
            degree: dict[str, int] = {}
            edge_kind_counts: dict[str, int] = {}
            edge_pair_counts: dict[str, int] = {}
            for (s, d, _k) in self._edges:
                degree[s] = degree.get(s, 0) + 1
                degree[d] = degree.get(d, 0) + 1
            for e in self._edges.values():
                kind = str(e.get("kind", "unknown"))
                edge_kind_counts[kind] = edge_kind_counts.get(kind, 0) + 1
                a = str(e.get("src", ""))
                b = str(e.get("dst", ""))
                pair_key = "->".join((a, b)) if a <= b else "->".join((b, a))
                edge_pair_counts[pair_key] = edge_pair_counts.get(pair_key, 0) + 1
            max_nodes = total_nodes if limit is None or int(limit) <= 0 else max(0, int(limit))
            ranked = sorted(
                self._nodes.values(), key=lambda n: (n["last_cycle"], n["seen_count"]), reverse=True
            )[:max_nodes]
            ids = {n["id"] for n in ranked}
            beliefs_by_node: dict[str, list[dict[str, Any]]] = {nid: [] for nid in ids}
            for (node_id, _key), belief in self._beliefs.items():
                if node_id in beliefs_by_node:
                    beliefs_by_node[node_id].append(_belief_snapshot(belief))
            for vals in beliefs_by_node.values():
                vals.sort(key=lambda b: (b["confidence"], b["evidence_count"]), reverse=True)
            nodes = [
                {
                    "id": n["id"],
                    "kind": n["kind"],
                    "salience": round(float(n["salience"]), 4),
                    "seen_count": int(n["seen_count"]),
                    "last_cycle": int(n["last_cycle"]),
                    "affect": round(float(n["affect"]), 4),
                    "degree": degree.get(n["id"], 0),
                    "appearance_hash": _appearance_hue(n["appearance"].tolist()),
                    "property_beliefs": beliefs_by_node.get(n["id"], [])[:8],
                    "unstable_property_count": sum(1 for b in beliefs_by_node.get(n["id"], []) if b["unstable"]),
                    "avg_property_confidence": round(
                        sum(float(b["confidence"]) for b in beliefs_by_node.get(n["id"], []))
                        / max(1, len(beliefs_by_node.get(n["id"], []))),
                        6,
                    )
                    if beliefs_by_node.get(n["id"])
                    else 0.0,
                }
                for n in ranked
            ]
            edges = [
                {
                    "source": e["src"],
                    "target": e["dst"],
                    "kind": e["kind"],
                    "weight": round(float(e["weight"]), 4),
                    "count": int(e.get("count", 1)),
                    "last_cycle": int(e.get("last_cycle", 0)),
                }
                for e in self._edges.values()
                if e["src"] in ids and e["dst"] in ids
            ]
            return {
                "nodes": nodes,
                "edges": edges,
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "rendered_nodes": len(nodes),
                "rendered_edges": len(edges),
                "snapshot_limit": max_nodes,
                "truncated_nodes": len(nodes) < total_nodes,
                "truncated_edges": len(edges) < total_edges,
                "edge_kind_counts": edge_kind_counts,
                "edge_pair_counts": edge_pair_counts,
                "total_property_beliefs": len(self._beliefs),
                "unstable_property_count": sum(1 for b in self._beliefs.values() if b.get("unstable")),
                "semantic": self.semantic_stats(),
            }

    def counts(self) -> tuple[int, int]:
        with self._lock:
            return len(self._nodes), len(self._edges)

    def belief_stats(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_cached_belief_stats_locked()
            return dict(self._cached_belief_stats)

    def cached_belief_stats(self) -> dict[str, Any]:
        """Last computed belief/semantic counters; cheap enough for hot telemetry."""
        with self._lock:
            return dict(self._cached_belief_stats)

    def match_cache_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": bool(self._match_cache_enabled),
                "size": int(len(self._match_cache_ids)),
                "hits": int(self._match_cache_hits),
                "misses": int(self._match_cache_misses),
                "last_ms": float(self._match_last_ms),
            }

    def semantic_stats(self) -> dict[str, Any]:
        return {
            "entities": len(self._semantic["entity"]),
            "events": len(self._semantic["event"]),
            "relationships": len(self._semantic["relationship"]),
            "correlations": len(self._semantic["correlation"]),
            "conclusions": len(self._semantic["conclusion"]),
            "values": len(self._semantic["value"]),
        }

    def clear(self) -> None:
        """Wipe the whole graph (agent reset)."""
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._beliefs.clear()
            for bucket in self._semantic.values():
                bucket.clear()
            self._next_id = 1
            self._next_semantic_id = 1
            self._mark_match_cache_dirty()
            self._refresh_cached_belief_stats_locked()
            if self._conn is not None:
                self._conn.execute("DELETE FROM nodes")
                self._conn.execute("DELETE FROM edges")
                self._conn.execute("DELETE FROM property_beliefs")
                self._conn.execute("DELETE FROM semantic_records")
                self._commit_locked()

    def prune_retention(self, *, cycle: int = 0) -> dict[str, int]:
        """Prune weak stale records so the persistent graph file plateaus."""
        if not C.ltm_retention_enabled():
            return {"nodes": 0, "edges": 0, "semantic_records": 0}
        with self._lock:
            node_pruned = 0
            edge_pruned = 0
            semantic_pruned = 0
            if len(self._nodes) > C.ltm_max_nodes():
                protected: set[str] = set()
                for src, dst, _kind in self._edges:
                    protected.add(src)
                    protected.add(dst)
                protected.update(node_id for node_id, _key in self._beliefs)
                stale_before = int(cycle) - C.ltm_prune_stale_cycles()
                candidates = [
                    n
                    for n in self._nodes.values()
                    if n["id"] not in protected
                    and int(n.get("last_cycle", 0) or 0) <= stale_before
                    and int(n.get("seen_count", 0) or 0) <= 1
                    and float(n.get("salience", 0.0) or 0.0) < 0.25
                ]
                overflow = max(0, len(self._nodes) - C.ltm_max_nodes())
                limit = min(C.ltm_prune_batch(), max(overflow, 1))
                for node in sorted(
                    candidates,
                    key=lambda n: (
                        float(n.get("salience", 0.0) or 0.0),
                        int(n.get("last_cycle", 0) or 0),
                    ),
                )[:limit]:
                    self._nodes.pop(str(node["id"]), None)
                    node_pruned += 1
                    if self._conn is not None:
                        self._conn.execute("DELETE FROM nodes WHERE id = ?", (str(node["id"]),))
            total_semantic = sum(len(bucket) for bucket in self._semantic.values())
            if total_semantic > C.ltm_max_semantic_records():
                candidates_sem: list[tuple[str, str, dict[str, Any]]] = []
                for cat, bucket in self._semantic.items():
                    if cat in ("conclusion", "value"):
                        continue
                    for rid, rec in bucket.items():
                        if rec.get("promoted"):
                            continue
                        candidates_sem.append((cat, rid, rec))
                overflow = max(0, total_semantic - C.ltm_max_semantic_records())
                limit = min(C.ltm_prune_batch(), max(overflow, 1))
                for cat, rid, _rec in sorted(
                    candidates_sem,
                    key=lambda item: (
                        float(item[2].get("confidence", 0.0) or 0.0),
                        float(item[2].get("evidence_count", 0.0) or 0.0),
                        int(item[2].get("last_cycle", 0) or 0),
                    ),
                )[:limit]:
                    self._semantic[cat].pop(rid, None)
                    semantic_pruned += 1
                    if self._conn is not None:
                        self._conn.execute(
                            "DELETE FROM semantic_records WHERE category = ? AND id = ?",
                            (cat, rid),
                        )
            if node_pruned or edge_pruned or semantic_pruned:
                self._mark_match_cache_dirty()
                self._refresh_cached_belief_stats_locked()
                self._ltm_pruned_nodes += node_pruned
                self._ltm_pruned_edges += edge_pruned
                self._ltm_pruned_semantic_records += semantic_pruned
                self._commit_locked()
            return {
                "nodes": node_pruned,
                "edges": edge_pruned,
                "semantic_records": semantic_pruned,
            }

    def _dump_memory(self, dest: sqlite3.Connection) -> None:
        for n in self._nodes.values():
            self._write_node(dest, n)
        for e in self._edges.values():
            self._write_edge(dest, e)
        for b in self._beliefs.values():
            self._write_belief(dest, b)
        for cat, bucket in self._semantic.items():
            for rec in bucket.values():
                self._write_semantic_record(dest, cat, rec)

    def backup_to(self, path: Path) -> None:
        """Snapshot the graph DB to ``path`` (SQLite online backup; in-memory store materialized)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dest = sqlite3.connect(str(path))
        configure_connection(dest)
        try:
            with self._lock:
                self._ensure_schema(dest)
                if self._conn is not None:
                    self._commit_locked()
                    wal_checkpoint_truncate(self._conn)
                    self._sqlite_wal_checkpoint_count += 1
                    self._conn.backup(dest)
                else:
                    self._dump_memory(dest)
            dest.commit()
        finally:
            dest.close()

    def restore_from(self, path: Path) -> None:
        """Replace the live graph with the SQLite file at ``path`` (no-op if missing)."""
        src_path = Path(path)
        if not src_path.is_file():
            return
        src = sqlite3.connect(str(src_path))
        configure_connection(src)
        try:
            self._ensure_schema(src)
            with self._lock:
                if self._conn is not None:
                    src.backup(self._conn)
                    self._commit_locked()
                    self._ensure_schema(self._conn)
                    self._load_all(self._conn)
                else:
                    self._load_all(src)
        finally:
            src.close()

    def persistence_metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sqlite_commit_count": int(self._sqlite_commit_count),
                "sqlite_batch_commit_count": int(self._sqlite_batch_commit_count),
                "sqlite_last_commit_ms": float(self._sqlite_last_commit_ms),
                "sqlite_wal_checkpoint_count": int(self._sqlite_wal_checkpoint_count),
                "ltm_pruned_nodes": int(self._ltm_pruned_nodes),
                "ltm_pruned_edges": int(self._ltm_pruned_edges),
                "ltm_pruned_semantic_records": int(self._ltm_pruned_semantic_records),
                **db_file_sizes(self._db_path),
            }
