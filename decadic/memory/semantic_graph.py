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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

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
    "hand",
    "wall",
    "building",
    "ball",
    "bear",
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
        self._next_id = 1
        self._conn: sqlite3.Connection | None = None
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._create_schema(self._conn)
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
        conn.commit()

    @staticmethod
    def _write_node(conn: sqlite3.Connection, node: dict[str, Any]) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO nodes "
            "(id, kind, appearance_json, salience, seen_count, first_cycle, "
            "last_cycle, position_json, affect) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                node["id"],
                node["kind"],
                json.dumps([float(x) for x in node["appearance"].tolist()]),
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

    def _load_all(self, conn: sqlite3.Connection) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._beliefs.clear()
        max_id = 0
        for row in conn.execute(
            "SELECT id, kind, appearance_json, salience, seen_count, first_cycle, "
            "last_cycle, position_json, affect FROM nodes"
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
        self._next_id = max_id + 1

    def _persist_node(self, node: dict[str, Any]) -> None:
        if self._conn is None:
            return
        self._write_node(self._conn, node)
        self._conn.commit()

    def _persist_edge(self, e: dict[str, Any]) -> None:
        if self._conn is None:
            return
        self._write_edge(self._conn, e)
        self._conn.commit()

    def _persist_belief(self, b: dict[str, Any]) -> None:
        if self._conn is None:
            return
        self._write_belief(self._conn, b)
        self._conn.commit()

    # ---- internal graph ops (assume lock held) -------------------------
    def _coin_id(self) -> str:
        nid = f"ent-{self._next_id:05d}"
        self._next_id += 1
        return nid

    def _match(
        self,
        appearance: np.ndarray | None,
        threshold: float,
        *,
        exclude: set[str] | None = None,
    ) -> str | None:
        if appearance is None or appearance.size == 0:
            return None
        exclude = exclude or set()
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
        return updated

    # ---- public API ----------------------------------------------------
    def match(self, appearance: Any, threshold: float | None = None) -> str | None:
        """Re-identify ``appearance`` against existing nodes; returns a node id or None."""
        a = None if appearance is None else np.asarray(appearance, dtype=np.float32).reshape(-1)
        with self._lock:
            return self._match(a, self._threshold if threshold is None else float(threshold))

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
                if str(getattr(s, "kind_hint", "object")) == "stuff":
                    continue
                try:
                    confidence = float(getattr(s, "confidence", 1.0) or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                if confidence < 0.2:
                    continue
                a_val = float(getattr(s, "affective_weight", 0.0) or affect.get(getattr(s, "entity_id", ""), 0.0))
                nid = self._upsert(
                    app,
                    str(getattr(s, "kind", "unknown")),
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

    def snapshot(self, limit: int = DEFAULT_SNAPSHOT_LIMIT) -> dict[str, Any]:
        """Windowed read-out for the dashboard: recent/most-seen nodes + their edges + totals."""
        with self._lock:
            total_nodes = len(self._nodes)
            total_edges = len(self._edges)
            degree: dict[str, int] = {}
            for (s, d, _k) in self._edges:
                degree[s] = degree.get(s, 0) + 1
                degree[d] = degree.get(d, 0) + 1
            ranked = sorted(
                self._nodes.values(), key=lambda n: (n["last_cycle"], n["seen_count"]), reverse=True
            )[: max(0, limit)]
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
                }
                for e in self._edges.values()
                if e["src"] in ids and e["dst"] in ids
            ]
            return {
                "nodes": nodes,
                "edges": edges,
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "total_property_beliefs": len(self._beliefs),
                "unstable_property_count": sum(1 for b in self._beliefs.values() if b.get("unstable")),
            }

    def counts(self) -> tuple[int, int]:
        with self._lock:
            return len(self._nodes), len(self._edges)

    def belief_stats(self) -> dict[str, Any]:
        with self._lock:
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
            }

    def clear(self) -> None:
        """Wipe the whole graph (agent reset)."""
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._beliefs.clear()
            self._next_id = 1
            if self._conn is not None:
                self._conn.execute("DELETE FROM nodes")
                self._conn.execute("DELETE FROM edges")
                self._conn.execute("DELETE FROM property_beliefs")
                self._conn.commit()

    def _dump_memory(self, dest: sqlite3.Connection) -> None:
        for n in self._nodes.values():
            self._write_node(dest, n)
        for e in self._edges.values():
            self._write_edge(dest, e)
        for b in self._beliefs.values():
            self._write_belief(dest, b)

    def backup_to(self, path: Path) -> None:
        """Snapshot the graph DB to ``path`` (SQLite online backup; in-memory store materialized)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dest = sqlite3.connect(str(path))
        try:
            with self._lock:
                self._create_schema(dest)
                if self._conn is not None:
                    self._conn.commit()
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
        try:
            self._create_schema(src)
            with self._lock:
                if self._conn is not None:
                    src.backup(self._conn)
                    self._conn.commit()
                    self._load_all(self._conn)
                else:
                    self._load_all(src)
        finally:
            src.close()
