"""WS4-M0.1: LanceDB + Kuzu smoke test on this machine (go/no-go gate).

Verifies both embedded engines on Windows inside the project venv:
create -> insert vectors -> similarity search -> close -> REOPEN -> data
survived. Kuzu failure here triggers the PRD's fallback path (LanceDB-only
WS4, graph stays on SQLite) - decided now, not discovered mid-migration.

The lancedb section uses the exact table-creation mechanism
``LanceEpisodicStore`` uses: an explicit pyarrow schema with fixed-size-list
vector columns, ``create_table(schema=...)``, then a batched ``table.add``.
(Letting lance infer the schema from plain python lists yields List<Float64>,
which it rejects as "not a vector" - the original M0.1 failure mode.)

The kuzu section prints the version first (known even on failure), closes
handles explicitly before the same-process reopen (``del`` alone does not
release the file lock on some versions), and retries under the repo's
``reports/`` folder if the temp-dir attempt fails (AV/lock interference).

Usage:  .venv\\Scripts\\python.exe scripts\\ws4_smoke.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

DIM = 80
KEY_DIM = 16
N = 500


def smoke_lancedb(root: Path) -> str:
    import lancedb
    import pyarrow as pa

    print(f"[info] lancedb version: {lancedb.__version__}")
    uri = str(root / "lance_smoke")
    rng = np.random.default_rng(7)
    vecs = rng.normal(size=(N, DIM)).astype(np.float32)
    keys = vecs[:, -KEY_DIM:].copy()

    # Explicit schema: fixed-size-list vector columns, exactly like
    # LanceEpisodicStore._arrow_schema (decadic/memory/lancedb_store.py).
    schema = pa.schema(
        [
            pa.field("cycle", pa.int64()),
            pa.field("salience", pa.float64()),
            pa.field("embedding", pa.list_(pa.float32(), DIM)),
            pa.field("percept_key", pa.list_(pa.float32(), KEY_DIM)),
        ]
    )
    db = lancedb.connect(uri)
    table = db.create_table("episodes", schema=schema)
    rows = [
        {
            "cycle": i,
            "salience": float(i % 10) / 10.0,
            "embedding": [float(x) for x in vecs[i]],
            "percept_key": [float(x) for x in keys[i]],
        }
        for i in range(N)
    ]
    table.add(pa.Table.from_pylist(rows, schema=schema))

    # full-vector search: nearest to row 42 must be row 42
    hits = table.search(vecs[42].tolist(), vector_column_name="embedding").limit(3).to_list()
    assert hits and hits[0]["cycle"] == 42, f"full-vector search wrong: {hits[0]['cycle']}"
    # percept-key sub-vector search (the WS3 novelty fix mechanism)
    khits = table.search(keys[7].tolist(), vector_column_name="percept_key").limit(3).to_list()
    assert khits and khits[0]["cycle"] == 7, f"percept-key search wrong: {khits[0]['cycle']}"
    del table, db

    # reopen: persistence check
    db2 = lancedb.connect(uri)
    t2 = db2.open_table("episodes")
    assert t2.count_rows() == N, f"rows lost on reopen: {t2.count_rows()}"

    return (
        f"lancedb {lancedb.__version__}: {N} rows via explicit fixed-size-list "
        f"schema, both vector columns searchable, survives reopen"
    )


def _close_kuzu_handles(*handles: object) -> None:
    """Best-effort explicit close (kuzu versions differ), then force GC."""
    import gc

    for h in handles:
        close = getattr(h, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    gc.collect()


def _kuzu_roundtrip(dbdir: str) -> str:
    """create -> populate -> assert -> close -> reopen -> assert; returns vector-index note."""
    import kuzu

    try:
        db = kuzu.Database(dbdir)
    except Exception as e:
        raise RuntimeError(
            f"kuzu cannot create a database on this machine ({dbdir}): "
            f"{type(e).__name__}: {e}"
        ) from e
    conn = kuzu.Connection(db)
    conn.execute(
        "CREATE NODE TABLE Entity(id INT64, appearance FLOAT[16], kind STRING, PRIMARY KEY(id))"
    )
    conn.execute("CREATE REL TABLE CO_PRESENT(FROM Entity TO Entity, weight DOUBLE)")
    rng = np.random.default_rng(11)
    apps = rng.normal(size=(60, KEY_DIM)).astype(np.float32)
    apps /= np.linalg.norm(apps, axis=1, keepdims=True)
    for i in range(60):
        conn.execute(
            "CREATE (e:Entity {id: $id, appearance: $app, kind: $kind})",
            {"id": i, "app": apps[i].tolist(), "kind": "npc" if i % 2 else "stuff"},
        )
    conn.execute(
        "MATCH (a:Entity {id: 1}), (b:Entity {id: 2}) CREATE (a)-[:CO_PRESENT {weight: 0.9}]->(b)"
    )
    res = conn.execute("MATCH (e:Entity) RETURN count(e)")
    count = res.get_next()[0]
    assert count == 60, f"node count wrong: {count}"
    res = conn.execute(
        "MATCH (a:Entity)-[r:CO_PRESENT]->(b:Entity) RETURN a.id, b.id, r.weight"
    )
    row = res.get_next()
    assert row[0] == 1 and row[1] == 2, f"edge wrong: {row}"

    # vector index (identity matching mechanism)
    vector_note = "native vector index OK"
    try:
        conn.execute("INSTALL vector; LOAD vector;")
        conn.execute(
            "CALL CREATE_VECTOR_INDEX('Entity', 'app_idx', 'appearance', metric := 'cosine')"
        )
        res = conn.execute(
            "CALL QUERY_VECTOR_INDEX('Entity', 'app_idx', $q, 1) RETURN node.id, distance",
            {"q": apps[5].tolist()},
        )
        row = res.get_next()
        assert row[0] == 5, f"vector index top-1 wrong: {row}"
    except Exception as e:  # index API varies by version - record, don't fail the gate
        vector_note = f"vector index unavailable ({str(e)[:80]}) - linear match fallback required"

    del res
    # Release the file lock BEFORE reopening in the same process: on some kuzu
    # versions `del conn, db` alone leaves the lock held, so the reopen below
    # fails with "Could not set lock on file".
    _close_kuzu_handles(conn, db)
    del conn, db

    try:
        db2 = kuzu.Database(dbdir)
        conn2 = kuzu.Connection(db2)
        count2 = conn2.execute("MATCH (e:Entity) RETURN count(e)").get_next()[0]
    except Exception as e:
        raise RuntimeError(
            f"kuzu reopen-after-close failed ({dbdir}): {type(e).__name__}: {e}"
        ) from e
    assert count2 == 60, f"rows lost on reopen: {count2}"
    _close_kuzu_handles(conn2, db2)
    del conn2, db2
    return vector_note


def smoke_kuzu(root: Path, fallback_base: Path | None = None) -> str:
    import kuzu

    # Version first: known even when everything below fails.
    print(f"[info] kuzu version: {kuzu.__version__}")

    try:
        note = _kuzu_roundtrip(str(root / "kuzu_smoke"))
        return f"kuzu {kuzu.__version__}: 60 nodes + rel, survives reopen (temp dir); {note}"
    except Exception as first:
        if fallback_base is None:
            raise
        first_msg = f"{type(first).__name__}: {first}"
        print(f"[warn] kuzu temp-dir attempt failed: {first_msg}")

    # Second attempt outside the OS temp dir (temp-dir locking / AV
    # interference is environmental; the repo tree may behave differently).
    fallback_base.mkdir(parents=True, exist_ok=True)
    alt = Path(tempfile.mkdtemp(prefix="ws4_smoke_kuzu_", dir=str(fallback_base)))
    try:
        note = _kuzu_roundtrip(str(alt / "kuzu_smoke"))
    except Exception as second:
        raise RuntimeError(
            f"both attempts failed; temp dir: {first_msg}; "
            f"reports dir: {type(second).__name__}: {second}"
        ) from second
    finally:
        shutil.rmtree(alt, ignore_errors=True)
    return (
        f"kuzu {kuzu.__version__}: 60 nodes + rel, survives reopen "
        f"(reports-dir fallback; temp-dir attempt failed: {first_msg}); {note}"
    )


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="ws4_smoke_"))
    fallback_base = Path(__file__).resolve().parents[1] / "reports"
    results: list[tuple[str, bool, str]] = []
    for name, fn, args in (
        ("lancedb", smoke_lancedb, (root,)),
        ("kuzu", smoke_kuzu, (root, fallback_base)),
    ):
        try:
            results.append((name, True, fn(*args)))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))
    shutil.rmtree(root, ignore_errors=True)
    ok = True
    for name, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok = ok and passed
    print("WS4_SMOKE:", "GO" if ok else ("LANCEDB_ONLY" if results[0][1] else "NO_GO"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
