"""WS4-M0.3: backend seam for the memory stores (M5 cutover: lance+kuzu default).

Construction of the episodic store and the semantic (long-term) graph goes
through these factories so the storage engine can be swapped by environment
variable without touching any cognition-side caller:

- ``DECADIC_MEMORY_BACKEND``: ``lancedb`` (default) | ``sqlite`` (legacy)
- ``DECADIC_GRAPH_BACKEND``:  ``kuzu`` (default) | ``sqlite`` (legacy)

WS4-M5 cutover (2026-07-03): with no env vars set the factories return the
LanceDB episodic store (with its full-mirror L1 recall cache) and the Kuzu
semantic graph. ``sqlite`` remains fully supported as the explicit legacy
value and returns exactly the classes the codebase constructed before the
seam existed (the parity baseline): a bare :class:`EpisodicStore` /
:class:`LongTermGraph` from ``make_episodic_store`` / ``make_semantic_graph``,
and the write-behind wrappers from the ``*_runtime_*`` variants that
:class:`decadic.agents.runtime.AgentRuntime` uses.

Backend imports are lazy so selecting ``sqlite`` never imports ``lancedb`` (and
importing this module works without the optional dependencies installed).
Tests that exercise a concrete class keep constructing it directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MEMORY_BACKEND_ENV = "DECADIC_MEMORY_BACKEND"
GRAPH_BACKEND_ENV = "DECADIC_GRAPH_BACKEND"

_MEMORY_BACKENDS = ("sqlite", "lancedb")
_GRAPH_BACKENDS = ("sqlite", "kuzu")


def memory_backend() -> str:
    """Resolve the episodic-store backend name from the environment.

    Default is ``lancedb`` (WS4-M5 cutover); ``sqlite`` is the explicit
    legacy value.
    """
    value = os.environ.get(MEMORY_BACKEND_ENV, "lancedb").strip().lower() or "lancedb"
    if value not in _MEMORY_BACKENDS:
        raise ValueError(
            f"{MEMORY_BACKEND_ENV}={value!r} is not a known memory backend; "
            f"expected one of {_MEMORY_BACKENDS}"
        )
    return value


def graph_backend() -> str:
    """Resolve the semantic-graph backend name from the environment.

    Default is ``kuzu`` (WS4-M5 cutover); ``sqlite`` is the explicit legacy
    value.
    """
    value = os.environ.get(GRAPH_BACKEND_ENV, "kuzu").strip().lower() or "kuzu"
    if value not in _GRAPH_BACKENDS:
        raise ValueError(
            f"{GRAPH_BACKEND_ENV}={value!r} is not a known graph backend; "
            f"expected one of {_GRAPH_BACKENDS}"
        )
    return value


def make_episodic_store(db_path: Path | None = None) -> Any:
    """Episodic store for the selected backend (bare store, no write-behind)."""
    backend = memory_backend()
    if backend == "lancedb":
        from decadic.memory.lancedb_store import LanceEpisodicStore

        return LanceEpisodicStore(db_path)
    from decadic.memory.episodic_store import EpisodicStore

    return EpisodicStore(db_path)


def make_semantic_graph(db_path: Path | None = None, **kwargs: Any) -> Any:
    """Semantic (long-term) graph for the selected backend.

    ``kwargs`` are forwarded to the concrete class (e.g. ``match_threshold``).
    """
    backend = graph_backend()
    if backend == "kuzu":
        from decadic.memory.kuzu_graph import KuzuLongTermGraph

        return KuzuLongTermGraph(db_path, **kwargs)
    from decadic.memory.semantic_graph import LongTermGraph

    return LongTermGraph(db_path, **kwargs)


def make_runtime_episodic_store(
    db_path: Path | None = None, *, enabled: bool = True
) -> Any:
    """Episodic store as the agent runtime builds it.

    lancedb (default): a :class:`LanceEpisodicStore`, whose internal write
    micro-batching plays the write-behind role (``set_async``/``flush``/
    ``close`` match the wrapper's duck type, so every ``getattr``-guarded
    runtime call still works).
    sqlite (legacy): the :class:`WriteBehindEpisodicStore` wrapper,
    byte-identical to the pre-seam construction (``enabled`` is the
    async-persistence birth default).
    """
    backend = memory_backend()
    if backend == "lancedb":
        from decadic.memory.lancedb_store import LanceEpisodicStore

        store = LanceEpisodicStore(db_path)
        store.set_async(bool(enabled))
        return store
    from decadic.memory.write_behind import WriteBehindEpisodicStore

    return WriteBehindEpisodicStore(db_path, enabled=enabled)


def make_runtime_ltm_graph(db_path: Path | None = None, **kwargs: Any) -> Any:
    """Long-term graph as the agent runtime builds it (write-behind wrapper).

    ``kwargs`` are forwarded to the write-behind class
    (``match_threshold``/``max_queue``/``enabled``). ``kuzu`` (the default)
    returns :class:`WriteBehindKuzuLongTermGraph` -- the same write-behind
    layer (diamond subclass) over the kuzu graph, so async consolidation,
    retention and runtime metrics behave identically to the legacy sqlite
    wrapper (:class:`WriteBehindLongTermGraph`).
    """
    backend = graph_backend()
    if backend == "kuzu":
        from decadic.memory.kuzu_graph import WriteBehindKuzuLongTermGraph

        return WriteBehindKuzuLongTermGraph(db_path, **kwargs)
    from decadic.memory.ltm_write_behind import WriteBehindLongTermGraph

    return WriteBehindLongTermGraph(db_path, **kwargs)
