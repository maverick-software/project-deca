"""WS4-M0.3: backend seam for the memory stores.

Construction of the episodic store and the semantic (long-term) graph goes
through these factories so the storage engine can be swapped by environment
variable without touching any cognition-side caller:

- ``DECADIC_MEMORY_BACKEND``: ``sqlite`` (default) | ``lancedb``
- ``DECADIC_GRAPH_BACKEND``:  ``sqlite`` (default) | ``kuzu`` (reserved, WS4-M2)

With no env vars set every factory returns exactly the classes the codebase
constructed before the seam existed (the parity baseline): a bare
:class:`EpisodicStore` / :class:`LongTermGraph` from ``make_episodic_store`` /
``make_semantic_graph``, and the write-behind wrappers from the ``*_runtime_*``
variants that :class:`decadic.agents.runtime.AgentRuntime` uses.

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

_KUZU_MESSAGE = (
    "DECADIC_GRAPH_BACKEND=kuzu is reserved for WS4-M2 (Kuzu semantic-graph "
    "backend, not yet implemented); unset it or use 'sqlite'."
)


def memory_backend() -> str:
    """Resolve the episodic-store backend name from the environment."""
    value = os.environ.get(MEMORY_BACKEND_ENV, "sqlite").strip().lower() or "sqlite"
    if value not in _MEMORY_BACKENDS:
        raise ValueError(
            f"{MEMORY_BACKEND_ENV}={value!r} is not a known memory backend; "
            f"expected one of {_MEMORY_BACKENDS}"
        )
    return value


def graph_backend() -> str:
    """Resolve the semantic-graph backend name from the environment."""
    value = os.environ.get(GRAPH_BACKEND_ENV, "sqlite").strip().lower() or "sqlite"
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
        raise NotImplementedError(_KUZU_MESSAGE)
    from decadic.memory.semantic_graph import LongTermGraph

    return LongTermGraph(db_path, **kwargs)


def make_runtime_episodic_store(
    db_path: Path | None = None, *, enabled: bool = True
) -> Any:
    """Episodic store as the agent runtime builds it.

    sqlite: the :class:`WriteBehindEpisodicStore` wrapper, byte-identical to the
    pre-seam construction (``enabled`` is the async-persistence birth default).
    lancedb: a :class:`LanceEpisodicStore`, whose internal write micro-batching
    plays the write-behind role (``set_async``/``flush``/``close`` match the
    wrapper's duck type, so every ``getattr``-guarded runtime call still works).
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
    """Long-term graph as the agent runtime builds it (write-behind on sqlite).

    ``kwargs`` are forwarded to :class:`WriteBehindLongTermGraph`
    (``match_threshold``/``max_queue``/``enabled``). ``kuzu`` is reserved for
    WS4-M2 and raises :class:`NotImplementedError` for now.
    """
    backend = graph_backend()
    if backend == "kuzu":
        raise NotImplementedError(_KUZU_MESSAGE)
    from decadic.memory.ltm_write_behind import WriteBehindLongTermGraph

    return WriteBehindLongTermGraph(db_path, **kwargs)
