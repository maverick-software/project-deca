"""Write-behind episodic store: move the per-cycle SQLite write off the hot path.

The cognitive cycle calls ``episodic.append`` once per cycle inside the agent
lock; the underlying SQLite ``INSERT``+``commit`` (an fsync) blocks the cycle. When
async mode is ON this wrapper hands the write to a single background worker thread
so the cycle returns immediately, while every read (``recent`` / ``search_similar`` /
``retrieval_context_vector``) is inherited unchanged from :class:`EpisodicStore` and
runs against the same thread-safe connection.

This store is **always** used by the runtime so the mode can be flipped live from
the dashboard (Agent Settings) without recreating the store or migrating the SQLite
connection. ``set_async(False)`` drains pending writes and stops the worker, after
which ``append`` is a plain synchronous write -- byte-identical to a bare
:class:`EpisodicStore`. The worker is started lazily, so an agent that never enables
async never spawns a thread (the test baseline, pinned OFF, has zero overhead).

Correctness:
- No write is ever lost: if the bounded queue is full, ``append`` falls back to a
  synchronous write (backpressure) rather than dropping the record.
- ``append`` and ``set_async`` both run under the agent lock (cycle vs. configure),
  so a mode flip can never race an in-flight append.
- ``clear`` / ``backup_to`` / ``restore_from`` ``flush()`` first, so a snapshot or
  wipe never races a pending write.
- Visibility: while async is ON a just-appended episode becomes queryable once the
  worker drains it (sub-ms to a few ms later). Episodic recall excludes the current
  cycle and is associative, so this lag is immaterial.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

from decadic import config as C
from decadic.memory.episodic_store import EpisodicRecord, EpisodicStore

logger = logging.getLogger(__name__)

_SENTINEL = object()


class WriteBehindEpisodicStore(EpisodicStore):
    """:class:`EpisodicStore` whose ``append`` can be persisted on a background thread."""

    def __init__(
        self, db_path: Path | None = None, *, max_queue: int = 4096, enabled: bool = True
    ) -> None:
        super().__init__(db_path)
        self._max_queue = max(1, int(max_queue))
        self._queue: queue.Queue | None = None
        self._worker: threading.Thread | None = None
        self._async_enabled = False
        if enabled:
            self.set_async(True)

    @property
    def async_enabled(self) -> bool:
        return self._async_enabled

    def set_async(self, enabled: bool) -> None:
        """Turn background persistence on/off (drains + stops the worker when off)."""
        enabled = bool(enabled)
        if enabled == self._async_enabled:
            return
        if enabled:
            self._queue = queue.Queue(maxsize=self._max_queue)
            self._async_enabled = True
            self._worker = threading.Thread(
                target=self._drain_loop,
                args=(self._queue,),
                name="episodic-write-behind",
                daemon=True,
            )
            self._worker.start()
        else:
            # Flip to synchronous first so any subsequent append writes inline, then
            # drain the backlog and retire the worker.
            self._async_enabled = False
            old_queue, old_worker = self._queue, self._worker
            self._queue, self._worker = None, None
            if old_queue is not None:
                old_queue.join()
                old_queue.put(_SENTINEL)
            if old_worker is not None:
                old_worker.join(timeout=2.0)

    def append(self, record: EpisodicRecord) -> None:
        """Enqueue for background persistence when async; else write synchronously."""
        q = self._queue if self._async_enabled else None
        if q is None:
            super().append(record)
            return
        # Recall visibility is immediate and RAM-backed; only disk persistence is
        # deferred. This prevents the hot recall path from forcing a write flush.
        self._cache_record(record)
        try:
            q.put_nowait(record)
        except queue.Full:
            # Never drop a memory: the queue is saturated, so persist inline. This
            # is the safety valve, not the expected path.
            logger.warning("episodic write-behind queue full; writing synchronously")
            self._append_persist(record, update_cache=False)

    def _drain_loop(self, q: queue.Queue) -> None:
        while True:
            item = q.get()
            try:
                if item is _SENTINEL:
                    return
                batch = [item]
                deadline = time.perf_counter() + (C.episodic_write_batch_ms() / 1000.0)
                while len(batch) < C.episodic_write_batch_size():
                    timeout = max(0.0, deadline - time.perf_counter())
                    if timeout <= 0.0:
                        break
                    try:
                        nxt = q.get(timeout=timeout)
                    except queue.Empty:
                        break
                    if nxt is _SENTINEL:
                        q.task_done()
                        try:
                            self._append_many_persist(batch, update_cache=False)
                        except Exception:  # pragma: no cover
                            logger.exception("episodic write-behind batch persist failed")
                        return
                    batch.append(nxt)
                try:
                    self._append_many_persist(batch, update_cache=False)
                except Exception:  # pragma: no cover - persistence must not kill the worker
                    logger.exception("episodic write-behind persist failed")
            finally:
                for _ in range(1 if item is _SENTINEL else len(batch)):
                    q.task_done()

    def flush(self) -> None:
        """Block until all queued writes have been persisted (no-op when sync)."""
        q = self._queue
        if q is not None:
            q.join()

    def clear(self) -> None:
        self.flush()
        super().clear()

    def backup_to(self, path: Path) -> None:
        self.flush()
        super().backup_to(path)

    def restore_from(self, path: Path) -> None:
        self.flush()
        super().restore_from(path)

    def close(self) -> None:
        """Drain and stop the worker; the store stays usable in synchronous mode."""
        self.set_async(False)
