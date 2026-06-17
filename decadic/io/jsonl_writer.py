"""Background JSONL appender: move per-cycle trace/probe file writes off the hot path.

A single process-wide daemon thread drains ``(path, line)`` items from a queue and
appends each ``line`` (plus a newline) to the target file. The cognitive cycle and the
interpretability probes still build the JSON line on-thread -- that compute is cheap and
keeps the file *content* byte-identical to before -- and hand only the actual disk write
here, so the open/append/flush never blocks the loop thread.

This backs two existing, already-gated sinks (the cycle-trace dump and probe capture);
it adds no new default-on flag and is inert until one of those sinks fires.

Correctness:
- No line is ever lost: if the bounded queue is full, ``append`` writes inline
  (backpressure) rather than dropping the record. Trace/probe records are
  self-contained (each carries its own cycle index), so the rare inline write reordering
  under saturation is immaterial.
- The worker is started lazily on first ``append`` and is a daemon, so an idle process
  spawns no thread and the worker never keeps the interpreter alive.
- ``flush()`` blocks until the queue is drained (used by tests / shutdown); ``close()``
  drains and retires the worker.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_SENTINEL = object()


class JsonlWriter:
    """A single daemon thread that appends queued lines to JSONL files off-thread."""

    def __init__(self, *, max_queue: int = 8192) -> None:
        self._max_queue = max(1, int(max_queue))
        self._queue: queue.Queue = queue.Queue(maxsize=self._max_queue)
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        # Serializes the actual file append so the worker and an inline backpressure
        # write never open/append the same path concurrently (which loses lines on
        # Windows, where O_APPEND seek+write is not atomic across handles).
        self._io_lock = threading.Lock()

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._drain_loop, name="jsonl-writer", daemon=True
                )
                self._worker.start()

    def append(self, path: str | Path, line: str) -> None:
        """Enqueue one already-serialized JSONL ``line`` for ``path`` (newline added here)."""
        self._ensure_worker()
        item = (str(path), str(line))
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Never drop a record: persist inline when the queue is saturated. This is
            # the safety valve, not the expected path.
            logger.warning("jsonl writer queue full; writing inline path=%s", item[0])
            self._write(item[0], item[1])

    def _write(self, path: str, line: str) -> None:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with self._io_lock:
                with p.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:  # pragma: no cover - a bad sink must not kill the worker
            logger.exception("jsonl writer append failed path=%s", path)

    def _drain_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                self._write(item[0], item[1])
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Block until every queued line has been written."""
        self._queue.join()

    def close(self) -> None:
        """Drain and stop the worker (the writer stays usable; it relazy-starts)."""
        worker = self._worker
        if worker is not None and worker.is_alive():
            self._queue.join()
            self._queue.put(_SENTINEL)
            worker.join(timeout=2.0)
        self._worker = None


_writer: JsonlWriter | None = None
_writer_lock = threading.Lock()


def get_jsonl_writer() -> JsonlWriter:
    """Process-wide :class:`JsonlWriter` singleton (lazily constructed)."""
    global _writer
    if _writer is None:
        with _writer_lock:
            if _writer is None:
                _writer = JsonlWriter()
    return _writer
