"""Structured JSON logging to stdout and optional log directory.

Logging is non-blocking: the root logger holds a single ``QueueHandler`` that only
enqueues records, while a background ``QueueListener`` thread owns the real stream +
rotating-file handlers and does the actual (potentially slow, especially a Windows
console) I/O. This keeps per-cycle ``logger.info`` calls off the cognitive loop's
critical path. Output format and ordering are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import queue
from datetime import datetime, timezone
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path

# Module-level so a re-init (tests / hot reload) can retire the prior worker.
_listener: QueueListener | None = None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            # Wall-clock event time (ISO-8601, UTC) so the log answers "when",
            # not just "what". Derived from the record's own creation stamp.
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


def setup_logging(log_dir: Path | None = None) -> QueueListener:
    """Configure root logger: JSON lines to stdout (+ rotating file) via a background
    listener so emitting a log never blocks the calling thread. Returns the listener
    so the caller can ``stop()`` it on shutdown."""
    global _listener
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    # Retire a listener from a previous setup_logging (idempotent re-init).
    stop_logging()

    targets: list[logging.Handler] = []
    stream = logging.StreamHandler()
    stream.setFormatter(JsonFormatter())
    targets.append(stream)

    if log_dir is None:
        env = os.environ.get("DECADIC_LOG_DIR")
        if env:
            log_dir = Path(env)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "decadic_server.jsonl",
            maxBytes=8_000_000,
            backupCount=4,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonFormatter())
        targets.append(file_handler)

    # Unbounded queue: a log call only enqueues (microseconds); the listener thread
    # drains to the real handlers. respect_handler_level keeps per-handler levels.
    log_queue: queue.Queue = queue.Queue(-1)
    root.addHandler(QueueHandler(log_queue))
    _listener = QueueListener(log_queue, *targets, respect_handler_level=True)
    _listener.start()
    return _listener


def stop_logging() -> None:
    """Flush and stop the background logging listener (no-op if not running)."""
    global _listener
    if _listener is not None:
        try:
            _listener.stop()
        finally:
            _listener = None
