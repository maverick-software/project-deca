"""Non-blocking logging: the root logger only enqueues; a listener does the I/O.

``setup_logging`` must (1) leave the root logger holding a single ``QueueHandler`` so a
per-cycle ``logger.info`` never blocks on the stream/file write, (2) drain every record
to the JSONL file through the background listener, and (3) flush cleanly on
``stop_logging``. The on-disk format is unchanged (ISO-UTC ``time`` + level/logger/msg).
"""

from __future__ import annotations

import json
import logging
from logging.handlers import QueueHandler, QueueListener

from decadic.logging import setup_logging, stop_logging


def _reset_root() -> None:
    stop_logging()
    logging.getLogger().handlers.clear()


def test_root_holds_only_a_queue_handler(tmp_path):
    try:
        listener = setup_logging(tmp_path / "logs")
        root = logging.getLogger()
        assert isinstance(listener, QueueListener)
        # The loop thread only ever touches this one cheap handler.
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], QueueHandler)
    finally:
        _reset_root()


def test_records_drain_to_file_and_flush_on_stop(tmp_path):
    log_dir = tmp_path / "logs"
    try:
        setup_logging(log_dir)
        logging.getLogger("decadic.test").info("async_log_marker cycle=%d", 7)
        stop_logging()  # flush + retire the listener thread
        text = (log_dir / "decadic_server.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        marker = [r for r in rows if "async_log_marker cycle=7" in r["message"]]
        assert marker, "the logged record never reached the file"
        rec = marker[0]
        assert rec["level"] == "INFO"
        assert rec["logger"] == "decadic.test"
        assert "time" in rec and rec["time"].endswith("+00:00")
    finally:
        _reset_root()


def test_setup_is_idempotent(tmp_path):
    try:
        first = setup_logging(tmp_path / "logs")
        second = setup_logging(tmp_path / "logs")
        # Re-init retires the old listener and installs exactly one fresh QueueHandler.
        assert first is not second
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], QueueHandler)
    finally:
        _reset_root()
