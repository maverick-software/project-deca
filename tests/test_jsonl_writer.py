"""Background JSONL writer: off-thread file appends with lossless flush.

The cognitive cycle and the probes serialize a line on-thread and hand the append to
this writer, so: every queued line must land after ``flush``, nothing may be dropped
under backpressure (the synchronous fallback), the content must be byte-for-byte what
was enqueued, and the parent directory must be created on demand. CPU-only and fast.
"""

from __future__ import annotations

import json

from decadic.io import get_jsonl_writer
from decadic.io.jsonl_writer import JsonlWriter


def test_flush_writes_all_lines(tmp_path):
    w = JsonlWriter()
    path = tmp_path / "nested" / "trace.jsonl"  # parent dir does not exist yet
    try:
        for i in range(40):
            w.append(path, json.dumps({"cycle": i}))
        w.flush()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 40
        cycles = [json.loads(line)["cycle"] for line in lines]
        assert sorted(cycles) == list(range(40))  # nothing lost
    finally:
        w.close()


def test_content_is_byte_identical_to_enqueued(tmp_path):
    w = JsonlWriter()
    path = tmp_path / "out.jsonl"
    try:
        payload = json.dumps({"a": 1, "b": [1, 2, 3], "c": "x y z"})
        w.append(path, payload)
        w.flush()
        assert path.read_text(encoding="utf-8") == payload + "\n"
    finally:
        w.close()


def test_backpressure_never_drops(tmp_path):
    # A 1-slot queue forces the inline synchronous fallback for most appends.
    w = JsonlWriter(max_queue=1)
    path = tmp_path / "bp.jsonl"
    try:
        for i in range(60):
            w.append(path, json.dumps({"i": i}))
        w.flush()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 60  # every record landed despite the saturated queue
    finally:
        w.close()


def test_singleton_is_stable():
    assert get_jsonl_writer() is get_jsonl_writer()


def test_close_then_reuse_relazy_starts(tmp_path):
    w = JsonlWriter()
    path = tmp_path / "reuse.jsonl"
    w.append(path, json.dumps({"n": 1}))
    w.close()  # drains + retires the worker
    w.append(path, json.dumps({"n": 2}))  # must relazy-start the worker
    w.flush()
    try:
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    finally:
        w.close()
