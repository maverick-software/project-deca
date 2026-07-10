"""Periodic rolling auto-save: config toggle (on by default), consistent CPU
snapshot, and atomic overwrite behaviour.

These cover the pure/unit surface of the feature (no full agent build): the
default-on flag, the deep CPU-clone helper that makes off-thread writes safe,
and the atomic temp+replace save that overwrites this agent's single rolling
snapshot rather than accumulating history.
"""

import torch
import pytest

from decadic import config as C
from decadic.nn.bundle import NeuralBundle, _to_cpu


def test_autosave_default_on(monkeypatch):
    monkeypatch.delenv("DECADIC_AUTOSAVE", raising=False)
    assert C.autosave_enabled() is True


@pytest.mark.parametrize(
    "val,expected",
    [("0", False), ("off", False), ("false", False), ("no", False),
     ("1", True), ("on", True), ("yes", True), ("true", True)],
)
def test_autosave_toggle_env(monkeypatch, val, expected):
    monkeypatch.setenv("DECADIC_AUTOSAVE", val)
    assert C.autosave_enabled() is expected


def test_autosave_interval_default_floor_and_bad_value(monkeypatch):
    monkeypatch.delenv("DECADIC_AUTOSAVE_INTERVAL_S", raising=False)
    assert C.autosave_interval_s() == C.DEFAULT_AUTOSAVE_INTERVAL_S
    # floored at 10s to bound write pressure
    monkeypatch.setenv("DECADIC_AUTOSAVE_INTERVAL_S", "1")
    assert C.autosave_interval_s() == 10.0
    # non-numeric falls back to the default (never raises)
    monkeypatch.setenv("DECADIC_AUTOSAVE_INTERVAL_S", "not-a-number")
    assert C.autosave_interval_s() == C.DEFAULT_AUTOSAVE_INTERVAL_S


def test_to_cpu_deep_clones_nested_tensors():
    src = {
        "a": torch.ones(3),
        "nested": {"b": torch.zeros(2), "meta": "keep"},
        "list": [torch.arange(4), 7],
        "tuple": (torch.tensor([1.0]),),
        "scalar": 5,
    }
    out = _to_cpu(src)
    # non-tensor payload preserved verbatim
    assert out["nested"]["meta"] == "keep"
    assert out["scalar"] == 5
    assert out["list"][1] == 7
    assert isinstance(out["tuple"], tuple)
    # every tensor lands on CPU
    assert out["a"].device.type == "cpu"
    assert out["nested"]["b"].device.type == "cpu"
    assert out["tuple"][0].device.type == "cpu"
    # and is a real clone: mutating the source must not touch the snapshot
    src["a"].add_(100.0)
    assert torch.equal(out["a"], torch.ones(3))


def test_atomic_save_payload_roundtrip_and_rolling_overwrite(tmp_path):
    path = tmp_path / "agent_x_autosave_brain.pt"
    NeuralBundle.atomic_save_payload({"v": torch.arange(5), "preset": "tiny"}, path)
    assert path.is_file()
    loaded = torch.load(path, weights_only=False)
    assert torch.equal(loaded["v"], torch.arange(5))

    # a second save overwrites the previous snapshot in place (rolling, not additive)
    NeuralBundle.atomic_save_payload({"v": torch.arange(3), "preset": "tiny"}, path)
    loaded2 = torch.load(path, weights_only=False)
    assert torch.equal(loaded2["v"], torch.arange(3))

    # the temp file is consumed by os.replace, and there is still exactly one brain
    assert not (tmp_path / "agent_x_autosave_brain.pt.tmp").exists()
    assert len(list(tmp_path.glob("*_autosave_brain.pt"))) == 1


def _write_lib_save(store, save_id, *, kind, agent_id, created_at):
    store.create_dir(save_id)
    store.write_manifest(
        save_id,
        {"save_id": save_id, "kind": kind, "source_agent_id": agent_id, "created_at": created_at},
    )


def test_prune_saves_keeps_only_newest_autosave_per_agent(tmp_path):
    """Rolling behaviour: an auto-save keeps exactly one entry per agent, newest
    by created_at, and never touches other agents' saves or manual saves."""
    from decadic.api.saved_agents.store import SavedAgentStore

    store = SavedAgentStore(tmp_path)
    _write_lib_save(store, "aaaaaaaaaaaa", kind="autosave", agent_id="agent-1",
                    created_at="2026-07-09T10:00:00")
    _write_lib_save(store, "bbbbbbbbbbbb", kind="autosave", agent_id="agent-1",
                    created_at="2026-07-09T11:00:00")  # newest for agent-1
    _write_lib_save(store, "cccccccccccc", kind="autosave", agent_id="agent-2",
                    created_at="2026-07-09T10:30:00")  # different agent
    _write_lib_save(store, "dddddddddddd", kind="manual", agent_id="agent-1",
                    created_at="2026-07-09T09:00:00")  # manual save

    removed = store.prune_saves(source_agent_id="agent-1", kind="autosave", keep=1)
    assert removed == 1
    ids = {m["save_id"] for m in store.list()}
    assert "bbbbbbbbbbbb" in ids       # newest agent-1 auto-save kept
    assert "aaaaaaaaaaaa" not in ids   # older agent-1 auto-save pruned
    assert "cccccccccccc" in ids       # other agent untouched
    assert "dddddddddddd" in ids       # manual save never pruned
