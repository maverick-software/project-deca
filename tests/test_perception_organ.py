from __future__ import annotations

import base64
import io

import numpy as np

from decadic.perception.bootstrap import (
    ObjectFileTarget,
    PerceptionBootstrapFrame,
    read_bootstrap_jsonl,
    strip_teacher_fields,
    write_bootstrap_jsonl,
)
from decadic.perception.object_files import HEALTH_STATES, object_files_from_proposals
from decadic.perception.organ import PerceptionOrgan


def _obs(gray: np.ndarray, ts: str = "t") -> dict:
    from PIL import Image

    arr = np.clip(gray * 255, 0, 255).astype("uint8")
    bio = io.BytesIO()
    Image.fromarray(arr, mode="L").save(bio, format="PNG")
    return {
        "timestamp": ts,
        "vision": {"data": base64.b64encode(bio.getvalue()).decode("ascii")},
        "proprioception": {"contacts": []},
    }


def test_retinotopic_map_preserves_image_position():
    img = np.zeros((16, 16), dtype=np.float32)
    img[2, 13] = 1.0
    organ = PerceptionOrgan(grid_size=16)
    _props, _diag, ret = organ.process(_obs(img), [])
    intensity = np.asarray(ret["intensity"], dtype=np.float32)
    y, x = np.unravel_index(intensity.argmax(), intensity.shape)
    assert x >= 12
    assert y <= 3


def test_flow_confidence_separates_global_from_local_motion():
    organ = PerceptionOrgan(grid_size=16)
    organ.process(_obs(np.zeros((16, 16), dtype=np.float32), "t0"), [])

    global_frame = np.full((16, 16), 0.2, dtype=np.float32)
    props = [{"idx": 0, "presence": 0.9, "uv": [0.5, 0.5], "spread": 0.1, "appearance": [1.0]}]
    _props, global_diag, _ret = organ.process(_obs(global_frame, "t1"), props)

    local_frame = global_frame.copy()
    local_frame[7:10, 7:10] = 1.0
    props, local_diag, _ret = organ.process(_obs(local_frame, "t2"), props)
    assert global_diag["flow_confidence"] <= local_diag["flow_confidence"]
    assert props[0]["local_motion"] > global_diag["global_motion"]


def test_body_candidate_uses_motion_motor_and_touch():
    organ = PerceptionOrgan(grid_size=16)
    organ.process(_obs(np.zeros((16, 16), dtype=np.float32), "t0"), [])
    img = np.zeros((16, 16), dtype=np.float32)
    img[6:10, 6:10] = 1.0
    obs = _obs(img, "t1")
    obs["proprioception"]["contacts"] = [80.0]
    props = [{"idx": 0, "presence": 0.9, "uv": [0.5, 0.5], "spread": 0.1, "appearance": [1.0]}]
    enriched, diag, _ret = organ.process(obs, props, prev_motor=[0.5, 0.0])
    assert enriched[0]["kind_hint"] == "body_part_candidate"
    assert diag["body_candidate_count"] == 1
    files = object_files_from_proposals(enriched)
    assert files[0].kind_hint == "body_part_candidate"
    assert files[0].agency > 0.0


def test_bootstrap_frames_are_offline_and_label_free(tmp_path):
    frame = PerceptionBootstrapFrame(
        timestamp="2026-06-20T00:00:00Z",
        targets=[
            ObjectFileTarget(
                target_id="teacher-1",
                centroid_uv=[0.25, 0.5],
                relative=[1.0, 0.0, 0.0],
                depth=1.0,
                motion=[0.1, 0.0],
                kind_hint="object",
            )
        ],
    )
    path = tmp_path / "perception.jsonl"
    write_bootstrap_jsonl(path, [frame])
    loaded = read_bootstrap_jsonl(path)
    assert loaded[0].targets[0].centroid_uv == [0.25, 0.5]
    stripped = strip_teacher_fields(loaded[0])
    assert "food" not in str(stripped).lower()
    assert "label" not in stripped[0]


def test_health_state_contract_lists_required_states():
    assert {"healthy", "low_confidence", "collapsed", "no_objects", "teacher_only", "stale_frame"} <= set(HEALTH_STATES)

