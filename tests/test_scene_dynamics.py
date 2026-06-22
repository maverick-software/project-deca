import torch


def test_scene_dynamics_config_default_on_and_disable(monkeypatch):
    from decadic import config as C

    monkeypatch.delenv("DECADIC_SCENE_DYNAMICS_ENABLED", raising=False)
    assert C.scene_dynamics_enabled() is True
    monkeypatch.setenv("DECADIC_SCENE_DYNAMICS_ENABLED", "0")
    assert C.scene_dynamics_enabled() is False


def test_scene_dynamics_head_shapes_and_zero_skip_loss():
    from decadic.nn.scene_dynamics import (
        SCENE_DYNAMICS_FEATURE_DIM,
        SCENE_DYNAMICS_OUTPUT_DIM,
        SceneDynamicsHead,
        scene_dynamics_loss,
    )

    head = SceneDynamicsHead(feature_dim=SCENE_DYNAMICS_FEATURE_DIM, motor_dim=3, hidden=16)
    features = torch.zeros(4, SCENE_DYNAMICS_FEATURE_DIM)
    motor = torch.zeros(1, 3)
    raw = head(features, motor)
    assert raw.shape == (4, SCENE_DYNAMICS_OUTPUT_DIM)

    target = torch.zeros_like(features)
    mask = torch.zeros(4, dtype=torch.bool)
    loss = scene_dynamics_loss(raw, features, target, mask)
    assert loss.item() == 0.0


def test_scene_dynamics_builds_only_when_discovered_and_enabled(monkeypatch):
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.faculties import CognitionFaculties
    from decadic.nn.neural_stack import NeuralCognitiveStack

    cfg = neural_config_from_env("tiny")
    fac = CognitionFaculties(
        perception_feedback=False,
        perception_mode="discovered",
        encoder_mode="zeros",
    )
    monkeypatch.delenv("DECADIC_SCENE_DYNAMICS_ENABLED", raising=False)
    stack = NeuralCognitiveStack(cfg, faculties=fac)
    assert stack.has_scene_dynamics is True

    monkeypatch.setenv("DECADIC_SCENE_DYNAMICS_ENABLED", "0")
    disabled = NeuralCognitiveStack(cfg, faculties=fac)
    assert disabled.has_scene_dynamics is False

    oracle = NeuralCognitiveStack(
        cfg,
        faculties=CognitionFaculties(
            perception_feedback=False,
            perception_mode="oracle",
            encoder_mode="zeros",
        ),
    )
    assert oracle.has_scene_dynamics is False


def test_scene_workspace_uses_prediction_for_reidentification():
    from decadic.perception.scene_workspace import SceneWorkspace

    ws = SceneWorkspace()
    ws.update(
        [
            {
                "confidence": 0.9,
                "centroid_uv": [0.2, 0.2],
                "appearance": [0.1, 0.2, 0.3],
                "kind_hint": "object",
            }
        ]
    )
    eid = next(iter(ws.entities))
    ws.update(
        [
            {
                "confidence": 0.9,
                "centroid_uv": [0.8, 0.2],
                "appearance": [0.1, 0.2, 0.3],
                "kind_hint": "object",
            }
        ],
        predictions=[
            {
                "entity_id": eid,
                "centroid_uv": [0.8, 0.2],
                "relative": [0.0, 0.0, 1.0],
                "motion": [0.6, 0.0],
                "visibility": 0.9,
                "persistence": 0.9,
                "uncertainty": 0.1,
            }
        ],
    )
    snap = ws.snapshot()
    assert snap["entity_count"] == 1
    assert snap["reidentified_count"] == 1
    assert snap["prediction_assisted_count"] == 1
    assert snap["prediction_error"] == 0.0


def test_scene_dynamics_entity_feature_rejects_semantic_property_keys():
    from decadic.nn.scene_dynamics import entity_feature

    feat = entity_feature(
        {
            "visible": True,
            "centroid_uv": [0.5, 0.25],
            "property_evidence": {
                "roundness": 0.8,
                "food_label": 1.0,
                "class": 2.0,
            },
        }
    )
    assert len(feat) == 32
    assert 0.8 in feat
    assert 1.0 not in feat[21:]
    assert 2.0 not in feat[21:]
