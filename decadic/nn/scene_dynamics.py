"""Anonymous scene-dynamics prediction head.

This is a perception-side world model. It predicts the next anonymous scene
entity state from the previous anonymous state plus efference copy. It never
uses labels, object classes, task rewards, or simulator semantics.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

SCENE_DYNAMICS_FEATURE_DIM = 32
SCENE_DYNAMICS_OUTPUT_DIM = 10

FORBIDDEN_SCENE_DYNAMICS_TOKENS = (
    "label",
    "class",
    "kind_name",
    "food",
    "water",
    "hand",
    "wall",
    "building",
    "ball",
    "bear",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _vec(raw: Any, n: int) -> list[float]:
    if not isinstance(raw, list):
        return [0.0] * n
    vals: list[float] = []
    for item in raw[:n]:
        vals.append(_finite(item))
    while len(vals) < n:
        vals.append(0.0)
    return vals


def _numeric_property_values(raw: Any, cap: int) -> list[float]:
    if not isinstance(raw, dict):
        return [0.0] * cap
    vals: list[float] = []
    for key in sorted(raw):
        low = str(key).lower()
        if any(tok in low for tok in FORBIDDEN_SCENE_DYNAMICS_TOKENS):
            continue
        value = raw.get(key)
        if isinstance(value, (int, float)):
            vals.append(_finite(value))
        elif isinstance(value, list):
            for item in value:
                vals.append(_finite(item))
                if len(vals) >= cap:
                    break
        if len(vals) >= cap:
            break
    while len(vals) < cap:
        vals.append(0.0)
    return vals[:cap]


def entity_feature(entity: dict[str, Any]) -> list[float]:
    """Encode one anonymous scene entity into a fixed-width numeric vector."""
    kind = str(entity.get("kind_hint", "object"))
    uv = _vec(entity.get("centroid_uv") or entity.get("uv"), 2)
    rel = _vec(entity.get("relative"), 3)
    motion = _vec(entity.get("motion") or entity.get("flow"), 2)
    depth = entity.get("depth")
    if depth is None:
        depth = math.sqrt(sum(x * x for x in rel))
    props = _numeric_property_values(entity.get("property_evidence"), 11)
    feat = [
        1.0 if bool(entity.get("visible", True)) else 0.0,
        1.0 if bool(entity.get("occluded", False)) else 0.0,
        uv[0],
        uv[1],
        rel[0],
        rel[1],
        rel[2],
        _finite(depth),
        motion[0],
        motion[1],
        _finite(entity.get("confidence"), _finite(entity.get("presence"), 0.0)),
        _finite(entity.get("persistence"), _finite(entity.get("confidence"), 0.0)),
        _finite(entity.get("agency")),
        _finite(entity.get("looming")),
        _finite(entity.get("local_motion")),
        _finite(entity.get("retina_contrast")),
        1.0 if kind == "object" else 0.0,
        1.0 if kind == "stuff" else 0.0,
        1.0 if kind == "body_part_candidate" else 0.0,
        min(1.0, max(0.0, _finite(entity.get("occlusion_age")) / 16.0)),
        min(1.0, max(0.0, _finite(entity.get("seen_count")) / 32.0)),
        *props,
    ]
    return feat[:SCENE_DYNAMICS_FEATURE_DIM]


def entities_to_features(entities: list[dict[str, Any]]) -> torch.Tensor:
    if not entities:
        return torch.zeros(0, SCENE_DYNAMICS_FEATURE_DIM, dtype=torch.float32)
    return torch.as_tensor([entity_feature(e) for e in entities], dtype=torch.float32)


def decode_scene_prediction(
    prev_features: torch.Tensor,
    raw: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Decode residual predictions into anonymous perceptual fields."""
    prev_uv = prev_features[:, 2:4]
    prev_rel = prev_features[:, 4:7]
    prev_motion = prev_features[:, 8:10]
    uv = torch.clamp(prev_uv + 0.25 * torch.tanh(raw[:, 0:2]), 0.0, 1.0)
    rel = prev_rel + 0.75 * torch.tanh(raw[:, 2:5])
    motion = prev_motion + 0.25 * torch.tanh(raw[:, 5:7])
    visibility = torch.sigmoid(raw[:, 7:8])
    persistence = torch.sigmoid(raw[:, 8:9])
    uncertainty = F.softplus(raw[:, 9:10]) + 1e-4
    return {
        "uv": uv,
        "relative": rel,
        "motion": motion,
        "visibility": visibility,
        "persistence": persistence,
        "uncertainty": uncertainty,
    }


def prediction_rows_to_dicts(
    entity_ids: list[str],
    prev_features: torch.Tensor,
    raw: torch.Tensor,
) -> list[dict[str, Any]]:
    decoded = decode_scene_prediction(prev_features.detach(), raw.detach())
    out: list[dict[str, Any]] = []
    for i, eid in enumerate(entity_ids[: int(raw.shape[0])]):
        out.append(
            {
                "entity_id": str(eid),
                "centroid_uv": [float(x) for x in decoded["uv"][i].cpu().tolist()],
                "relative": [float(x) for x in decoded["relative"][i].cpu().tolist()],
                "motion": [float(x) for x in decoded["motion"][i].cpu().tolist()],
                "visibility": float(decoded["visibility"][i].cpu().item()),
                "persistence": float(decoded["persistence"][i].cpu().item()),
                "uncertainty": float(decoded["uncertainty"][i].cpu().item()),
            }
        )
    return out


class SceneDynamicsHead(nn.Module):
    """Small residual predictor over anonymous scene-entity features."""

    def __init__(self, *, feature_dim: int, motor_dim: int, hidden: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.motor_dim = int(motor_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.feature_dim + self.motor_dim),
            nn.Linear(self.feature_dim + self.motor_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, SCENE_DYNAMICS_OUTPUT_DIM),
        )
        with torch.no_grad():
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                final.weight.zero_()
                final.bias.zero_()

    def forward(self, features: torch.Tensor, motor: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            features = features.reshape(-1, self.feature_dim)
        if motor.ndim == 1:
            motor = motor.reshape(1, -1)
        if motor.shape[0] == 1 and features.shape[0] != 1:
            motor = motor.expand(features.shape[0], -1)
        if motor.shape[-1] < self.motor_dim:
            motor = F.pad(motor, (0, self.motor_dim - motor.shape[-1]))
        motor = motor[:, : self.motor_dim]
        return self.net(torch.cat([features, motor.to(features.dtype)], dim=-1))


def scene_dynamics_loss(
    raw: torch.Tensor,
    prev_features: torch.Tensor,
    target_features: torch.Tensor,
    match_mask: torch.Tensor,
    *,
    uncertainty_weight: float = 0.05,
) -> torch.Tensor:
    """Self-supervised loss for anonymous next-scene prediction."""
    if raw.numel() == 0 or prev_features.numel() == 0 or target_features.numel() == 0:
        return raw.new_zeros(())
    mask = match_mask.to(device=raw.device, dtype=torch.bool).reshape(-1)
    n = min(raw.shape[0], prev_features.shape[0], target_features.shape[0], mask.shape[0])
    if n <= 0 or not bool(mask[:n].any()):
        return raw.new_zeros(())
    raw = raw[:n]
    prev_features = prev_features[:n].to(device=raw.device, dtype=raw.dtype)
    target_features = target_features[:n].to(device=raw.device, dtype=raw.dtype)
    mask = mask[:n]
    pred = decode_scene_prediction(prev_features, raw)
    visible = target_features[:, 0:1].clamp(0.0, 1.0)
    pos_mask = (mask.reshape(-1, 1) & (visible > 0.5)).to(raw.dtype)
    any_pos = pos_mask.sum().clamp_min(1.0)
    uv_loss = (((pred["uv"] - target_features[:, 2:4]) ** 2) * pos_mask).sum() / any_pos
    rel_loss = (((pred["relative"] - target_features[:, 4:7]) ** 2) * pos_mask).sum() / any_pos
    mot_loss = (((pred["motion"] - target_features[:, 8:10]) ** 2) * pos_mask).sum() / any_pos
    bce = F.binary_cross_entropy(pred["visibility"], visible, reduction="none")
    vis_loss = (bce.reshape(-1)[mask]).mean()
    pers_loss = F.mse_loss(pred["persistence"][mask], target_features[:, 11:12][mask])
    realized = (pred["uv"].detach() - target_features[:, 2:4]).pow(2).mean(dim=1, keepdim=True)
    unc_loss = F.mse_loss(pred["uncertainty"][mask], realized[mask])
    return uv_loss + 0.5 * rel_loss + 0.5 * mot_loss + vis_loss + pers_loss + float(uncertainty_weight) * unc_loss
