"""Offline perception bootstrap scaffolds.

These types describe teacher targets for training the perceptual object-file
builder. They are intentionally not imported by the live cycle and carry no path
for semantic labels into cognition.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ObjectFileTarget:
    """A label-free teacher target for perceptual separation."""

    target_id: str
    centroid_uv: list[float]
    relative: list[float] | None = None
    mask_rle: str | None = None
    depth: float | None = None
    motion: list[float] | None = None
    kind_hint: str = "object"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerceptionBootstrapFrame:
    """One offline training frame for the object-file builder."""

    timestamp: str
    targets: list[ObjectFileTarget] = field(default_factory=list)
    source: str = "scaffold"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["targets"] = [t.to_dict() for t in self.targets]
        return data


@dataclass(frozen=True)
class PerceptionBootstrapLossWeights:
    """Offline-only loss weights for the perception organ."""

    localization: float = 1.0
    mask_diversity: float = 0.2
    temporal_consistency: float = 0.5
    background_rejection: float = 0.4
    agency_correlation: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerceptionCheckpointManifest:
    """Metadata for saved perception-organ weights/checkpoints."""

    checkpoint_id: str
    created_at: str
    source_frames: int
    loss_weights: PerceptionBootstrapLossWeights
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["loss_weights"] = self.loss_weights.to_dict()
        return data


def strip_teacher_fields(frame: PerceptionBootstrapFrame) -> list[dict[str, Any]]:
    """Return anonymous targets suitable for metric/eval code, not cognition."""
    out: list[dict[str, Any]] = []
    for t in frame.targets:
        out.append(
            {
                "target_id": t.target_id,
                "centroid_uv": list(t.centroid_uv),
                "relative": list(t.relative) if t.relative is not None else None,
                "depth": t.depth,
                "motion": list(t.motion) if t.motion is not None else None,
                "kind_hint": t.kind_hint,
            }
        )
    return out


def write_bootstrap_jsonl(path: Path, frames: list[PerceptionBootstrapFrame]) -> None:
    """Persist offline scaffold frames. This is not read by live cognition."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for frame in frames:
            f.write(json.dumps(frame.to_dict(), separators=(",", ":")) + "\n")


def read_bootstrap_jsonl(path: Path) -> list[PerceptionBootstrapFrame]:
    """Load offline scaffold frames for perception-only training/evaluation."""
    out: list[PerceptionBootstrapFrame] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            targets = [
                ObjectFileTarget(
                    target_id=str(t.get("target_id", "")),
                    centroid_uv=[float(x) for x in t.get("centroid_uv", [0.0, 0.0])[:2]],
                    relative=(
                        [float(x) for x in t["relative"][:3]]
                        if isinstance(t.get("relative"), list)
                        else None
                    ),
                    mask_rle=t.get("mask_rle") if isinstance(t.get("mask_rle"), str) else None,
                    depth=float(t["depth"]) if t.get("depth") is not None else None,
                    motion=(
                        [float(x) for x in t["motion"][:2]]
                        if isinstance(t.get("motion"), list)
                        else None
                    ),
                    kind_hint=str(t.get("kind_hint", "object")),
                )
                for t in raw.get("targets", [])
                if isinstance(t, dict)
            ]
            out.append(
                PerceptionBootstrapFrame(
                    timestamp=str(raw.get("timestamp", "")),
                    targets=targets,
                    source=str(raw.get("source", "scaffold")),
                )
            )
    return out
