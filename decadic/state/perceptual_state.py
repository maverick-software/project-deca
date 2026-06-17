"""Maintained perceptual synthesis from streaming observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from decadic.state.working_memory import WorkingMemory
from decadic.state.world_graph import (
    edges_from_nodes,
    egocentric_nodes_from_perception,
    egocentric_nodes_from_world_state,
    update_entity_affect,
)

if TYPE_CHECKING:
    from decadic.perception.discovery_metrics import DiscoveryEvaluator


def _new_discovery_evaluator() -> "DiscoveryEvaluator":
    # Lazy import: decadic.perception.__init__ pulls in this module, so importing
    # the evaluator at module load would create a circular import.
    from decadic.perception.discovery_metrics import DiscoveryEvaluator

    return DiscoveryEvaluator()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _audio_stats(audio: dict[str, Any]) -> tuple[float, float] | None:
    """(duration_s, rms) of a pcm16 base64 audio blob; None if undecodable."""
    import base64
    import binascii

    data = audio.get("data")
    if not isinstance(data, str) or not data.strip():
        return None
    try:
        blob = base64.b64decode(data)
    except (ValueError, binascii.Error):
        return None
    if len(blob) < 2:
        return None
    wav = np.frombuffer(blob, dtype="<i2").astype(np.float32) / 32768.0
    sr = int(audio.get("sample_rate", 16000) or 16000)
    return len(wav) / max(1, sr), float(np.sqrt(np.mean(np.square(wav))))


@dataclass
class PerceptualState:
    """Running integrator state (Phase 1: lightweight fusion stub)."""

    last_timestamp_iso: str | None = None
    vision_resolution: list[int] | None = None
    audio_duration_s: float | None = None
    audio_rms: float | None = None
    proprio_position: list[float] | None = None
    proprio_orientation: list[float] | None = None
    proprio_velocity: list[float] | None = None
    proprio_joints: list[float] | None = None
    proprio_contacts: list[float] | None = None
    current_action_observed: str | None = None
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    recent_events_cap: int = 32
    # Stub fused embedding updated each integration tick
    fused_stub_emb: np.ndarray = field(
        default_factory=lambda: np.zeros((32,), dtype=np.float32)
    )
    integration_ticks: int = 0
    egocentric_nodes: list[dict[str, Any]] = field(default_factory=list)
    egocentric_edges: list[dict[str, Any]] = field(default_factory=list)
    # Decaying per-entity survival valence (entity_id -> signed weight), feeds affective edges.
    entity_affect: dict[str, float] = field(default_factory=dict)
    # Bounded decaying store giving the graph object permanence (condition 3).
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    # "oracle" (legacy) vs "discovered" (graph emerges from perception/memory).
    perception_mode: str = "oracle"
    # Discovered mode only: world_state.entities kept as eval-only ground truth
    # (never fed to cognition) + a running discovery-quality evaluator.
    oracle_truth: list[dict[str, Any]] = field(default_factory=list)
    discovery_eval: "DiscoveryEvaluator" = field(default_factory=_new_discovery_evaluator)

    def integrate_observation(self, obs: dict[str, Any]) -> None:
        """Incorporate one validated observation dict (JSON-shaped)."""
        self.last_timestamp_iso = obs.get("timestamp") or _utc_now_iso()
        vision = obs.get("vision") or {}
        res = vision.get("resolution")
        if isinstance(res, list) and len(res) >= 2:
            self.vision_resolution = [int(res[0]), int(res[1])]
        audio = obs.get("audio")
        if isinstance(audio, dict):
            stats = _audio_stats(audio)
            if stats is not None:
                self.audio_duration_s, self.audio_rms = round(stats[0], 4), round(stats[1], 5)
        prop = obs.get("proprioception") or {}
        if "position" in prop:
            self.proprio_position = [float(x) for x in prop["position"]]
        if "orientation" in prop:
            self.proprio_orientation = [float(x) for x in prop["orientation"]]
        if "velocity" in prop:
            self.proprio_velocity = [float(x) for x in prop["velocity"]]
        if isinstance(prop.get("joints"), list):
            self.proprio_joints = [float(x) for x in prop["joints"]]
        if isinstance(prop.get("contacts"), list):
            self.proprio_contacts = [float(x) for x in prop["contacts"]]
        if "current_action" in prop:
            self.current_action_observed = str(prop["current_action"])
        events = obs.get("events") or []
        if isinstance(events, list):
            for e in events[-8:]:
                if isinstance(e, dict):
                    self.recent_events.append(e)
            while len(self.recent_events) > self.recent_events_cap:
                self.recent_events.pop(0)
        ws = obs.get("world_state")
        if self.perception_mode == "discovered":
            # The graph is owned by the neural cycle's slot-discovery pass (it
            # populates working memory from the camera). Here we only refresh the
            # self node from sensed proprioception and re-derive nodes/edges from
            # whatever working memory currently holds. world_state.entities is
            # stashed as eval-only truth and never reaches cognition.
            self.oracle_truth = self._extract_oracle_truth(ws)
            self.rebuild_discovered_graph(obs)
        else:
            update_entity_affect(self.entity_affect, events if isinstance(events, list) else [])
            observed_nodes = egocentric_nodes_from_world_state(ws)
            # Persist entities across observations so the graph isn't rebuilt from scratch.
            self.working_memory.integrate(
                observed_nodes,
                self.entity_affect,
                events=events if isinstance(events, list) else [],
            )
            self_nodes = [n for n in observed_nodes if n.get("role") == "self"]
            context_nodes = [n for n in observed_nodes if n.get("role") == "context"]
            new_nodes = self_nodes + self.working_memory.entity_nodes() + context_nodes
            new_edges = edges_from_nodes(new_nodes, affect=self.entity_affect)
            self.egocentric_nodes = new_nodes
            self.egocentric_edges = new_edges
        # Stub fusion: shift embedding based on proprio + event count + scene complexity
        noise = np.zeros_like(self.fused_stub_emb, dtype=np.float32)
        if self.proprio_position:
            noise[: min(3, len(noise))] += np.array(
                self.proprio_position[:3], dtype=np.float32
            )
        noise[-1] += float(len(self.recent_events))
        noise[min(8, len(noise) - 1)] += float(len(self.egocentric_nodes)) * 0.01
        self.fused_stub_emb = np.tanh(self.fused_stub_emb * 0.95 + 0.05 * noise)
        self.integration_ticks += 1

    def _self_node_from_proprio(self) -> dict[str, Any]:
        """The minimal self: a node the agent senses via its own proprioception."""
        node: dict[str, Any] = {"role": "self", "id": "self"}
        if self.proprio_position is not None:
            node["position"] = [float(x) for x in self.proprio_position[:3]]
        if self.proprio_orientation is not None:
            node["orientation"] = [float(x) for x in self.proprio_orientation[:3]]
        return node

    @staticmethod
    def _extract_oracle_truth(world_state: Any) -> list[dict[str, Any]]:
        """Pull world_state.entities out as eval-only ground truth (never to cognition)."""
        if not isinstance(world_state, dict):
            return []
        ents = world_state.get("entities")
        out: list[dict[str, Any]] = []
        if isinstance(ents, list):
            for raw in ents:
                if isinstance(raw, dict):
                    out.append(dict(raw))
        return out

    def rebuild_discovered_graph(self, obs: dict[str, Any] | None = None) -> None:
        """Rebuild nodes/edges from the proprioceptive self + working-memory object files.

        Entities/agency live entirely in working memory, which the neural cycle's
        slot-discovery pass populates from the camera. This re-derives the graph
        and refreshes the discovery evaluator against the stashed oracle truth.
        """
        self_node = self._self_node_from_proprio()
        new_nodes = egocentric_nodes_from_perception(
            self_node, self.working_memory.entity_nodes()
        )
        new_edges = edges_from_nodes(new_nodes, affect=self.entity_affect)
        self.egocentric_nodes = new_nodes
        self.egocentric_edges = new_edges
        body_parts = None
        if isinstance(obs, dict):
            et = obs.get("eval_truth")
            if isinstance(et, dict) and isinstance(et.get("body_parts"), dict):
                body_parts = {
                    str(k): [float(x) for x in v]
                    for k, v in et["body_parts"].items()
                    if isinstance(v, list) and len(v) >= 3
                }
        self.discovery_eval.update(
            new_nodes,
            self.oracle_truth,
            self_pos=self.proprio_position,
            body_parts_truth=body_parts,
        )

    def snapshot_dict(self) -> dict[str, Any]:
        return {
            "last_timestamp_iso": self.last_timestamp_iso,
            "vision_resolution": self.vision_resolution,
            "audio_duration_s": self.audio_duration_s,
            "audio_rms": self.audio_rms,
            "proprio_position": self.proprio_position,
            "proprio_orientation": self.proprio_orientation,
            "proprio_velocity": self.proprio_velocity,
            "proprio_joints": self.proprio_joints,
            "proprio_contacts": self.proprio_contacts,
            "current_action_observed": self.current_action_observed,
            "recent_events": list(self.recent_events),
            "fused_stub_emb": self.fused_stub_emb.tolist(),
            "integration_ticks": self.integration_ticks,
            "egocentric_nodes": list(self.egocentric_nodes),
            "egocentric_edges": list(self.egocentric_edges),
            "egocentric_graph": {
                "nodes": list(self.egocentric_nodes),
                "edges": list(self.egocentric_edges),
            },
            "working_memory": self.working_memory.snapshot(),
            "perception_mode": self.perception_mode,
            "discovery": (
                self.discovery_eval.snapshot()
                if self.perception_mode == "discovered"
                else None
            ),
        }
