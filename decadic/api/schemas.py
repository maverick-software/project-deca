"""Pydantic schemas for WebSocket / REST payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "noop"
    parameters: dict[str, Any] = Field(default_factory=dict)


class PredictedOutcome(BaseModel):
    model_config = ConfigDict(extra="allow")

    embedding: list[float] = Field(default_factory=list)
    expected_position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class ActionMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: str
    action: ActionPayload | dict[str, Any]
    predicted_outcome: PredictedOutcome | dict[str, Any] | None = None
    trace: list[dict[str, Any]] | None = None


class ObservationMessage(BaseModel):
    """Inbound observation (environment → server)."""

    model_config = ConfigDict(extra="allow")

    timestamp: str | None = None
    vision: dict[str, Any] | None = None
    audio: dict[str, Any] | None = None
    proprioception: dict[str, Any] | None = None
    events: list[dict[str, Any]] | None = None
    world_state: dict[str, Any] | None = None


class AgentCreateResponse(BaseModel):
    agent_id: str


class AgentStateResponse(BaseModel):
    agent_id: str
    payload: dict[str, Any]


class MemoryQueryResponse(BaseModel):
    agent_id: str
    episodes: list[dict[str, Any]]


class MemorySimilarResponse(BaseModel):
    agent_id: str
    matches: list[dict[str, Any]]


class MetricsResponse(BaseModel):
    agent_id: str
    metrics: dict[str, Any]


class CheckpointResponse(BaseModel):
    agent_id: str
    path: str
    neural_brain: str | None = None


class SaveAgentRequest(BaseModel):
    """Body for POST /agent/{id}/save (Saved Agents library)."""

    name: str = Field(..., min_length=1, max_length=120)
    notes: str | None = Field(default=None, max_length=2000)


class SavedAgentRecord(BaseModel):
    """A single saved-agent manifest, as surfaced to the dashboard."""

    model_config = ConfigDict(extra="allow")

    save_id: str
    name: str
    created_at: str
    source_agent_id: str | None = None
    preset: str | None = None
    encoder_mode: str | None = None
    viability_mode: str | None = None
    viability: float | None = None
    cycle_index: int | None = None
    has_memory: bool = True
    notes: str | None = None
    schema_version: int | None = None


class SavedAgentListResponse(BaseModel):
    saves: list[SavedAgentRecord] = Field(default_factory=list)


class LoadSavedResponse(BaseModel):
    agent_id: str
    save_id: str


class CreateAgentPresetRequest(BaseModel):
    """Body for POST /agent-presets (a named scenario/body config)."""

    name: str = Field(..., min_length=1, max_length=120)
    elements: list[str] = Field(default_factory=list)
    vision: bool = True
    audio: bool = False
    # Whether the manual joint-brace scaffold starts engaged.
    braces: bool = False
    # A disembodied mind: no body/world is spawned for this preset.
    mind_only: bool = False


class AgentPresetRecord(BaseModel):
    """A single agent preset, as surfaced to the dashboard dropdown."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    elements: list[str] = Field(default_factory=list)
    vision: bool = True
    audio: bool = False
    braces: bool = False
    mind_only: bool = False
    builtin: bool = False
    created_at: str | None = None


class AgentPresetListResponse(BaseModel):
    presets: list[AgentPresetRecord] = Field(default_factory=list)
