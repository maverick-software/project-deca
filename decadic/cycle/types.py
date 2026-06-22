"""Typed objects passed through the Decadic Cycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decadic.memory.episodic_store import EpisodicStore
from decadic.memory.semantic_graph import LongTermGraph
from decadic.state.perceptual_state import PerceptualState
from decadic.state.state_bus import StateBus
from decadic.state.viability import Homeostasis, ViabilityState


@dataclass
class StageTrace:
    stage: int
    name: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class CycleContext:
    state_bus: StateBus
    perceptual: PerceptualState
    viability: ViabilityState
    episodic: EpisodicStore
    # Persistent, unbounded long-term relational graph (the hippocampal index).
    # Optional: None on stub/test paths and when DECADIC_LTM_GRAPH=0 (parity gate);
    # stages skip consolidation/reinstatement when it is absent.
    ltm_graph: LongTermGraph | None = None
    # Per-reservoir homeostatic state (hydration/energy/integrity). Optional so
    # stub/test paths without a body keep working; when None the interoceptive
    # drive reads as "full reservoirs" -> zero drive (parity).
    homeostasis: Homeostasis | None = None
    latents: dict[str, Any] = field(default_factory=dict)
    last_observation: dict[str, Any] | None = None
    # Up to K observations buffered since the previous cycle (parallel sessions).
    pending_observations: list[dict[str, Any]] = field(default_factory=list)
    # Runtime scheduling mode for incoming perception. The Decadic cycle remains
    # serialized; persistent mode means object/scene perception may be committed
    # continuously before the cycle samples the current state.
    perceptual_processing_mode: str = "batching_observations"
    # Manual assist-harness level: None -> follow the fading curriculum; a float
    # (0/1/2/3) -> pin the harness gain to that level for this cycle.
    assist_override: float | None = None
    # Which body support system to run: "guided" (assist-as-needed harness) or
    # "legacy" (training-wheels assist). None -> leave the body on its env default.
    curriculum_mode: str | None = None
    # "oracle" vs "discovered" perception. In discovered mode the cycle runs the
    # slot-attention object-discovery pass and owns the egocentric graph.
    perception_mode: str = "oracle"
    # Read-only observation toggles (per-agent overrides). None -> fall back to the
    # process-env default. They gate interpretability work only; never cognition.
    cognition_trace: bool | None = None
    probe_capture: bool | None = None
    # Global-workspace competition (self-model program, Phase 2). None -> fall back
    # to the process-env default (DECADIC_GWT_ENABLED). When on, the working-memory
    # EMA blend into A is replaced by winner-take-all + ignition + broadcast; when
    # off the cycle keeps the legacy EMA (byte-identical). A live pipeline branch
    # (not an architecture change) -> toggling it does NOT rebuild the brain.
    gwt_enabled: bool | None = None
    # Temporal-integration window in ms (self-model program, Phase 3). None -> env
    # default (DECADIC_INTEGRATION_WINDOW_MS). > 0 binds a span of percepts into one
    # committed "now"; 0 = off = the freshest percept is always now (byte-identical).
    integration_window_ms: float | None = None
    # Live curriculum overrides for active-inference weights. None -> use the
    # process-env default (exact parity). They reweight the EXISTING self-
    # supervised objective (drive-reduction pull, deprivation priority, motor
    # exploration); they never introduce a new loss term.
    ai_intero_pref_weight: float | None = None
    drive_priority_gain: float | None = None
    motor_babble_sigma: float | None = None
    # Runtime-precomputed episodic recall context. When provided, Stage 3 uses
    # this vector instead of querying episodic memory inside the cognitive lock.
    cached_memory_context: list[float] | None = None
    cached_memory_query: list[float] | None = None
    memory_recall_on_critical_path: bool = True
