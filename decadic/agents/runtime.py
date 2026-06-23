"""Per-agent runtime: shared state, queues, and background cycle worker."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decadic import config as C
from decadic.config import (
    DEFAULT_CYCLE_INTERVAL_S,
    DEFAULT_PARALLEL_SESSIONS,
    DEFAULT_REVIVE_VIABILITY,
    FAST_PATH_COLLISION_THRESHOLD,
    MAX_PARALLEL_SESSIONS,
    PROCESSING_BATCHING,
    PROCESSING_PERSISTENT_PERCEPTION,
    PROCESSING_SERIAL_PREFETCH,
    PROCESSING_STAGE_PIPELINE,
    PERCEPTUAL_PROCESSING_BATCHING,
    PERCEPTUAL_PROCESSING_MODES,
    PERCEPTUAL_PROCESSING_PERSISTENT,
    cognition_history_len,
    cognition_trace_enabled,
    consolidation_enabled,
    cycle_profile_enabled,
    episodic_async_enabled,
    consolidation_prune_min_salience,
    damage_grace_floor,
    energy_empty_s,
    effort_drain_enabled,
    effort_energy_scale,
    effort_max_energy_drain_per_obs,
    food_credit,
    fatigue_pain_gain,
    goal_abandon_cycles,
    goal_max_cycles,
    goal_onset_deficit,
    goal_satisfy_level,
    gwt_enabled,
    her_enabled,
    integration_window_ms,
    her_relabel_k,
    heal_min_reserve,
    hydration_empty_s,
    integrity_heal_full_s,
    landscape_batch,
    landscape_enabled,
    landscape_grid,
    landscape_interval_s,
    landscape_seed,
    landscape_span,
    ltm_async_enabled,
    ltm_consolidation_async_enabled,
    ltm_consolidation_queue_max,
    ltm_graph_enabled,
    ltm_match_threshold,
    ltm_snapshot_limit,
    max_integrity_damage_per_obs,
    metabolic_compression,
    processing_mode,
    perceptual_processing_mode,
    metabolic_tick_s,
    plasticity_log_every,
    probe_capture_enabled,
    randomize_resources_enabled,
    replay_buffer_size,
    scene_dynamics_enabled,
    sf_gamma,
    sf_lambda,
    stress_gain,
    strain_pain_gain,
    tombstone_keep,
    viability_mode_default,
    water_credit,
    work_energy_scale,
)
from decadic.consolidation.consolidator import ConsolidationManager
from decadic.consolidation.landscape import LossLandscapeProbe
from decadic.consolidation.episodes import (
    EpisodeAccumulator,
    achieved_feature,
    build_hindsight_copies,
)
from decadic.consolidation.replay_buffer import ReplayBuffer, Transition
from decadic.consolidation.stub_loop import consolidation_stub_loop
from decadic.cycle.neural_pipeline import run_neural_cycle
from decadic.cycle.pipeline import run_cycle as run_stub_cycle
from decadic.cycle.stage_pipeline import DecadicSession, SerialPrefetchSupervisor
from decadic.cycle.types import CycleContext
from decadic.io import get_jsonl_writer
from decadic.memory.episodic_store import EpisodicStore
from decadic.memory.ltm_write_behind import WriteBehindLongTermGraph
from decadic.memory.semantic_graph import LongTermGraph
from decadic.memory.write_behind import WriteBehindEpisodicStore
from decadic.nn.bundle import NeuralBundle
from decadic.nn.faculties import CognitionFaculties
from decadic.nn.plastic import PlasticityFlags
from decadic.perception.integration import PerceptualIntegrator
from decadic.perception.object_files import evaluate_discovery_health, object_files_from_proposals
from decadic.perception.organ import PerceptionOrgan
from decadic.state.goal_lifecycle import GoalState
from decadic.state.perceptual_state import PerceptualState
from decadic.state.state_bus import StateBus
from decadic.state.body_map import most_pained_part, normalize_body_map, normalize_effort
from decadic.state.viability import (
    Homeostasis,
    ViabilityState,
    apply_pain_pleasure_to_B,
    classify_events,
    ema_affect,
    passive_metabolism,
    viability_delta_to_signals,
)

logger = logging.getLogger(__name__)

# Rolling window (in observations) for locomotion/gait telemetry. Eval-only.
LOCO_WINDOW = 64
# A foot is "in contact" for the gait phase when its load exceeds this (sim
# force units); the margin avoids flicker when both feet share weight.
LOCO_FOOT_CONTACT_N = 5.0


def _gait_regularity(phases: list[int]) -> float:
    """Left/right contact-alternation score in [0, 1] from a foot-phase sequence.

    ``phases`` is a sequence of +1 (left foot leading), -1 (right foot leading),
    or 0 (double support / airborne). A steady walk alternates the leading foot,
    so among consecutive single-support frames the fraction that *flip* sign is
    the regularity. Standing still (all 0) or dragging one foot scores ~0; a
    clean alternating gait scores ~1. Purely observational.
    """
    signed = [p for p in phases if p != 0]
    if len(signed) < 2:
        return 0.0
    flips = sum(1 for a, b in zip(signed, signed[1:]) if a != b)
    return float(flips) / float(len(signed) - 1)


def _compact_cognitive(trace: dict[str, Any]) -> dict[str, Any]:
    """Tiny per-cycle summary kept in the temporal ring buffer (sparklines)."""
    intent = trace.get("intent") or {}
    drivers = intent.get("drivers") or []
    top = drivers[0] if drivers else {}
    affect = trace.get("affect") or {}
    surprise = trace.get("self_surprise") or {}
    return {
        "cycle": trace.get("cycle"),
        "intent": intent.get("summary", ""),
        "top_goal": top.get("goal"),
        "pain": affect.get("pain"),
        "pleasure": affect.get("pleasure"),
        "risk": affect.get("risk"),
        "surprise": surprise.get("mean_abs_residual"),
    }


class AgentRuntime:
    """Holds cognitive state and orchestrates the asynchronous cycle loop."""

    def __init__(
        self,
        agent_id: str,
        *,
        cycle_interval_s: float | None = None,
        episodic_db_path: Path | None = None,
        graph_db_path: Path | None = None,
        preset: str | None = None,
        flags: PlasticityFlags | None = None,
        faculties: CognitionFaculties | None = None,
    ) -> None:
        self.agent_id = agent_id
        if cycle_interval_s is None:
            cycle_interval_s = float(
                os.environ.get("DECADIC_CYCLE_INTERVAL_S", str(DEFAULT_CYCLE_INTERVAL_S))
            )
        self.cycle_interval_s = cycle_interval_s
        self.lock = asyncio.Lock()
        self.state_bus = StateBus()
        # Core cognitive faculties this agent was built with (perception-feedback
        # loop, perception mode, encoder mode). Resolved to a concrete object so
        # reset()/configure() rebuild to the agent's faculties rather than
        # silently re-reading the process env.
        self.faculties = faculties if faculties is not None else CognitionFaculties.from_env()
        # Perception mode: "oracle" (graph handed by the sim) vs "discovered"
        # (graph emerges from the agent's own camera + proprioception + memory).
        self.perception_mode = self.faculties.perception_mode
        self.perceptual = PerceptualState(perception_mode=self.perception_mode)
        self.perceptual_integrator = PerceptualIntegrator()
        self.viability = ViabilityState()
        # Episodic store is always the write-behind wrapper so async persistence can
        # be toggled live per-agent from the dashboard (Agent Settings). When async is
        # OFF it is byte-identical to a bare EpisodicStore and spawns no worker thread;
        # DECADIC_EPISODIC_ASYNC sets the birth default (ON in production, OFF in tests).
        self.episodic: EpisodicStore = WriteBehindEpisodicStore(
            episodic_db_path, enabled=episodic_async_enabled()
        )
        # Long-term knowledge graph (the hippocampal index): persistent, unbounded
        # relational memory that the bounded working memory consolidates into and
        # reinstates from. ON by default; None only when explicitly disabled
        # (DECADIC_LTM_GRAPH=0) so the oracle/no-LTM path stays byte-identical.
        # Always the write-behind wrapper (like episodic) so async consolidation can
        # be toggled live per-agent; when async is OFF it is byte-identical to a bare
        # LongTermGraph and spawns no worker. DECADIC_LTM_ASYNC sets the birth default.
        self.ltm_graph: LongTermGraph | None = (
            WriteBehindLongTermGraph(
                graph_db_path,
                match_threshold=ltm_match_threshold(),
                max_queue=ltm_consolidation_queue_max(),
                enabled=ltm_async_enabled() and ltm_consolidation_async_enabled(),
            )
            if ltm_graph_enabled()
            else None
        )
        # Interpretability/observation toggles (read-only; never feed cognition).
        # Seeded from the env so the UI shows the right initial state; live-settable
        # per agent via configure() and threaded into the cycle through CycleContext.
        self.cognition_trace = cognition_trace_enabled()
        self.probe_capture = probe_capture_enabled()
        # Global-workspace competition (self-model program, Phase 2). Live per-agent
        # toggle threaded into the cycle through CycleContext; off => the legacy EMA.
        self.gwt_enabled = gwt_enabled()
        # Temporal-integration window in ms (self-model program, Phase 3). Live
        # per-agent setting; 0 => the freshest percept is always "now".
        self.integration_window_ms = integration_window_ms()
        # Neuroplasticity flags this agent was built with (None -> env defaults).
        # Stored so reset() preserves the agent's plasticity config rather than
        # silently re-reading the process env.
        self.plastic_flags = flags
        self.neural: NeuralBundle | None = NeuralBundle.try_build(
            agent_id, preset=preset, flags=flags, faculties=self.faculties
        )
        self.preset: str | None = self.neural.preset if self.neural else None
        self._memory_context_vector: list[float] | None = None
        self._memory_context_query: list[float] | None = None
        self._memory_context_refresh_cycle = 0
        self._memory_context_worker_ms = 0.0
        self._memory_context_task: asyncio.Task[None] | None = None
        self._cycle_deadline_s = time.perf_counter()
        self._last_observation: dict[str, Any] | None = None
        self._wait_for_observation_after_reset = False
        self._debug_views: dict[str, str] = {}
        self._last_cycle_trace: dict[str, Any] | None = None
        # Cognitive trace ("why" monitoring): the latest structured explanation
        # plus a bounded ring buffer of compact per-cycle summaries for the
        # temporal view. Read-only observation; never feeds cognition.
        self._last_cognitive_trace: dict[str, Any] | None = None
        self._cognitive_history: deque[dict[str, Any]] = deque(maxlen=cognition_history_len())
        # Edge-trigger for the curiosity event log: tracks whether the agent is
        # currently in the curiosity-driven "investigate" priority, so we log only
        # the enter/leave transitions (never per-cycle). False when curiosity is off.
        self._curiosity_investigating = False
        self.out_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self.control_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
        self.running = True
        self.paused = False
        # Mortality lifecycle: "alive" -> "dead" (viability<=0) -> "alive" (revive/reset).
        self.status = "alive"
        self.died_at_cycle: int | None = None
        k = int(os.environ.get("DECADIC_PARALLEL_SESSIONS", str(DEFAULT_PARALLEL_SESSIONS)))
        self.parallel_sessions = max(1, min(MAX_PARALLEL_SESSIONS, k))
        self.processing_mode = processing_mode()
        self.perceptual_processing_mode = self.processing_mode
        # Manual assist-harness override: None -> follow the fading curriculum;
        # a float (0/1/2/3) -> hold that level regardless of training progress.
        # Default 0 -> no training-wheel assist unless the operator opts in.
        self.assist_override: float | None = 0.0
        # Which body support system runs: "guided" (assist-as-needed harness) or
        # "legacy" (training-wheels assist). Seeded from the launch env so the UI
        # shows the right initial state; the dashboard can switch it per-agent.
        _cm = os.environ.get("DECADIC_CURRICULUM_MODE", "legacy").strip().lower()
        self.curriculum_mode: str = "guided" if _cm == "guided" else "legacy"
        # Homeostasis: viability is the running minimum of three human-like
        # reservoirs. "immortal" mode pins them at full and disables death.
        self.viability_mode = viability_mode_default()
        self.homeostasis = Homeostasis()
        self.metabolic_compression = metabolic_compression()
        # Live curriculum knobs: per-agent overrides for active-inference weights
        # the cognitive cycle otherwise reads from the process env. None -> follow
        # the env default (exact parity); a value is forwarded through CycleContext
        # so the curriculum can retune drive/exploration without a restart. These
        # only reweight the EXISTING self-supervised objective; they never add a
        # new term to the loss.
        self.ai_intero_pref_weight_override: float | None = None
        self.drive_priority_gain_override: float | None = None
        self.motor_babble_sigma_override: float | None = None
        self.stress = 0.0
        self._threat_stress = 0.0
        self._last_metab_monotonic: float | None = None
        self._metabolic_task: asyncio.Task[None] | None = None
        # Legacy batching buffer. Persistent Parallel Perceptual Processing uses
        # the queue below; batching mode drains this buffer once per cycle.
        self._obs_buffer: deque[dict[str, Any]] = deque(maxlen=self.parallel_sessions)
        self.prefetch_queue_max = C.prefetch_queue_max_frames(self.parallel_sessions)
        self.prefetch_overload_policy = C.prefetch_overload_policy()
        self.ready_coalesce_policy = C.ready_coalesce_policy()
        self.prefetch_backpressure_warn_ms = C.prefetch_backpressure_warn_ms()
        self.prefetch_oldest_unfolded_warn_ms = C.prefetch_oldest_unfolded_warn_ms()
        self._perception_queue: asyncio.Queue[
            tuple[int, dict[str, Any], float, dict[str, Any] | None]
        ] = asyncio.Queue(maxsize=self.prefetch_queue_max)
        self._perception_workers: list[asyncio.Task[None]] = []
        self._perception_ready: dict[
            int, tuple[dict[str, Any] | None, float, float, dict[str, Any] | None]
        ] = {}
        self._perception_commit_lock = asyncio.Lock()
        self._perception_seq = 0
        self._perception_next_commit = 1
        self._perception_inflight = 0
        self._perception_committed = 0
        self._perception_dropped = 0
        self._perception_ingested = 0
        self._perception_started_at = time.perf_counter()
        self._perception_last_commit_s: float | None = None
        self._runtime_perception_organ = PerceptionOrgan()
        self.stage_pipeline = SerialPrefetchSupervisor(
            capacity=self.parallel_sessions,
            coalesce_policy=self.ready_coalesce_policy,
        )
        self._state_bus_version = 0
        self._scene_version = 0
        self._workspace_version = 0
        self._weight_version = 0
        self._commit_index = 0
        # Read-only locomotion / gait telemetry (curriculum evaluation only; NEVER
        # read by cognition). Tracks how far the body has travelled, its net
        # displacement from the run origin, a rolling fall-rate, and a gait
        # regularity score derived from left/right foot-contact alternation.
        self._loco_last_xy: tuple[float, float] | None = None
        self._loco_origin_xy: tuple[float, float] | None = None
        self._loco_distance: float = 0.0
        self._loco_fall_buf: deque[int] = deque(maxlen=LOCO_WINDOW)
        self._loco_foot_phase_buf: deque[int] = deque(maxlen=LOCO_WINDOW)
        self._started_perf = time.perf_counter()
        self._cycle_task: asyncio.Task[None] | None = None
        self._consolidation_task: asyncio.Task[None] | None = None
        # Dual-network consolidation (Option B): a salience-prioritized replay
        # buffer fed by the live cycle plus a cloned consolidator that replays it
        # and Polyak-syncs back. Both stay null/no-op unless the feature flag is on,
        # so the baseline is byte-identical.
        self.replay_buffer: ReplayBuffer | None = (
            ReplayBuffer(replay_buffer_size(), min_salience=consolidation_prune_min_salience())
            if (consolidation_enabled() and self.neural is not None)
            else None
        )
        # Explicit goal lifecycle: latches the dominant homeostatic deficit as the
        # active goal so credit assignment has crisp episode boundaries. Always
        # constructed (telemetry-only until the episode/return pathway grows it).
        self.goal_state = GoalState(
            onset_deficit=goal_onset_deficit(),
            satisfy_level=goal_satisfy_level(),
            abandon_cycles=goal_abandon_cycles(),
            max_cycles=goal_max_cycles(),
        )
        # Ordered goal-episode accumulator: collects the live transitions of each
        # open goal and writes lambda-returns back into them (in the replay buffer)
        # when it closes -- the distal credit-assignment timeline.
        self._episode_acc = EpisodeAccumulator(gamma=sf_gamma(), lam=sf_lambda())
        self._consolidator: ConsolidationManager | None = None
        # Live loss-landscape probe (visualization only): a flagged background task
        # evaluates a filter-normalized 2D slice of the live objective on its own
        # throwaway clone and caches the surface for the dashboard. Null/no-op unless
        # the flag is on; it never touches the live weights, so the baseline is
        # byte-identical.
        self._landscape_task: asyncio.Task[None] | None = None
        self._landscape_probe: LossLandscapeProbe | None = None
        self._last_landscape: dict[str, Any] | None = None
        self.metrics: dict[str, Any] = self._initial_metrics()
        # Skill Dojo metadata is written by decadic.training.supervisor and read
        # only when packaging replay transitions. It never changes live action
        # selection or live-cycle losses.
        self.dojo_training: dict[str, Any] | None = None
        self.metrics["perception_mode"] = self.perception_mode
        self.metrics["discovered_perception"] = self.faculties.discovered
        self._refresh_hardware_metrics()
        self._refresh_perception_pipeline_metrics()
        self._refresh_homeostasis_metrics()
        self._refresh_plasticity_metrics()

    @staticmethod
    def _initial_metrics() -> dict[str, Any]:
        return {
            "cycles_completed": 0,
            "last_cycle_wall_ms": 0.0,
            "last_observation_iso": None,
            "fast_path_hits": 0,
            "prediction_error_last": 0.0,
            "prediction_error_ema": 0.0,
            "drive_reward_last": 0.0,
            # Legacy metric aliases (identical values; kept for back-compat consumers).
            "prediction_error_stub_last": 0.0,
            "prediction_error_stub_ema": 0.0,
            "reward_stub_last": 0.0,
            "last_stage_timing_ms_total": 0.0,
            "approx_cycles_per_sec": 0.0,
            "consolidation_stub_ticks": 0,
            # Dual-network consolidation telemetry (0 unless the feature is enabled).
            # replay_count: consolidator gradient steps; consolidator_loss: last
            # replay loss; last_sync_cycle: live cycle index of the last soft-sync;
            # replay_buffer_size: current transitions held.
            "replay_count": 0,
            "consolidator_loss": 0.0,
            "last_sync_cycle": 0,
            "consolidation_sync_delta_mean": 0.0,
            "consolidation_sync_delta_max": 0.0,
            "consolidation_sync_moved_params": 0,
            "consolidation_sync_reset_params": 0,
            "replay_buffer_size": 0,
            "neural_pc_loss_last": 0.0,
            "loss_total": 0.0,
            "loss_dominant_term": "",
            "loss_dominant_fraction": 0.0,
            "loss_terms": {},
            "loss_canary_state": "warming",
            "loss_canary_reason": "",
            "loss_canary_pressure": 0.0,
            "loss_canary_optimizer_action": "normal",
            "loss_canary_step_scale": 1.0,
            "loss_canary_ema": None,
            "loss_canary_pc_ema": None,
            "loss_canary_slope_ema": 0.0,
            "loss_canary_pc_slope_ema": 0.0,
            "loss_canary_jump_ratio": 1.0,
            "drive_priority_gain_configured": 0.0,
            "drive_priority_gain_effective": 0.0,
            "learning_rate": 0.0,
            "gpu_memory_max_allocated": 0,
            "hardware_cuda_available": False,
            "hardware_cuda_device": "",
            "hardware_torch_version": "",
            "hardware_python_executable": "",
            "neural_device": "none",
            "cuda_required": C.require_cuda(),
            "cuda_warning": "",
            "parallel_sessions": 0,
            "processing_mode": PROCESSING_SERIAL_PREFETCH,
            "stage_pipeline_enabled": True,
            "perceptual_processing_mode": PROCESSING_SERIAL_PREFETCH,
            "pipeline_sessions": 0,
            "stage_pipeline_active_sessions": 0,
            "stage_pipeline_ready_sessions": 0,
            "stage_pipeline_committed_sessions": 0,
            "stage_pipeline_committed_per_s": 0.0,
            "stage_pipeline_dropped_sessions": 0,
            "stage_pipeline_stale_sessions": 0,
            "stage_pipeline_failed_sessions": 0,
            "stage_pipeline_queue_depths": {},
            "stage_pipeline_inflight": {},
            "stage_pipeline_latency_ms": {},
            "stage_pipeline_recent_sessions": [],
            "stage_pipeline_selected_session": None,
            "stage_pipeline_arbitration_reason": "",
            "frames_received": 0,
            "frames_prefetched": 0,
            "frames_folded": 0,
            "frames_deep_processed": 0,
            "coalesced_sessions": 0,
            "information_loss": 0,
            "producer_overlap_ratio": 0.0,
            "decode_on_consume_ms": 0.0,
            "consume_wait_ms": 0.0,
            "ready_queue_depth": 0,
            "ready_coalesce_policy": "freshest",
            "fold_lag_ms": 0.0,
            "memory_recall_ms": 0.0,
            "memory_recall_worker_ms": 0.0,
            "memory_recall_cache_size": 0,
            "memory_recall_cache_hits": 0,
            "memory_recall_cache_misses": 0,
            "memory_recall_refresh_cycle": 0,
            "memory_recall_staleness_cycles": 0,
            "memory_recall_on_critical_path": False,
            "api_snapshot_cache_enabled": C.api_snapshot_cache_enabled(),
            "metrics_payload_lightweight": C.metrics_lightweight_enabled(),
            "metrics_snapshot_age_ms": 0.0,
            "cycle_scheduler_mode": C.cycle_scheduler_mode(),
            "cycle_interval_ms": 0.0,
            "cycle_idle_ms": 0.0,
            "cycle_overrun_ms": 0.0,
            "cycle_compute_ratio": 0.0,
            "prefetch_queue_depth": 0,
            "prefetch_queue_max": 0,
            "prefetch_overload_policy": "block",
            "prefetch_backpressure_events": 0,
            "prefetch_backpressure_ms": 0.0,
            "oldest_unfolded_age_ms": 0.0,
            "prefetch_backpressure_warning": False,
            "oldest_unfolded_warning": False,
            "perception_queue_depth": 0,
            "perception_inflight": 0,
            "perception_ingest_hz": 0.0,
            "perception_commit_hz": 0.0,
            "frames_committed": 0,
            "frames_dropped": 0,
            "commit_lag_ms": 0.0,
            "sample_age_ms": 0.0,
            "batching_fallback": False,
            "working_memory_slots": 0,
            "encode_phase_ms": 0.0,
            "forward_model_error": 0.0,
            # Tactile world-model prediction error (full-body touch active inference).
            "tactile_pred_error": 0.0,
            "assist_gain": 0.0,
            "assist_override": 0.0,
            # Joint-brace guidance telemetry reported up from the body. rom_mean:
            # mean per-joint range of motion earned (0 welded -> 1 free).
            # brace_engaged: mean brace tightness (1 fully welded -> 0 free).
            # joint_rom: per-hinge ROM fraction for the dashboard bars.
            "rom_mean": 0.0,
            "brace_engaged": 0.0,
            "joint_rom": [],
            "braces_enabled": False,
            # Active joint-brace stance/posture and (for motion stances) its phase.
            "stance": "stand",
            "stance_phase": 0.0,
            # Hold mode: keep the active movement welded + looping until disabled.
            "movement_hold": False,
            "foot_load_l": 0.0,
            "foot_load_r": 0.0,
            "hand_load_l": 0.0,
            "hand_load_r": 0.0,
            # Full-body touch: per-part contact loads (short name -> force/weight),
            # live in all modes for the dashboard contact map.
            "part_loads": {},
            "body_map": {},
            "effort": {},
            "effort_total": 0.0,
            "work_total": 0.0,
            "strain_total": 0.0,
            "fatigue_total": 0.0,
            "pain_total": 0.0,
            "support_effort": 0.0,
            "effort_energy_delta": 0.0,
            "fatigue_pain": 0.0,
            "strain_pain": 0.0,
            "most_pained_part": "",
            "net_energy_return": 0.0,
            "effort_pred_error": 0.0,
            "resource_relief_events": 0,
            "ltm_property_beliefs": 0,
            "ltm_avg_property_confidence": 0.0,
            "ltm_consolidation_queue_depth": 0,
            "ltm_consolidation_worker_ms": 0.0,
            "ltm_consolidation_jobs_enqueued": 0,
            "ltm_consolidation_jobs_completed": 0,
            "ltm_consolidation_sync_fallbacks": 0,
            "ltm_match_ms": 0.0,
            "ltm_match_cache_size": 0,
            "ltm_match_cache_hits": 0,
            "ltm_match_cache_misses": 0,
            "ltm_semantic_jobs_skipped_by_interval": 0,
            "sqlite_commit_count": 0,
            "sqlite_batch_commit_count": 0,
            "sqlite_last_commit_ms": 0.0,
            "sqlite_wal_checkpoint_count": 0,
            "episodic_write_batch_size_last": 0,
            "episodic_db_rows": 0,
            "episodic_db_pruned_rows": 0,
            "ltm_write_batch_size_last": 0,
            "ltm_pruned_nodes": 0,
            "ltm_pruned_edges": 0,
            "ltm_pruned_semantic_records": 0,
            "memory_db_bytes": 0,
            "memory_wal_bytes": 0,
            "motor_babble_sigma": 0.0,
            "motor_activity_rms": 0.0,
            "motor_command": [],
            # Need-gated curiosity (autonomous epistemic drive; 0 unless enabled).
            # curiosity_drive: gated epistemic drive [0,1] added to the babble gate.
            # curiosity_pleasure: pleasure-side affect folded into element B.
            # curiosity_learning_progress: recent fall in forward-model error [0,1].
            "curiosity_drive": 0.0,
            "curiosity_pleasure": 0.0,
            "curiosity_learning_progress": 0.0,
            # Successor-features value (Layer-2): sf_value is the predicted value of
            # the chosen action; sf_value_weight is the active (ramped) shaping
            # weight. Both 0 until the SF head learns and the ramp opens.
            "sf_value": 0.0,
            "sf_value_weight": 0.0,
            # Locomotion / gait telemetry (eval-only; never read by cognition).
            # distance_traveled: cumulative XY path length (m). net_displacement:
            # straight-line distance from the run origin (m). fall_rate: fraction
            # of the last LOCO_WINDOW observations carrying a fall/collision.
            # gait_regularity: left/right foot-contact alternation score [0,1].
            "distance_traveled": 0.0,
            "net_displacement": 0.0,
            "fall_rate": 0.0,
            "gait_regularity": 0.0,
            # Cumulative count of observations carrying a food/water consumption
            # (the act->relief contingency firing). Eval-only; the curriculum reads
            # its rate-of-change as a foraging-success gate.
            "consume_events": 0,
            # Anti-camping: RNG seed of the current life's resource scatter (-1 if
            # randomization is disabled or no scatter has happened yet).
            "resource_seed": -1,
            # Goal lifecycle (explicit latched intent for credit assignment).
            # goal: active goal label or "none"; goal_status: idle|active;
            # goal_dwell: cycles the current goal has been open; goal_episodes:
            # total closed episodes this life; goal_last_outcome: last close reason.
            "goal": "none",
            "goal_status": "idle",
            "goal_dwell": 0,
            "goal_episodes": 0,
            "goal_last_outcome": "",
            # Episodic replay timeline (closed goal episodes annotated with returns).
            # episodes_closed: total annotated episodes; episode_last_len: steps in
            # the last episode; episode_last_return: its lambda-return at onset.
            "episodes_closed": 0,
            "episode_last_len": 0,
            "episode_last_return": 0.0,
            # Hindsight relabeling (HER): cumulative relabeled transitions pushed and
            # the count from the most recent failed-but-relief episode.
            "her_relabels": 0,
            "her_last": 0,
            # NaN firewall: cumulative count of cycles where a non-finite forward
            # pass / recurrent state was detected, recovered, and the update skipped.
            "nan_recovery_events": 0,
            "nan_recovery_last": False,
            "hydration": 100.0,
            "energy": 100.0,
            "integrity": 100.0,
            "stress": 0.0,
            "viability_mode": "metabolic",
            "time_to_death_s": None,
            # Neuroplasticity (A/B/C) telemetry; defaults reflect a non-plastic stack.
            "plasticity_enabled": False,
            "sparse_enabled": False,
            "growth_enabled": False,
            "plasticity_alpha": 0.0,
            "plasticity_alpha_configured": 0.0,
            "plasticity_alpha_effective": 0.0,
            "plasticity_guardian_state": "inactive",
            "plasticity_guardian_action": "none",
            "plasticity_pc_ema": None,
            "plasticity_pc_slope_ema": 0.0,
            "plasticity_overlay_ratio_mean": 0.0,
            "plasticity_overlay_ratio_max": 0.0,
            "plasticity_freeze_count": 0,
            "plasticity_thaw_count": 0,
            "plasticity_warmup_blocked_reason": "",
            "sparse_density": 1.0,
            "awake_neurons": 0,
            "allocated_neurons": 0,
            "active_connections": 0,
            "max_neurons": 0,
            "rewire_events": 0,
            "growth_events": 0,
            "plasticity_frozen": False,
            # Discovered-perception telemetry (None/0 in oracle mode).
            "perception_mode": "oracle",
            "discovered_perception": False,
            "slots_present": 0,
            "slot_recon_error": 0.0,
            "discovered_objects": 0,
            "self_parts": 0,
            "agency_mean": 0.0,
            "agency_loss": 0.0,
            # Skill Dojo telemetry (consolidation-only teacher hints).
            "teacher_override_fraction": 0.0,
            "teacher_live_assist": 0.0,
            "teacher_motor_agreement": 1.0,
            "teacher_support_active": False,
            "teacher_support_force": 0.0,
            "teacher_support_torque": 0.0,
            "teacher_drop_m": 0.0,
            "teacher_target_drop_m": 0.25,
            "teacher_height_error_m": 0.0,
            "teacher_vertical_velocity": 0.0,
            "teacher_support_mode": "off",
            "root_height": 0.0,
            "torso_tilt": 0.0,
        }

    def ensure_cycle_worker(self) -> None:
        """Start continuous cognitive + consolidation stub loops (Phase 1 plan layout)."""
        if self._cycle_task is None or self._cycle_task.done():
            self._cycle_task = asyncio.create_task(
                self._cycle_loop(), name=f"decadic-cycle-{self.agent_id}"
            )
        if self._stage_pipeline_enabled():
            self.stage_pipeline.start()
        self._ensure_perception_workers()
        interval = float(os.environ.get("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "10"))
        # The real dual-network consolidator starts whenever the feature is on
        # (regardless of the stub interval); otherwise the no-op heartbeat runs only
        # when its interval is positive.
        consolidation_on = consolidation_enabled() and self.replay_buffer is not None
        if (consolidation_on or interval > 0) and (
            self._consolidation_task is None or self._consolidation_task.done()
        ):
            self._consolidation_task = asyncio.create_task(
                self._consolidation_runner(),
                name=f"decadic-consolidation-{self.agent_id}",
            )
        if metabolic_tick_s() > 0 and (
            self._metabolic_task is None or self._metabolic_task.done()
        ):
            self._metabolic_task = asyncio.create_task(
                self._metabolic_loop(),
                name=f"decadic-metabolic-{self.agent_id}",
            )
        # Loss-landscape probe (visualization only): only when enabled AND there is a
        # neural stack + replay buffer to score against. Zero cost when the flag is off.
        if (
            landscape_enabled()
            and self.neural is not None
            and self.replay_buffer is not None
            and (self._landscape_task is None or self._landscape_task.done())
        ):
            self._landscape_task = asyncio.create_task(
                self._landscape_runner(),
                name=f"decadic-landscape-{self.agent_id}",
            )

    async def suspend_cycle_worker(self) -> bool:
        """Temporarily stop the cycle task for reset/restore barriers."""
        restart_cycle = self._cycle_task is not None
        current = asyncio.current_task()
        if self._cycle_task is not None and self._cycle_task is not current:
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                pass
            self._cycle_task = None
        return restart_cycle

    def resume_cycle_worker(self, restart_cycle: bool) -> None:
        if restart_cycle and self.running:
            self.ensure_cycle_worker()

    def _perception_pipeline_enabled(self) -> bool:
        return self.processing_mode == PROCESSING_PERSISTENT_PERCEPTION

    def _stage_pipeline_enabled(self) -> bool:
        return self.processing_mode in (PROCESSING_SERIAL_PREFETCH, PROCESSING_STAGE_PIPELINE)

    def _serial_prefetch_enabled(self) -> bool:
        return self.processing_mode in (PROCESSING_SERIAL_PREFETCH, PROCESSING_STAGE_PIPELINE)

    def _ensure_perception_workers(self) -> None:
        if not (self._perception_pipeline_enabled() or self._serial_prefetch_enabled()):
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        alive = [t for t in self._perception_workers if not t.done()]
        target = max(1, min(MAX_PARALLEL_SESSIONS, int(self.parallel_sessions)))
        self._perception_workers = alive[:target]
        for idx in range(len(self._perception_workers), target):
            self._perception_workers.append(
                asyncio.create_task(
                    self._perception_worker_loop(idx),
                    name=f"decadic-perception-{self.agent_id}-{idx}",
                )
            )

    def _clear_perception_pipeline(self) -> None:
        while True:
            try:
                self._perception_queue.get_nowait()
                self._perception_queue.task_done()
            except asyncio.QueueEmpty:
                break
        self._perception_ready.clear()
        self._perception_seq = 0
        self._perception_next_commit = 1
        self._perception_inflight = 0
        self._perception_last_commit_s = None

    def _refresh_prefetch_config(self) -> None:
        self.prefetch_queue_max = C.prefetch_queue_max_frames(self.parallel_sessions)
        self.prefetch_overload_policy = C.prefetch_overload_policy()
        self.ready_coalesce_policy = C.ready_coalesce_policy()
        self.prefetch_backpressure_warn_ms = C.prefetch_backpressure_warn_ms()
        self.prefetch_oldest_unfolded_warn_ms = C.prefetch_oldest_unfolded_warn_ms()
        self.stage_pipeline.set_coalesce_policy(self.ready_coalesce_policy)

    def _rebuild_perception_queue_if_needed(self) -> None:
        self._refresh_prefetch_config()
        if self._perception_queue.maxsize == self.prefetch_queue_max:
            return
        # Keep the queue object stable: perception workers may be awaiting
        # ``get()`` on it. Resizing the bounded queue in place avoids stranding
        # those tasks on an abandoned queue instance.
        self._perception_queue._maxsize = self.prefetch_queue_max

    async def _resize_perception_workers(self) -> None:
        if not (self._perception_pipeline_enabled() or self._serial_prefetch_enabled()):
            for task in self._perception_workers:
                task.cancel()
            if self._perception_workers:
                await asyncio.gather(*self._perception_workers, return_exceptions=True)
            self._perception_workers = []
            return
        target = max(1, min(MAX_PARALLEL_SESSIONS, int(self.parallel_sessions)))
        live = [t for t in self._perception_workers if not t.done()]
        extra = live[target:]
        for task in extra:
            task.cancel()
        if extra:
            await asyncio.gather(*extra, return_exceptions=True)
        self._perception_workers = live[:target]
        self._ensure_perception_workers()

    async def _enqueue_perception_observation(self, obs: dict[str, Any]) -> None:
        if not (self._perception_pipeline_enabled() or self._serial_prefetch_enabled()):
            return
        self._ensure_perception_workers()
        if self._serial_prefetch_enabled():
            async with self.lock:
                snapshots = self._session_snapshots_locked()
            sess = await self.stage_pipeline.enqueue_observation(obs, snapshots=snapshots)
            item = (sess.frame_seq, dict(obs), time.perf_counter(), None)
            if self.prefetch_overload_policy == "drop_oldest" and self._perception_queue.full():
                try:
                    dropped_seq, _dropped_obs, _dropped_s, _dropped_prepared = self._perception_queue.get_nowait()
                    self._perception_queue.task_done()
                    self._perception_dropped += 1
                    await self.stage_pipeline.mark_failed(
                        dropped_seq,
                        "prefetch_queue_drop_oldest",
                    )
                except asyncio.QueueEmpty:
                    pass
            put_started = time.perf_counter()
            await self._perception_queue.put(item)
            waited_s = time.perf_counter() - put_started
            if waited_s * 1000.0 >= self.prefetch_backpressure_warn_ms:
                await self.stage_pipeline.record_prefetch_backpressure(elapsed_s=waited_s)
            self._perception_ingested += 1
        else:
            self._perception_seq += 1
            seq = self._perception_seq
            try:
                self._perception_queue.put_nowait((seq, dict(obs), time.perf_counter(), None))
                self._perception_ingested += 1
            except asyncio.QueueFull:
                self._perception_dropped += 1
                now = time.perf_counter()
                self._perception_ready[seq] = (None, now, now, None)
                await self._drain_ready_perception()
        self._refresh_perception_pipeline_metrics()

    def _prepare_perception_slot_evidence(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Compute read-only slot proposals outside the cognition lock."""
        prepared: dict[str, Any] = {"slot_proposals": [], "slot_error": None}
        if self.perception_mode != "discovered":
            return prepared
        bundle = self.neural
        if bundle is None or not getattr(bundle.stack, "has_slots", False):
            return prepared
        proposals: list[dict[str, Any]] = []
        try:
            import torch

            from decadic.cycle.discovery import extract_proposals
            from decadic.cycle.neural_pipeline import _slot_mask_entropies

            with torch.no_grad():
                patch_tokens = bundle.encoders.vision_patch_tokens(obs)
                if patch_tokens is not None:
                    patch_tokens = patch_tokens.to(device=bundle.device)
                    slot_out = bundle.stack.slot_encode(patch_tokens)
                    centroids = bundle.stack.slots_module.centroids(slot_out["attn"])
                    slots_np = slot_out["slots"][0].detach().float().cpu().numpy()
                    presence_np = slot_out["presence"][0].detach().float().cpu().numpy()
                    centroids_np = centroids[0].detach().float().cpu().numpy()
                    mask_entropies = _slot_mask_entropies(slot_out["attn"])
                    proposals = extract_proposals(
                        slots_np,
                        presence_np,
                        centroids_np,
                        threshold=C.slot_presence_threshold(),
                    )
                    for p in proposals:
                        idx = int(p.get("idx", -1))
                        if 0 <= idx < len(mask_entropies):
                            p["mask_entropy"] = mask_entropies[idx]
        except Exception as exc:
            prepared["slot_error"] = type(exc).__name__
            logger.debug(
                "perception_slot_prepare_failed agent_id=%s",
                self.agent_id,
                exc_info=True,
            )
        prepared["slot_proposals"] = proposals
        return prepared

    def _prepare_perception_fold(
        self,
        obs: dict[str, Any],
        prepared: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare anonymous object-file evidence outside the cognition lock.

        This runs in ordered fold position because the perception organ carries
        frame-difference state. It does not mutate StateBus, action, optimizer,
        replay, episodic memory, or LTM.
        """
        prepared = dict(prepared or {})
        if self.perception_mode != "discovered":
            prepared["skip"] = True
            return prepared
        bundle = self.neural
        organ = getattr(bundle, "_perception_organ", None) if bundle is not None else None
        if organ is None:
            organ = self._runtime_perception_organ
            if bundle is not None:
                bundle._perception_organ = organ
        prev_motor = getattr(bundle, "prev_motor", None) if bundle is not None else None
        proposals = [dict(p) for p in prepared.get("slot_proposals", []) if isinstance(p, dict)]
        proposals, organ_diag, ret_map = organ.process(obs, proposals, prev_motor=prev_motor)
        object_files = object_files_from_proposals(proposals)
        prepared.update(
            {
                "skip": False,
                "proposals": proposals,
                "organ_diag": organ_diag,
                "retinotopic_map": ret_map,
                "object_files": object_files,
            }
        )
        return prepared

    async def _perception_worker_loop(self, worker_idx: int) -> None:
        del worker_idx
        while self.running:
            seq, obs, enqueued_s, prepared = await self._perception_queue.get()
            self._perception_inflight += 1
            try:
                if self._serial_prefetch_enabled():
                    await self.stage_pipeline.mark_prefetching(seq)
                prefetch_started = time.perf_counter()
                if self.neural is not None:
                    encoders = getattr(self.neural, "encoders", None)
                    predecode = getattr(encoders, "predecode", None)
                    if callable(predecode):
                        try:
                            await asyncio.to_thread(predecode, obs)
                        except Exception:
                            logger.debug(
                                "perception_predecode_failed agent_id=%s seq=%s",
                                self.agent_id,
                                seq,
                                exc_info=True,
                            )
                prefetch_elapsed = time.perf_counter() - prefetch_started
                if self._serial_prefetch_enabled():
                    await self.stage_pipeline.mark_prefetched(
                        seq,
                        elapsed_s=prefetch_elapsed,
                    )
                    prepared = await asyncio.to_thread(self._prepare_perception_slot_evidence, obs)
                self._perception_ready[seq] = (obs, enqueued_s, time.perf_counter(), prepared)
                await self._drain_ready_perception()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._perception_ready[seq] = (None, enqueued_s, time.perf_counter(), None)
                if self._serial_prefetch_enabled():
                    await self.stage_pipeline.mark_failed(seq, "prefetch_worker_failed")
                logger.debug(
                    "perception_worker_failed agent_id=%s seq=%s",
                    self.agent_id,
                    seq,
                    exc_info=True,
                )
                await self._drain_ready_perception()
            finally:
                self._perception_inflight = max(0, self._perception_inflight - 1)
                self._perception_queue.task_done()
                self._refresh_perception_pipeline_metrics()

    async def _drain_ready_perception(self) -> None:
        async with self._perception_commit_lock:
            while self._perception_next_commit in self._perception_ready:
                obs, enqueued_s, _ready_s, prepared = self._perception_ready.pop(
                    self._perception_next_commit
                )
                self._perception_next_commit += 1
                if obs is None:
                    continue
                if self._serial_prefetch_enabled():
                    await self.stage_pipeline.mark_folding(self._perception_next_commit - 1)
                fold_started = time.perf_counter()
                if self._serial_prefetch_enabled():
                    prepared = self._prepare_perception_fold(obs, prepared)
                async with self.lock:
                    if self._serial_prefetch_enabled():
                        self.perceptual_integrator.integrate(self.perceptual, obs)
                    if self._serial_prefetch_enabled():
                        self._apply_prepared_perception_observation_locked(obs, prepared)
                    else:
                        self._commit_perception_observation_locked(obs)
                    if self._serial_prefetch_enabled():
                        self._scene_version += 1
                    self._perception_committed += 1
                    now = time.perf_counter()
                    self._perception_last_commit_s = now
                    self.metrics["commit_lag_ms"] = (now - enqueued_s) * 1000.0
                if self._serial_prefetch_enabled():
                    await self.stage_pipeline.mark_folded(
                        self._perception_next_commit - 1,
                        elapsed_s=time.perf_counter() - fold_started,
                    )
        self._refresh_perception_pipeline_metrics()

    def _commit_perception_observation_locked(self, obs: dict[str, Any]) -> None:
        """Commit anonymous perceptual object files from one frame, under lock."""
        prepared = self._prepare_perception_slot_evidence(obs)
        prepared = self._prepare_perception_fold(obs, prepared)
        self._apply_prepared_perception_observation_locked(obs, prepared)

    def _apply_prepared_perception_observation_locked(
        self,
        obs: dict[str, Any],
        prepared: dict[str, Any] | None,
    ) -> None:
        """Apply prepared anonymous perceptual evidence to runtime state."""
        if self.perception_mode != "discovered":
            return
        prepared = prepared or {}
        object_files = list(prepared.get("object_files") or [])
        organ_diag = prepared.get("organ_diag")
        ret_map = prepared.get("retinotopic_map")
        self.perceptual.object_files = [f.to_dict() for f in object_files]
        scene_dynamics_report = {
            "enabled": bool(C.scene_dynamics_enabled()),
            "model_active": False,
            "loss": None,
            "uncertainty": None,
            "prediction_count": 0,
            "matched_count": 0,
        }
        if C.scene_workspace_enabled() and hasattr(self.perceptual, "update_scene_workspace"):
            self.perceptual.update_scene_workspace(
                homeostasis=self.homeostasis,
                state_bus=self.state_bus,
                dynamics_report=scene_dynamics_report,
            )
        scene_focus_proposals = None
        scene_ws = getattr(self.perceptual, "scene_workspace", None)
        if scene_ws is not None:
            scene_focus_proposals = scene_ws.focus_proposals()
        wm_disc = getattr(self.perceptual, "working_memory", None)
        matched = []
        if wm_disc is not None:
            matched = wm_disc.integrate_discovered(
                scene_focus_proposals
                if scene_focus_proposals is not None
                else [f.to_working_memory_proposal() for f in object_files],
                events=(obs.get("events") if isinstance(obs.get("events"), list) else []),
                appearance_weight=C.assoc_appearance_weight(),
                match_threshold=C.assoc_match_threshold(),
                appearance_ema=C.appearance_ema(),
                reidentify=(self.ltm_graph.match if self.ltm_graph is not None else None),
            )
        del matched
        stable_count = (
            sum(
                1
                for s in getattr(wm_disc, "slots", {}).values()
                if int(getattr(s, "seen_count", 0)) >= C.ltm_consolidate_min_seen()
            )
            if wm_disc is not None
            else 0
        )
        health = evaluate_discovery_health(
            object_files,
            tracked_count=len(getattr(wm_disc, "slots", {}) or {}) if wm_disc is not None else 0,
            stable_tracked_objects=stable_count,
        )
        if wm_disc is not None:
            from decadic.cycle.neural_pipeline import _stable_object_file_snapshots

            self.perceptual.object_files = _stable_object_file_snapshots(wm_disc)
            if C.scene_workspace_enabled() and hasattr(self.perceptual, "update_scene_workspace"):
                self.perceptual.update_scene_workspace(
                    homeostasis=self.homeostasis,
                    state_bus=self.state_bus,
                    dynamics_report=scene_dynamics_report,
                )
        self.perceptual.discovery_health = health.to_dict()
        self.perceptual.ltm_consolidation = {
            "status": "not_evaluated",
            "reason": health.reason,
        }
        self.perceptual.perception_organ = organ_diag
        self.perceptual.retinotopic_map = ret_map

    def _refresh_perception_pipeline_metrics(self) -> None:
        elapsed = max(1e-6, time.perf_counter() - self._perception_started_at)
        self.metrics["processing_mode"] = self.processing_mode
        self.metrics["stage_pipeline_enabled"] = self._stage_pipeline_enabled()
        self.metrics["perceptual_processing_mode"] = self.processing_mode
        self.metrics["pipeline_sessions"] = int(self.parallel_sessions)
        self.metrics["prefetch_queue_depth"] = int(self._perception_queue.qsize())
        self.metrics["prefetch_queue_max"] = int(self.prefetch_queue_max)
        self.metrics["prefetch_overload_policy"] = str(self.prefetch_overload_policy)
        self.metrics["ready_coalesce_policy"] = str(self.ready_coalesce_policy)
        self.metrics["perception_queue_depth"] = int(self._perception_queue.qsize())
        self.metrics["perception_inflight"] = int(self._perception_inflight)
        self.metrics["perception_ingest_hz"] = float(self._perception_ingested / elapsed)
        self.metrics["perception_commit_hz"] = float(self._perception_committed / elapsed)
        self.metrics["frames_committed"] = int(self._perception_committed)
        self.metrics["frames_dropped"] = int(self._perception_dropped)
        self.metrics["batching_fallback"] = (
            self.processing_mode == PROCESSING_BATCHING
        )
        if self._perception_last_commit_s is not None:
            self.metrics["sample_age_ms"] = (
                time.perf_counter() - self._perception_last_commit_s
            ) * 1000.0
        self._refresh_stage_pipeline_metrics()

    def _session_snapshots_locked(self) -> dict[str, Any]:
        scene_ws = getattr(self.perceptual, "scene_workspace", None)
        return {
            "scene_version": int(self._scene_version),
            "state_bus_version": int(self._state_bus_version),
            "workspace_version": int(self._workspace_version),
            "weight_version": int(self._weight_version),
            "commit_index": int(self._commit_index),
            "cycle_index": int(self.state_bus.cycle_index),
            "priority_label": str(self.state_bus.priority_label),
            "pain": float(self.state_bus.pain_scalar),
            "pleasure": float(self.state_bus.pleasure_scalar),
            "scene": scene_ws.snapshot() if scene_ws is not None else None,
            "working_memory": self.perceptual.working_memory.snapshot(),
        }

    def _refresh_stage_pipeline_metrics(self) -> None:
        m = self.stage_pipeline.metrics()
        self.metrics["stage_pipeline_active_sessions"] = int(m["active_sessions"])
        self.metrics["stage_pipeline_ready_sessions"] = int(m["ready_sessions"])
        self.metrics["stage_pipeline_committed_sessions"] = int(m["committed_sessions"])
        self.metrics["stage_pipeline_committed_per_s"] = float(m["committed_sessions_per_s"])
        self.metrics["stage_pipeline_dropped_sessions"] = int(m["dropped_sessions"])
        self.metrics["stage_pipeline_stale_sessions"] = int(m["stale_sessions"])
        self.metrics["stage_pipeline_failed_sessions"] = int(m["failed_sessions"])
        self.metrics["stage_pipeline_queue_depths"] = m["stage_queue_depths"]
        self.metrics["stage_pipeline_inflight"] = m["stage_inflight"]
        self.metrics["stage_pipeline_latency_ms"] = m["stage_latency_ms"]
        self.metrics["stage_pipeline_recent_sessions"] = m["recent_sessions"]
        self.metrics["stage_pipeline_selected_session"] = m["selected_session"]
        self.metrics["frames_received"] = int(m.get("frames_received", 0))
        self.metrics["frames_prefetched"] = int(m.get("frames_prefetched", 0))
        self.metrics["frames_folded"] = int(m.get("frames_folded", 0))
        self.metrics["frames_deep_processed"] = int(m.get("frames_deep_processed", 0))
        self.metrics["coalesced_sessions"] = int(m.get("coalesced_sessions", 0))
        self.metrics["information_loss"] = int(m.get("information_loss", 0))
        self.metrics["producer_overlap_ratio"] = float(m.get("producer_overlap_ratio", 0.0))
        self.metrics["decode_on_consume_ms"] = float(m.get("decode_on_consume_ms") or 0.0)
        self.metrics["consume_wait_ms"] = float(m.get("consume_wait_ms") or 0.0)
        self.metrics["ready_queue_depth"] = int(m.get("ready_queue_depth", 0))
        self.metrics["ready_coalesce_policy"] = str(m.get("ready_coalesce_policy", self.ready_coalesce_policy))
        self.metrics["prefetch_backpressure_events"] = int(m.get("prefetch_backpressure_events", 0))
        self.metrics["prefetch_backpressure_ms"] = float(m.get("prefetch_backpressure_ms") or 0.0)
        self.metrics["oldest_unfolded_age_ms"] = float(m.get("oldest_unfolded_age_ms") or 0.0)
        self.metrics["prefetch_backpressure_warning"] = bool(
            self.metrics["prefetch_backpressure_ms"] >= self.prefetch_backpressure_warn_ms
            and self.metrics["prefetch_backpressure_events"] > 0
        )
        self.metrics["oldest_unfolded_warning"] = bool(
            self.metrics["oldest_unfolded_age_ms"] >= self.prefetch_oldest_unfolded_warn_ms
        )
        self.metrics["fold_lag_ms"] = float(m.get("fold_lag_ms") or 0.0)
        sel = m["selected_session"] or {}
        self.metrics["stage_pipeline_arbitration_reason"] = str(
            sel.get("arbitration_reason", "")
        )
        if m.get("commit_lag_ms") is not None:
            self.metrics["commit_lag_ms"] = float(m["commit_lag_ms"])

    def _cached_memory_context_for_cycle(self) -> list[float] | None:
        if not C.memory_context_async_enabled() or self.neural is None:
            return None
        if self._memory_context_vector is not None:
            return list(self._memory_context_vector)
        return [0.0] * int(self.neural.cfg.memory_context_dim)

    def _refresh_memory_recall_metrics(self) -> None:
        stats = getattr(self.episodic, "recall_cache_stats", lambda: {})()
        if isinstance(stats, dict):
            self.metrics["memory_recall_cache_size"] = int(stats.get("size", 0) or 0)
            self.metrics["memory_recall_cache_hits"] = int(stats.get("hits", 0) or 0)
            self.metrics["memory_recall_cache_misses"] = int(stats.get("misses", 0) or 0)
        pm_getter = getattr(self.episodic, "persistence_metrics", None)
        pm = pm_getter() if callable(pm_getter) else {}
        if isinstance(pm, dict):
            self.metrics["episodic_sqlite_commit_count"] = int(pm.get("sqlite_commit_count", 0) or 0)
            self.metrics["episodic_sqlite_batch_commit_count"] = int(pm.get("sqlite_batch_commit_count", 0) or 0)
            self.metrics["episodic_sqlite_last_commit_ms"] = float(pm.get("sqlite_last_commit_ms", 0.0) or 0.0)
            self.metrics["episodic_sqlite_wal_checkpoint_count"] = int(pm.get("sqlite_wal_checkpoint_count", 0) or 0)
            self.metrics["episodic_write_batch_size_last"] = int(pm.get("episodic_write_batch_size_last", 0) or 0)
            self.metrics["episodic_db_rows"] = int(pm.get("episodic_db_rows", 0) or 0)
            self.metrics["episodic_db_pruned_rows"] = int(pm.get("episodic_db_pruned_rows", 0) or 0)
            self.metrics["episodic_db_bytes"] = int(pm.get("memory_db_bytes", 0) or 0)
            self.metrics["episodic_wal_bytes"] = int(pm.get("memory_wal_bytes", 0) or 0)
            self.metrics["sqlite_commit_count"] = self.metrics["episodic_sqlite_commit_count"]
            self.metrics["sqlite_batch_commit_count"] = self.metrics["episodic_sqlite_batch_commit_count"]
            self.metrics["sqlite_last_commit_ms"] = self.metrics["episodic_sqlite_last_commit_ms"]
            self.metrics["sqlite_wal_checkpoint_count"] = self.metrics["episodic_sqlite_wal_checkpoint_count"]
            self.metrics["memory_db_bytes"] = self.metrics["episodic_db_bytes"]
            self.metrics["memory_wal_bytes"] = self.metrics["episodic_wal_bytes"]
        self.metrics["memory_recall_worker_ms"] = float(self._memory_context_worker_ms)
        self.metrics["memory_recall_refresh_cycle"] = int(self._memory_context_refresh_cycle)
        self.metrics["memory_recall_staleness_cycles"] = max(
            0, int(self.state_bus.cycle_index) - int(self._memory_context_refresh_cycle)
        )

    def _refresh_ltm_metrics(self) -> None:
        if self.ltm_graph is None:
            return
        try:
            stats_getter = getattr(self.ltm_graph, "cached_belief_stats", None)
            stats = stats_getter() if callable(stats_getter) else self.ltm_graph.belief_stats()
            if isinstance(stats, dict):
                self.metrics["ltm_property_beliefs"] = int(
                    stats.get("total_property_beliefs", 0)
                )
                self.metrics["ltm_avg_property_confidence"] = float(
                    stats.get("avg_property_confidence", 0.0)
                )
        except Exception:
            pass
        try:
            metrics_getter = getattr(self.ltm_graph, "runtime_metrics", None)
            ltm_metrics = metrics_getter() if callable(metrics_getter) else {}
            if isinstance(ltm_metrics, dict):
                self.metrics.update(ltm_metrics)
                self.metrics["ltm_sqlite_commit_count"] = int(ltm_metrics.get("sqlite_commit_count", 0) or 0)
                self.metrics["ltm_sqlite_batch_commit_count"] = int(ltm_metrics.get("sqlite_batch_commit_count", 0) or 0)
                self.metrics["ltm_sqlite_last_commit_ms"] = float(ltm_metrics.get("sqlite_last_commit_ms", 0.0) or 0.0)
                self.metrics["ltm_sqlite_wal_checkpoint_count"] = int(ltm_metrics.get("sqlite_wal_checkpoint_count", 0) or 0)
                self.metrics["ltm_db_bytes"] = int(ltm_metrics.get("memory_db_bytes", 0) or 0)
                self.metrics["ltm_wal_bytes"] = int(ltm_metrics.get("memory_wal_bytes", 0) or 0)
                self.metrics["sqlite_commit_count"] = int(self.metrics.get("episodic_sqlite_commit_count", 0) or 0) + int(ltm_metrics.get("sqlite_commit_count", 0) or 0)
                self.metrics["sqlite_batch_commit_count"] = int(self.metrics.get("episodic_sqlite_batch_commit_count", 0) or 0) + int(ltm_metrics.get("sqlite_batch_commit_count", 0) or 0)
                self.metrics["sqlite_last_commit_ms"] = max(
                    float(self.metrics.get("episodic_sqlite_last_commit_ms", 0.0) or 0.0),
                    float(ltm_metrics.get("sqlite_last_commit_ms", 0.0) or 0.0),
                )
                self.metrics["sqlite_wal_checkpoint_count"] = int(self.metrics.get("episodic_sqlite_wal_checkpoint_count", 0) or 0) + int(ltm_metrics.get("sqlite_wal_checkpoint_count", 0) or 0)
                self.metrics["memory_db_bytes"] = int(self.metrics.get("episodic_db_bytes", 0) or 0) + int(ltm_metrics.get("memory_db_bytes", 0) or 0)
                self.metrics["memory_wal_bytes"] = int(self.metrics.get("episodic_wal_bytes", 0) or 0) + int(ltm_metrics.get("memory_wal_bytes", 0) or 0)
        except Exception:
            pass

    def _maybe_schedule_memory_context_refresh(self, query: Any, cycle: int) -> None:
        if not C.memory_context_async_enabled() or self.neural is None or query is None:
            return
        if self._memory_context_task is not None and not self._memory_context_task.done():
            return
        if int(cycle) - int(self._memory_context_refresh_cycle) < C.memory_context_refresh_cycles():
            return
        q_list = [float(x) for x in list(query)]
        out_dim = int(self.neural.cfg.memory_context_dim)
        top_k = int(os.environ.get("DECADIC_MEMORY_TOP_K", "5"))
        min_salience = float(os.environ.get("DECADIC_MEMORY_MIN_SALIENCE", "0"))

        async def refresh() -> None:
            started = time.perf_counter()
            try:
                import numpy as np

                q = np.asarray(q_list, dtype=np.float32)
                vec = await asyncio.to_thread(
                    self.episodic.retrieval_context_vector,
                    q,
                    out_dim,
                    top_k=top_k,
                    min_salience=min_salience,
                    exclude_cycle=int(cycle),
                )
                self._memory_context_vector = [float(x) for x in vec.tolist()]
                self._memory_context_query = list(q_list)
                self._memory_context_refresh_cycle = int(cycle)
                self._memory_context_worker_ms = (time.perf_counter() - started) * 1000.0
            except Exception:
                logger.debug("memory_context_refresh_failed agent_id=%s", self.agent_id, exc_info=True)
            finally:
                self._refresh_memory_recall_metrics()

        self._memory_context_task = asyncio.create_task(
            refresh(), name=f"decadic-memory-context-{self.agent_id}"
        )

    async def _consolidation_runner(self) -> None:
        # Real dual-network consolidation when enabled: clone the current stack and
        # run the prioritized-replay + soft-sync loop. Otherwise keep the historical
        # no-op heartbeat (parity).
        if consolidation_enabled() and self.neural is not None and self.replay_buffer is not None:
            self._consolidator = ConsolidationManager(self.neural, lock=self.lock)
            await self._consolidator.run_loop(
                self.replay_buffer,
                should_continue=lambda: self.running,
                current_cycle=lambda: int(self.state_bus.cycle_index),
                on_sync=self._on_consolidation_sync,
            )
            return
        await consolidation_stub_loop(
            self.agent_id,
            should_continue=lambda: self.running,
            on_tick=self._on_consolidation_tick,
        )

    def _on_consolidation_tick(self) -> None:
        self.metrics["consolidation_stub_ticks"] = (
            int(self.metrics.get("consolidation_stub_ticks", 0)) + 1
        )

    def _on_consolidation_sync(
        self,
        replay_steps: int,
        loss: float,
        cycle: int,
        sync_metrics: dict[str, float | int] | None = None,
    ) -> None:
        self.metrics["replay_count"] = int(replay_steps)
        self.metrics["consolidator_loss"] = float(loss)
        self.metrics["last_sync_cycle"] = int(cycle)
        if sync_metrics:
            self.metrics["consolidation_sync_delta_mean"] = float(
                sync_metrics.get("delta_mean", 0.0)
            )
            self.metrics["consolidation_sync_delta_max"] = float(
                sync_metrics.get("delta_max", 0.0)
            )
            self.metrics["consolidation_sync_moved_params"] = int(
                sync_metrics.get("moved_params", 0)
            )
            self.metrics["consolidation_sync_reset_params"] = int(
                sync_metrics.get("reset_params", 0)
            )
        if self.replay_buffer is not None:
            self.metrics["replay_buffer_size"] = len(self.replay_buffer)

    async def _landscape_runner(self) -> None:
        """Periodically recompute the filter-normalized loss surface (off the loop).

        Each refresh samples a replay batch, evaluates the grid on a throwaway clone
        in a worker thread (never the live weights, never the event loop), and caches
        the surface for ``GET /agent/{id}/brain/landscape``. Re-entrancy-safe: the
        probe owns its own clone, so it is independent of consolidation's clone.
        """
        if self.neural is None or self.replay_buffer is None:
            return
        self._landscape_probe = LossLandscapeProbe(self.neural, seed=landscape_seed())
        interval = landscape_interval_s()
        while self.running:
            await asyncio.sleep(interval)
            if not self.running:
                break
            batch = self.replay_buffer.sample(landscape_batch())
            if not batch:
                continue
            t0 = time.perf_counter()
            surface = await asyncio.to_thread(
                self._landscape_probe.compute,
                batch,
                grid=landscape_grid(),
                span=landscape_span(),
                cycle=int(self.state_bus.cycle_index),
            )
            if surface is None:
                continue
            surface["wall_ms"] = (time.perf_counter() - t0) * 1000.0
            self._last_landscape = surface
            logger.info(
                "landscape_compute agent_id=%s cycle=%s grid=%s batch=%s "
                "center=%.5f z_min=%.5f z_max=%.5f wall_ms=%.1f",
                self.agent_id,
                surface["cycle"],
                surface["grid"],
                surface["batch"],
                surface["center_loss"],
                surface["z_min"],
                surface["z_max"],
                surface["wall_ms"],
            )

    async def _metabolic_loop(self) -> None:
        """Wall-clock homeostatic clock; frozen while paused, dead, or immortal."""
        tick = metabolic_tick_s()
        if tick <= 0:
            return
        while self.running:
            await asyncio.sleep(tick)
            if not self.running:
                break
            now = time.monotonic()
            # No metabolism unless alive, running, and in metabolic mode. Reset
            # the baseline so paused/immortal time never accrues as a debt.
            if self.paused or self.status == "dead" or self.viability_mode != "metabolic":
                self._last_metab_monotonic = now
                continue
            last = self._last_metab_monotonic
            self._last_metab_monotonic = now
            if last is None:
                continue
            dt = max(0.0, now - last)
            if dt <= 0.0:
                continue
            async with self.lock:
                self.stress = self._compute_stress()
                self._threat_stress *= 0.85
                passive_metabolism(
                    self.homeostasis,
                    dt,
                    self.stress,
                    hydration_empty_s=hydration_empty_s(),
                    energy_empty_s=energy_empty_s(),
                    integrity_heal_full_s=integrity_heal_full_s(),
                    heal_min_reserve=heal_min_reserve(),
                    stress_gain=stress_gain(),
                    compression=self.metabolic_compression,
                )
                self.viability.value = self.homeostasis.viability
                self._refresh_homeostasis_metrics()
                self._check_death()

    def _compute_stress(self) -> float:
        """Blend pain, lingering threat, and prediction error into a 0..1 load."""
        pain = min(1.0, float(self.state_bus.pain_scalar))
        pc = float(self.metrics.get("neural_pc_loss_last", 0.0) or 0.0)
        threat = min(1.0, float(self._threat_stress))
        raw = 0.4 * pain + 0.35 * threat + 0.25 * min(1.0, pc * 2.0)
        return max(0.0, min(1.0, raw))

    def _damage_grace(self) -> float:
        """Discount injury while the joint braces are still holding the body.

        A learner that cannot yet stand will fall constantly; like a toddler,
        those tumbles must not be lethal. ``brace_engaged`` is the mean per-joint
        brace tightness: ~1 (fully welded, body still on its training braces)
        yields a heavy discount toward ``damage_grace_floor``; ~0 (braces fully
        loosened, the body moving on its own) yields full damage. Once the body
        has earned its range of motion, falls hurt for real.
        """
        brace = float(self.metrics.get("brace_engaged", 1.0) or 0.0)
        held = max(0.0, min(1.0, brace))
        return max(damage_grace_floor(), 1.0 - held)

    def _time_to_death_s(self) -> float | None:
        """Estimated seconds until the fastest-draining reservoir empties."""
        if self.viability_mode != "metabolic":
            return None
        h = self.homeostasis
        span = h.max_value - h.min_value
        if span <= 0:
            return None
        drain_mult = self.metabolic_compression * (1.0 + stress_gain() * self.stress)
        if drain_mult <= 0:
            return None
        hyd_rate = span / hydration_empty_s() * drain_mult
        eng_rate = span / energy_empty_s() * drain_mult
        candidates: list[float] = []
        if hyd_rate > 0:
            candidates.append((h.hydration - h.min_value) / hyd_rate)
        if eng_rate > 0:
            candidates.append((h.energy - h.min_value) / eng_rate)
        # Integrity heals rather than passively draining, so it does not bound
        # lifespan here unless damage events take it to zero between ticks.
        if not candidates:
            return None
        return round(float(min(candidates)), 1)

    def _refresh_plasticity_metrics(self) -> None:
        """Mirror the neural stack's structural state into metrics (Brain Map / panels)."""
        b = self.neural
        stack = getattr(b, "stack", None) if b is not None else None
        if b is None or stack is None or not getattr(stack, "has_plastic", False):
            self.metrics["plasticity_enabled"] = False
            self.metrics["sparse_enabled"] = False
            self.metrics["growth_enabled"] = False
            return
        f = b.flags
        self.metrics["plasticity_enabled"] = bool(f.plastic)
        self.metrics["sparse_enabled"] = bool(f.sparse)
        self.metrics["growth_enabled"] = bool(f.growth)
        self.metrics["plasticity_alpha"] = round(stack.plastic_alpha_mean(), 6)
        self.metrics["plasticity_alpha_configured"] = round(stack.plastic_alpha_mean(), 6)
        self.metrics["plasticity_alpha_effective"] = round(stack.plastic_effective_alpha_mean(), 6)
        overlay_mean, overlay_max = stack.plastic_overlay_ratio_stats()
        self.metrics["plasticity_overlay_ratio_mean"] = round(overlay_mean, 6)
        self.metrics["plasticity_overlay_ratio_max"] = round(overlay_max, 6)
        self.metrics["sparse_density"] = round(stack.connection_density(), 6)
        self.metrics["awake_neurons"] = stack.awake_neurons()
        self.metrics["allocated_neurons"] = stack.allocated_neurons()
        self.metrics["active_connections"] = stack.active_connections()
        ps = b.plasticity_state
        if ps is not None:
            self.metrics["max_neurons"] = int(ps.max_neurons)
            self.metrics["rewire_events"] = int(ps.rewire_events)
            self.metrics["growth_events"] = int(ps.growth_events)
            self.metrics["plasticity_frozen"] = bool(ps.frozen)
            self.metrics["plasticity_guardian_state"] = str(ps.guardian_state)
            self.metrics["plasticity_guardian_action"] = str(ps.last_action)
            self.metrics["plasticity_pc_ema"] = ps.pc_ema
            self.metrics["plasticity_pc_slope_ema"] = float(ps.pc_slope_ema)
            self.metrics["plasticity_freeze_count"] = int(ps.freeze_count)
            self.metrics["plasticity_thaw_count"] = int(ps.thaw_count)
            self.metrics["plasticity_warmup_blocked_reason"] = str(ps.blocked_reason)

    def _refresh_homeostasis_metrics(self) -> None:
        h = self.homeostasis
        self.metrics["hydration"] = round(h.hydration, 4)
        self.metrics["energy"] = round(h.energy, 4)
        self.metrics["integrity"] = round(h.integrity, 4)
        self.metrics["stress"] = round(float(self.stress), 4)
        self.metrics["viability_mode"] = self.viability_mode
        self.metrics["time_to_death_s"] = self._time_to_death_s()
        self._refresh_ltm_metrics()

    def _capture_locomotion_telemetry(self, obs: dict[str, Any]) -> None:
        """Mirror the body's joint-brace signals into agent metrics.

        Dashboard-only readout (mean ROM earned, brace tightness, per-joint ROM
        bars, per-part load); inert to cognition.
        """
        world_state = obs.get("world_state")
        if not isinstance(world_state, dict):
            return
        body = world_state.get("body")
        if not isinstance(body, dict):
            return
        for key in (
            "rom_mean",
            "brace_engaged",
            "foot_load_l",
            "foot_load_r",
            "hand_load_l",
            "hand_load_r",
            "teacher_support_force",
            "teacher_support_torque",
            "teacher_drop_m",
            "teacher_target_drop_m",
            "teacher_height_error_m",
            "teacher_vertical_velocity",
            "caregiver_delivery_count",
        ):
            value = body.get(key)
            if value is not None:
                self.metrics[key] = float(value)
        teacher_support_active = body.get("teacher_support_active")
        if teacher_support_active is not None:
            self.metrics["teacher_support_active"] = bool(teacher_support_active)
        teacher_support_mode = body.get("teacher_support_mode")
        if teacher_support_mode is not None:
            self.metrics["teacher_support_mode"] = str(teacher_support_mode)
        for key in (
            "caregiver_status",
            "caregiver_kind",
            "caregiver_request_kind",
            "caregiver_last_offer_item",
        ):
            value = body.get(key)
            if value is not None:
                self.metrics[key] = str(value)
        for key in ("caregiver_parent_present", "caregiver_pending_request"):
            value = body.get(key)
            if value is not None:
                self.metrics[key] = bool(value)
        if "caregiver_parent_present" in body:
            self.metrics["caregiver_missing_parent"] = not bool(
                body.get("caregiver_parent_present")
            )
        rom_frac = body.get("rom_frac")
        if isinstance(rom_frac, list):
            self.metrics["joint_rom"] = [float(v) for v in rom_frac]
        braces_enabled = body.get("braces_enabled")
        if braces_enabled is not None:
            self.metrics["braces_enabled"] = bool(braces_enabled)
        stance = body.get("stance")
        if stance is not None:
            self.metrics["stance"] = str(stance)
        stance_phase = body.get("stance_phase")
        if stance_phase is not None:
            self.metrics["stance_phase"] = float(stance_phase)
        movement_hold = body.get("movement_hold")
        if movement_hold is not None:
            self.metrics["movement_hold"] = bool(movement_hold)
        # Full-body touch map (short name -> load). Live in all modes; the
        # dashboard renders one bar per part for full-body contact awareness.
        part_loads = body.get("part_loads")
        if isinstance(part_loads, dict):
            self.metrics["part_loads"] = {
                str(k): float(v) for k, v in part_loads.items()
            }
        prop = obs.get("proprioception") if isinstance(obs, dict) else None
        if isinstance(prop, dict):
            body_map = normalize_body_map(prop.get("body_map"))
            effort = normalize_effort(prop.get("effort"))
            self.metrics["body_map"] = body_map
            self.metrics["effort"] = effort
            for key in (
                "effort_total",
                "work_total",
                "strain_total",
                "fatigue_total",
                "pain_total",
                "support_effort",
            ):
                self.metrics[key] = float(effort.get(key, 0.0) or 0.0)
            part, pain = most_pained_part(body_map)
            self.metrics["most_pained_part"] = part
            self.metrics["most_pained_part_pain"] = round(float(pain), 5)
        self._capture_gait_and_motion(obs, part_loads)

    def _capture_gait_and_motion(
        self, obs: dict[str, Any], part_loads: Any
    ) -> None:
        """Eval-only locomotion/gait readouts from the streamed body state.

        Distance/displacement come from the proprioceptive root position;
        fall-rate from this observation's events; gait regularity from the
        left/right foot loads. All of this is dashboard/dojo telemetry and
        is NEVER fed back into cognition.
        """
        proprio = obs.get("proprioception")
        pos = proprio.get("position") if isinstance(proprio, dict) else None
        if isinstance(pos, list) and len(pos) >= 2:
            try:
                xy = (float(pos[0]), float(pos[1]))
                if len(pos) >= 3:
                    self.metrics["root_height"] = round(float(pos[2]), 4)
            except (TypeError, ValueError):
                xy = None
            if xy is not None:
                if self._loco_origin_xy is None:
                    self._loco_origin_xy = xy
                if self._loco_last_xy is not None:
                    step = math.hypot(
                        xy[0] - self._loco_last_xy[0], xy[1] - self._loco_last_xy[1]
                    )
                    # Guard against teleports (recenter) polluting the path length.
                    if step < 2.0:
                        self._loco_distance += step
                self._loco_last_xy = xy
                ox, oy = self._loco_origin_xy
                self.metrics["distance_traveled"] = round(self._loco_distance, 4)
                self.metrics["net_displacement"] = round(
                    math.hypot(xy[0] - ox, xy[1] - oy), 4
                )

        events = obs.get("events")
        fell = 0
        if isinstance(events, list):
            for ev in events:
                if isinstance(ev, dict) and str(ev.get("type", "")).lower() in (
                    "fall",
                    "collision",
                ):
                    fell = 1
                    break
        self._loco_fall_buf.append(fell)
        if self._loco_fall_buf:
            self.metrics["fall_rate"] = round(
                sum(self._loco_fall_buf) / len(self._loco_fall_buf), 4
            )

        if isinstance(part_loads, dict):
            try:
                left = float(part_loads.get("left_foot", 0.0))
                right = float(part_loads.get("right_foot", 0.0))
            except (TypeError, ValueError):
                left = right = 0.0
            phase = 0
            if max(left, right) >= LOCO_FOOT_CONTACT_N:
                if left > right:
                    phase = 1
                elif right > left:
                    phase = -1
            self._loco_foot_phase_buf.append(phase)
            self.metrics["gait_regularity"] = round(
                _gait_regularity(list(self._loco_foot_phase_buf)), 4
            )
        orient = proprio.get("orientation") if isinstance(proprio, dict) else None
        if isinstance(orient, list) and len(orient) >= 2:
            try:
                roll = float(orient[0])
                pitch = float(orient[1])
                self.metrics["torso_tilt"] = round(math.hypot(roll, pitch), 4)
            except (TypeError, ValueError):
                pass

    def pause(self) -> None:
        """Freeze the cognitive cycle loop (state and weights retained)."""
        self.paused = True

    def resume(self) -> None:
        """Resume cycling; restarts the worker task if it ever died."""
        self.paused = False
        self.ensure_cycle_worker()

    def configure(
        self,
        *,
        parallel_sessions: int | None = None,
        processing_mode: str | None = None,
        stage_pipeline_enabled: bool | None = None,
        perceptual_processing_mode: str | None = None,
        working_memory_slots: int | None = None,
        working_memory_decay: float | None = None,
        assist_override: float | None = None,
        curriculum_mode: str | None = None,
        viability_mode: str | None = None,
        metabolic_compression: float | None = None,
        ai_intero_pref_weight: float | None = None,
        drive_priority_gain: float | None = None,
        motor_babble_sigma: float | None = None,
        plasticity_alpha: float | None = None,
        sparse_density: float | None = None,
        max_neurons: int | None = None,
        perception_mode: str | None = None,
        perception_feedback: bool | None = None,
        self_model_feedback: bool | None = None,
        predictive_affect: bool | None = None,
        represented_self: bool | None = None,
        encoder_mode: str | None = None,
        cognition_trace: bool | None = None,
        probe_capture: bool | None = None,
        gwt_enabled: bool | None = None,
        integration_window_ms: float | None = None,
        episodic_async: bool | None = None,
        ltm_async: bool | None = None,
    ) -> dict[str, Any]:
        """Live-tune workspace capacity (K), working-memory slots (S), and decay.

        ``assist_override``: a negative sentinel (e.g. ``-1``) clears back to Auto
        (curriculum); any value ``>= 0`` pins the assist harness to that level.
        ``viability_mode``: ``"immortal"`` pins all reservoirs at full and
        disables death; ``"metabolic"`` runs the full wall-clock model.
        ``metabolic_compression``: time-acceleration of the metabolic clock.
        ``plasticity_alpha`` / ``sparse_density`` / ``max_neurons``: live A/B/C
        knobs. ``max_neurons`` raises the growth cap (the controller grows into
        it organically) or, if lowered below the current width, sleeps neurons
        immediately. Enabling/disabling a subsystem entirely takes effect on the
        next ``reset()`` (mirrors preset-switch semantics).
        ``perception_feedback`` / ``perception_mode`` / ``encoder_mode`` /
        ``self_model_feedback``: the core cognitive faculties. Each changes the
        model's module set / state_dict shape, so toggling one rebuilds the brain
        with fresh weights (reset semantics for cognition; episodic + working
        memory are left intact). ``self_model_feedback`` adds the self-state
        feedback spine (the previous cycle's A‖C‖E shapes the next cycle).
        ``predictive_affect`` adds the affect forward model (the predicted next-step
        affect colours perception). ``represented_self`` adds the represented-self
        ingress (interoception/affect/capability written onto the self-node + fed
        back). All default off and rebuild the brain on toggle.
        ``cognition_trace`` / ``probe_capture``: read-only observation toggles that
        apply live (no rebuild); they never feed cognition.
        ``gwt_enabled``: live toggle for the global-workspace competition (replaces
        the working-memory EMA blend into A with winner-take-all + ignition +
        broadcast). A pipeline branch, not an architecture change, so no rebuild.
        ``integration_window_ms``: live temporal-integration window (ms). > 0 binds a
        span of percepts into one committed "now"; 0 = off. Pipeline branch, no
        rebuild.
        ``episodic_async``: live toggle for write-behind episodic persistence (moves
        the per-cycle SQLite write off the cognitive lock); drains+stops the worker
        when turned off. No rebuild; no write is lost.
        ``ltm_async``: live toggle for write-behind LTM consolidation (moves stage 10's
        WM->LTM commit off the cognitive lock); drains+stops the worker when turned
        off. No rebuild; no consolidation is lost.
        """
        requested_mode = processing_mode
        if requested_mode is None and stage_pipeline_enabled is not None:
            requested_mode = (
                PROCESSING_SERIAL_PREFETCH
                if bool(stage_pipeline_enabled)
                else PROCESSING_PERSISTENT_PERCEPTION
            )
        if requested_mode is None:
            requested_mode = perceptual_processing_mode
        if requested_mode is not None:
            mode = str(requested_mode).strip().lower()
            if mode == PROCESSING_STAGE_PIPELINE:
                mode = PROCESSING_SERIAL_PREFETCH
            if mode == PERCEPTUAL_PROCESSING_PERSISTENT:
                mode = PROCESSING_PERSISTENT_PERCEPTION
            if mode in PERCEPTUAL_PROCESSING_MODES and mode != self.processing_mode:
                self.processing_mode = mode
                self.perceptual_processing_mode = mode
                self._clear_perception_pipeline()
                self.stage_pipeline.clear()
                self._rebuild_perception_queue_if_needed()
                if self._stage_pipeline_enabled():
                    self.stage_pipeline.start()
        if parallel_sessions is not None:
            k = max(1, min(MAX_PARALLEL_SESSIONS, int(parallel_sessions)))
            self.parallel_sessions = k
            self._obs_buffer = deque(self._obs_buffer, maxlen=k)
            self.stage_pipeline.set_capacity(k)
            self._rebuild_perception_queue_if_needed()
            extra = self._perception_workers[k:]
            for task in extra:
                task.cancel()
            self._perception_workers = self._perception_workers[:k]
            self._ensure_perception_workers()
        wm = self.perceptual.working_memory
        if working_memory_slots is not None:
            wm.capacity = max(1, int(working_memory_slots))
        if working_memory_decay is not None:
            wm.decay = float(min(0.9999, max(0.0, working_memory_decay)))
        if assist_override is not None:
            self.assist_override = None if assist_override < 0 else float(assist_override)
        if curriculum_mode is not None:
            mode = str(curriculum_mode).strip().lower()
            # Accept "standard" as a friendly alias for the legacy training-wheels system.
            if mode in ("guided", "legacy", "standard"):
                self.curriculum_mode = "guided" if mode == "guided" else "legacy"
        # Core cognitive faculties. Changing any of these alters the model's
        # module set / state_dict shape, so we rebuild the brain once at the end
        # rather than mutating the live stack in place.
        arch_changed = False
        if perception_mode is not None:
            mode = str(perception_mode).strip().lower()
            if mode in ("oracle", "discovered") and mode != self.faculties.perception_mode:
                self.faculties.perception_mode = mode
                self.perception_mode = mode
                self.perceptual.perception_mode = mode
                arch_changed = True
        if perception_feedback is not None:
            pf = bool(perception_feedback)
            if pf != self.faculties.perception_feedback:
                self.faculties.perception_feedback = pf
                arch_changed = True
        if self_model_feedback is not None:
            smf = bool(self_model_feedback)
            if smf != self.faculties.self_model_feedback:
                self.faculties.self_model_feedback = smf
                arch_changed = True
        if predictive_affect is not None:
            pa = bool(predictive_affect)
            if pa != self.faculties.predictive_affect:
                self.faculties.predictive_affect = pa
                arch_changed = True
        if represented_self is not None:
            rs = bool(represented_self)
            if rs != self.faculties.represented_self:
                self.faculties.represented_self = rs
                arch_changed = True
        if encoder_mode is not None:
            enc = str(encoder_mode).strip().lower()
            if enc in ("zeros", "hf") and enc != self.faculties.encoder_mode:
                self.faculties.encoder_mode = enc
                arch_changed = True
        if cognition_trace is not None:
            self.cognition_trace = bool(cognition_trace)
        if probe_capture is not None:
            self.probe_capture = bool(probe_capture)
        if gwt_enabled is not None:
            self.gwt_enabled = bool(gwt_enabled)
        if integration_window_ms is not None:
            self.integration_window_ms = max(0.0, float(integration_window_ms))
        if episodic_async is not None:
            set_async = getattr(self.episodic, "set_async", None)
            if callable(set_async):
                set_async(bool(episodic_async))
        if ltm_async is not None and self.ltm_graph is not None:
            set_ltm_async = getattr(self.ltm_graph, "set_async", None)
            if callable(set_ltm_async):
                set_ltm_async(bool(ltm_async))
        if viability_mode is not None:
            mode = str(viability_mode).strip().lower()
            if mode in ("metabolic", "immortal"):
                self.viability_mode = mode
                if mode == "immortal":
                    # Entering immortal pins every reservoir at full and, if the
                    # agent had already died, brings the same mind back to life
                    # (no separate Revive click needed for long learning runs).
                    if self.status == "dead":
                        self.status = "alive"
                        self.died_at_cycle = None
                        self.stress = 0.0
                        self._threat_stress = 0.0
                        self._last_metab_monotonic = time.monotonic()
                        self.ensure_cycle_worker()
                    self.homeostasis.reset(self.homeostasis.max_value)
                    self.viability.value = self.homeostasis.max_value
                else:
                    # Resume metabolism without charging for the paused interval.
                    self._last_metab_monotonic = time.monotonic()
                    self.viability.value = self.homeostasis.viability
                self._refresh_homeostasis_metrics()
        if metabolic_compression is not None:
            self.metabolic_compression = max(0.0, float(metabolic_compression))
            self._refresh_homeostasis_metrics()
        # Active-inference live knobs: a negative sentinel clears back to the env
        # default (None); any value >= 0 pins the per-agent override.
        if ai_intero_pref_weight is not None:
            self.ai_intero_pref_weight_override = (
                None if ai_intero_pref_weight < 0 else float(ai_intero_pref_weight)
            )
        if drive_priority_gain is not None:
            self.drive_priority_gain_override = (
                None if drive_priority_gain < 0 else float(drive_priority_gain)
            )
        if motor_babble_sigma is not None:
            self.motor_babble_sigma_override = (
                None if motor_babble_sigma < 0 else float(motor_babble_sigma)
            )
        if arch_changed:
            self._rebuild_brain()
        self._configure_plasticity(
            plasticity_alpha=plasticity_alpha,
            sparse_density=sparse_density,
            max_neurons=max_neurons,
        )
        self._refresh_perception_pipeline_metrics()
        return self.capacity_config()

    def _rebuild_brain(self) -> None:
        """Rebuild the neural bundle to the current faculties with fresh weights.

        Architecture faculties (perception-feedback loop, perception mode, encoder
        mode) change the model's module set / state_dict shape, so toggling one
        cannot mutate the live stack in place — the brain is rebuilt (reset
        semantics for cognition). Episodic and working memory are intentionally
        left intact so the agent keeps its experiences across the toggle.
        """
        self.neural = NeuralBundle.try_build(
            self.agent_id, self.preset, flags=self.plastic_flags, faculties=self.faculties
        )
        self.preset = self.neural.preset if self.neural else self.preset
        self._brain_topology_cache = None
        self.metrics["perception_mode"] = self.perception_mode
        self.metrics["discovered_perception"] = self.faculties.discovered
        self._refresh_plasticity_metrics()

    def _configure_plasticity(
        self,
        *,
        plasticity_alpha: float | None,
        sparse_density: float | None,
        max_neurons: int | None,
    ) -> None:
        b = self.neural
        stack = getattr(b, "stack", None) if b is not None else None
        if b is None or stack is None or not getattr(stack, "has_plastic", False):
            return
        if plasticity_alpha is not None:
            stack.set_alpha_all(max(0.0, float(plasticity_alpha)))
            if b.plasticity_state is not None:
                b.plasticity_state.configured_alpha = max(0.0, float(plasticity_alpha))
                b.plasticity_state.effective_alpha = min(
                    float(b.plasticity_state.effective_alpha),
                    float(b.plasticity_state.configured_alpha),
                )
                stack.set_effective_alpha_all(float(b.plasticity_state.effective_alpha))
        if sparse_density is not None:
            changed = stack.set_density_all(float(sparse_density))
            if changed:
                b.reset_optimizer_state(changed)
                if b.plasticity_state is not None:
                    b.plasticity_state.density = stack.connection_density()
                self._brain_topology_cache = None
        if max_neurons is not None and b.plasticity_state is not None:
            n = max(1, int(max_neurons))
            b.plasticity_state.max_neurons = n
            # Immediate shrink only; growth toward a raised cap is organic (the
            # per-cycle controller wakes neurons while pc-loss stays high).
            changed: list[Any] = []
            for blk in stack.plastic_blocks():
                if blk.growth and blk.awake_count() > min(n, blk.hidden_ceiling):
                    if blk.set_awake_ceiling(n):
                        changed.extend(blk.structural_params())
            if changed:
                b.reset_optimizer_state(changed)
                self._brain_topology_cache = None
        self._refresh_plasticity_metrics()

    def _refresh_hardware_metrics(self) -> None:
        self.metrics["hardware_python_executable"] = sys.executable
        self.metrics["cuda_required"] = bool(C.require_cuda())
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            self.metrics["hardware_torch_version"] = str(torch.__version__)
            self.metrics["hardware_cuda_available"] = cuda_available
            self.metrics["hardware_cuda_device"] = (
                str(torch.cuda.get_device_name(0)) if cuda_available else ""
            )
            if cuda_available:
                try:
                    self.metrics["hardware_cuda_bf16"] = bool(torch.cuda.is_bf16_supported())
                except Exception:
                    self.metrics["hardware_cuda_bf16"] = False
        except Exception as exc:
            self.metrics["hardware_torch_version"] = ""
            self.metrics["hardware_cuda_available"] = False
            self.metrics["hardware_cuda_device"] = ""
            self.metrics["cuda_warning"] = f"torch unavailable: {type(exc).__name__}"
        device = getattr(self.neural, "device", None)
        self.metrics["neural_device"] = str(device) if device is not None else "none"
        warning = ""
        if self.neural is not None and str(device) != "cuda":
            warning = "neural bundle is not using CUDA"
        if self.neural is not None and int(self.metrics.get("gpu_memory_max_allocated", 0) or 0) == 0:
            warning = warning or "torch GPU allocation is zero"
        self.metrics["cuda_warning"] = warning

    def capacity_config(self) -> dict[str, Any]:
        wm = self.perceptual.working_memory
        cfg: dict[str, Any] = {
            "parallel_sessions": self.parallel_sessions,
            "processing_mode": self.processing_mode,
            "stage_pipeline_enabled": self._stage_pipeline_enabled(),
            "perceptual_processing_mode": self.processing_mode,
            "prefetch_queue_max": int(self.prefetch_queue_max),
            "prefetch_overload_policy": self.prefetch_overload_policy,
            "ready_coalesce_policy": self.ready_coalesce_policy,
            "working_memory_slots": wm.capacity,
            "working_memory_decay": wm.decay,
            "assist_override": self.assist_override,
            "curriculum_mode": self.curriculum_mode,
            "viability_mode": self.viability_mode,
            "metabolic_compression": self.metabolic_compression,
            "ai_intero_pref_weight": self.ai_intero_pref_weight_override,
            "drive_priority_gain": self.drive_priority_gain_override,
            "motor_babble_sigma": self.motor_babble_sigma_override,
            "perception_mode": self.perception_mode,
            "perception_feedback": self.faculties.perception_feedback,
            "self_model_feedback": self.faculties.self_model_feedback,
            "predictive_affect": self.faculties.predictive_affect,
            "represented_self": self.faculties.represented_self,
            "encoder_mode": self.faculties.encoder_mode,
            "scene_dynamics_enabled": bool(self.faculties.discovered and scene_dynamics_enabled()),
            "cognition_trace": self.cognition_trace,
            "probe_capture": self.probe_capture,
            "gwt_enabled": self.gwt_enabled,
            "integration_window_ms": float(self.integration_window_ms),
            "episodic_async": bool(getattr(self.episodic, "async_enabled", False)),
            "ltm_async": bool(getattr(self.ltm_graph, "async_enabled", False)),
            "ltm_consolidation_async": bool(getattr(self.ltm_graph, "async_enabled", False)),
            "ltm_consolidation_queue_max": int(C.ltm_consolidation_queue_max()),
            "ltm_semantic_evidence_interval": int(C.ltm_semantic_evidence_interval()),
            "ltm_scene_edge_max_per_job": int(C.ltm_scene_edge_max_per_job()),
            "ltm_match_cache_enabled": bool(C.ltm_match_cache_enabled()),
            "ltm_match_recent_cap": int(C.ltm_match_recent_cap()),
            "ltm_match_salient_cap": int(C.ltm_match_salient_cap()),
        }
        b = self.neural
        stack = getattr(b, "stack", None) if b is not None else None
        has_plastic = bool(b is not None and stack is not None and getattr(stack, "has_plastic", False))
        cfg["plasticity"] = {
            "available": has_plastic,
            "plasticity_enabled": bool(has_plastic and b.flags.plastic),
            "sparse_enabled": bool(has_plastic and b.flags.sparse),
            "growth_enabled": bool(has_plastic and b.flags.growth),
            "plasticity_alpha": round(stack.plastic_alpha_mean(), 6) if has_plastic else 0.0,
            "plasticity_alpha_effective": round(stack.plastic_effective_alpha_mean(), 6)
            if has_plastic
            else 0.0,
            "sparse_density": round(stack.connection_density(), 6) if has_plastic else 1.0,
            "awake_neurons": stack.awake_neurons() if has_plastic else 0,
            "allocated_neurons": stack.allocated_neurons() if has_plastic else 0,
            "max_neurons": int(b.plasticity_state.max_neurons)
            if (has_plastic and b.plasticity_state is not None)
            else 0,
        }
        return cfg

    def _resolve_backups_dir(self) -> Path:
        default = Path(__file__).resolve().parents[2] / "backups"
        return Path(os.environ.get("DECADIC_BACKUPS_DIR", default))

    def _check_death(self) -> None:
        """Trip the mortality transition once any reservoir bottoms out."""
        if self.viability_mode == "immortal":
            return
        if self.status == "alive" and self.homeostasis.viability <= self.homeostasis.min_value:
            self.die()

    def die(self) -> None:
        """Existential cessation: freeze the mind, write a tombstone, emit death."""
        if self.status == "dead":
            return
        self.status = "dead"
        self.died_at_cycle = int(self.state_bus.cycle_index)
        # Close any open goal episode as "died" so the journey still earns credit
        # (truncated returns) and can feed hindsight relabeling -- the body failed,
        # but the path it took is still a real example of how it moved.
        try:
            death_events = self.goal_state.update(
                self._reservoirs_norm(), int(self.state_bus.cycle_index), alive=False
            )
            self._accumulate_episode(None, death_events)
        except Exception:  # death handling must never raise
            pass
        self._write_tombstone()
        try:
            self.out_queue.put_nowait(
                {
                    "type": "death",
                    "agent_id": self.agent_id,
                    "cycle": self.died_at_cycle,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        except asyncio.QueueFull:
            pass
        logger.info(
            "agent_death agent_id=%s cycle=%s viability=%.4f",
            self.agent_id,
            self.died_at_cycle,
            self.viability.value,
        )

    def _write_tombstone(self) -> None:
        """Persist frozen weights + final snapshot for study / reincarnation."""
        try:
            backups_dir = self._resolve_backups_dir()
            backups_dir.mkdir(parents=True, exist_ok=True)
            brain_name = self.save_brain(backups_dir)
            payload = self.checkpoint_payload()
            payload["status"] = "dead"
            payload["died_at_cycle"] = self.died_at_cycle
            payload["neural_brain"] = brain_name
            payload["tombstone"] = True
            path = backups_dir / f"agent_{self.agent_id}_tombstone.json"
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            self._prune_tombstones(backups_dir)
        except Exception:
            logger.exception("tombstone_write_failed agent_id=%s", self.agent_id)

    def _prune_tombstones(self, backups_dir: Path) -> None:
        """Keep only the N most-recent dead-agent brain tombstones; delete older sets.

        Brain ``.pt`` files are only ever written on death, so pruning by mtime can
        never touch a live agent's data. The just-written tombstone is always newest.
        """
        keep = tombstone_keep()
        try:
            brains = sorted(
                backups_dir.glob("agent_*_brain.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            removed = 0
            for stale in brains[keep:]:
                stale_id = stale.name[len("agent_") : -len("_brain.pt")]
                for suffix in ("_brain.pt", "_tombstone.json", "_checkpoint.json"):
                    (backups_dir / f"agent_{stale_id}{suffix}").unlink(missing_ok=True)
                removed += 1
            if removed:
                logger.info("tombstone_prune kept=%d removed=%d", min(keep, len(brains)), removed)
        except Exception:
            logger.exception("tombstone_prune_failed agent_id=%s", self.agent_id)

    def revive(self, restore_to: float | None = None) -> None:
        """Admin resurrection: same weights/state, viability restored, cycling resumes."""
        if self.status != "dead":
            return
        self.status = "alive"
        self.died_at_cycle = None
        target = DEFAULT_REVIVE_VIABILITY if restore_to is None else float(restore_to)
        target = float(
            min(self.homeostasis.max_value, max(self.homeostasis.min_value + 1e-6, target))
        )
        self.homeostasis.reset(target)
        self.viability.value = self.homeostasis.viability
        self.stress = 0.0
        self._threat_stress = 0.0
        self._last_metab_monotonic = time.monotonic()
        self._refresh_homeostasis_metrics()
        # A new life: re-pose the body upright (preserving earned ROM) and scatter
        # resources to fresh positions so the agent cannot camp a known location.
        # The mind state (weights, memory, curriculum) is untouched -- only the
        # body pose and the world's resource layout reset.
        if randomize_resources_enabled():
            seed = random.randint(0, 2**31 - 1)
            self.queue_body_command("recenter")
            if self.queue_body_command(f"randomize_resources:{seed}"):
                self.metrics["resource_seed"] = seed
        # Drop any dangling open episode from the prior life (death already closed
        # it); the new life starts the goal timeline fresh.
        self._episode_acc.reset()
        self.ensure_cycle_worker()

    async def reset(self, preset: str | None = None) -> None:
        """Fresh mind: new weights, zeroed state bus / viability / perception, wiped episodes."""
        restart_cycle = await self.suspend_cycle_worker()
        if self._memory_context_task is not None and not self._memory_context_task.done():
            self._memory_context_task.cancel()
            try:
                await self._memory_context_task
            except asyncio.CancelledError:
                pass
            self._memory_context_task = None
        async with self.lock:
            self.state_bus = StateBus()
            self.perceptual = PerceptualState(perception_mode=self.perception_mode)
            self.perceptual_integrator = PerceptualIntegrator()
            self.viability = ViabilityState()
            self.homeostasis = Homeostasis()
            self.stress = 0.0
            self._threat_stress = 0.0
            self._last_metab_monotonic = None
            self.episodic.clear()
            if self.ltm_graph is not None:
                self.ltm_graph.clear()
            self.neural = NeuralBundle.try_build(
                self.agent_id, preset or self.preset, flags=self.plastic_flags, faculties=self.faculties
            )
            self.preset = self.neural.preset if self.neural else None
            self._last_observation = None
            self._memory_context_vector = None
            self._memory_context_query = None
            self._memory_context_refresh_cycle = 0
            self._memory_context_worker_ms = 0.0
            self._cycle_deadline_s = time.perf_counter()
            self._wait_for_observation_after_reset = True
            self._debug_views = {}
            self._last_cycle_trace = None
            self._last_cognitive_trace = None
            self._curiosity_investigating = False
            self._last_landscape = None
            self._landscape_probe = None
            # Fresh mind: wipe the goal timeline and any open episode so credit
            # assignment starts from a clean slate alongside the new weights.
            self.goal_state = GoalState(
                onset_deficit=goal_onset_deficit(),
                satisfy_level=goal_satisfy_level(),
                abandon_cycles=goal_abandon_cycles(),
                max_cycles=goal_max_cycles(),
            )
            self._episode_acc = EpisodeAccumulator(gamma=sf_gamma(), lam=sf_lambda())
            self._cognitive_history.clear()
            self._obs_buffer.clear()
            self._clear_perception_pipeline()
            self.stage_pipeline.clear()
            self._runtime_perception_organ = PerceptionOrgan()
            self._loco_last_xy = None
            self._loco_origin_xy = None
            self._loco_distance = 0.0
            self._loco_fall_buf.clear()
            self._loco_foot_phase_buf.clear()
            self.status = "alive"
            self.died_at_cycle = None
            self.metrics = self._initial_metrics()
            self._refresh_hardware_metrics()
            self._refresh_perception_pipeline_metrics()
            self._refresh_homeostasis_metrics()
            self._refresh_plasticity_metrics()
            self._brain_topology_cache = None
            self._started_perf = time.perf_counter()
            while not self.out_queue.empty():
                try:
                    self.out_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            while not self.control_queue.empty():
                try:
                    self.control_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        self.resume_cycle_worker(restart_cycle)

    async def stop(self) -> None:
        self.running = False
        if self._consolidation_task is not None:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
        if self._landscape_task is not None:
            self._landscape_task.cancel()
            try:
                await self._landscape_task
            except asyncio.CancelledError:
                pass
        if self._metabolic_task is not None:
            self._metabolic_task.cancel()
            try:
                await self._metabolic_task
            except asyncio.CancelledError:
                pass
        for task in self._perception_workers:
            task.cancel()
        if self._perception_workers:
            await asyncio.gather(*self._perception_workers, return_exceptions=True)
        self._perception_workers = []
        await self.stage_pipeline.stop()
        if self._cycle_task is not None:
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                pass
        # Drain + stop the write-behind episodic worker (no-op for the sync store).
        close = getattr(self.episodic, "close", None)
        if callable(close):
            close()
        # Drain + stop the write-behind LTM worker (no-op when disabled / sync graph).
        if self.ltm_graph is not None:
            close_ltm = getattr(self.ltm_graph, "close", None)
            if callable(close_ltm):
                close_ltm()

    def save_brain(self, backups_dir: Path) -> str | None:
        if self.neural is None:
            return None
        path = backups_dir / f"agent_{self.agent_id}_brain.pt"
        self.neural.save(path)
        return path.name

    def load_brain(self, backups_dir: Path) -> None:
        if self.neural is None:
            return
        path = backups_dir / f"agent_{self.agent_id}_brain.pt"
        if path.is_file():
            try:
                self.neural.load(path)
            except ValueError as e:
                logger.warning("skipping brain checkpoint for %s: %s", self.agent_id, e)

    def save_brain_to(self, path: Path) -> str | None:
        """Save the neural bundle to an explicit ``path`` (Saved Agents library).

        Unlike :meth:`save_brain`, the filename is caller-chosen so the save is
        not coupled to the volatile agent id. Returns ``None`` when the agent
        has no neural stack.
        """
        if self.neural is None:
            return None
        self.neural.save(path)
        return path.name

    def load_brain_from(self, path: Path) -> None:
        """Load a neural bundle from an explicit ``path``.

        Lets a preset/architecture mismatch raise (``ValueError``) so the caller
        can surface it; callers should align the preset first (e.g. via
        :meth:`reset`).
        """
        if self.neural is None:
            raise RuntimeError("agent has no neural stack to load into")
        self.neural.load(path)

    def backup_memory_to(self, path: Path) -> None:
        """Snapshot episodic memory to an explicit sqlite ``path``."""
        self.episodic.backup_to(path)

    def restore_memory_from(self, path: Path) -> None:
        """Replace episodic memory with the sqlite snapshot at ``path``."""
        self.episodic.restore_from(path)

    def backup_graph_to(self, path: Path) -> None:
        """Snapshot the long-term knowledge graph to an explicit sqlite ``path`` (no-op if disabled)."""
        if self.ltm_graph is not None:
            self.ltm_graph.backup_to(path)

    def restore_graph_from(self, path: Path) -> None:
        """Replace the long-term knowledge graph with the sqlite snapshot at ``path`` (no-op if disabled/missing)."""
        if self.ltm_graph is not None:
            self.ltm_graph.restore_from(path)

    def _apply_cycle_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        pe = float(
            diagnostics.get(
                "prediction_error_delta",
                diagnostics.get("stub_prediction_error_delta", 0.0),
            )
        )
        self.metrics["prediction_error_last"] = pe
        self.metrics["prediction_error_stub_last"] = pe  # legacy alias
        mag = abs(pe)
        alpha = float(
            os.environ.get(
                "DECADIC_PE_EMA_ALPHA", os.environ.get("DECADIC_PE_STUB_EMA_ALPHA", "0.08")
            )
        )
        prev = float(self.metrics.get("prediction_error_ema", 0.0))
        ema = (1.0 - alpha) * prev + alpha * mag
        self.metrics["prediction_error_ema"] = ema
        self.metrics["prediction_error_stub_ema"] = ema  # legacy alias
        rw = float(
            diagnostics.get("drive_reward_delta", diagnostics.get("stub_reward_delta", 0.0))
        )
        self.metrics["drive_reward_last"] = rw
        self.metrics["reward_stub_last"] = rw  # legacy alias
        self.metrics["last_stage_timing_ms_total"] = float(
            diagnostics.get("stage_timing_ms_total", 0.0)
        )
        if "neural_pc_loss" in diagnostics:
            self.metrics["neural_pc_loss_last"] = float(diagnostics["neural_pc_loss"])
        for key in (
            "loss_total",
            "loss_dominant_fraction",
            "loss_canary_pressure",
            "loss_canary_step_scale",
            "loss_canary_slope_ema",
            "loss_canary_pc_slope_ema",
            "loss_canary_jump_ratio",
            "drive_priority_gain_configured",
            "drive_priority_gain_effective",
        ):
            if diagnostics.get(key) is not None:
                self.metrics[key] = float(diagnostics[key])
        for key in (
            "loss_dominant_term",
            "loss_canary_state",
            "loss_canary_reason",
            "loss_canary_optimizer_action",
        ):
            if diagnostics.get(key) is not None:
                self.metrics[key] = str(diagnostics[key])
        for key in ("loss_canary_ema", "loss_canary_pc_ema"):
            self.metrics[key] = None if diagnostics.get(key) is None else float(diagnostics[key])
        if isinstance(diagnostics.get("loss_terms"), dict):
            self.metrics["loss_terms"] = diagnostics["loss_terms"]
        if diagnostics.get("learning_rate") is not None:
            self.metrics["learning_rate"] = float(diagnostics["learning_rate"])
        if diagnostics.get("gpu_memory_max_allocated") is not None:
            self.metrics["gpu_memory_max_allocated"] = int(
                diagnostics["gpu_memory_max_allocated"]
            )
        if diagnostics.get("memory_recall_ms") is not None:
            self.metrics["memory_recall_ms"] = float(diagnostics["memory_recall_ms"])
        if diagnostics.get("memory_recall_on_critical_path") is not None:
            self.metrics["memory_recall_on_critical_path"] = bool(
                diagnostics["memory_recall_on_critical_path"]
            )
        for key in ("parallel_sessions", "working_memory_slots"):
            if key == "parallel_sessions" and self._perception_pipeline_enabled():
                self.metrics[key] = int(self.parallel_sessions)
            elif diagnostics.get(key) is not None:
                self.metrics[key] = int(diagnostics[key])
        if diagnostics.get("encode_phase_ms") is not None:
            self.metrics["encode_phase_ms"] = float(diagnostics["encode_phase_ms"])
        self._refresh_perception_pipeline_metrics()
        # Embodied motor learning (active inference) telemetry.
        for key in (
            "forward_model_error",
            "tactile_pred_error",
            "effort_pred_error",
            "assist_gain",
            "motor_babble_sigma",
            "motor_activity_rms",
            "curiosity_drive",
            "curiosity_pleasure",
            "curiosity_learning_progress",
            # Successor-features value telemetry (Layer-2 incentive salience).
            "sf_value",
            "sf_value_weight",
        ):
            if diagnostics.get(key) is not None:
                self.metrics[key] = float(diagnostics[key])
        if isinstance(diagnostics.get("motor_command"), list):
            self.metrics["motor_command"] = [float(x) for x in diagnostics["motor_command"]]
        self._refresh_hardware_metrics()
        self._refresh_memory_recall_metrics()
        # NaN firewall telemetry: count cycles the firewall recovered from.
        nan_recovery = bool(diagnostics.get("nan_recovery", False))
        self.metrics["nan_recovery_last"] = nan_recovery
        if nan_recovery:
            self.metrics["nan_recovery_events"] = (
                int(self.metrics.get("nan_recovery_events", 0)) + 1
            )
        # Neuroplasticity (A/B/C) telemetry straight from the per-cycle hook.
        for key in (
            "plasticity_alpha",
            "sparse_density",
            "awake_neurons",
            "allocated_neurons",
            "active_connections",
            "rewire_events",
            "growth_events",
            "plasticity_alpha_configured",
            "plasticity_alpha_effective",
            "plasticity_pc_ema",
            "plasticity_pc_slope_ema",
            "plasticity_guardian_state",
            "plasticity_guardian_action",
            "plasticity_warmup_blocked_reason",
            "plasticity_stable_cycles",
            "plasticity_freeze_count",
            "plasticity_thaw_count",
            "plasticity_frozen_since_cycle",
            "plasticity_last_thaw_cycle",
            "plasticity_thaw_eligible",
            "plasticity_thaw_cycles_remaining",
            "plasticity_overlay_ratio_mean",
            "plasticity_overlay_ratio_max",
        ):
            if diagnostics.get(key) is not None:
                self.metrics[key] = diagnostics[key]
        if diagnostics.get("plasticity_frozen") is not None:
            self.metrics["plasticity_frozen"] = bool(diagnostics["plasticity_frozen"])
        # Persist structural-plasticity events to the server log. These are
        # edge-triggered (True only on the cycle the event fires), so they are
        # naturally low-volume: rewires every ~250 cycles, growth every ~500, a
        # freeze at most once. Without this the timeline lives only in the
        # in-memory metrics and is lost when the server stops.
        cyc = int(self.state_bus.cycle_index)
        if diagnostics.get("rewired"):
            logger.info(
                "plasticity_rewire agent_id=%s cycle=%s connections=%s active=%s "
                "density=%.4f total_rewires=%s",
                self.agent_id,
                cyc,
                int(diagnostics.get("connections_rewired", 0)),
                int(diagnostics.get("active_connections", 0)),
                float(diagnostics.get("sparse_density", 0.0)),
                int(diagnostics.get("rewire_events", 0)),
            )
        if diagnostics.get("grew"):
            logger.info(
                "plasticity_growth agent_id=%s cycle=%s neurons_woken=%s awake=%s "
                "allocated=%s total_grows=%s",
                self.agent_id,
                cyc,
                int(diagnostics.get("neurons_woken", 0)),
                int(diagnostics.get("awake_neurons", 0)),
                int(diagnostics.get("allocated_neurons", 0)),
                int(diagnostics.get("growth_events", 0)),
            )
        if diagnostics.get("froze"):
            logger.warning(
                "plasticity_frozen agent_id=%s cycle=%s reason=%s "
                "pc_loss=%.6f effective_alpha=%.6f total_rewires=%s total_grows=%s",
                self.agent_id,
                cyc,
                str(diagnostics.get("plasticity_freeze_reason") or "pc_ema_diverged_or_nonfinite"),
                float(diagnostics.get("neural_pc_loss", 0.0) or 0.0),
                float(diagnostics.get("plasticity_alpha_effective", 0.0) or 0.0),
                int(diagnostics.get("rewire_events", 0)),
                int(diagnostics.get("growth_events", 0)),
            )
        # Edge-triggered curiosity event log. The curiosity drive is a continuous
        # per-cycle signal (in metrics/telemetry), so logging it every cycle would
        # flood the log; instead we record only the transitions into and out of the
        # curiosity-driven "investigate" priority -- the moments curiosity actually
        # takes the wheel. Naturally inert when curiosity is off (the label is never
        # "investigate"), so the baseline log stays byte-identical.
        investigating = self.state_bus.priority_label == "investigate"
        if investigating != self._curiosity_investigating:
            self._curiosity_investigating = investigating
            if investigating:
                logger.info(
                    "curiosity_investigate_enter agent_id=%s cycle=%s drive=%.4f "
                    "learning_progress=%.4f pleasure=%.4f",
                    self.agent_id,
                    cyc,
                    float(diagnostics.get("curiosity_drive", 0.0)),
                    float(diagnostics.get("curiosity_learning_progress", 0.0)),
                    float(diagnostics.get("curiosity_pleasure", 0.0)),
                )
            else:
                logger.info(
                    "curiosity_investigate_exit agent_id=%s cycle=%s drive=%.4f "
                    "learning_progress=%.4f",
                    self.agent_id,
                    cyc,
                    float(diagnostics.get("curiosity_drive", 0.0)),
                    float(diagnostics.get("curiosity_learning_progress", 0.0)),
                )
        # Optional periodic snapshot (env DECADIC_PLASTICITY_LOG_EVERY; 0 = off):
        # a continuous time series of structural state for trend analysis. Gated
        # on a plastic stack (plasticity_frozen is present only then) so it is a
        # no-op cost when disabled or when the stack is dense.
        log_every = plasticity_log_every()
        if (
            log_every > 0
            and diagnostics.get("plasticity_frozen") is not None
            and cyc % log_every == 0
        ):
            logger.info(
                "plasticity_snapshot agent_id=%s cycle=%s awake=%s allocated=%s "
                "active=%s density=%.4f alpha=%.6f effective_alpha=%.6f "
                "state=%s action=%s rewires=%s grows=%s frozen=%s",
                self.agent_id,
                cyc,
                int(diagnostics.get("awake_neurons", 0)),
                int(diagnostics.get("allocated_neurons", 0)),
                int(diagnostics.get("active_connections", 0)),
                float(diagnostics.get("sparse_density", 0.0)),
                float(diagnostics.get("plasticity_alpha_configured", 0.0) or 0.0),
                float(diagnostics.get("plasticity_alpha_effective", 0.0) or 0.0),
                str(diagnostics.get("plasticity_guardian_state", "")),
                str(diagnostics.get("plasticity_guardian_action", "")),
                int(diagnostics.get("rewire_events", 0)),
                int(diagnostics.get("growth_events", 0)),
                bool(diagnostics.get("plasticity_frozen", False)),
            )
        # Perception feedback loop telemetry (gate/pred-error are None when off).
        if diagnostics.get("perception_feedback") is not None:
            self.metrics["perception_feedback"] = bool(diagnostics["perception_feedback"])
        for key in ("precision_gate_mean", "perceptual_pred_error"):
            if diagnostics.get(key) is not None:
                self.metrics[key] = float(diagnostics[key])
        # Homeostatic drive-reduction telemetry (drive/pred-error are None when off).
        if diagnostics.get("homeostatic_drive") is not None:
            self.metrics["homeostatic_drive"] = bool(diagnostics["homeostatic_drive"])
        for key in ("intero_drive", "intero_pred_error"):
            if diagnostics.get(key) is not None:
                self.metrics[key] = float(diagnostics[key])
        # Discovered-perception telemetry (object discovery + body-self agency).
        if diagnostics.get("perception_mode") is not None:
            self.metrics["perception_mode"] = str(diagnostics["perception_mode"])
        if diagnostics.get("discovered_perception") is not None:
            self.metrics["discovered_perception"] = bool(diagnostics["discovered_perception"])
        for key in ("slots_present", "discovered_objects", "self_parts"):
            if diagnostics.get(key) is not None:
                self.metrics[key] = int(diagnostics[key])
        for key in ("slot_recon_error", "agency_mean", "agency_loss"):
            if diagnostics.get(key) is not None:
                self.metrics[key] = float(diagnostics[key])
        # A structural change (rewire/grow) must invalidate the cached topology so
        # the Brain Map re-reads the new awake-neuron ring on its next poll.
        if diagnostics.get("structural_change"):
            self._brain_topology_cache = None
        # Reflect the current manual override (None = automatic schedule) every cycle.
        self.metrics["assist_override"] = self.assist_override
        self._refresh_homeostasis_metrics()

    def _maybe_dump_cycle_trace(
        self, outbound: dict[str, Any], diagnostics: dict[str, Any]
    ) -> None:
        every = int(os.environ.get("DECADIC_CYCLE_TRACE_EVERY", "0"))
        if every <= 0 or self.state_bus.cycle_index % every != 0:
            return
        base = os.environ.get("DECADIC_LOG_DIR")
        if not base:
            return
        path = Path(base) / f"cycle_trace_{self.agent_id}.jsonl"
        record = {
            "cycle": self.state_bus.cycle_index,
            "agent_id": self.agent_id,
            "outbound": outbound,
            "diagnostics": diagnostics,
        }
        # Build the line on-cycle (content unchanged) but hand the file append to the
        # background JSONL writer so the disk write leaves the cognitive critical path.
        get_jsonl_writer().append(path, json.dumps(record, default=str))

    def _reservoirs_norm(self) -> tuple[float, float, float]:
        """Homeostatic reservoirs normalized to 0..1 (1.0 == full), goal-label order."""
        h = self.homeostasis
        span = max(1e-6, h.max_value - h.min_value)
        return (
            (h.hydration - h.min_value) / span,
            (h.energy - h.min_value) / span,
            (h.integrity - h.min_value) / span,
        )

    def _advance_goal(self, transition: "dict[str, Any] | None") -> list:
        """Advance the explicit goal lifecycle one cycle and surface telemetry.

        Feeds the normalized reservoir vector to the GoalState (which latches /
        holds / closes the dominant deficit as the active goal) and mirrors the
        result into metrics. Returns the lifecycle events; the episodic-replay
        pathway consumes them to open/close return-annotated episodes.
        """
        gs = self.goal_state
        cycle = int(self.state_bus.cycle_index)
        events = gs.update(self._reservoirs_norm(), cycle, alive=(self.status == "alive"))
        self.metrics["goal"] = gs.goal_id or "none"
        self.metrics["goal_status"] = gs.status
        self.metrics["goal_dwell"] = gs.dwell(cycle)
        self.metrics["goal_episodes"] = gs.episodes
        if gs.last_outcome:
            self.metrics["goal_last_outcome"] = gs.last_outcome
        return events

    def _accumulate_episode(self, transition: Any, events: list) -> None:
        """Route the current transition + lifecycle events through the episode timeline.

        The current step is added to whatever episode is open FIRST (so an
        achieving step's reward lands inside the episode it completes), then the
        events are applied: a close annotates the ordered episode with
        lambda-returns, an open starts the next one.
        """
        acc = self._episode_acc
        acc.add(transition)
        for ev in events:
            if ev.kind == "closed":
                closed = acc.on_close(ev.outcome or "")
                self._on_episode_closed(closed, ev.outcome or "")
            elif ev.kind == "opened":
                acc.on_open(ev.goal_id, ev.onset_cycle)

    def _on_episode_closed(self, steps: list, outcome: str) -> None:
        """Surface closed-episode stats; hindsight relabeling hooks in here (Phase 5)."""
        acc = self._episode_acc
        self.metrics["episodes_closed"] = acc.episodes_closed
        self.metrics["episode_last_len"] = acc.last_len
        self.metrics["episode_last_return"] = round(acc.last_return, 5)
        self._maybe_hindsight_relabel(steps, outcome)

    def _maybe_hindsight_relabel(self, steps: list, outcome: str) -> None:
        """Hindsight-relabel a failed/abandoned episode that still found relief.

        A non-achieved episode whose trajectory nonetheless accrued reservoir gain
        in some channel (e.g. latched on thirst but actually ate) is relabeled with
        that achieved terminal feature and re-pushed (with recomputed SF/return
        targets) so the SF head trains more on the real, off-goal relief it found --
        the literal "the journey still taught me." Gated; no fabricated reward.
        """
        if outcome == "achieved" or not steps:
            return
        if self.replay_buffer is None or not her_enabled() or her_relabel_k() <= 0:
            return
        achieved = achieved_feature(steps)
        if not achieved or max(achieved) <= 1e-6:
            return  # nothing was actually achieved -> no hindsight success to learn
        copies = build_hindsight_copies(
            steps, achieved, gamma=sf_gamma(), lam=sf_lambda(), k=her_relabel_k()
        )
        pushed = sum(1 for c in copies if self.replay_buffer.push(c))
        if pushed:
            self.metrics["her_relabels"] = int(self.metrics.get("her_relabels", 0)) + pushed
            self.metrics["her_last"] = pushed

    async def _cycle_loop(self) -> None:
        while self.running:
            scheduler = C.cycle_scheduler_mode()
            self.metrics["cycle_scheduler_mode"] = scheduler
            self.metrics["cycle_interval_ms"] = float(self.cycle_interval_s * 1000.0)
            if scheduler == "fixed_sleep":
                await asyncio.sleep(self.cycle_interval_s)
                cycle_start_target = time.perf_counter()
                self.metrics["cycle_idle_ms"] = float(self.cycle_interval_s * 1000.0)
            else:
                now_sched = time.perf_counter()
                if self._cycle_deadline_s <= 0:
                    self._cycle_deadline_s = now_sched
                idle_s = max(0.0, self._cycle_deadline_s - now_sched)
                if idle_s > 0:
                    await asyncio.sleep(idle_s)
                cycle_start_target = self._cycle_deadline_s
                self.metrics["cycle_idle_ms"] = idle_s * 1000.0
            # A dead mind is frozen; only an admin revive/reset reanimates it.
            if self.paused or self.status == "dead":
                await asyncio.sleep(self.cycle_interval_s)
                self._cycle_deadline_s = time.perf_counter() + self.cycle_interval_s
                continue
            t0 = time.perf_counter()
            selected_session: DecadicSession | None = None
            if self._serial_prefetch_enabled():
                selected_session, _commit_bundle = await self.stage_pipeline.pop_commit_candidate()
                if selected_session is None:
                    self._refresh_stage_pipeline_metrics()
                    m = self.stage_pipeline.metrics()
                    active = int(m.get("active_sessions", 0) or 0) + int(
                        m.get("ready_sessions", 0) or 0
                    )
                    if active > 0 or self._last_observation is not None:
                        await asyncio.sleep(0.001)
                        continue
            async with self.lock:
                if self._serial_prefetch_enabled():
                    pending = []
                    latest_observation = selected_session.observation if selected_session else None
                elif self.processing_mode == PROCESSING_BATCHING:
                    pending = list(self._obs_buffer)
                    self._obs_buffer.clear()
                    latest_observation = self._last_observation
                else:
                    pending = []
                    latest_observation = self._last_observation
                if (
                    self._wait_for_observation_after_reset
                    and not pending
                    and latest_observation is None
                ):
                    await asyncio.sleep(0.001)
                    continue
                self._wait_for_observation_after_reset = False
                ctx = CycleContext(
                    state_bus=self.state_bus,
                    perceptual=self.perceptual,
                    viability=self.viability,
                    episodic=self.episodic,
                    ltm_graph=self.ltm_graph,
                    homeostasis=self.homeostasis,
                    last_observation=latest_observation,
                    pending_observations=pending,
                    perceptual_processing_mode=self.processing_mode,
                    assist_override=self.assist_override,
                    curriculum_mode=self.curriculum_mode,
                    perception_mode=self.perception_mode,
                    cognition_trace=self.cognition_trace,
                    probe_capture=self.probe_capture,
                    gwt_enabled=self.gwt_enabled,
                    integration_window_ms=self.integration_window_ms,
                    ai_intero_pref_weight=self.ai_intero_pref_weight_override,
                    drive_priority_gain=self.drive_priority_gain_override,
                    motor_babble_sigma=self.motor_babble_sigma_override,
                    cached_memory_context=self._cached_memory_context_for_cycle(),
                    memory_recall_on_critical_path=not C.memory_context_async_enabled(),
                )
                if self.neural is not None:
                    msg = run_neural_cycle(ctx, self.neural)
                else:
                    msg = run_stub_cycle(ctx)
                diagnostics = msg.pop("_diagnostics", {})
                cognitive = msg.pop("_cognitive", None)
                transition = msg.pop("_transition", None)
                # Dual-network consolidation: feed the realized transition to the
                # replay buffer (no-op unless the feature is enabled).
                tr = None
                if transition is not None and self.replay_buffer is not None:
                    if selected_session is not None:
                        transition = {
                            **transition,
                            "session_id": selected_session.session_id,
                            "frame_seq": selected_session.frame_seq,
                            "commit_index": self._commit_index,
                            "stage_timings": dict(selected_session.timings_ms),
                            "snapshot_versions": dict(
                                selected_session.to_dict().get("snapshots", {})
                            ),
                            "selected_status": "selected",
                        }
                    dojo = self.dojo_training if isinstance(self.dojo_training, dict) else None
                    if dojo is not None:
                        transition = {
                            **transition,
                            "skill_id": str(dojo.get("skill_id", "")),
                            "origin": str(dojo.get("origin", "self")),
                            "expert_motor": dojo.get("expert_motor"),
                            "demo_weight": float(dojo.get("demo_weight", 0.0) or 0.0),
                            "success": bool(dojo.get("success", False)),
                        }
                        self.metrics["teacher_override_fraction"] = max(
                            0.0, min(1.0, float(dojo.get("demo_weight", 0.0) or 0.0))
                        )
                    tr = Transition(**transition)
                    self.replay_buffer.push(tr)
                # Advance the explicit goal lifecycle (telemetry) and route the
                # transition through the goal-episode accumulator so closed episodes
                # get lambda-return credit assignment (in-place on the same buffer
                # refs -- no second push, existing one-step consolidation unchanged).
                events = self._advance_goal(transition)
                self._accumulate_episode(tr, events)
                outbound = dict(msg)
                # Piggyback the agent's reservoir levels (normalized 0..1) onto the
                # outbound action so the body's scripted parent can provision on a
                # need threshold. This is body-only telemetry; cognition already
                # produced the action and never reads this field back.
                act = outbound.get("action")
                if isinstance(act, dict) and isinstance(act.get("parameters"), dict):
                    h = self.homeostasis
                    span = max(1e-6, h.max_value - h.min_value)
                    act = dict(act)
                    params = dict(act["parameters"])
                    params["reservoirs"] = {
                        "hydration": round((h.hydration - h.min_value) / span, 4),
                        "energy": round((h.energy - h.min_value) / span, 4),
                        "integrity": round((h.integrity - h.min_value) / span, 4),
                    }
                    act["parameters"] = params
                    outbound["action"] = act
                self._apply_live_teacher(outbound)
                trace = outbound.get("trace")
                if isinstance(trace, list):
                    self._last_cycle_trace = {
                        "cycle": int(self.state_bus.cycle_index),
                        "stages": trace,
                    }
                if isinstance(cognitive, dict):
                    self._last_cognitive_trace = cognitive
                    self._cognitive_history.append(_compact_cognitive(cognitive))
                self._check_death()
                self._state_bus_version += 1
                self._workspace_version += 1
                self._weight_version += 1
                self._commit_index += 1
            wall_ms = (time.perf_counter() - t0) * 1000.0
            if scheduler == "deadline":
                self._cycle_deadline_s = cycle_start_target + self.cycle_interval_s
                while self._cycle_deadline_s < time.perf_counter() - self.cycle_interval_s:
                    self._cycle_deadline_s += self.cycle_interval_s
                self.metrics["cycle_overrun_ms"] = max(
                    0.0, (time.perf_counter() - (cycle_start_target + self.cycle_interval_s)) * 1000.0
                )
            else:
                self.metrics["cycle_overrun_ms"] = max(0.0, wall_ms - self.cycle_interval_s * 1000.0)
            self.metrics["cycle_compute_ratio"] = wall_ms / max(1e-6, self.cycle_interval_s * 1000.0)
            elapsed = time.perf_counter() - self._started_perf
            self.metrics["cycles_completed"] = int(self.state_bus.cycle_index)
            self.metrics["last_cycle_wall_ms"] = wall_ms
            if elapsed > 1e-6:
                self.metrics["approx_cycles_per_sec"] = self.state_bus.cycle_index / elapsed
            self._apply_cycle_diagnostics(diagnostics)
            self._maybe_schedule_memory_context_refresh(
                diagnostics.get("memory_query_vector"), int(self.state_bus.cycle_index)
            )
            if selected_session is not None:
                act_type = outbound.get("action", {}).get("type") if isinstance(outbound, dict) else None
                await self.stage_pipeline.mark_committed(
                    selected_session.session_id,
                    action_type=str(act_type) if act_type is not None else None,
                )
                self._refresh_stage_pipeline_metrics()
            self._maybe_dump_cycle_trace(outbound, diagnostics)
            if cycle_profile_enabled():
                logger.info(
                    "cycle_profile agent_id=%s cycle=%s total_ms=%.2f encoders_ms=%.2f "
                    "fwd_ms=%.2f bwd_ms=%.2f mem_recall_ms=%.2f stage10_ms=%.2f "
                    "pc_loss=%.6f gpu_mem_mb=%.1f",
                    self.agent_id,
                    self.state_bus.cycle_index,
                    wall_ms,
                    float(diagnostics.get("encoders_ms", 0.0)),
                    float(diagnostics.get("neural_forward_ms", 0.0)),
                    float(diagnostics.get("neural_backward_ms", 0.0)),
                    float(diagnostics.get("memory_recall_ms", 0.0)),
                    float(diagnostics.get("stage10_ms", 0.0)),
                    float(diagnostics.get("neural_pc_loss", 0.0) or 0.0),
                    float(diagnostics.get("gpu_memory_max_allocated", 0)) / (1024.0 * 1024.0),
                )
            logger.info(
                "cycle_completed agent_id=%s cycle=%s wall_ms=%.3f action_type=%s",
                self.agent_id,
                self.state_bus.cycle_index,
                wall_ms,
                outbound.get("action", {}).get("type"),
            )
            if not self._put_outbound(outbound, drop_oldest=False):
                logger.warning(
                    "out_queue_full_drop agent_id=%s cycle=%s",
                    self.agent_id,
                    self.state_bus.cycle_index,
                )
            await asyncio.sleep(0)

    async def handle_observation_dict(self, obs: dict[str, Any]) -> None:
        """Integrate observation under lock and apply collision fast-path."""
        # Spectator camera frames are dashboard-only: keep them out of cognition/memory.
        views = obs.pop("debug_views", None)
        if isinstance(views, dict):
            self._debug_views = {
                str(k): v for k, v in views.items() if isinstance(v, str)
            }

        events = obs.get("events") or []
        if not isinstance(events, list):
            events = []

        effects = classify_events(events, FAST_PATH_COLLISION_THRESHOLD)
        damage = effects["integrity_damage"]
        threat_damage = effects.get("threat_damage", 0.0)
        energy_gain = effects["energy_gain"]
        hydration_gain = effects["hydration_gain"]
        threat = effects["stress"]
        prop = obs.get("proprioception") if isinstance(obs, dict) else None
        effort = normalize_effort(prop.get("effort") if isinstance(prop, dict) else None)
        body_map = normalize_body_map(prop.get("body_map") if isinstance(prop, dict) else None)
        effort_cost = 0.0
        fatigue_pain = 0.0
        strain_pain = 0.0
        if effort_drain_enabled():
            effort_cost = (
                float(effort.get("effort_total", 0.0) or 0.0) * effort_energy_scale()
                + float(effort.get("work_total", 0.0) or 0.0) * work_energy_scale()
            )
            effort_cost = min(effort_cost, effort_max_energy_drain_per_obs())
            fatigue_pain = min(
                1.0, float(effort.get("fatigue_total", 0.0) or 0.0) * fatigue_pain_gain()
            )
            strain_pain = min(
                1.0, float(effort.get("strain_total", 0.0) or 0.0) * strain_pain_gain()
            )

        # Legacy batching has no producer worker, so keep its decode-on-ingest
        # optimization. Serial prefetch and persistent perception do this in the
        # producer workers instead.
        if self.processing_mode == PROCESSING_BATCHING and self.neural is not None:
            encoders = getattr(self.neural, "encoders", None)
            predecode = getattr(encoders, "predecode", None)
            if callable(predecode):
                try:
                    predecode(obs)
                except Exception:
                    logger.debug(
                        "predecode_failed agent_id=%s", self.agent_id, exc_info=True
                    )

        async with self.lock:
            self._wait_for_observation_after_reset = False
            if not self._serial_prefetch_enabled():
                self.perceptual_integrator.integrate(self.perceptual, obs)
            self.metrics["last_observation_iso"] = obs.get("timestamp")
            self._capture_locomotion_telemetry(obs)

            # Anticipatory threat raises the lingering stress accumulator.
            if threat > 0.0:
                self._threat_stress = min(1.0, self._threat_stress + 0.5 * threat)

            metabolic = self.viability_mode == "metabolic"
            if damage > 0.0:
                # Learning grace + hard ceiling. The harness discount applies only
                # to the scaffold's own tumbles (falls/collisions while the body
                # is still being held up) so a learner that cannot yet stand is not
                # punished to death for exploring. External threats (the "bear
                # bite": explicit damage / combat / environmental hits) are EXEMPT -
                # they always teach at full strength so the agent learns to avoid
                # them. A grace floor keeps even tumbles costing a little (so falls
                # still teach), and no single observation may empty the reservoir.
                threat_part = min(threat_damage, damage)
                graceable = max(0.0, damage - threat_part)
                damage = min(
                    graceable * self._damage_grace() + threat_part,
                    max_integrity_damage_per_obs(),
                )
            if damage > 0.0:
                self.metrics["fast_path_hits"] = int(self.metrics["fast_path_hits"]) + 1
                pain, pleasure = viability_delta_to_signals(-damage)
                self.state_bus.emotion_physio = apply_pain_pleasure_to_B(
                    self.state_bus.emotion_physio, pain, pleasure
                )
                self.state_bus.pain_scalar = ema_affect(
                    self.state_bus.pain_scalar, pain, retain=0.95
                )
                if metabolic:
                    self.homeostasis.apply_reservoir_deltas(integrity=-damage)
                    logger.info(
                        "fast_path_damage agent_id=%s damage=%.4f integrity=%.4f viability=%.4f",
                        self.agent_id,
                        damage,
                        self.homeostasis.integrity,
                        self.homeostasis.viability,
                    )
            if energy_gain > 0.0 or hydration_gain > 0.0:
                # Eval-only: count the act->relief contingency firing (foraging gate).
                self.metrics["consume_events"] = (
                    int(self.metrics.get("consume_events", 0)) + 1
                )
                credit = energy_gain + hydration_gain
                pain, pleasure = viability_delta_to_signals(credit)
                self.state_bus.emotion_physio = apply_pain_pleasure_to_B(
                    self.state_bus.emotion_physio, pain, pleasure
                )
                self.state_bus.pleasure_scalar = ema_affect(
                    self.state_bus.pleasure_scalar, pleasure, retain=0.95
                )
                if metabolic:
                    self.homeostasis.apply_reservoir_deltas(
                        energy=energy_gain, hydration=hydration_gain
                    )
                    logger.info(
                        "nourishment agent_id=%s energy=+%.4f hydration=+%.4f viability=%.4f",
                        self.agent_id,
                        energy_gain,
                        hydration_gain,
                        self.homeostasis.viability,
                    )
            if metabolic:
                self.viability.value = self.homeostasis.viability
            if metabolic and effort_cost > 0.0:
                self.homeostasis.apply_reservoir_deltas(energy=-effort_cost)
                pain, pleasure = viability_delta_to_signals(-effort_cost)
                localized = min(1.0, pain + fatigue_pain + strain_pain)
                self.state_bus.emotion_physio = apply_pain_pleasure_to_B(
                    self.state_bus.emotion_physio, localized, pleasure
                )
                self.state_bus.pain_scalar = ema_affect(
                    self.state_bus.pain_scalar, localized, retain=0.95
                )
                self.viability.value = self.homeostasis.viability
            self.metrics["effort_energy_delta"] = round(-effort_cost if metabolic else 0.0, 6)
            self.metrics["fatigue_pain"] = round(float(fatigue_pain), 6)
            self.metrics["strain_pain"] = round(float(strain_pain), 6)
            self.metrics["net_energy_return"] = round(float(energy_gain - (effort_cost if metabolic else 0.0)), 6)
            if energy_gain > 0.0 or hydration_gain > 0.0:
                self.metrics["resource_relief_events"] = int(
                    self.metrics.get("resource_relief_events", 0)
                ) + 1
            self._refresh_homeostasis_metrics()
            self._last_observation = dict(obs)
            if self.processing_mode == PROCESSING_BATCHING:
                self._obs_buffer.append(dict(obs))
            # Fast-path damage can kill between cognitive cycles.
            self._check_death()
        if self._serial_prefetch_enabled():
            await self._enqueue_perception_observation(obs)
        elif self.processing_mode == PROCESSING_PERSISTENT_PERCEPTION:
            await self._enqueue_perception_observation(obs)

    def last_vision_png(self, camera: str | None = None) -> bytes | None:
        """Latest frame for a camera, decoded from base64 (dashboard endpoint).

        ``None`` / ``"egocentric"`` selects the brain's own vision frame;
        anything else looks up the spectator ``debug_views`` cameras.
        """
        if camera in (None, "", "egocentric"):
            obs = self._last_observation or {}
            vis = obs.get("vision") or {}
            data = vis.get("data")
        else:
            data = self._debug_views.get(camera)
        if not isinstance(data, str) or not data.strip():
            return None
        import base64
        import binascii

        try:
            return base64.b64decode(data)
        except (ValueError, binascii.Error):
            return None

    def last_audio_wav(self) -> bytes | None:
        """Latest observed audio window wrapped as a WAV file (dashboard endpoint)."""
        obs = self._last_observation or {}
        audio = obs.get("audio") or {}
        data = audio.get("data")
        if not isinstance(data, str) or not data.strip():
            return None
        import base64
        import binascii
        import io
        import wave

        try:
            pcm = base64.b64decode(data)
        except (ValueError, binascii.Error):
            return None
        sr = int(audio.get("sample_rate", 16000) or 16000)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm)
        return buf.getvalue()

    def encoder_mode(self) -> str | None:
        """Sensory encoder mode of the neural bundle (None when stub cognition)."""
        if self.neural is None:
            return None
        return getattr(self.neural.encoders, "mode", None)

    def has_body(self) -> bool:
        """True once at least one observation has streamed in from a body."""
        return self._last_observation is not None

    def brain_topology(self) -> dict[str, Any] | None:
        """Topology of the neural stack for the Brain Map (cached per rebuild)."""
        if self.neural is None:
            return None
        cached = getattr(self, "_brain_topology_cache", None)
        if cached is not None and cached[0] is self.neural.stack:
            return cached[1]
        from decadic.nn.brain_map import brain_topology

        topo = brain_topology(
            self.neural.stack, preset=getattr(self.neural, "preset", None)
        )
        self._brain_topology_cache = (self.neural.stack, topo)
        return topo

    def brain_landscape(self) -> dict[str, Any] | None:
        """Latest cached loss-landscape surface (None if disabled or still warming up)."""
        return self._last_landscape

    def vision_views(self) -> list[str]:
        """Camera names with frames available right now (egocentric first)."""
        views: list[str] = []
        obs = self._last_observation or {}
        if (obs.get("vision") or {}).get("data"):
            views.append("egocentric")
        views.extend(sorted(self._debug_views))
        return views

    async def give_resource(self, resource: str, amount: float | None = None) -> dict[str, Any]:
        """Admin top-up: credit a reservoir directly, as if the agent had consumed it.

        This mirrors the nourishment branch of ``handle_observation_dict`` (same
        affect + reservoir + viability bookkeeping) but needs no body and no
        proximity event. It is an intervention that bypasses the act->relief
        contingency the agent normally learns; the "place nearby" path is the
        self-learned alternative. Returns a small status dict for the API.
        """
        kind = str(resource).strip().lower()
        if kind not in ("water", "food"):
            raise ValueError(f"unknown resource: {resource!r}")
        if amount is None:
            credit = water_credit() if kind == "water" else food_credit()
        else:
            credit = float(amount)
        credit = max(0.0, credit)

        async with self.lock:
            metabolic = self.viability_mode == "metabolic"
            if credit > 0.0:
                pain, pleasure = viability_delta_to_signals(credit)
                self.state_bus.emotion_physio = apply_pain_pleasure_to_B(
                    self.state_bus.emotion_physio, pain, pleasure
                )
                self.state_bus.pleasure_scalar = ema_affect(
                    self.state_bus.pleasure_scalar, pleasure, retain=0.95
                )
                if metabolic:
                    if kind == "water":
                        self.homeostasis.apply_reservoir_deltas(hydration=credit)
                    else:
                        self.homeostasis.apply_reservoir_deltas(energy=credit)
                    logger.info(
                        "nourishment agent_id=%s %s=+%.4f viability=%.4f source=admin",
                        self.agent_id,
                        kind,
                        credit,
                        self.homeostasis.viability,
                    )
            if metabolic:
                self.viability.value = self.homeostasis.viability
            self._refresh_homeostasis_metrics()
            return {
                "resource": kind,
                "amount": round(credit, 4),
                "hydration": round(self.homeostasis.hydration, 4),
                "energy": round(self.homeostasis.energy, 4),
                "viability": round(self.homeostasis.viability, 4),
            }

    def queue_body_command(self, command: str) -> bool:
        """Push a body command (e.g. recenter) to the connected environment."""
        msg = {
            "type": "body_command",
            "command": command,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        return self._put_control_outbound(msg)

    def _apply_live_teacher(self, outbound: dict[str, Any]) -> None:
        """Attach Skill Dojo teacher metadata without suppressing student action.

        The teacher provides live body support through the MuJoCo spotter and
        replay/consolidation hints through ``expert_motor``. It must not replace
        the live student motor command with a neutral teacher target, or high
        assist freezes exploration exactly when the learner needs to practice.
        """
        dojo = self.dojo_training if isinstance(self.dojo_training, dict) else None
        if dojo is None:
            self.metrics["teacher_motor_agreement"] = 1.0
            self.metrics["teacher_live_assist"] = 0.0
            return
        action = outbound.get("action")
        if not isinstance(action, dict) or action.get("type") != "motor":
            return
        params = action.get("parameters")
        if not isinstance(params, dict):
            return
        student = params.get("ctrl")
        teacher = dojo.get("expert_motor")
        if not isinstance(student, list) or not isinstance(teacher, list):
            return
        n = max(len(student), len(teacher), 1)

        def _vec(xs: list[Any]) -> list[float]:
            out: list[float] = []
            for i in range(n):
                try:
                    v = float(xs[i]) if i < len(xs) else 0.0
                except (TypeError, ValueError):
                    v = 0.0
                out.append(max(-1.0, min(1.0, v)))
            return out

        s = _vec(student)
        t = _vec(teacher)
        assist = max(0.0, min(1.0, float(dojo.get("demo_weight", 0.0) or 0.0)))
        mean_abs_diff = sum(abs(sv - tv) for sv, tv in zip(s, t)) / max(1, n)
        agreement = max(0.0, min(1.0, 1.0 - mean_abs_diff / 2.0))
        params = dict(params)
        params["student_ctrl"] = [round(v, 5) for v in s]
        params["teacher_ctrl"] = [round(v, 5) for v in t]
        params["ctrl"] = [round(v, 5) for v in s]
        params["teacher_assist"] = round(assist, 5)
        params["teacher_origin"] = str(dojo.get("origin", "self"))
        params["assist_reason"] = str(dojo.get("assist_reason", ""))
        params["objective_confidence"] = round(float(dojo.get("objective_confidence", 0.0) or 0.0), 5)
        params["confidence_reason"] = str(dojo.get("confidence_reason", ""))
        params["teacher_live"] = bool(dojo.get("teacher_live", False) and assist > 0.0)
        action["parameters"] = params
        outbound["action"] = action
        self.metrics["teacher_motor_agreement"] = round(agreement, 5)
        self.metrics["teacher_live_assist"] = round(assist, 5)

    def _put_outbound(self, msg: dict[str, Any], *, drop_oldest: bool) -> bool:
        """Queue an outbound websocket message.

        Normal cycle outputs are high-rate and disposable if the body is slow.
        Body/control commands use ``control_queue`` and are drained first by the
        websocket sender.
        """
        try:
            self.out_queue.put_nowait(msg)
            return True
        except asyncio.QueueFull:
            if not drop_oldest:
                return False

        # Make bounded room for control messages without blocking the API
        # handler. Dropping old motor outputs is preferable to losing a manual
        # recenter/viewer/stance command.
        for _ in range(min(32, self.out_queue.maxsize or 32)):
            try:
                self.out_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                self.out_queue.put_nowait(msg)
                return True
            except asyncio.QueueFull:
                continue
        return False

    def _put_control_outbound(self, msg: dict[str, Any]) -> bool:
        """Queue a low-rate control message on the priority lane."""
        try:
            self.control_queue.put_nowait(msg)
            return True
        except asyncio.QueueFull:
            return False

    def snapshot_state(self) -> dict[str, Any]:
        # The bounded "now" graph (working memory) plus the unbounded long-term
        # relational graph (the hippocampal index) ride along in the perceptual
        # payload so the dashboard can render both side by side.
        perceptual = self.perceptual.snapshot_dict()
        if self.ltm_graph is not None:
            perceptual["ltm_graph"] = self.ltm_graph.snapshot(ltm_snapshot_limit())
        return {
            "agent_id": self.agent_id,
            "neural_enabled": self.neural is not None,
            "status": self.status,
            "died_at_cycle": self.died_at_cycle,
            "paused": self.paused,
            "capacity": self.capacity_config(),
            "state_bus": self.state_bus.snapshot_dict(),
            "perceptual": perceptual,
            "viability": {"value": self.viability.value},
            "homeostasis": self.homeostasis.snapshot(),
            "viability_mode": self.viability_mode,
            "metrics": dict(self.metrics),
            "vision_views": self.vision_views(),
            "last_cycle_trace": self._last_cycle_trace,
            "cognitive_trace": self._last_cognitive_trace,
        }

    def explain(
        self, *, history: int = 0, attribution: bool = False, counterfactuals: bool = False
    ) -> dict[str, Any]:
        """Read-only cognitive-trace report for the ``/explain`` endpoint.

        ``attribution`` / ``counterfactuals`` request the on-demand (gradient /
        rollout) extras; they are computed lazily by the cognition layer and are
        no-ops until that gated path is wired.
        """
        hist: list[dict[str, Any]] = []
        if history > 0 and self._cognitive_history:
            hist = list(self._cognitive_history)[-history:]
        extras = self._cognitive_on_demand(attribution=attribution, counterfactuals=counterfactuals)
        return {
            "agent_id": self.agent_id,
            "cycle": int(self.state_bus.cycle_index),
            "trace": self._last_cognitive_trace,
            "history": hist,
            "on_demand": extras,
        }

    def _cognitive_on_demand(
        self, *, attribution: bool = False, counterfactuals: bool = False
    ) -> dict[str, Any] | None:
        """Compute on-demand counterfactual rollouts for ``/explain`` (read-only).

        Uses the last buffered state/command (``prev_state``/``prev_motor``) and
        the current reservoirs to preview the survival-objective landscape for
        alternative motor commands. Input attribution is the cheaper sampled
        in-cycle path and is already embedded in the latest trace's ``salient``.
        """
        if not counterfactuals or self.neural is None:
            return None
        bundle = self.neural
        if bundle.prev_state is None or bundle.prev_motor is None:
            return None
        try:
            from decadic.cycle import cognition_trace

            drive_on = getattr(bundle.stack, "has_intero_model", False) and self.homeostasis is not None
            rollout = cognition_trace.counterfactual_rollout(
                bundle=bundle,
                z5=bundle.prev_state,
                base_motor=bundle.prev_motor,
                homeostasis=self.homeostasis,
                fwd_dim=int(bundle.cfg.forward_pred_dim),
                drive_on=bool(drive_on),
            )
        except Exception:
            return None
        return {"counterfactuals": rollout} if rollout else None

    def checkpoint_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "state_bus": self.state_bus.snapshot_dict(),
            "perceptual": self.perceptual.snapshot_dict(),
            "viability": {"value": self.viability.value},
            "homeostasis": self.homeostasis.snapshot(),
            "viability_mode": self.viability_mode,
        }
        if self.neural is not None:
            payload["neural_brain"] = f"agent_{self.agent_id}_brain.pt"
        return payload

    def apply_checkpoint_payload(self, payload: dict[str, Any]) -> None:
        """Restore vectors/scalars from checkpoint (Phase 1 simple merge)."""
        import numpy as np

        sb = payload.get("state_bus") or {}
        if "A_state_of_mind" in sb:
            self.state_bus.state_of_mind = np.asarray(sb["A_state_of_mind"], dtype=np.float32)
        if "B_emotion_physio" in sb:
            self.state_bus.emotion_physio = np.asarray(sb["B_emotion_physio"], dtype=np.float32)
        if "B_pain_scalar" in sb:
            self.state_bus.pain_scalar = float(sb["B_pain_scalar"])
        if "B_pleasure_scalar" in sb:
            self.state_bus.pleasure_scalar = float(sb["B_pleasure_scalar"])
        if "prev_drive_pressure" in sb:
            self.state_bus.prev_drive_pressure = float(sb["prev_drive_pressure"])
        if "C_narrative_emb" in sb:
            self.state_bus.narrative_emb = np.asarray(sb["C_narrative_emb"], dtype=np.float32)
        if "E_metacognition" in sb:
            self.state_bus.metacognition = np.asarray(sb["E_metacognition"], dtype=np.float32)
        if "D_priority_scalar" in sb:
            self.state_bus.priority_scalar = float(sb["D_priority_scalar"])
        if "D_priority_label" in sb:
            self.state_bus.priority_label = str(sb["D_priority_label"])
        if "cycle_index" in sb:
            self.state_bus.cycle_index = int(sb["cycle_index"])

        perc = payload.get("perceptual") or {}
        if "fused_stub_emb" in perc:
            self.perceptual.fused_stub_emb = np.asarray(
                perc["fused_stub_emb"], dtype=np.float32
            )
        if "integration_ticks" in perc:
            self.perceptual.integration_ticks = int(perc["integration_ticks"])

        via = payload.get("viability") or {}
        if "value" in via:
            self.viability.value = float(via["value"])

        hs = payload.get("homeostasis") or {}
        if "hydration" in hs:
            self.homeostasis.hydration = float(hs["hydration"])
        if "energy" in hs:
            self.homeostasis.energy = float(hs["energy"])
        if "integrity" in hs:
            self.homeostasis.integrity = float(hs["integrity"])
        if hs:
            self.viability.value = self.homeostasis.viability
        mode = payload.get("viability_mode")
        if isinstance(mode, str) and mode.strip().lower() in ("metabolic", "immortal"):
            self.viability_mode = mode.strip().lower()
