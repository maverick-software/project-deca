"""Central constants for tensor/vector dimensions (Phase 1 stubs)."""

import os

from decadic.state.body_map import EFFORT_VECTOR_DIM

STATE_OF_MIND_DIM = 64
EMOTION_DIM = 32
NARRATIVE_EMB_DIM = 48
METACOG_DIM = 24
PRIORITY_LABEL_MAX = 64
ACTION_HISTORY_MAX = 32

# Stub stage latent sizes
STAGE_LATENT_DIM = 32

DEFAULT_CYCLE_INTERVAL_S = 0.05
FAST_PATH_COLLISION_THRESHOLD = 0.35

# Working memory + parallel-session ("global workspace") capacity.
DEFAULT_WORKING_MEMORY_SLOTS = 12  # bounded entity slots (Baddeley-style capacity)
DEFAULT_WM_DECAY = 0.9  # per-integration salience decay → object permanence then fade
DEFAULT_WM_MIN_SALIENCE = 0.05  # slots below this are dropped from the live graph
DEFAULT_PARALLEL_SESSIONS = 10  # K: perceptual pipeline capacity / batched fallback frames
MAX_PARALLEL_SESSIONS = 16
DEFAULT_PREFETCH_QUEUE_MAX_FRAMES = 32
DEFAULT_PREFETCH_QUEUE_HARD_MAX = 128
DEFAULT_PREFETCH_BACKPRESSURE_WARN_MS = 1.0
DEFAULT_PREFETCH_OLDEST_UNFOLDED_WARN_MS = 1000.0
DEFAULT_REQUIRE_CUDA = True
DEFAULT_EPISODIC_RECALL_CACHE_ENABLED = True
DEFAULT_EPISODIC_RECALL_RECENT_CAP = 2048
DEFAULT_EPISODIC_RECALL_SALIENT_CAP = 2048
DEFAULT_EPISODIC_RECALL_SQL_FALLBACK_CAP = 512
DEFAULT_SQLITE_WAL = True
DEFAULT_SQLITE_SYNCHRONOUS = "NORMAL"
DEFAULT_SQLITE_WAL_AUTOCHECKPOINT = 1000
DEFAULT_SQLITE_VECTOR_BLOB_ENABLED = True
DEFAULT_SQLITE_WRITE_LEGACY_JSON_VECTORS = False
DEFAULT_EPISODIC_WRITE_BATCH_SIZE = 64
DEFAULT_EPISODIC_WRITE_BATCH_MS = 250.0
DEFAULT_EPISODIC_DB_RETENTION_ENABLED = True
DEFAULT_EPISODIC_DB_RECENT_CAP = 100_000
DEFAULT_EPISODIC_DB_SALIENT_CAP = 25_000
DEFAULT_EPISODIC_DB_PRUNE_INTERVAL_WRITES = 5_000
DEFAULT_EPISODIC_DB_PRUNE_BATCH = 5_000
DEFAULT_MEMORY_CONTEXT_REFRESH_CYCLES = 4
DEFAULT_MEMORY_CONTEXT_ASYNC = True
DEFAULT_API_SNAPSHOT_CACHE = True
DEFAULT_METRICS_LIGHTWEIGHT = True
DEFAULT_CYCLE_SCHEDULER = "deadline"
PROCESSING_SERIAL_PREFETCH = "serial_prefetch"
PROCESSING_STAGE_PIPELINE = "stage_pipeline"
PROCESSING_PERSISTENT_PERCEPTION = "persistent_parallel_perception"
PROCESSING_BATCHING = "batching_observations"
PREFETCH_OVERLOAD_POLICIES = {"block", "drop_oldest"}
READY_COALESCE_POLICIES = {"freshest", "oldest"}
PERCEPTUAL_PROCESSING_PERSISTENT = "persistent_parallel"
PERCEPTUAL_PROCESSING_BATCHING = "batching_observations"
PERCEPTUAL_PROCESSING_MODES = {
    PROCESSING_SERIAL_PREFETCH,
    PROCESSING_STAGE_PIPELINE,
    PROCESSING_PERSISTENT_PERCEPTION,
    PROCESSING_BATCHING,
    PERCEPTUAL_PROCESSING_PERSISTENT,
    PERCEPTUAL_PROCESSING_BATCHING,
}
DEFAULT_REVIVE_VIABILITY = 100.0  # viability restored by an admin revive

# Pooling of parallel-session encodes into the deliberative pass, and the
# persistent scene latent held in working memory. Env overrides:
# DECADIC_SESSION_RECENCY, DECADIC_WM_SCENE_ALPHA, DECADIC_WM_SCENE_BLEND.
DEFAULT_SESSION_RECENCY = 0.7  # gamma: weight decay per frame of age in the pooled percept
DEFAULT_WM_SCENE_ALPHA = 0.3  # EMA rate of the persisting scene latent (new evidence share)
DEFAULT_WM_SCENE_BLEND = 0.5  # scene-latent share of the attention vector vs entity hashes


def processing_mode() -> str:
    """Runtime scheduling mode for incoming observations."""
    explicit = os.environ.get("DECADIC_PROCESSING_MODE")
    if explicit:
        mode = explicit.strip().lower()
        if mode == PROCESSING_STAGE_PIPELINE:
            return PROCESSING_SERIAL_PREFETCH
        if mode in (PROCESSING_SERIAL_PREFETCH, PROCESSING_PERSISTENT_PERCEPTION, PROCESSING_BATCHING):
            return mode
        if mode == PERCEPTUAL_PROCESSING_PERSISTENT:
            return PROCESSING_PERSISTENT_PERCEPTION
    stage_flag = os.environ.get("DECADIC_STAGE_PIPELINING_ENABLED", "1").strip().lower()
    if stage_flag not in ("0", "false", "no", "off"):
        return PROCESSING_SERIAL_PREFETCH
    legacy = os.environ.get("DECADIC_PERCEPTUAL_PROCESSING_MODE")
    if legacy:
        mode = legacy.strip().lower()
        if mode == PERCEPTUAL_PROCESSING_PERSISTENT:
            return PROCESSING_PERSISTENT_PERCEPTION
        if mode == PERCEPTUAL_PROCESSING_BATCHING:
            return PROCESSING_BATCHING
    flag = os.environ.get("DECADIC_PERSISTENT_PARALLEL_PERCEPTION", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return PROCESSING_BATCHING
    return PROCESSING_PERSISTENT_PERCEPTION


def perceptual_processing_mode() -> str:
    """Compatibility alias for older dashboard/runtime callers."""
    return processing_mode()


def prefetch_queue_max_frames(parallel_sessions: int | None = None) -> int:
    """Bound accepted-but-unfolded perception work.

    The default preserves evidence by blocking ingress under sustained overload,
    while preventing unbounded queue growth. The effective size is never below
    ``max(32, parallel_sessions * 3)`` and is hard-clamped to 128.
    """
    try:
        requested = int(
            os.environ.get("DECADIC_PREFETCH_QUEUE_MAX_FRAMES", str(DEFAULT_PREFETCH_QUEUE_MAX_FRAMES))
        )
    except (TypeError, ValueError):
        requested = DEFAULT_PREFETCH_QUEUE_MAX_FRAMES
    try:
        k = int(parallel_sessions if parallel_sessions is not None else DEFAULT_PARALLEL_SESSIONS)
    except (TypeError, ValueError):
        k = DEFAULT_PARALLEL_SESSIONS
    floor = max(DEFAULT_PREFETCH_QUEUE_MAX_FRAMES, max(1, k) * 3)
    return max(1, min(DEFAULT_PREFETCH_QUEUE_HARD_MAX, max(requested, floor)))


def prefetch_overload_policy() -> str:
    raw = os.environ.get("DECADIC_PREFETCH_OVERLOAD_POLICY", "block").strip().lower()
    return raw if raw in PREFETCH_OVERLOAD_POLICIES else "block"


def ready_coalesce_policy() -> str:
    raw = os.environ.get("DECADIC_READY_COALESCE_POLICY", "freshest").strip().lower()
    return raw if raw in READY_COALESCE_POLICIES else "freshest"


def prefetch_backpressure_warn_ms() -> float:
    return max(
        0.0,
        float(
            os.environ.get(
                "DECADIC_PREFETCH_BACKPRESSURE_WARN_MS",
                str(DEFAULT_PREFETCH_BACKPRESSURE_WARN_MS),
            )
        ),
    )


def prefetch_oldest_unfolded_warn_ms() -> float:
    return max(
        0.0,
        float(
            os.environ.get(
                "DECADIC_PREFETCH_OLDEST_UNFOLDED_WARN_MS",
                str(DEFAULT_PREFETCH_OLDEST_UNFOLDED_WARN_MS),
            )
        ),
    )


def require_cuda() -> bool:
    return os.environ.get("DECADIC_REQUIRE_CUDA", "1" if DEFAULT_REQUIRE_CUDA else "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def episodic_recall_cache_enabled() -> bool:
    return os.environ.get(
        "DECADIC_EPISODIC_RECALL_CACHE_ENABLED",
        "1" if DEFAULT_EPISODIC_RECALL_CACHE_ENABLED else "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def episodic_recall_recent_cap() -> int:
    return max(1, int(os.environ.get("DECADIC_EPISODIC_RECALL_RECENT_CAP", str(DEFAULT_EPISODIC_RECALL_RECENT_CAP))))


def episodic_recall_salient_cap() -> int:
    return max(1, int(os.environ.get("DECADIC_EPISODIC_RECALL_SALIENT_CAP", str(DEFAULT_EPISODIC_RECALL_SALIENT_CAP))))


def episodic_recall_sql_fallback_cap() -> int:
    return max(1, int(os.environ.get("DECADIC_EPISODIC_RECALL_SQL_FALLBACK_CAP", str(DEFAULT_EPISODIC_RECALL_SQL_FALLBACK_CAP))))


def sqlite_wal_enabled() -> bool:
    return _env_bool("DECADIC_SQLITE_WAL", DEFAULT_SQLITE_WAL)


def sqlite_synchronous() -> str:
    value = os.environ.get("DECADIC_SQLITE_SYNCHRONOUS", DEFAULT_SQLITE_SYNCHRONOUS).strip().upper()
    return value if value in ("OFF", "NORMAL", "FULL", "EXTRA") else DEFAULT_SQLITE_SYNCHRONOUS


def sqlite_wal_autocheckpoint() -> int:
    return max(1, int(os.environ.get("DECADIC_SQLITE_WAL_AUTOCHECKPOINT", str(DEFAULT_SQLITE_WAL_AUTOCHECKPOINT))))


def sqlite_vector_blob_enabled() -> bool:
    return _env_bool("DECADIC_SQLITE_VECTOR_BLOB_ENABLED", DEFAULT_SQLITE_VECTOR_BLOB_ENABLED)


def sqlite_write_legacy_json_vectors() -> bool:
    return _env_bool("DECADIC_SQLITE_WRITE_LEGACY_JSON_VECTORS", DEFAULT_SQLITE_WRITE_LEGACY_JSON_VECTORS)


def episodic_write_batch_size() -> int:
    return max(1, int(os.environ.get("DECADIC_EPISODIC_WRITE_BATCH_SIZE", str(DEFAULT_EPISODIC_WRITE_BATCH_SIZE))))


def episodic_write_batch_ms() -> float:
    return max(0.0, float(os.environ.get("DECADIC_EPISODIC_WRITE_BATCH_MS", str(DEFAULT_EPISODIC_WRITE_BATCH_MS))))


def episodic_db_retention_enabled() -> bool:
    return _env_bool("DECADIC_EPISODIC_DB_RETENTION_ENABLED", DEFAULT_EPISODIC_DB_RETENTION_ENABLED)


def episodic_db_recent_cap() -> int:
    return max(1, int(os.environ.get("DECADIC_EPISODIC_DB_RECENT_CAP", str(DEFAULT_EPISODIC_DB_RECENT_CAP))))


def episodic_db_salient_cap() -> int:
    return max(1, int(os.environ.get("DECADIC_EPISODIC_DB_SALIENT_CAP", str(DEFAULT_EPISODIC_DB_SALIENT_CAP))))


def episodic_db_prune_interval_writes() -> int:
    return max(1, int(os.environ.get("DECADIC_EPISODIC_DB_PRUNE_INTERVAL_WRITES", str(DEFAULT_EPISODIC_DB_PRUNE_INTERVAL_WRITES))))


def episodic_db_prune_batch() -> int:
    return max(1, int(os.environ.get("DECADIC_EPISODIC_DB_PRUNE_BATCH", str(DEFAULT_EPISODIC_DB_PRUNE_BATCH))))


def memory_context_refresh_cycles() -> int:
    return max(1, int(os.environ.get("DECADIC_MEMORY_CONTEXT_REFRESH_CYCLES", str(DEFAULT_MEMORY_CONTEXT_REFRESH_CYCLES))))


def memory_context_async_enabled() -> bool:
    return os.environ.get(
        "DECADIC_MEMORY_CONTEXT_ASYNC",
        "1" if DEFAULT_MEMORY_CONTEXT_ASYNC else "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def api_snapshot_cache_enabled() -> bool:
    return os.environ.get(
        "DECADIC_API_SNAPSHOT_CACHE",
        "1" if DEFAULT_API_SNAPSHOT_CACHE else "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def metrics_lightweight_enabled() -> bool:
    return os.environ.get(
        "DECADIC_METRICS_LIGHTWEIGHT",
        "1" if DEFAULT_METRICS_LIGHTWEIGHT else "0",
    ).strip().lower() in ("1", "true", "yes", "on")


def cycle_scheduler_mode() -> str:
    mode = os.environ.get("DECADIC_CYCLE_SCHEDULER", DEFAULT_CYCLE_SCHEDULER).strip().lower()
    return mode if mode in ("deadline", "fixed_sleep") else DEFAULT_CYCLE_SCHEDULER

# Persistent, anonymous scene workspace. This is pre-cognitive sensory state: it
# integrates object files into an egocentric scene model before bounded attention
# and global-workspace broadcast.
DEFAULT_SCENE_ENTITY_TTL_CYCLES = 12
DEFAULT_PERCEPTION_CANDIDATE_CAPACITY = 64
DEFAULT_SCENE_ENTITY_CAPACITY = 128
DEFAULT_ATTENTION_FOCUS_CAPACITY = 7
DEFAULT_WM_FOCUS_CAPACITY = 7
DEFAULT_LTM_CONSOLIDATE_FROM_SCENE = True
DEFAULT_DRIVE_ATTENTION_ENABLED = True
DEFAULT_DRIVE_ATTENTION_WEIGHT = 1.0
DEFAULT_NOVELTY_ATTENTION_WEIGHT = 1.0
DEFAULT_THREAT_ATTENTION_WEIGHT = 1.0
DEFAULT_RELIEF_ATTENTION_WEIGHT = 1.0
DEFAULT_SCENE_DYNAMICS_ENABLED = True
DEFAULT_SCENE_DYNAMICS_WEIGHT = 0.05
DEFAULT_SCENE_DYNAMICS_MAX_ENTITIES = 12
DEFAULT_SCENE_DYNAMICS_MATCH_THRESHOLD = 0.35
DEFAULT_SCENE_DYNAMICS_UNCERTAINTY_WEIGHT = 0.05

# --- Embodied motor learning (active inference) -----------------------------
# The motor head emits one normalized PD target per actuator; the MuJoCo
# humanoid has 21 torque actuators (3 abdomen + 5/leg incl. ankle pitch+roll +
# 3/arm). The body warns if model.nu != n_actuators(); they must match.
DEFAULT_N_ACTUATORS = 21
# Controllable proprioceptive state the forward model predicts:
# roll, pitch, yaw, root height, vx, vy, vz  (+ joint qpos appended after).
CONTROLLABLE_PROPRIO_BASE = 7
DEFAULT_ASSIST_DECAY_CYCLES = 4000  # cycles over which the assist harness fades 1 -> 0
DEFAULT_MOTOR_BABBLE_SIGMA = 0.3  # MAX exploration noise on normalized PD targets (gate scales it)
DEFAULT_MOTOR_BABBLE_FLOOR = 0.05  # min exploration while ANY reservoir is below full
DEFAULT_BABBLE_ERROR_HALFSAT = 0.5  # forward-model error giving a 0.5 exploration gate (saturating)
DEFAULT_AI_FWD_WEIGHT = 1.0  # weight of the forward-model prediction-error loss
DEFAULT_TOMBSTONE_KEEP = 5  # most-recent dead-agent brain tombstones to retain in backups/


def n_actuators() -> int:
    return max(1, int(os.environ.get("DECADIC_N_ACTUATORS", str(DEFAULT_N_ACTUATORS))))


def tombstone_keep() -> int:
    """How many recent dead-agent brain tombstones to keep; older ones are pruned."""
    return max(1, int(os.environ.get("DECADIC_TOMBSTONE_KEEP", str(DEFAULT_TOMBSTONE_KEEP))))


def assist_gain_for_cycle(cycle: int) -> float:
    """Linear training-wheels schedule: full assist at cycle 0, none past horizon."""
    horizon = float(os.environ.get("DECADIC_ASSIST_DECAY_CYCLES", str(DEFAULT_ASSIST_DECAY_CYCLES)))
    if horizon <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - cycle / horizon))


def motor_exploration_sigma(
    *, drive: float, fwd_error: float, sigma_max: float | None = None
) -> float:
    """Need-and-error-gated motor exploration noise (replaces the clock decay).

    The old schedule faded exploration to zero on a fixed cycle horizon regardless
    of whether the agent had learned anything - a direct route into the dark room
    (a still policy that perfectly predicts its own stillness has nothing left to
    do). Here exploration scales with how badly the agent is doing:

    - ``drive``: unmet homeostatic need in [0, 1] (the deprivation-pain scalar).
    - ``fwd_error``: forward-model surprise (MSE); high while the world model is
      still wrong, ~0 once it predicts well.

    The two are combined into a gate in [0, 1] and scaled by the max sigma. While
    any reservoir is below full (``drive > 0``) the result is floored above zero,
    so a deprived agent keeps trying actions until it discovers the action->relief
    contingency. A sated agent whose world model predicts well explores ~0 (it may
    legitimately rest).

    ``sigma_max`` is a live override for the ceiling (the curriculum's exploration
    knob); when None the process-env ``DECADIC_MOTOR_BABBLE_SIGMA`` default is used
    (exact parity)."""
    if sigma_max is None:
        sigma_max = float(
            os.environ.get("DECADIC_MOTOR_BABBLE_SIGMA", str(DEFAULT_MOTOR_BABBLE_SIGMA))
        )
    sigma_max = max(0.0, float(sigma_max))
    if sigma_max <= 0.0:
        return 0.0
    drive = min(1.0, max(0.0, float(drive)))
    err = max(0.0, float(fwd_error))
    halfsat = max(
        1e-6, float(os.environ.get("DECADIC_BABBLE_ERROR_HALFSAT", str(DEFAULT_BABBLE_ERROR_HALFSAT)))
    )
    err_term = err / (err + halfsat)  # saturating in [0, 1)
    gate = min(1.0, drive + err_term)
    sigma = sigma_max * gate
    if drive > 0.0:
        floor = max(
            0.0, float(os.environ.get("DECADIC_MOTOR_BABBLE_FLOOR", str(DEFAULT_MOTOR_BABBLE_FLOOR)))
        )
        sigma = max(sigma, min(sigma_max, floor))
    return float(min(sigma_max, max(0.0, sigma)))


def ai_fwd_weight() -> float:
    return float(os.environ.get("DECADIC_AI_FWD_WEIGHT", str(DEFAULT_AI_FWD_WEIGHT)))


def scene_workspace_enabled() -> bool:
    return _env_bool("DECADIC_SCENE_WORKSPACE_ENABLED", True)


def scene_entity_ttl_cycles() -> int:
    return max(1, int(os.environ.get("DECADIC_SCENE_ENTITY_TTL_CYCLES", str(DEFAULT_SCENE_ENTITY_TTL_CYCLES))))


def perception_candidate_capacity() -> int:
    return max(1, int(os.environ.get("DECADIC_PERCEPTION_CANDIDATE_CAPACITY", str(DEFAULT_PERCEPTION_CANDIDATE_CAPACITY))))


def scene_entity_capacity() -> int:
    return max(1, int(os.environ.get("DECADIC_SCENE_ENTITY_CAPACITY", str(DEFAULT_SCENE_ENTITY_CAPACITY))))


def scene_relation_enabled() -> bool:
    return _env_bool("DECADIC_SCENE_RELATION_ENABLED", True)


def scene_prediction_enabled() -> bool:
    return _env_bool("DECADIC_SCENE_PREDICTION_ENABLED", True)


def scene_dynamics_enabled() -> bool:
    return _env_bool("DECADIC_SCENE_DYNAMICS_ENABLED", DEFAULT_SCENE_DYNAMICS_ENABLED)


def scene_dynamics_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_SCENE_DYNAMICS_WEIGHT", str(DEFAULT_SCENE_DYNAMICS_WEIGHT))))


def scene_dynamics_max_entities() -> int:
    return max(1, int(os.environ.get("DECADIC_SCENE_DYNAMICS_MAX_ENTITIES", str(DEFAULT_SCENE_DYNAMICS_MAX_ENTITIES))))


def scene_dynamics_match_threshold() -> float:
    return max(0.0, min(1.0, float(os.environ.get("DECADIC_SCENE_DYNAMICS_MATCH_THRESHOLD", str(DEFAULT_SCENE_DYNAMICS_MATCH_THRESHOLD)))))


def scene_dynamics_uncertainty_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_SCENE_DYNAMICS_UNCERTAINTY_WEIGHT", str(DEFAULT_SCENE_DYNAMICS_UNCERTAINTY_WEIGHT))))


def attention_focus_capacity() -> int:
    legacy = os.environ.get("DECADIC_ATTENTION_FOCUS_CAPACITY")
    return max(1, int(os.environ.get("DECADIC_WM_FOCUS_CAPACITY", legacy or str(DEFAULT_WM_FOCUS_CAPACITY))))


def ltm_consolidate_from_scene() -> bool:
    return _env_bool("DECADIC_LTM_CONSOLIDATE_FROM_SCENE", DEFAULT_LTM_CONSOLIDATE_FROM_SCENE)


def drive_attention_enabled() -> bool:
    return _env_bool("DECADIC_DRIVE_ATTENTION_ENABLED", DEFAULT_DRIVE_ATTENTION_ENABLED)


def drive_attention_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_DRIVE_ATTENTION_WEIGHT", str(DEFAULT_DRIVE_ATTENTION_WEIGHT))))


def novelty_attention_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_NOVELTY_ATTENTION_WEIGHT", str(DEFAULT_NOVELTY_ATTENTION_WEIGHT))))


def threat_attention_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_THREAT_ATTENTION_WEIGHT", str(DEFAULT_THREAT_ATTENTION_WEIGHT))))


def relief_attention_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_RELIEF_ATTENTION_WEIGHT", str(DEFAULT_RELIEF_ATTENTION_WEIGHT))))


def scene_attention_weights() -> dict[str, float]:
    return {
        "drive": drive_attention_weight(),
        "novelty": novelty_attention_weight(),
        "threat": threat_attention_weight(),
        "relief": relief_attention_weight(),
    }


# --- Joint-brace guidance system (replaces the external support harness) -----
# The body is no longer held up by an external world-space force (that unloaded
# the feet and let the body skate/glide). Instead every hinge is braced toward a
# tuned upright standing pose with a stiff, semi-implicit joint spring+damper
# (MuJoCo native jnt_stiffness / dof_damping, integrated by implicitfast). Each
# joint starts welded (tightness 1.0) and loosens monotonically -- its brace
# fading toward the model's native softness -- only as the brain's proprioceptive
# forward-model error for THAT joint falls. The feet keep 100% of the weight, so
# friction is always real and all travel must come from genuine limb push-off.
# The braces live body-side in scripts/mujoco_decadic_adapter.py; these are the
# canonical defaults (the adapter reads the same DECADIC_* env vars).
DEFAULT_BRACE_STIFFNESS = 1500.0  # fully-braced joint spring stiffness (N*m/rad)
DEFAULT_BRACE_DAMPING = 30.0  # fully-braced joint damping (N*m*s/rad)
DEFAULT_BRACE_PE_THRESH = 0.02  # per-joint forward-model PE EMA below which ROM may widen
DEFAULT_BRACE_PE_TAU = 0.05  # low-pass blend rate for the per-joint PE EMA
DEFAULT_BRACE_DWELL_S = 4.0  # seconds of sustained low PE to earn a ROM step
DEFAULT_BRACE_LOOSEN_STEP = 0.04  # tightness drop per earned ROM step (monotonic)
DEFAULT_STAND_ROOT_Z = 1.30  # spawn root height where the soles rest on the floor


def brace_stiffness() -> float:
    return float(os.environ.get("DECADIC_BRACE_STIFFNESS", str(DEFAULT_BRACE_STIFFNESS)))


def brace_damping() -> float:
    return float(os.environ.get("DECADIC_BRACE_DAMPING", str(DEFAULT_BRACE_DAMPING)))


# Legacy support-harness defaults retained only for backward-compatible imports;
# the joint-brace system above supersedes them and they no longer drive the body.
DEFAULT_CURRICULUM_MODE = "legacy"
DEFAULT_SUPPORT_CAP0 = 0.85


def curriculum_mode() -> str:
    mode = os.environ.get("DECADIC_CURRICULUM_MODE", DEFAULT_CURRICULUM_MODE).strip().lower()
    return mode if mode in ("legacy", "guided") else DEFAULT_CURRICULUM_MODE


# --- Neuroplasticity: A) Hebbian plasticity, B) sparse training, C) growth ----
# These are inherent faculties of the architecture and default ON. With every
# flag turned off (via the UI / env) the neural stack is numerically and
# structurally identical to the dense, fixed-topology baseline (full parity),
# which is what the ablation/parity tests pin explicitly. Only the four interior
# MLP blocks (stage1, stage3, risk_mlp, motor) are made plastic/sparse/growable;
# external in/out dims never change.
DEFAULT_PLASTICITY_ENABLED = True
DEFAULT_PLASTICITY_ALPHA = 0.001  # configured ceiling for the Hebbian overlay gain
DEFAULT_PLASTICITY_ALPHA_START = 0.0  # guardian-controlled effective alpha starts here
DEFAULT_PLASTICITY_ETA = 0.1  # Hebbian trace blend rate per cycle
DEFAULT_PLASTICITY_INSTABILITY_PCLOSS = 50.0  # pc-loss EMA above this auto-freezes plasticity
DEFAULT_PLASTICITY_HEALTHY_PCEMA = 3.0
DEFAULT_PLASTICITY_THROTTLE_PCEMA = 10.0
DEFAULT_PLASTICITY_FREEZE_PCEMA = 50.0
DEFAULT_PLASTICITY_THAW_PCEMA = 5.0
DEFAULT_PLASTICITY_STABLE_CYCLES_TO_INCREASE = 50
DEFAULT_PLASTICITY_STABLE_CYCLES_TO_THAW = 100
DEFAULT_PLASTICITY_ALPHA_INCREASE_STEP = 0.00025
DEFAULT_PLASTICITY_ALPHA_DECREASE_FACTOR = 0.5
DEFAULT_PLASTICITY_OVERLAY_MAX_FRAC = 0.05
DEFAULT_PLASTICITY_SLOPE_EMA_BETA = 0.9
DEFAULT_PLASTICITY_RISING_SLOPE = 2.0

# Live objective-health canary. This guards optimizer stability only: it never
# adds reward, labels, or cognition inputs. Jump/non-finite detectors are active
# immediately; EMA thresholds use the warmup window.
DEFAULT_LOSS_CANARY_ENABLED = True
DEFAULT_LOSS_CANARY_WARMUP_CYCLES = 20
DEFAULT_LOSS_CANARY_WARN_JUMP_RATIO = 10.0
DEFAULT_LOSS_CANARY_HARD_JUMP_RATIO = 25.0
DEFAULT_LOSS_CANARY_WARN_PCEMA = 10.0
DEFAULT_LOSS_CANARY_HARD_PCEMA = 50.0
DEFAULT_LOSS_CANARY_WARNING_STEP_SCALE = 0.25

DEFAULT_SPARSE_ENABLED = True
DEFAULT_SPARSE_DENSITY = 0.5  # fraction of connections kept active (1.0 == dense parity)
DEFAULT_SPARSE_REWIRE_INTERVAL = 250  # cycles between prune/grow rewires
DEFAULT_SPARSE_REWIRE_FRACTION = 0.1  # fraction of active edges reallocated per rewire

DEFAULT_GROWTH_ENABLED = True
DEFAULT_GROWABLE_HIDDEN_CEILING = 512  # per-block hidden allocation when growth is enabled
DEFAULT_MAX_NEURONS = 256  # awake-hidden-neuron cap N (per growable block), <= ceiling
DEFAULT_GROWTH_INTERVAL = 500  # cycles between growth evaluations
DEFAULT_GROWTH_STEP = 8  # dormant neurons woken per growth event (per block)
DEFAULT_GROWTH_PCLOSS_THRESHOLD = 1.0  # grow only while pc-loss EMA exceeds this
DEFAULT_GROWTH_VIABILITY_COST = 0.0  # viability debited per growth event (metabolic price)
# Growth governance (2026-07-05): the 1-h embodied soak grew 8 times because a
# high-but-flat pc-loss re-triggered growth every interval -- in an open world,
# absolute loss never converges, so growth must be gated on PROGRESS, not level.
# Growth now requires (a) learning to have STALLED at current capacity (relative
# EMA improvement over the last growth interval below MIN_PROGRESS), and (b) the
# PREVIOUS growth event to have PAID (EMA improved by at least MIN_GAIN since
# that event) -- an unpaid growth blocks further growth: capacity was not the
# bottleneck, and more neurons will not make an irreducibly surprising world
# predictable.
DEFAULT_GROWTH_MIN_PROGRESS = 0.01  # relative EMA improvement/interval that still counts as learning
DEFAULT_GROWTH_MIN_GAIN = 0.02  # relative EMA improvement a growth event must yield to re-arm growth

# Periodic structural-plasticity snapshot logging. 0 disables it; N>0 emits a
# plasticity_snapshot log line every N cycles (edge events rewire/grow/freeze are
# always logged regardless of this). Off by default to avoid log volume.
DEFAULT_PLASTICITY_LOG_EVERY = 0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def plasticity_enabled() -> bool:
    return _env_bool("DECADIC_PLASTICITY_ENABLED", DEFAULT_PLASTICITY_ENABLED)


def plasticity_alpha() -> float:
    return max(0.0, float(os.environ.get("DECADIC_PLASTICITY_ALPHA", str(DEFAULT_PLASTICITY_ALPHA))))


def plasticity_alpha_start() -> float:
    return max(
        0.0,
        float(os.environ.get("DECADIC_PLASTICITY_ALPHA_START", str(DEFAULT_PLASTICITY_ALPHA_START))),
    )


def plasticity_eta() -> float:
    return min(
        1.0, max(0.0, float(os.environ.get("DECADIC_PLASTICITY_ETA", str(DEFAULT_PLASTICITY_ETA))))
    )


def plasticity_instability_pcloss() -> float:
    return float(
        os.environ.get(
            "DECADIC_PLASTICITY_INSTABILITY_PCLOSS", str(DEFAULT_PLASTICITY_INSTABILITY_PCLOSS)
        )
    )


def plasticity_healthy_pcema() -> float:
    return float(os.environ.get("DECADIC_PLASTICITY_HEALTHY_PCEMA", str(DEFAULT_PLASTICITY_HEALTHY_PCEMA)))


def plasticity_throttle_pcema() -> float:
    return float(os.environ.get("DECADIC_PLASTICITY_THROTTLE_PCEMA", str(DEFAULT_PLASTICITY_THROTTLE_PCEMA)))


def plasticity_freeze_pcema() -> float:
    return float(os.environ.get("DECADIC_PLASTICITY_FREEZE_PCEMA", str(DEFAULT_PLASTICITY_FREEZE_PCEMA)))


def plasticity_thaw_pcema() -> float:
    return float(os.environ.get("DECADIC_PLASTICITY_THAW_PCEMA", str(DEFAULT_PLASTICITY_THAW_PCEMA)))


def plasticity_stable_cycles_to_increase() -> int:
    return max(
        1,
        int(
            os.environ.get(
                "DECADIC_PLASTICITY_STABLE_CYCLES_TO_INCREASE",
                str(DEFAULT_PLASTICITY_STABLE_CYCLES_TO_INCREASE),
            )
        ),
    )


def plasticity_stable_cycles_to_thaw() -> int:
    return max(
        1,
        int(
            os.environ.get(
                "DECADIC_PLASTICITY_STABLE_CYCLES_TO_THAW",
                str(DEFAULT_PLASTICITY_STABLE_CYCLES_TO_THAW),
            )
        ),
    )


def plasticity_alpha_increase_step() -> float:
    return max(
        0.0,
        float(
            os.environ.get(
                "DECADIC_PLASTICITY_ALPHA_INCREASE_STEP",
                str(DEFAULT_PLASTICITY_ALPHA_INCREASE_STEP),
            )
        ),
    )


def plasticity_alpha_decrease_factor() -> float:
    return min(
        1.0,
        max(
            0.0,
            float(
                os.environ.get(
                    "DECADIC_PLASTICITY_ALPHA_DECREASE_FACTOR",
                    str(DEFAULT_PLASTICITY_ALPHA_DECREASE_FACTOR),
                )
            ),
        ),
    )


def plasticity_overlay_max_frac() -> float:
    return max(
        0.0,
        float(
            os.environ.get(
                "DECADIC_PLASTICITY_OVERLAY_MAX_FRAC",
                str(DEFAULT_PLASTICITY_OVERLAY_MAX_FRAC),
            )
        ),
    )


def plasticity_slope_ema_beta() -> float:
    return min(
        0.999,
        max(
            0.0,
            float(os.environ.get("DECADIC_PLASTICITY_SLOPE_EMA_BETA", str(DEFAULT_PLASTICITY_SLOPE_EMA_BETA))),
        ),
    )


def plasticity_rising_slope() -> float:
    return float(os.environ.get("DECADIC_PLASTICITY_RISING_SLOPE", str(DEFAULT_PLASTICITY_RISING_SLOPE)))


def loss_canary_enabled() -> bool:
    return _env_bool("DECADIC_LOSS_CANARY_ENABLED", DEFAULT_LOSS_CANARY_ENABLED)


def loss_canary_warmup_cycles() -> int:
    return max(
        1,
        int(os.environ.get("DECADIC_LOSS_CANARY_WARMUP_CYCLES", str(DEFAULT_LOSS_CANARY_WARMUP_CYCLES))),
    )


def loss_canary_warn_jump_ratio() -> float:
    return max(
        1.0,
        float(os.environ.get("DECADIC_LOSS_CANARY_WARN_JUMP_RATIO", str(DEFAULT_LOSS_CANARY_WARN_JUMP_RATIO))),
    )


def loss_canary_hard_jump_ratio() -> float:
    return max(
        1.0,
        float(os.environ.get("DECADIC_LOSS_CANARY_HARD_JUMP_RATIO", str(DEFAULT_LOSS_CANARY_HARD_JUMP_RATIO))),
    )


def loss_canary_warn_pcema() -> float:
    return max(0.0, float(os.environ.get("DECADIC_LOSS_CANARY_WARN_PCEMA", str(DEFAULT_LOSS_CANARY_WARN_PCEMA))))


def loss_canary_hard_pcema() -> float:
    return max(0.0, float(os.environ.get("DECADIC_LOSS_CANARY_HARD_PCEMA", str(DEFAULT_LOSS_CANARY_HARD_PCEMA))))


def loss_canary_warning_step_scale() -> float:
    return min(
        1.0,
        max(
            0.0,
            float(
                os.environ.get(
                    "DECADIC_LOSS_CANARY_WARNING_STEP_SCALE",
                    str(DEFAULT_LOSS_CANARY_WARNING_STEP_SCALE),
                )
            ),
        ),
    )


def sparse_enabled() -> bool:
    return _env_bool("DECADIC_SPARSE_ENABLED", DEFAULT_SPARSE_ENABLED)


def sparse_density() -> float:
    return min(1.0, max(0.01, float(os.environ.get("DECADIC_SPARSE_DENSITY", str(DEFAULT_SPARSE_DENSITY)))))


def sparse_rewire_interval() -> int:
    return max(1, int(os.environ.get("DECADIC_SPARSE_REWIRE_INTERVAL", str(DEFAULT_SPARSE_REWIRE_INTERVAL))))


def sparse_rewire_fraction() -> float:
    return min(
        1.0, max(0.0, float(os.environ.get("DECADIC_SPARSE_REWIRE_FRACTION", str(DEFAULT_SPARSE_REWIRE_FRACTION))))
    )


def growth_enabled() -> bool:
    return _env_bool("DECADIC_GROWTH_ENABLED", DEFAULT_GROWTH_ENABLED)


def growable_hidden_ceiling() -> int:
    return max(1, int(os.environ.get("DECADIC_GROWABLE_HIDDEN_CEILING", str(DEFAULT_GROWABLE_HIDDEN_CEILING))))


def max_neurons() -> int:
    return max(1, int(os.environ.get("DECADIC_MAX_NEURONS", str(DEFAULT_MAX_NEURONS))))


def growth_interval() -> int:
    return max(1, int(os.environ.get("DECADIC_GROWTH_INTERVAL", str(DEFAULT_GROWTH_INTERVAL))))


def growth_step() -> int:
    return max(1, int(os.environ.get("DECADIC_GROWTH_STEP", str(DEFAULT_GROWTH_STEP))))


def growth_pcloss_threshold() -> float:
    return float(os.environ.get("DECADIC_GROWTH_PCLOSS_THRESHOLD", str(DEFAULT_GROWTH_PCLOSS_THRESHOLD)))


def growth_viability_cost() -> float:
    return max(0.0, float(os.environ.get("DECADIC_GROWTH_VIABILITY_COST", str(DEFAULT_GROWTH_VIABILITY_COST))))


def growth_min_progress() -> float:
    return max(0.0, float(os.environ.get("DECADIC_GROWTH_MIN_PROGRESS", str(DEFAULT_GROWTH_MIN_PROGRESS))))


def growth_min_gain() -> float:
    return max(0.0, float(os.environ.get("DECADIC_GROWTH_MIN_GAIN", str(DEFAULT_GROWTH_MIN_GAIN))))


def plasticity_log_every() -> int:
    """Cycles between periodic plasticity_snapshot log lines (0 disables)."""
    return max(0, int(os.environ.get("DECADIC_PLASTICITY_LOG_EVERY", str(DEFAULT_PLASTICITY_LOG_EVERY))))


# --- Perception feedback loop (top-down predictive perception) ---------------
# Closes the loop so history shapes the percept: a learned top-down prediction
# of z0 is blended with the bottom-up encode under a learned precision gate, and
# episodic recall is keyed by perceptual likeness (see decadic/memory/embeddings).
# Default ON: this is a core faculty, not an experiment. Turning it off makes the
# gate a no-op so z0_eff == ingress(fused) (full parity), which the parity tests
# pin explicitly. It is an architecture variant that takes effect on the next
# reset(), exactly like the A/B/C neuroplasticity subsystems. Nothing here is
# hand-tuned steering: top_down and precision_gate are trainable modules
# optimized by the existing self-supervised objective; the init below is only a
# safe starting prior (near pure bottom-up) that gradient descent is free to move.
DEFAULT_PERCEPTION_FEEDBACK_ENABLED = True
DEFAULT_PERCEPTION_PRED_WEIGHT = 0.5  # weight of the self-supervised perceptual PE term
DEFAULT_PRECISION_GATE_INIT = 3.0  # gate bias logit; sigmoid(3) ~= 0.95 -> starts bottom-up


def perception_feedback_enabled() -> bool:
    return _env_bool("DECADIC_PERCEPTION_FEEDBACK_ENABLED", DEFAULT_PERCEPTION_FEEDBACK_ENABLED)


# Self-state feedback spine (self-model program, Phase 1). When ON the stack
# gains a self_ingress projection that additively injects the previous cycle's
# self-report (A state-of-mind || C narrative || E metacognition) into the
# stage-3 fuse, so the channels that "sound like inner life" actually shape the
# next cycle's processing instead of being emitted-and-discarded. Default OFF:
# the projection is not built, so the state_dict + numerics are byte-identical to
# the baseline (parity). It is an architecture variant that takes effect on the
# next rebuild (configure() / reset()), exactly like perception_feedback. The
# projection is zero-initialized, so even with the flag ON the first cycle is
# byte-identical until learning moves it off parity; the fed-back vector is
# detached (no cross-cycle BPTT). Default ON (the self-model program ships
# enabled for every new agent); tests pin it OFF via conftest for deterministic
# baselines, and it is still a per-agent toggle in the dashboard.
DEFAULT_SELF_MODEL_FEEDBACK = True


def self_model_feedback_enabled() -> bool:
    return _env_bool("DECADIC_SELF_MODEL_FEEDBACK", DEFAULT_SELF_MODEL_FEEDBACK)


# Real global workspace (self-model program, Phase 2). When ON the post-hoc EMA
# blend of the working-memory attention summary into A is replaced by a
# capacity-limited winner-take-all competition with an ignition threshold: only a
# dominant coalition (share of the salience mass >= threshold) "ignites" and is
# globally broadcast (blended into A, fed back via the spine, boosts the episodic
# salience, and is described by the narrative). Below threshold there is no
# ignition: A holds its prior (nothing reaches global broadcast). Default ON (the
# off-branch is the legacy EMA blend). Unlike the zero-init faculties this changes
# cognition immediately, so it is a deliberate behavioral default. It is a live
# per-agent toggle (a pipeline branch, not an architecture change -> no rebuild);
# tests pin it OFF via conftest so the EMA-parity assertions still hold.
DEFAULT_GWT_ENABLED = True


def gwt_enabled() -> bool:
    return _env_bool("DECADIC_GWT_ENABLED", DEFAULT_GWT_ENABLED)


def gwt_ignition_threshold() -> float:
    """Min share of the salience mass a coalition must command to ignite ([0,1])."""
    v = float(os.environ.get("DECADIC_GWT_IGNITION_THRESHOLD", "0.5"))
    return min(1.0, max(0.0, v))


def gwt_capacity() -> int:
    """How many slots may join the winning coalition (workspace breadth)."""
    return max(1, int(os.environ.get("DECADIC_GWT_CAPACITY", "1")))


def gwt_temperature() -> float:
    """Softmax temperature for combining the winning coalition's content."""
    return max(1e-3, float(os.environ.get("DECADIC_GWT_TEMPERATURE", "1.0")))


def gwt_salience_boost() -> float:
    """How much a strong ignition lifts the stored episode's salience."""
    return max(0.0, float(os.environ.get("DECADIC_GWT_SALIENCE_BOOST", "1.0")))


# Explicit temporal-integration window (self-model program, Phase 3). When > 0
# the agent accumulates the bottom-up percept over this many wall-clock
# milliseconds (or DECADIC_INTEGRATION_WINDOW_MAX_FRAMES cycles, whichever comes
# first) and commits ONE bound "now" latent on close; between commits it acts on
# the last committed moment (perception is held). 0 = off = the freshest percept
# is always "now". Default 200 ms (a human-scale integration window; capped at
# DECADIC_INTEGRATION_WINDOW_MAX_FRAMES cycles so fast cycles still commit). A
# live per-agent setting; tests pin it to 0 via conftest for deterministic timing.
DEFAULT_INTEGRATION_WINDOW_MS = 200.0


def integration_window_ms() -> float:
    return max(0.0, float(os.environ.get("DECADIC_INTEGRATION_WINDOW_MS", str(DEFAULT_INTEGRATION_WINDOW_MS))))


def integration_window_max_frames() -> int:
    return max(1, int(os.environ.get("DECADIC_INTEGRATION_WINDOW_MAX_FRAMES", "8")))


# Predictive affect (self-model program, Phase 4). When ON a small forward model
# predicts the next-step affective context (viability/pain/pleasure/priority) from
# the previous cycle's actual affect, and the predicted delta is added to the
# episodic proxy before it is projected into the stack -- so the agent perceives
# in light of how it expects to feel. Default ON; the predictor's output layer is
# zero-init so on is byte-identical until it learns. Rebuilds the brain on toggle
# (the predictor is a stack submodule, checkpointed with the brain). Tests pin it
# OFF via conftest for deterministic baselines.
DEFAULT_PREDICTIVE_AFFECT = True


def predictive_affect_enabled() -> bool:
    return _env_bool("DECADIC_PREDICTIVE_AFFECT", DEFAULT_PREDICTIVE_AFFECT)


def predictive_affect_gain() -> float:
    """How strongly the predicted affect delta colours the episodic proxy."""
    return max(0.0, float(os.environ.get("DECADIC_PREDICTIVE_AFFECT_GAIN", "1.0")))


# Represented self (self-model program, Phase 5). When ON the agent's interoception
# (reservoirs), affect, and capability (the discovered body schema) are written as
# content onto the egocentric self-node, "controls" edges bind the self to its
# learned body parts, and a compact self-node embedding is fed back through a
# dedicated zero-init spine ingress -- so the self becomes a represented object the
# agent models, not just an implicit process. Default ON; the ingress is zero-init
# (byte-identical until learned). Rebuilds the brain on toggle (stack submodule).
# Tests pin it OFF via conftest for deterministic baselines.
DEFAULT_REPRESENTED_SELF = True


def represented_self_enabled() -> bool:
    return _env_bool("DECADIC_REPRESENTED_SELF", DEFAULT_REPRESENTED_SELF)


# WS5-M1 (relational binding): expose working memory to the stack as a slot
# TENSOR (K entity tokens read by keyed cross-attention) instead of only the
# pooled scene latent. Research pathway: default OFF, byte-identical when off;
# the ingress is zero-init so even flag-on is byte-identical until learned.
def wm_slot_tensor_enabled() -> bool:
    # Default ON (owner decision 2026-07-04): every validated cognition
    # upgrade runs in a full cognition run. Tests pin OFF via conftest.
    return _env_bool("DECADIC_WM_SLOT_TENSOR", True)


def wm_slot_k() -> int:
    """Slots exposed to the stack -- the neural WM window. A COGNITIVE
    parameter (human WM holds ~4 chunks regardless of brain size), deliberately
    small and preset-independent; see ws5_relational_binding_prd.md open
    decisions."""
    return max(1, int(os.environ.get("DECADIC_WM_SLOT_K", "6")))


# WS5-M2 (relational binding): recalled episodes enter the stack as TOKENS
# read by keyed cross-attention, beside the legacy mean-pooled context vector
# (five remembered situations stop entering as their average). Research
# pathway: default OFF, byte-identical when off; zero-init ingress.
def memory_tokens_enabled() -> bool:
    # Default ON (owner decision 2026-07-04). Tests pin OFF via conftest.
    return _env_bool("DECADIC_MEMORY_TOKENS", True)


# WS5-M3 (relational binding): the relational core -- a small transformer over
# [slot tokens ; memory tokens ; interoceptive token] whose pooled summary
# augments the stage-4 risk input (zero-init ingress). Runs on DELIBERATIVE
# cycles only (a gate skip never pays for it). Default OFF.
def relational_core_enabled() -> bool:
    # Default ON (owner decision 2026-07-04; measured +1.5 ms on the full
    # preset, ~2% of the cycle envelope). Tests pin OFF via conftest.
    return _env_bool("DECADIC_RELATIONAL_CORE", True)


# WS6-M0.5 (speech loop): the continuous auditory intake organ (PRD 3.8, G9).
# "mic" captures the system microphone into the server-side ring buffer; "bus"
# runs the same ring fed only by mix_in() (the Rig-1 mixing bus / loopback
# tap); "off" disables the organ entirely (byte-identical parity).
def audio_intake_mode() -> str:
    # Default "mic" (owner decision 2026-07-04: hearing is a property of the
    # body, default-on -- the DEFAULT_REPRESENTED_SELF precedent). Tests pin
    # "off" via conftest. Without sounddevice or a device, mic mode degrades
    # gracefully to an inert capture (the bus tap keeps working), so the
    # default is safe on headless boxes.
    raw = os.environ.get("DECADIC_AUDIO_INTAKE", "mic").strip().lower()
    return raw if raw in ("off", "mic", "bus") else "mic"


# WS6-M2.1 (speech loop): the vocal motor organ -- the stack emits a VOICE_DIM
# articulatory parameter vector beside the motor command.
def voice_enabled() -> bool:
    # Default ON (owner decision 2026-07-04: every validated cognition upgrade
    # runs in a full cognition run -- the DEFAULT_REPRESENTED_SELF precedent).
    # The head is zero-init, so ON is behaviorally silent until learning moves
    # it: the newborn does not speak. Tests pin OFF via conftest.
    return _env_bool("DECADIC_VOICE", True)


# WS6-M2.2 (speech loop): whether the rendered voice also reaches the
# OPERATOR's speakers (the monitor tee, PRD 3.7). NOTE: loopback self-hearing
# is NOT a playback mode -- whenever voice is on and the intake is running the
# rendered waveform is ALWAYS mixed back into the intake (the loop IS the
# design); playback only decides whether the room hears it too.
def voice_playback_mode() -> str:
    # "auto" plays through a physical output device when one exists and
    # degrades silently otherwise; "device" is the same sink, named for
    # explicitness in rig scripts; "off" keeps the room silent.
    raw = os.environ.get("DECADIC_VOICE_PLAYBACK", "auto").strip().lower()
    return raw if raw in ("off", "device", "auto") else "auto"


# Memory-efficient training path (self-model program, Phase 6 — hardware-gated).
# When ON the per-cycle training step uses (a) an 8-bit Adam optimizer when
# bitsandbytes is importable on CUDA (halving the optimizer-moment memory, the
# single largest training cost for the heavy presets) and (b) a bf16 autocast
# around the stack forward on CUDA (cutting activation memory). Both fall back
# silently to the fp32 path when unavailable (no bnb / CPU), so on a CPU/test box
# it is byte-identical regardless. Default ON to fit the 250m/500m/1b tiers on a
# single consumer GPU; on CUDA with bitsandbytes this switches to 8-bit Adam +
# bf16 forward.
DEFAULT_MEMORY_EFFICIENT_TRAINING = True


def memory_efficient_training_enabled() -> bool:
    return _env_bool("DECADIC_MEMORY_EFFICIENT_TRAINING", DEFAULT_MEMORY_EFFICIENT_TRAINING)


def perception_pred_weight() -> float:
    return max(
        0.0,
        float(os.environ.get("DECADIC_PERCEPTION_PRED_WEIGHT", str(DEFAULT_PERCEPTION_PRED_WEIGHT))),
    )


def precision_gate_init() -> float:
    return float(os.environ.get("DECADIC_PRECISION_GATE_INIT", str(DEFAULT_PRECISION_GATE_INIT)))


# --- Sensory encoder mode (frozen CLIP/Whisper vs synthetic fallback) --------
# "hf" loads real frozen CLIP + Whisper (downloads ~1 GB on first run) and is the
# only mode that yields the spatial patch tokens discovered perception needs.
# "zeros" is the cheap synthetic fallback (no download); it keeps cycles fast but
# makes discovered perception inert (no patch tokens). Default "hf": the faculties
# are inherent, so the honest default streams real perceptual features.
DEFAULT_ENCODER_MODE = "hf"


def encoder_mode() -> str:
    """Resolve the sensory encoder mode from the env; unknown values -> hf."""
    mode = os.environ.get("DECADIC_ENCODER_MODE", DEFAULT_ENCODER_MODE).strip().lower()
    return mode if mode in ("zeros", "hf") else DEFAULT_ENCODER_MODE


# --- Homeostatic drive reduction (self-learned thirst/hunger seeking) --------
# Closes the motivational loop: a depleted reservoir is felt as innate pain
# (tonic deprivation valence), and a learned interoceptive world model lets the
# policy act to reduce its *predicted* internal drive toward a full-reservoir
# setpoint. The satisfier (water) is never labeled or reward-shaped; it is
# discovered from the agent's own experienced transitions. Always on: the
# interoceptive head is built unconditionally and the drive is the root
# motivation whenever a body streams reservoirs. The setpoint and the
# aversiveness of deprivation are innate substrate (what phylogeny gives a
# newborn), not steering; the action->relief contingency stays learned.
INTERO_PRED_DIM = 3  # reservoirs the interoceptive world model predicts: [hydration, energy, integrity]
# Reference level from which deprivation is measured. 100 (full) means NO dead
# zone: any dip from full registers a little pain, growing convexly as the
# reservoir empties (drive theory; the comfort setpoint is innate substrate).
DEFAULT_DRIVE_COMFORT_SETPOINT = 100.0  # reservoir level (0..100) treated as "satisfied"
DEFAULT_DRIVE_PAIN_GAIN = 1.0  # scale of tonic deprivation pain (summed deficits -> pain, clamped 0..1)
DEFAULT_DRIVE_PAIN_EXPONENT = 2.0  # convexity: >1 makes slight need faint and severe need dominate
DEFAULT_AI_INTERO_FWD_WEIGHT = 1.0  # weight of the interoceptive forward-model PE loss
DEFAULT_AI_INTERO_PREF_WEIGHT = 0.5  # weight of the preferred-interoceptive-state (drive-reduction) loss
DEFAULT_DRIVE_PRIORITY_GAIN = 2.0  # how strongly deprivation severity up-weights drive reduction
DEFAULT_DRIVE_PRIORITY_GAIN_RAMP_CYCLES = 300
DEFAULT_DRIVE_PRIORITY_GAIN_RAMP_FLOOR = 0.25

# --- Tactile world model (full-body touch) ---------------------------------
# The body streams a soft, normalized per-part contact load for every touch
# sensor (feet, hands, shins, thighs, arms, torso, head, waist, butt). A tactile
# forward-model head predicts the NEXT per-part load from (state, action), so the
# brain forms tactile expectations and learns from prediction error which actions
# load which body part -- the per-limb credit-assignment signal for learning to
# push off. Touch has no innate setpoint, so this is PE-only (no preference term).
TACTILE_PRED_DIM = 16  # per-part loads the tactile world model predicts (= touch sensor count)
DEFAULT_AI_TACTILE_FWD_WEIGHT = 1.0  # weight of the tactile forward-model PE loss
EFFORT_PRED_DIM = EFFORT_VECTOR_DIM  # body-map effort/work/strain/fatigue/pain + aggregate totals
DEFAULT_AI_EFFORT_FWD_WEIGHT = 0.5
DEFAULT_AI_EFFORT_COST_WEIGHT = 0.02
DEFAULT_EFFORT_DRAIN_ENABLED = True
# System A (body-reported effort energy drain) RETIRED 2026-07-08: it was
# per-observation (frame-count, not time) and ~100x too aggressive (killed the
# agent in ~100 min). Replaced by the motor-signal cost below. Scales default 0;
# getters retained for A/B revert.
DEFAULT_EFFORT_ENERGY_SCALE = 0.0
DEFAULT_WORK_ENERGY_SCALE = 0.0
DEFAULT_FATIGUE_RECOVERY_S = 8.0
DEFAULT_FATIGUE_PAIN_GAIN = 0.35
DEFAULT_STRAIN_PAIN_GAIN = 0.25
DEFAULT_EFFORT_MAX_ENERGY_DRAIN_PER_OBS = 0.08
DEFAULT_EFFORT_DRAIN_GRACE_MODE = "dojo_or_braced"
# Motor-signal energy cost: energy is spent on the SIGNALS the brain sends to
# its joints (Sum_j |u_emit_j|), integrated over real time (dt-scaled, respects
# compression, cycle-rate-independent). Each actuated joint costs energy, so the
# agent can associate its own motor output with getting hungry. Scale is a
# starting point; calibrate so typical activity empties energy in ~7 days.
DEFAULT_MOTOR_ENERGY_ENABLED = True
DEFAULT_MOTOR_ENERGY_SCALE = 1e-5  # energy per (unit-Sum|u| x second); calibrate
DEFAULT_MOTOR_ENERGY_MODE = "l1"   # "l1" = Sum|u| (per-joint); "l2" = Sum u^2


def drive_comfort_setpoint() -> float:
    return min(
        100.0,
        max(1.0, float(os.environ.get("DECADIC_DRIVE_COMFORT_SETPOINT", str(DEFAULT_DRIVE_COMFORT_SETPOINT)))),
    )


def drive_pain_gain() -> float:
    return max(0.0, float(os.environ.get("DECADIC_DRIVE_PAIN_GAIN", str(DEFAULT_DRIVE_PAIN_GAIN))))


def drive_pain_exponent() -> float:
    """Convexity of the deprivation-pain curve; clamped to >= 1 (>1 = convex)."""
    return max(
        1.0, float(os.environ.get("DECADIC_DRIVE_PAIN_EXPONENT", str(DEFAULT_DRIVE_PAIN_EXPONENT)))
    )


def ai_intero_fwd_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_AI_INTERO_FWD_WEIGHT", str(DEFAULT_AI_INTERO_FWD_WEIGHT))))


def ai_tactile_fwd_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_AI_TACTILE_FWD_WEIGHT", str(DEFAULT_AI_TACTILE_FWD_WEIGHT))))


def ai_effort_fwd_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_AI_EFFORT_FWD_WEIGHT", str(DEFAULT_AI_EFFORT_FWD_WEIGHT))))


def ai_effort_cost_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_AI_EFFORT_COST_WEIGHT", str(DEFAULT_AI_EFFORT_COST_WEIGHT))))


def ai_intero_pref_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_AI_INTERO_PREF_WEIGHT", str(DEFAULT_AI_INTERO_PREF_WEIGHT))))


def drive_priority_gain() -> float:
    """How strongly deprivation severity scales the drive-reduction loss weight."""
    return max(
        0.0, float(os.environ.get("DECADIC_DRIVE_PRIORITY_GAIN", str(DEFAULT_DRIVE_PRIORITY_GAIN)))
    )


def drive_priority_gain_ramp_cycles() -> int:
    return max(
        0,
        int(
            os.environ.get(
                "DECADIC_DRIVE_PRIORITY_GAIN_RAMP_CYCLES",
                str(DEFAULT_DRIVE_PRIORITY_GAIN_RAMP_CYCLES),
            )
        ),
    )


def drive_priority_gain_ramp_floor() -> float:
    return min(
        1.0,
        max(
            0.0,
            float(
                os.environ.get(
                    "DECADIC_DRIVE_PRIORITY_GAIN_RAMP_FLOOR",
                    str(DEFAULT_DRIVE_PRIORITY_GAIN_RAMP_FLOOR),
                )
            ),
        ),
    )


def drive_priority_gain_effective(cycle_index: int, configured: float | None = None) -> float:
    gain = drive_priority_gain() if configured is None else max(0.0, float(configured))
    ramp_cycles = drive_priority_gain_ramp_cycles()
    if ramp_cycles <= 0:
        return gain
    floor = drive_priority_gain_ramp_floor()
    progress = min(1.0, max(0.0, float(cycle_index) / float(ramp_cycles)))
    return gain * (floor + (1.0 - floor) * progress)


def effort_drain_enabled() -> bool:
    return _env_bool("DECADIC_EFFORT_DRAIN_ENABLED", DEFAULT_EFFORT_DRAIN_ENABLED)


def effort_energy_scale() -> float:
    return max(0.0, float(os.environ.get("DECADIC_EFFORT_ENERGY_SCALE", str(DEFAULT_EFFORT_ENERGY_SCALE))))


def work_energy_scale() -> float:
    return max(0.0, float(os.environ.get("DECADIC_WORK_ENERGY_SCALE", str(DEFAULT_WORK_ENERGY_SCALE))))


def motor_energy_enabled() -> bool:
    return _env_bool("DECADIC_MOTOR_ENERGY_ENABLED", DEFAULT_MOTOR_ENERGY_ENABLED)


def motor_energy_scale() -> float:
    return max(0.0, float(os.environ.get("DECADIC_MOTOR_ENERGY_SCALE", str(DEFAULT_MOTOR_ENERGY_SCALE))))


def motor_energy_mode() -> str:
    m = str(os.environ.get("DECADIC_MOTOR_ENERGY_MODE", DEFAULT_MOTOR_ENERGY_MODE)).strip().lower()
    return m if m in ("l1", "l2") else "l1"


def fatigue_recovery_s() -> float:
    return max(1e-6, float(os.environ.get("DECADIC_FATIGUE_RECOVERY_S", str(DEFAULT_FATIGUE_RECOVERY_S))))


def fatigue_pain_gain() -> float:
    return max(0.0, float(os.environ.get("DECADIC_FATIGUE_PAIN_GAIN", str(DEFAULT_FATIGUE_PAIN_GAIN))))


def strain_pain_gain() -> float:
    return max(0.0, float(os.environ.get("DECADIC_STRAIN_PAIN_GAIN", str(DEFAULT_STRAIN_PAIN_GAIN))))


def effort_max_energy_drain_per_obs() -> float:
    return max(
        0.0,
        float(
            os.environ.get(
                "DECADIC_EFFORT_MAX_ENERGY_DRAIN_PER_OBS",
                str(DEFAULT_EFFORT_MAX_ENERGY_DRAIN_PER_OBS),
            )
        ),
    )


def effort_drain_grace_mode() -> str:
    mode = os.environ.get("DECADIC_EFFORT_DRAIN_GRACE_MODE", DEFAULT_EFFORT_DRAIN_GRACE_MODE).strip().lower()
    return mode if mode in {"none", "dojo_or_braced"} else DEFAULT_EFFORT_DRAIN_GRACE_MODE


# --- Cycle affect: real predictive-coding surprise + homeostatic relief ------
# The phasic pain/pleasure assembled in the neural cycle (decadic/cycle/
# neural_pipeline.py) must be grounded in the agent's own dynamics, never in
# arbitrary scaffolding:
#   * pe_stub_weight() blends in the legacy cycle-counter PE oscillation. The
#     real surprise signal is the predictive-coding loss; that oscillation is a
#     Phase-1 confound, so the production default is 0.0 (removed). Tests pin it
#     to its legacy 0.25 for a byte-identical neural baseline.
#   * the drive-reduction (homeostatic relief) reward is the positive complement
#     to the tonic interoceptive_drive_pain: phasic pleasure proportional to the
#     per-cycle reduction in drive pressure (Keramati & Gutkin homeostatic RL).
#     It replaces the old fixed periodic placeholder reward. ON by default in
#     production; pinned OFF in tests, where the legacy placeholder is retained
#     so the deterministic baseline is unchanged.
DEFAULT_DRIVE_REWARD_ENABLED = True
DEFAULT_DRIVE_REWARD_GAIN = 1.0  # relief scale; symmetric with DEFAULT_DRIVE_PAIN_GAIN


def pe_stub_weight() -> float:
    """Blend weight of the legacy cycle-counter PE oscillation in ``pe_delta``.

    Defaults to 0.0 (the placeholder confound is removed; the predictive-coding
    loss is the genuine surprise term). Pinned to 0.25 in tests for a
    byte-identical neural baseline.
    """
    return max(0.0, float(os.environ.get("DECADIC_PE_STUB_WEIGHT", "0.0")))


def drive_reward_enabled() -> bool:
    return _env_bool("DECADIC_DRIVE_REWARD_ENABLED", DEFAULT_DRIVE_REWARD_ENABLED)


def drive_reward_gain() -> float:
    return max(0.0, float(os.environ.get("DECADIC_DRIVE_REWARD_GAIN", str(DEFAULT_DRIVE_REWARD_GAIN))))


# --- Need-gated curiosity (autonomous epistemic drive) ----------------------
# Curiosity is a learning-progress epistemic drive folded into element B as a
# pleasure-side affect (see decadic/state/curiosity.py). It is ON by default in
# production: it rewards the *reduction* of forward-model prediction error (not
# raw surprise -> no noisy-TV trap) and is gated by survival urgency (suppressed
# under threat/deprivation, expressed when safe and sated). The test suite pins it
# OFF in tests/conftest.py for a deterministic baseline; set DECADIC_CURIOSITY_ENABLED=0
# to recover the byte-identical no-curiosity cycle.
DEFAULT_CURIOSITY_ENABLED = True
DEFAULT_CURIOSITY_GAIN = 1.0  # pleasure scale of a fully-permitted, fully-learning state (-> [0,1] affect)
DEFAULT_CURIOSITY_PROGRESS_WINDOW = 8  # forward-model PE samples used to estimate learning progress
DEFAULT_CURIOSITY_SAFETY_SHARPNESS = 2.0  # >1: curiosity falls off fast as threat/deprivation rises


def curiosity_enabled() -> bool:
    return _env_bool("DECADIC_CURIOSITY_ENABLED", DEFAULT_CURIOSITY_ENABLED)


def curiosity_gain() -> float:
    return max(0.0, float(os.environ.get("DECADIC_CURIOSITY_GAIN", str(DEFAULT_CURIOSITY_GAIN))))


def curiosity_progress_window() -> int:
    return max(
        4,
        int(os.environ.get("DECADIC_CURIOSITY_PROGRESS_WINDOW", str(DEFAULT_CURIOSITY_PROGRESS_WINDOW))),
    )


def curiosity_safety_sharpness() -> float:
    return max(
        1.0,
        float(os.environ.get("DECADIC_CURIOSITY_SAFETY_SHARPNESS", str(DEFAULT_CURIOSITY_SAFETY_SHARPNESS))),
    )


# --- Dual-network memory consolidation (Option B) ---------------------------
# A cloned consolidator stack replays salience-prioritized transitions on its own
# optimizer and is Polyak-blended back into the live stack (decadic/consolidation/).
# ON by default in production -> a live agent replays and consolidates from its own
# salient experience. The test suite pins it OFF in tests/conftest.py for determinism;
# set DECADIC_CONSOLIDATION_ENABLED=0 to keep the no-op stub heartbeat and byte-identical
# live weights.
DEFAULT_CONSOLIDATION_ENABLED = True
DEFAULT_REPLAY_BUFFER_SIZE = 2048  # max transitions retained (lowest-salience evicted)
DEFAULT_CONSOLIDATION_REPLAY_BATCH = 32  # transitions per replay gradient step
DEFAULT_CONSOLIDATION_STEPS_PER_BURST = 4  # replay steps per wake-up before a sync
DEFAULT_CONSOLIDATION_SYNC_TAU = 0.05  # Polyak blend rate (live <- (1-tau)*live + tau*cons)
DEFAULT_CONSOLIDATION_SYNC_INTERVAL_S = 30.0  # seconds between replay+sync bursts
DEFAULT_CONSOLIDATION_PRUNE_MIN_SALIENCE = 0.0  # transitions below this are never stored
DEFAULT_CONSOLIDATION_GRAD_CLIP = 1.0
DEFAULT_CONSOLIDATION_SYNC_RESET_REL_EPS = 0.01


def consolidation_enabled() -> bool:
    return _env_bool("DECADIC_CONSOLIDATION_ENABLED", DEFAULT_CONSOLIDATION_ENABLED)


def replay_buffer_size() -> int:
    return max(
        1, int(os.environ.get("DECADIC_REPLAY_BUFFER_SIZE", str(DEFAULT_REPLAY_BUFFER_SIZE)))
    )


def consolidation_replay_batch() -> int:
    return max(
        1,
        int(os.environ.get("DECADIC_CONSOLIDATION_REPLAY_BATCH", str(DEFAULT_CONSOLIDATION_REPLAY_BATCH))),
    )


def consolidation_steps_per_burst() -> int:
    return max(
        1,
        int(os.environ.get("DECADIC_CONSOLIDATION_STEPS_PER_BURST", str(DEFAULT_CONSOLIDATION_STEPS_PER_BURST))),
    )


def consolidation_sync_tau() -> float:
    return min(
        1.0,
        max(0.0, float(os.environ.get("DECADIC_CONSOLIDATION_SYNC_TAU", str(DEFAULT_CONSOLIDATION_SYNC_TAU)))),
    )


def consolidation_sync_interval_s() -> float:
    return max(
        0.0,
        float(os.environ.get("DECADIC_CONSOLIDATION_SYNC_INTERVAL_S", str(DEFAULT_CONSOLIDATION_SYNC_INTERVAL_S))),
    )


def consolidation_prune_min_salience() -> float:
    return max(
        0.0,
        float(os.environ.get("DECADIC_CONSOLIDATION_PRUNE_MIN_SALIENCE", str(DEFAULT_CONSOLIDATION_PRUNE_MIN_SALIENCE))),
    )


def consolidation_grad_clip() -> float:
    return max(
        0.0,
        float(os.environ.get("DECADIC_CONSOLIDATION_GRAD_CLIP", str(DEFAULT_CONSOLIDATION_GRAD_CLIP))),
    )


def consolidation_sync_reset_rel_eps() -> float:
    return max(
        0.0,
        float(
            os.environ.get(
                "DECADIC_CONSOLIDATION_SYNC_RESET_REL_EPS",
                str(DEFAULT_CONSOLIDATION_SYNC_RESET_REL_EPS),
            )
        ),
    )


# --- Live loss-landscape probe (visualization only) -------------------------
# A flagged background probe that evaluates the agent's real objective over a 2D
# filter-normalized slice of weight space (decadic/consolidation/landscape.py) and
# caches the surface for the dashboard. OFF by default: it is purely a visualization,
# costs grid*grid*batch forward passes per refresh, and reuses the replay buffer (so
# it is only useful while consolidation is feeding transitions). Never touches the
# live weights, so enabling it is byte-identical to baseline cognition.
DEFAULT_LANDSCAPE_ENABLED = False
DEFAULT_LANDSCAPE_GRID = 15  # surface resolution (grid x grid points); capped at 41
DEFAULT_LANDSCAPE_SPAN = 1.0  # half-width of the alpha/beta sweep in filter-normalized units
DEFAULT_LANDSCAPE_BATCH = 8  # replay transitions scored at each grid point
DEFAULT_LANDSCAPE_INTERVAL_S = 20.0  # seconds between surface refreshes
DEFAULT_LANDSCAPE_SEED = 0  # fixes the random direction basis so refreshes are comparable


def landscape_enabled() -> bool:
    return _env_bool("DECADIC_LANDSCAPE_ENABLED", DEFAULT_LANDSCAPE_ENABLED)


def landscape_grid() -> int:
    return max(
        3, min(41, int(os.environ.get("DECADIC_LANDSCAPE_GRID", str(DEFAULT_LANDSCAPE_GRID))))
    )


def landscape_span() -> float:
    return abs(float(os.environ.get("DECADIC_LANDSCAPE_SPAN", str(DEFAULT_LANDSCAPE_SPAN)))) or 1.0


def landscape_batch() -> int:
    return max(1, int(os.environ.get("DECADIC_LANDSCAPE_BATCH", str(DEFAULT_LANDSCAPE_BATCH))))


def landscape_interval_s() -> float:
    return max(
        1.0, float(os.environ.get("DECADIC_LANDSCAPE_INTERVAL_S", str(DEFAULT_LANDSCAPE_INTERVAL_S)))
    )


def landscape_seed() -> int:
    return int(os.environ.get("DECADIC_LANDSCAPE_SEED", str(DEFAULT_LANDSCAPE_SEED)))


# --- Human-like homeostasis (metabolic viability) ---------------------------
# Viability is derived from three reservoirs: hydration, energy, integrity.
# Modes: "metabolic" runs the full wall-clock metabolic model; "immortal" pins
# every reservoir at full and disables death (long uninterrupted learning runs).
DEFAULT_VIABILITY_MODE = "metabolic"
# Real 1:1 human survival horizons: thirst ~3 days, starvation ~3 weeks.
DEFAULT_HYDRATION_EMPTY_S = 3 * 24 * 3600  # 259200 s to drain 100 -> 0
DEFAULT_ENERGY_EMPTY_S = 21 * 24 * 3600  # 1814400 s to drain 100 -> 0 (~7x slower)
DEFAULT_INTEGRITY_HEAL_FULL_S = 3 * 24 * 3600  # full wound recovery 0 -> 100
DEFAULT_METABOLIC_COMPRESSION = 1.0  # >1 fast-forwards the clock for testing
DEFAULT_HEAL_MIN_RESERVE = 25.0  # both food + water must exceed this to heal
DEFAULT_STRESS_GAIN = 1.5  # multiplier on depletion at full stress
DEFAULT_METABOLIC_TICK_S = 1.0  # wall-clock cadence of the metabolic loop
DEFAULT_WATER_CREDIT = 20.0  # hydration restored per glass consumed
DEFAULT_FOOD_CREDIT = 15.0  # energy restored per morsel consumed
DEFAULT_MEDICAL_KIT_CREDIT = 25.0  # integrity restored per medical kit consumed


def viability_mode_default() -> str:
    mode = os.environ.get("DECADIC_VIABILITY_MODE", DEFAULT_VIABILITY_MODE).strip().lower()
    return mode if mode in ("metabolic", "immortal") else DEFAULT_VIABILITY_MODE


def hydration_empty_s() -> float:
    return max(
        1.0, float(os.environ.get("DECADIC_HYDRATION_EMPTY_S", str(DEFAULT_HYDRATION_EMPTY_S)))
    )


def energy_empty_s() -> float:
    return max(1.0, float(os.environ.get("DECADIC_ENERGY_EMPTY_S", str(DEFAULT_ENERGY_EMPTY_S))))


def integrity_heal_full_s() -> float:
    return max(
        1.0,
        float(os.environ.get("DECADIC_INTEGRITY_HEAL_FULL_S", str(DEFAULT_INTEGRITY_HEAL_FULL_S))),
    )


def metabolic_compression() -> float:
    return max(
        0.0, float(os.environ.get("DECADIC_METABOLIC_COMPRESSION", str(DEFAULT_METABOLIC_COMPRESSION)))
    )


def heal_min_reserve() -> float:
    return float(os.environ.get("DECADIC_HEAL_MIN_RESERVE", str(DEFAULT_HEAL_MIN_RESERVE)))


def stress_gain() -> float:
    return max(0.0, float(os.environ.get("DECADIC_STRESS_GAIN", str(DEFAULT_STRESS_GAIN))))


def metabolic_tick_s() -> float:
    return float(os.environ.get("DECADIC_METABOLIC_TICK_S", str(DEFAULT_METABOLIC_TICK_S)))


def water_credit() -> float:
    return float(os.environ.get("DECADIC_WATER_CREDIT", str(DEFAULT_WATER_CREDIT)))


def food_credit() -> float:
    return float(os.environ.get("DECADIC_FOOD_CREDIT", str(DEFAULT_FOOD_CREDIT)))


def medical_kit_credit() -> float:
    return float(os.environ.get("DECADIC_MEDICAL_KIT_CREDIT", str(DEFAULT_MEDICAL_KIT_CREDIT)))


# --- Physical-injury calibration (impact-based, human-superficial) -----------
# Damage is event-driven (collisions / falls), wholly separate from the passive
# metabolic clock. Event intensities arrive on 0..1 from the body as an impact-
# energy proxy. A normal stumble costs a few integrity points and heals in
# hours; only hard, high-energy impacts do real harm. The per-observation cap
# guarantees no single tick can empty the reservoir, so nothing dies "instantly"
# from contact the way the old force-magnitude model did.
DEFAULT_COLLISION_DAMAGE_SCALE = 25.0  # integrity lost at a full-intensity (1.0) impact
DEFAULT_FALL_DAMAGE_SCALE = 3.0  # superficial "you went down" cost (intensity * this)
DEFAULT_GAME_DAMAGE_SCALE = 6.0  # explicit game damage (damage / environment / combat)
DEFAULT_MAX_INTEGRITY_DAMAGE_PER_OBS = 30.0  # hard ceiling on damage from one observation
DEFAULT_DAMAGE_GRACE_FLOOR = 0.15  # min damage fraction while the curriculum still holds


def collision_damage_scale() -> float:
    return max(
        0.0, float(os.environ.get("DECADIC_COLLISION_DAMAGE_SCALE", str(DEFAULT_COLLISION_DAMAGE_SCALE)))
    )


def fall_damage_scale() -> float:
    return max(0.0, float(os.environ.get("DECADIC_FALL_DAMAGE_SCALE", str(DEFAULT_FALL_DAMAGE_SCALE))))


def game_damage_scale() -> float:
    return max(0.0, float(os.environ.get("DECADIC_GAME_DAMAGE_SCALE", str(DEFAULT_GAME_DAMAGE_SCALE))))


def max_integrity_damage_per_obs() -> float:
    return max(
        0.0,
        float(
            os.environ.get(
                "DECADIC_MAX_INTEGRITY_DAMAGE_PER_OBS", str(DEFAULT_MAX_INTEGRITY_DAMAGE_PER_OBS)
            )
        ),
    )


def damage_grace_floor() -> float:
    return min(
        1.0,
        max(0.0, float(os.environ.get("DECADIC_DAMAGE_GRACE_FLOOR", str(DEFAULT_DAMAGE_GRACE_FLOOR)))),
    )


# --- Perception-derived world graph (discovered vs oracle) ------------------
# "oracle" (default): the egocentric graph is built from world_state.entities
# (id/kind/position handed to the agent by the simulator) - the legacy path,
# byte-identical to the prior behaviour. "discovered": the graph emerges from
# the agent's own egocentric camera (object-centric slot attention),
# proprioception, and memory; objects get coined anonymous ids, body parts are
# discovered from the efference<->motion contingency, and world_state.entities
# is demoted to an evaluation-only ground-truth channel that never feeds
# cognition. "discovered" is the default because a self-built, agent-indexed world
# graph is the point of the architecture; "oracle" remains as an eval scaffold.
# Switching builds the slot/agency modules and therefore takes effect on the next
# reset(), exactly like the A/B/C, perception-feedback, and homeostatic-drive
# subsystems. Discovered perception needs real CLIP patch tokens, so it is only
# meaningful with DECADIC_ENCODER_MODE=hf (the default).
DEFAULT_PERCEPTION_MODE = "discovered"

# Slot-attention object discovery (built only in discovered mode).
DEFAULT_SLOTS_K = 7  # K object slots competing for the egocentric feature map
DEFAULT_SLOT_DIM = 64  # per-slot latent width
DEFAULT_SLOT_ITERS = 3  # slot-attention refinement iterations per frame
DEFAULT_SLOT_PRESENCE_THRESHOLD = 0.12  # min presence for a slot to become a proposal
DEFAULT_SLOT_RECON_WEIGHT = 0.5  # weight of the self-supervised feature-reconstruction loss
DEFAULT_SLOT_DIVERSITY_WEIGHT = 0.02  # discourages multiple slots from claiming one region
DEFAULT_SLOT_ENTROPY_WEIGHT = 0.01  # encourages confident per-patch slot assignment
DEFAULT_SLOT_SPATIAL_SEPARATION_WEIGHT = 0.02  # discourages center-collapsed centroids

# Data association of discovered proposals into working-memory object files.
DEFAULT_ASSOC_APPEARANCE_WEIGHT = 0.6  # appearance-cosine share of the match score (vs position)
DEFAULT_ASSOC_MATCH_THRESHOLD = 0.35  # min combined score to bind a proposal to an existing slot
DEFAULT_APPEARANCE_EMA = 0.5  # EMA rate of a slot's appearance fingerprint
DEFAULT_PROVISIONAL_ENTRY_ENABLED = True
DEFAULT_ENTITY_ENTRY_CONFIDENCE_FLOOR = 0.05
DEFAULT_ENTITY_PRECISION_ETA = 0.18
DEFAULT_ENTITY_PROMOTION_PRECISION = 0.2
DEFAULT_ENTITY_PROVISIONAL_DECAY_BOOST = 1.25
DEFAULT_EPISTEMIC_MATURITY_ENABLED = False
DEFAULT_EPISTEMIC_S_MAX = 0.85

# Body-self / agency ("mine") discovery.
DEFAULT_AGENCY_WEIGHT = 0.3  # weight of the agency-head self-supervised motion loss
DEFAULT_AGENCY_EMA = 0.1  # EMA rate of a slot's running agency score
DEFAULT_AGENCY_THRESHOLD = 0.15  # agency score above which a persistent slot reads as "self_part"
DEFAULT_AGENCY_MIN_SEEN = 6  # min sightings before a slot can be promoted to a body part


def perception_mode() -> str:
    """Resolve the perception mode from the env; unknown values -> oracle."""
    mode = os.environ.get("DECADIC_PERCEPTION_MODE", DEFAULT_PERCEPTION_MODE).strip().lower()
    return mode if mode in ("oracle", "discovered") else DEFAULT_PERCEPTION_MODE


def discovered_perception_enabled() -> bool:
    """True when the slot/agency modules should be built (discovered mode)."""
    return perception_mode() == "discovered"


# --- Long-term knowledge graph (hippocampal index) -------------------------
# Persistent, unbounded relational memory that working memory consolidates into
# and reinstates from (decadic/memory/semantic_graph.py). ON by default: a fresh
# agent grows and re-identifies against its own long-term graph with zero config.
# Set DECADIC_LTM_GRAPH=0 only to prove the no-LTM path stays byte-identical.
DEFAULT_LTM_GRAPH_ENABLED = True
DEFAULT_LTM_MATCH_THRESHOLD = 0.6  # cosine over appearance embeddings for re-identification
DEFAULT_LTM_CONSOLIDATE_MIN_SEEN = 2  # cycles a slot must persist before it is committed
DEFAULT_LTM_SNAPSHOT_LIMIT = 64  # nodes returned to the dashboard (graph itself is unbounded)
DEFAULT_LTM_CONSOLIDATION_ASYNC = True
DEFAULT_LTM_CONSOLIDATION_QUEUE_MAX = 4096
DEFAULT_LTM_SEMANTIC_EVIDENCE_INTERVAL = 4
DEFAULT_LTM_SCENE_EDGE_MAX_PER_JOB = 32
DEFAULT_LTM_MATCH_CACHE_ENABLED = True
DEFAULT_LTM_MATCH_RECENT_CAP = 4096
DEFAULT_LTM_MATCH_SALIENT_CAP = 4096
DEFAULT_LTM_RETENTION_ENABLED = True
DEFAULT_LTM_MAX_NODES = 50_000
DEFAULT_LTM_MAX_SEMANTIC_RECORDS = 200_000
DEFAULT_LTM_PRUNE_STALE_CYCLES = 50_000
DEFAULT_LTM_PRUNE_BATCH = 2_000
# WS4C M2 relational hygiene (2026-07-07). Evidence: reports/ws4c_m1_verdict.md.
# -- 194k anonymous event records and 5.4k never-retired edges in a 34-entity
# world after 6 h. Events keyed by class collapse the dominant write factory;
# edge retention + degree cap bound the relation set below the flusher
# crossover by design.
DEFAULT_LTM_EVENT_KEYED = True
DEFAULT_LTM_EDGE_RETENTION_ENABLED = True
DEFAULT_LTM_EDGE_STALE_CYCLES = 2_000
DEFAULT_LTM_EDGE_DEGREE_CAP = 16
DEFAULT_LTM_EDGE_PRUNE_PREFIXES = "scene_"


def ltm_graph_enabled() -> bool:
    return _env_bool("DECADIC_LTM_GRAPH", DEFAULT_LTM_GRAPH_ENABLED)


def ltm_match_threshold() -> float:
    return float(os.environ.get("DECADIC_LTM_MATCH_THRESHOLD", str(DEFAULT_LTM_MATCH_THRESHOLD)))


def ltm_consolidate_min_seen() -> int:
    return max(1, int(os.environ.get("DECADIC_LTM_MIN_SEEN", str(DEFAULT_LTM_CONSOLIDATE_MIN_SEEN))))


def ltm_snapshot_limit() -> int:
    return max(1, int(os.environ.get("DECADIC_LTM_SNAPSHOT_LIMIT", str(DEFAULT_LTM_SNAPSHOT_LIMIT))))


def ltm_consolidation_async_enabled() -> bool:
    return _env_bool("DECADIC_LTM_CONSOLIDATION_ASYNC", DEFAULT_LTM_CONSOLIDATION_ASYNC)


def ltm_consolidation_queue_max() -> int:
    return max(
        1,
        int(os.environ.get("DECADIC_LTM_CONSOLIDATION_QUEUE_MAX", str(DEFAULT_LTM_CONSOLIDATION_QUEUE_MAX))),
    )


def ltm_semantic_evidence_interval() -> int:
    return max(
        1,
        int(os.environ.get("DECADIC_LTM_SEMANTIC_EVIDENCE_INTERVAL", str(DEFAULT_LTM_SEMANTIC_EVIDENCE_INTERVAL))),
    )


def ltm_scene_edge_max_per_job() -> int:
    return max(
        0,
        int(os.environ.get("DECADIC_LTM_SCENE_EDGE_MAX_PER_JOB", str(DEFAULT_LTM_SCENE_EDGE_MAX_PER_JOB))),
    )


def ltm_match_cache_enabled() -> bool:
    return _env_bool("DECADIC_LTM_MATCH_CACHE_ENABLED", DEFAULT_LTM_MATCH_CACHE_ENABLED)


def ltm_match_recent_cap() -> int:
    return max(1, int(os.environ.get("DECADIC_LTM_MATCH_RECENT_CAP", str(DEFAULT_LTM_MATCH_RECENT_CAP))))


def ltm_match_salient_cap() -> int:
    return max(1, int(os.environ.get("DECADIC_LTM_MATCH_SALIENT_CAP", str(DEFAULT_LTM_MATCH_SALIENT_CAP))))


def ltm_retention_enabled() -> bool:
    return _env_bool("DECADIC_LTM_RETENTION_ENABLED", DEFAULT_LTM_RETENTION_ENABLED)


def ltm_max_nodes() -> int:
    return max(1, int(os.environ.get("DECADIC_LTM_MAX_NODES", str(DEFAULT_LTM_MAX_NODES))))


def ltm_max_semantic_records() -> int:
    return max(1, int(os.environ.get("DECADIC_LTM_MAX_SEMANTIC_RECORDS", str(DEFAULT_LTM_MAX_SEMANTIC_RECORDS))))


def ltm_prune_stale_cycles() -> int:
    return max(0, int(os.environ.get("DECADIC_LTM_PRUNE_STALE_CYCLES", str(DEFAULT_LTM_PRUNE_STALE_CYCLES))))


def ltm_prune_batch() -> int:
    return max(1, int(os.environ.get("DECADIC_LTM_PRUNE_BATCH", str(DEFAULT_LTM_PRUNE_BATCH))))


def ltm_event_keyed_enabled() -> bool:
    """WS4C M2.3: semantic event records keyed by event_class (one aggregate
    record per class, evidence accumulates) instead of a fresh anonymous id
    per instance (unbounded row factory; see reports/ws4c_m1_verdict.md)."""
    return _env_bool("DECADIC_LTM_EVENT_KEYED", DEFAULT_LTM_EVENT_KEYED)


def ltm_edge_retention_enabled() -> bool:
    """WS4C M2.1: retire stale prunable-kind edges and cap per-node degree."""
    return _env_bool("DECADIC_LTM_EDGE_RETENTION_ENABLED", DEFAULT_LTM_EDGE_RETENTION_ENABLED)


def ltm_edge_stale_cycles() -> int:
    """Edges of prunable kinds unconfirmed for this many cycles are retired."""
    return max(1, int(os.environ.get("DECADIC_LTM_EDGE_STALE_CYCLES", str(DEFAULT_LTM_EDGE_STALE_CYCLES))))


def ltm_edge_degree_cap() -> int:
    """Backstop: keep at most this many prunable-kind edges per node
    (by weight, then recency); bounds a 34-node scene at ~550 edges."""
    return max(1, int(os.environ.get("DECADIC_LTM_EDGE_DEGREE_CAP", str(DEFAULT_LTM_EDGE_DEGREE_CAP))))


def ltm_edge_prune_prefixes() -> tuple[str, ...]:
    """Edge kinds subject to retention/degree cap (comma-separated prefixes).
    Default: scene_* only -- co_occurrence and custom kinds are exempt."""
    raw = os.environ.get("DECADIC_LTM_EDGE_PRUNE_PREFIXES", DEFAULT_LTM_EDGE_PRUNE_PREFIXES)
    return tuple(p.strip() for p in str(raw).split(",") if p.strip())


def slots_k() -> int:
    return max(1, int(os.environ.get("DECADIC_SLOTS_K", str(DEFAULT_SLOTS_K))))


def slot_dim() -> int:
    return max(4, int(os.environ.get("DECADIC_SLOT_DIM", str(DEFAULT_SLOT_DIM))))


def slot_iters() -> int:
    return max(1, int(os.environ.get("DECADIC_SLOT_ITERS", str(DEFAULT_SLOT_ITERS))))


def slot_presence_threshold() -> float:
    return min(
        1.0,
        max(0.0, float(os.environ.get("DECADIC_SLOT_PRESENCE_THRESHOLD", str(DEFAULT_SLOT_PRESENCE_THRESHOLD)))),
    )


def slot_diversity_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_SLOT_DIVERSITY_WEIGHT", str(DEFAULT_SLOT_DIVERSITY_WEIGHT))))


def slot_entropy_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_SLOT_ENTROPY_WEIGHT", str(DEFAULT_SLOT_ENTROPY_WEIGHT))))


def slot_spatial_separation_weight() -> float:
    return max(
        0.0,
        float(
            os.environ.get(
                "DECADIC_SLOT_SPATIAL_SEPARATION_WEIGHT",
                str(DEFAULT_SLOT_SPATIAL_SEPARATION_WEIGHT),
            )
        ),
    )


def slot_recon_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_SLOT_RECON_WEIGHT", str(DEFAULT_SLOT_RECON_WEIGHT))))


def assoc_appearance_weight() -> float:
    return min(
        1.0,
        max(0.0, float(os.environ.get("DECADIC_ASSOC_APPEARANCE_WEIGHT", str(DEFAULT_ASSOC_APPEARANCE_WEIGHT)))),
    )


def assoc_match_threshold() -> float:
    return float(os.environ.get("DECADIC_ASSOC_MATCH_THRESHOLD", str(DEFAULT_ASSOC_MATCH_THRESHOLD)))


def appearance_ema() -> float:
    return min(1.0, max(0.0, float(os.environ.get("DECADIC_APPEARANCE_EMA", str(DEFAULT_APPEARANCE_EMA)))))


def provisional_entry_enabled() -> bool:
    return _env_bool("DECADIC_PROVISIONAL_ENTRY_ENABLED", DEFAULT_PROVISIONAL_ENTRY_ENABLED)


def entity_entry_confidence_floor() -> float:
    return min(
        1.0,
        max(
            0.0,
            float(
                os.environ.get(
                    "DECADIC_ENTITY_ENTRY_CONFIDENCE_FLOOR",
                    str(DEFAULT_ENTITY_ENTRY_CONFIDENCE_FLOOR),
                )
            ),
        ),
    )


def entity_precision_eta() -> float:
    return min(
        1.0,
        max(0.0, float(os.environ.get("DECADIC_ENTITY_PRECISION_ETA", str(DEFAULT_ENTITY_PRECISION_ETA)))),
    )


def entity_promotion_precision() -> float:
    return min(
        1.0,
        max(
            0.0,
            float(
                os.environ.get(
                    "DECADIC_ENTITY_PROMOTION_PRECISION",
                    str(DEFAULT_ENTITY_PROMOTION_PRECISION),
                )
            ),
        ),
    )


def entity_provisional_decay_boost() -> float:
    return max(
        1.0,
        float(
            os.environ.get(
                "DECADIC_ENTITY_PROVISIONAL_DECAY_BOOST",
                str(DEFAULT_ENTITY_PROVISIONAL_DECAY_BOOST),
            )
        ),
    )


def epistemic_maturity_enabled() -> bool:
    return _env_bool("DECADIC_EPISTEMIC_MATURITY_ENABLED", DEFAULT_EPISTEMIC_MATURITY_ENABLED)


def epistemic_s_max() -> float:
    return min(0.99, max(0.0, float(os.environ.get("DECADIC_EPISTEMIC_S_MAX", str(DEFAULT_EPISTEMIC_S_MAX)))))


def agency_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_AGENCY_WEIGHT", str(DEFAULT_AGENCY_WEIGHT))))


def agency_ema() -> float:
    return min(1.0, max(0.0, float(os.environ.get("DECADIC_AGENCY_EMA", str(DEFAULT_AGENCY_EMA))))) 


def agency_threshold() -> float:
    return float(os.environ.get("DECADIC_AGENCY_THRESHOLD", str(DEFAULT_AGENCY_THRESHOLD)))


def agency_min_seen() -> int:
    return max(1, int(os.environ.get("DECADIC_AGENCY_MIN_SEEN", str(DEFAULT_AGENCY_MIN_SEEN))))


# --- Cognitive trace (interpretability / "why" monitoring) ------------------
# A per-cycle, human-readable explanation of the agent's behaviour, assembled
# read-only from tensors the cycle already computes (survival-intent
# decomposition, self-model surprise, episodic grounding) plus optional gated
# extras (input attribution, counterfactual rollouts, interpretability probes,
# and a templated/LM narrative). Nothing here feeds back into cognition; it is
# pure observation. Default on for the cheap parts; the gradient/LM parts are
# separately gated so the hot loop's cycle rate is preserved.
DEFAULT_COGNITION_TRACE = True
DEFAULT_COGNITION_ATTRIBUTION_INTERVAL = 8  # cycles between saliency passes; 0 disables
DEFAULT_COGNITION_HISTORY_LEN = 256  # compact per-cycle summaries kept for the temporal view
DEFAULT_NARRATIVE_MODE = "template"  # off | template | lm
DEFAULT_PROBE_PATH = ""  # path to trained interpretability probes (empty disables read-out)


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


def cognition_trace_enabled() -> bool:
    return _env_bool("DECADIC_COGNITION_TRACE", DEFAULT_COGNITION_TRACE)


def cognition_attribution_interval() -> int:
    return max(
        0,
        int(os.environ.get("DECADIC_COGNITION_ATTRIBUTION_INTERVAL", str(DEFAULT_COGNITION_ATTRIBUTION_INTERVAL))),
    )


def cognition_history_len() -> int:
    return max(
        1, int(os.environ.get("DECADIC_COGNITION_HISTORY_LEN", str(DEFAULT_COGNITION_HISTORY_LEN)))
    )


def narrative_mode() -> str:
    mode = os.environ.get("DECADIC_NARRATIVE_MODE", DEFAULT_NARRATIVE_MODE).strip().lower()
    return mode if mode in ("off", "template", "lm") else DEFAULT_NARRATIVE_MODE


def probe_path() -> str:
    return os.environ.get("DECADIC_PROBE_PATH", DEFAULT_PROBE_PATH).strip()


def probe_capture_enabled() -> bool:
    return _env_bool("DECADIC_PROBE_CAPTURE", False)


def probe_capture_path() -> str:
    return os.environ.get("DECADIC_PROBE_CAPTURE_PATH", "probe_capture.jsonl").strip()


# --- Performance / GPU knobs ------------------------------------------------
# Opt-in per-section cycle profiler: when on, the cycle times encoders vs.
# stack forward/backward vs. episodic write and logs the split (RULE #6: measure
# before optimizing). Near-zero overhead when off (the section context manager
# short-circuits). Default OFF.
def cycle_profile_enabled() -> bool:
    return _env_bool("DECADIC_CYCLE_PROFILE", False)


# Compute dtype for the FROZEN CLIP/Whisper encoders only. The trainable stack
# stays fp32 regardless. "auto" -> bf16 on a bf16-capable CUDA device (Ampere+),
# else fp32. CPU always resolves to fp32, so the test baseline is unchanged.
def encoder_autocast_dtype():
    import torch

    pref = os.environ.get("DECADIC_ENCODER_PRECISION", "auto").strip().lower()
    if pref == "fp32":
        return torch.float32
    if pref == "bf16":
        return torch.bfloat16
    if pref == "fp16":
        return torch.float16
    # auto: bf16 only when CUDA is present and supports it.
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float32


# Write-behind episodic persistence: when on, the per-cycle SQLite append is
# handed to a background worker so it never blocks the cognitive cycle. This is
# the BIRTH default for a new agent; it is also a live per-agent toggle in the
# dashboard (Agent Settings), so it can be flipped at runtime without a restart.
# Default ON (keeps the cognitive lock free); pinned OFF in tests for byte-identical
# determinism. No write is ever lost (sync fallback under backpressure).
def episodic_async_enabled() -> bool:
    return _env_bool("DECADIC_EPISODIC_ASYNC", True)


# Write-behind long-term-graph (LTM) consolidation: when on, stage 10's WM->LTM
# consolidate (which ends in an fsync via the SQLite commit) is handed to a
# background worker so it never blocks the cognitive cycle. Same contract as
# episodic-async: BIRTH default + live per-agent toggle (Agent Settings), no
# write lost (order-preserving sync fallback under backpressure). The graph is
# read next cycle in stage 3, so the worker carries a ~one-cycle visibility lag
# that is immaterial to associative re-identification. Default ON; pinned OFF in
# tests for byte-identical determinism.
def ltm_async_enabled() -> bool:
    return _env_bool("DECADIC_LTM_ASYNC", True)


# --- Per-life resource randomization (anti-camping) -------------------------
# Resources (food/water) are placed at deterministic XML positions, so an agent
# can learn to sit at a fixed spot ("camp") instead of foraging. When enabled,
# every food/water/gift body is re-scattered to fresh positions at each new life
# (revive) and at body spawn, so the location of relief is never memorizable --
# only the SKILL of seeking-and-reaching transfers. "arena" scatters uniformly
# in the arena disc (>= min-dist from origin so the agent is never spawned on
# top of relief); "zone" keeps each resource random within its habitat zone.
DEFAULT_RANDOMIZE_RESOURCES = True
DEFAULT_RESOURCE_PLACEMENT_MODE = "arena"  # arena | zone
DEFAULT_RESOURCE_MIN_DIST = 3.0  # m: keep scattered resources at least this far from spawn origin
DEFAULT_RESOURCE_FENCE_MARGIN = 1.5  # m: keep resources this far inside the arena fence


def randomize_resources_enabled() -> bool:
    return _env_bool("DECADIC_RANDOMIZE_RESOURCES", DEFAULT_RANDOMIZE_RESOURCES)


def resource_placement_mode() -> str:
    mode = os.environ.get("DECADIC_RESOURCE_PLACEMENT_MODE", DEFAULT_RESOURCE_PLACEMENT_MODE).strip().lower()
    return mode if mode in ("arena", "zone") else DEFAULT_RESOURCE_PLACEMENT_MODE


def resource_min_dist() -> float:
    return max(0.0, float(os.environ.get("DECADIC_RESOURCE_MIN_DIST", str(DEFAULT_RESOURCE_MIN_DIST))))


def resource_fence_margin() -> float:
    return max(0.0, float(os.environ.get("DECADIC_RESOURCE_FENCE_MARGIN", str(DEFAULT_RESOURCE_FENCE_MARGIN))))


# --- Goal lifecycle (explicit intent with onset/achievement/abandonment) ----
# The homeostatic drive is always-on and overlapping; to give credit assignment
# crisp episode boundaries, an explicit GoalState latches the dominant deficit as
# the active goal at onset, holds it while pursued, and closes it on achievement
# (reservoir recovered past the satisfy level / matching consume) or abandonment
# (the dominant deficit flips for N cycles, or death). The closed [onset->close]
# window is the episode the return-based learner trains on. Thresholds are in
# normalized reservoir units (0..1).
DEFAULT_GOAL_ONSET_DEFICIT = 0.15  # weighted deficit (1.0 - reservoir) above which a goal latches
DEFAULT_GOAL_SATISFY_LEVEL = 0.92  # reservoir level (0..1) at/above which the goal is achieved
DEFAULT_GOAL_ABANDON_CYCLES = 40  # cycles the dominant deficit may differ before abandoning the goal
DEFAULT_GOAL_MAX_CYCLES = 4000  # hard cap on an open episode (truncate so returns always resolve)


def goal_onset_deficit() -> float:
    return min(1.0, max(0.0, float(os.environ.get("DECADIC_GOAL_ONSET_DEFICIT", str(DEFAULT_GOAL_ONSET_DEFICIT)))))


def goal_satisfy_level() -> float:
    return min(1.0, max(0.0, float(os.environ.get("DECADIC_GOAL_SATISFY_LEVEL", str(DEFAULT_GOAL_SATISFY_LEVEL)))))


def goal_abandon_cycles() -> int:
    return max(1, int(os.environ.get("DECADIC_GOAL_ABANDON_CYCLES", str(DEFAULT_GOAL_ABANDON_CYCLES))))


def goal_max_cycles() -> int:
    return max(1, int(os.environ.get("DECADIC_GOAL_MAX_CYCLES", str(DEFAULT_GOAL_MAX_CYCLES))))


# --- Distal credit assignment + successor-features value --------------------
# The live policy objective is one-step (greedy drive reduction). To let reward
# reach the long sequence of postural/locomotor commands that PRECEDED relief,
# the consolidator computes lambda-returns over ordered goal-episodes and trains
# a successor-features (SF) head: psi(state, action) predicts the discounted sum
# of future controllable-intero features. A scalar value v = psi . w composes the
# learned (reward-free) prediction with the INNATE drive weights, so a seen
# resource inherits value from the future relief it predicts ("the object becomes
# a goal"). The SF head is zero-initialized and its policy-shaping influence ramps
# from 0, so a fresh agent is byte-identical to today until experience grows it --
# true to the experiment (nothing trained at the start).
DEFAULT_SF_ENABLED = True
# WS-FORAGE M1/M2 (owner: features ship ON; safety is zero-init/ramp + validation,
# not an off-switch). The SF value-learning runs the NORMALIZED, longer-horizon
# regime by default:
#   - SF_NORMALIZE_RETURNS scales returns/SF targets by (1-gamma) -> discounted
#     AVERAGE not SUM, so target magnitude is bounded and ~invariant as the
#     horizon grows (the prerequisite for a minutes-long horizon).
#   - GAMMA 0.995 (~50 s horizon at ~4 cyc/s) up from 0.97 (~8 s); LAMBDA 0.8
#     (lean more on the value bootstrap as the horizon lengthens -> lower variance).
#   - VALUE_WEIGHT 10.0 compensates the (1-gamma) target rescale so the policy-
#     shaping INFLUENCE is preserved; because normalization makes the value scale
#     gamma-independent, this stays a single default across gammas. (Adam absorbs
#     the training-target rescale, so SF_LOSS_WEIGHT is unchanged.)
# This value-regime bundle is the one WS-FORAGE change that is NOT birth-identical
# (it alters learning dynamics), so the M2 soak validates it; all env-tunable.
DEFAULT_SF_GAMMA = 0.995  # discount on future features (horizon ~ 1/(1-gamma))
DEFAULT_SF_LAMBDA = 0.8  # eligibility-trace / lambda-return decay (credit smear over the journey)
DEFAULT_SF_LOSS_WEIGHT = 1.0  # weight of the SF TD(lambda) loss in the consolidator
DEFAULT_SF_VALUE_WEIGHT = 10.0  # MAX weight of the value-advantage policy-shaping term (post-ramp)
DEFAULT_SF_VALUE_RAMP_CYCLES = 2000  # cycles over which the shaping weight climbs 0 -> max
DEFAULT_SF_NORMALIZE_RETURNS = True


def sf_enabled() -> bool:
    return _env_bool("DECADIC_SF_ENABLED", DEFAULT_SF_ENABLED)


def sf_normalize_returns() -> bool:
    return _env_bool("DECADIC_SF_NORMALIZE_RETURNS", DEFAULT_SF_NORMALIZE_RETURNS)


# WS-FORAGE M3: condition the motor policy on the active homeostatic goal (which
# need, how deprived; M4 adds an egocentric bearing to the remembered resource).
# Folded in via a ZERO-INIT ingress, so the agent is birth-identical and the
# capability emerges as experience trains the ingress -> ships ON by default.
DEFAULT_GOAL_CONDITIONED_POLICY = True


def goal_conditioned_policy_enabled() -> bool:
    return _env_bool("DECADIC_GOAL_CONDITIONED_POLICY", DEFAULT_GOAL_CONDITIONED_POLICY)


# WS-FORAGE M4: fill the goal vector's reserved bearing slots with an egocentric
# bearing to the remembered resource (recall-and-navigate, not just react-to-cue).
# Requires goal conditioning; the bearing is zero/masked when no target is
# remembered, so it stays birth-identical until memory has a target to point at.
DEFAULT_GOAL_BEARING = True
DEFAULT_GOAL_BEARING_MAX_DIST = 10.0  # metres that map to normalized distance 1.0


def goal_bearing_enabled() -> bool:
    return _env_bool("DECADIC_GOAL_BEARING", DEFAULT_GOAL_BEARING)


def goal_bearing_max_dist() -> float:
    try:
        return max(0.5, float(os.environ.get("DECADIC_GOAL_BEARING_MAX_DIST", str(DEFAULT_GOAL_BEARING_MAX_DIST))))
    except (TypeError, ValueError):
        return DEFAULT_GOAL_BEARING_MAX_DIST


# WS-FORAGE M5: dual-process control. When an active need's relief is REMEMBERED
# but NOT here (a remembered target beyond TYPE2_FAR_DISTANCE, normalized), trip
# an unconditional gate escalation into the deliberate (System-2) path so the
# agent pursues the resource from memory instead of only reacting to what it
# sees. The bearing (M4) already conditions the policy; this makes it deliberate.
DEFAULT_TYPE2_SEARCH = True
DEFAULT_TYPE2_FAR_DISTANCE = 0.15  # normalized distance beyond which a target is "not here"
# With goal conditioning now CONTINUOUS (2026-07-06), Type-2 needs its own
# deficit bar (deliberation costs compute; a 2% deficit shouldn't buy a memory
# search every cycle). Much lower than the old episode-latch onset (0.15): a
# graded System-2 economics gate, eventually to be replaced by the learned value
# arbitrating when a deliberate detour is worth it.
DEFAULT_TYPE2_MIN_DEFICIT = 0.05


def type2_search_enabled() -> bool:
    return _env_bool("DECADIC_TYPE2_SEARCH", DEFAULT_TYPE2_SEARCH)


def type2_min_deficit() -> float:
    try:
        v = float(os.environ.get("DECADIC_TYPE2_MIN_DEFICIT", str(DEFAULT_TYPE2_MIN_DEFICIT)))
    except (TypeError, ValueError):
        return DEFAULT_TYPE2_MIN_DEFICIT
    return min(1.0, max(0.0, v))


# Type-2 refractory (2026-07-06 calibration): the trigger condition (needy +
# remembered + not-here) is LEVEL-true for long stretches of a hungry life, and
# an unconditional level-triggered escalation re-deliberated the same intention
# every cycle (observed: 55% of all cycles -- perseveration, not thought). The
# deliberation's OUTPUT persists between fires (goal vector + bearing condition
# the policy continuously; the stage-4 precedent decays), so Type-2 now fires,
# latches its hysteresis, then holds a cooldown before re-firing on a merely
# persisting condition -- one intention, execution, periodic re-checks. With
# hysteresis k, type2-attributable deliberation is bounded near (1+k)/refractory.
DEFAULT_TYPE2_REFRACTORY_CYCLES = 32


def type2_refractory_cycles() -> int:
    try:
        return max(0, int(os.environ.get("DECADIC_TYPE2_REFRACTORY_CYCLES", str(DEFAULT_TYPE2_REFRACTORY_CYCLES))))
    except (TypeError, ValueError):
        return DEFAULT_TYPE2_REFRACTORY_CYCLES


def type2_far_distance() -> float:
    try:
        v = float(os.environ.get("DECADIC_TYPE2_FAR_DISTANCE", str(DEFAULT_TYPE2_FAR_DISTANCE)))
    except (TypeError, ValueError):
        return DEFAULT_TYPE2_FAR_DISTANCE
    return min(1.0, max(0.0, v))


# WS-EXPAND E2: multi-channel learning control. Four routed channels replace
# the single plasticity-modulation scalar: reward (== the old scalar), an
# expected-uncertainty rate scale (volatility raises it, plain noise lowers
# it), a transient surprise boost (also raises gate escalation propensity),
# and a viability-trend discount modulation CLAMPED to [LC_GAMMA_MIN,
# LC_GAMMA_MAX] and rate-limited to LC_GAMMA_STEP per LC_GAMMA_RATE_CYCLES
# (the meta-gradient instability guard). Neutral until LC_WARMUP cycles ->
# birth-identical with the flag ON; the flag is the kill switch. Tests pin OFF.
DEFAULT_LEARN_CONTROL_MULTI = True
DEFAULT_LC_WARMUP_CYCLES = 64
DEFAULT_LC_FAST_ALPHA = 0.2  # fast prediction-error EMA (recent level)
DEFAULT_LC_SLOW_ALPHA = 0.02  # slow EMA (baseline level; fast-slow = trend)
DEFAULT_LC_NOISE_ALPHA = 0.05  # EMA of |pc - fast| (noise scale)
DEFAULT_LC_TREND_ALPHA = 0.05  # viability per-cycle delta EMA
DEFAULT_LC_NOISE_FLOOR = 1e-3  # regularizer: quiet+converged reads neutral 1.0
DEFAULT_LC_ETA_MIN_SCALE = 0.25  # noise-dominated stream learns SLOWER, floor
DEFAULT_LC_ETA_MAX_SCALE = 3.0  # volatility+surprise cap (channels never compound past it)
DEFAULT_LC_TREND_MARGIN = 2.0  # trend must exceed this x noise before the rate rises
DEFAULT_LC_SPIKE_K = 3.0  # spike = pc beyond k noise-scales above recent level
DEFAULT_LC_SURPRISE_TAU = 16.0  # cycles for the transient boost to decay by 1/e
DEFAULT_LC_SURPRISE_GAIN = 1.0  # spike peak adds up to +100% rate (pre-cap)
DEFAULT_LC_GAMMA_MIN = 0.99  # hard discount band (default SF_GAMMA 0.995 is inside)
DEFAULT_LC_GAMMA_MAX = 0.997
DEFAULT_LC_GAMMA_STEP = 0.0002  # max discount move per rate window
DEFAULT_LC_GAMMA_RATE_CYCLES = 100  # rate-limit window
DEFAULT_LC_GAMMA_TREND_SCALE = 0.05  # viability pts/cycle for a full-band swing
DEFAULT_LC_GATE_SURPRISE_GAIN = 0.10  # threshold drop per unit surprise
DEFAULT_LC_GATE_UNCERTAINTY_GAIN = 0.05  # threshold drop per unit excess eta-scale
DEFAULT_LC_GATE_MAX_BIAS = 0.15  # gate keeps a floor: threshold 0.30 never below 0.15


def learn_control_multi_enabled() -> bool:
    return _env_bool("DECADIC_LEARN_CONTROL_MULTI", DEFAULT_LEARN_CONTROL_MULTI)


def _lc_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    try:
        v = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def lc_warmup_cycles() -> int:
    return max(1, int(_lc_float("DECADIC_LC_WARMUP_CYCLES", DEFAULT_LC_WARMUP_CYCLES, 1)))


def lc_fast_alpha() -> float:
    return _lc_float("DECADIC_LC_FAST_ALPHA", DEFAULT_LC_FAST_ALPHA, 0.001, 1.0)


def lc_slow_alpha() -> float:
    return _lc_float("DECADIC_LC_SLOW_ALPHA", DEFAULT_LC_SLOW_ALPHA, 0.0001, 1.0)


def lc_noise_alpha() -> float:
    return _lc_float("DECADIC_LC_NOISE_ALPHA", DEFAULT_LC_NOISE_ALPHA, 0.001, 1.0)


def lc_trend_alpha() -> float:
    return _lc_float("DECADIC_LC_TREND_ALPHA", DEFAULT_LC_TREND_ALPHA, 0.001, 1.0)


def lc_noise_floor() -> float:
    return _lc_float("DECADIC_LC_NOISE_FLOOR", DEFAULT_LC_NOISE_FLOOR, 1e-9)


def lc_eta_min_scale() -> float:
    return _lc_float("DECADIC_LC_ETA_MIN_SCALE", DEFAULT_LC_ETA_MIN_SCALE, 0.0, 1.0)


def lc_eta_max_scale() -> float:
    return _lc_float("DECADIC_LC_ETA_MAX_SCALE", DEFAULT_LC_ETA_MAX_SCALE, 1.0)


def lc_trend_margin() -> float:
    return _lc_float("DECADIC_LC_TREND_MARGIN", DEFAULT_LC_TREND_MARGIN, 1.0)


def lc_spike_k() -> float:
    return _lc_float("DECADIC_LC_SPIKE_K", DEFAULT_LC_SPIKE_K, 0.5)


def lc_surprise_tau() -> float:
    return _lc_float("DECADIC_LC_SURPRISE_TAU", DEFAULT_LC_SURPRISE_TAU, 1.0)


def lc_surprise_gain() -> float:
    return _lc_float("DECADIC_LC_SURPRISE_GAIN", DEFAULT_LC_SURPRISE_GAIN, 0.0)


def lc_gamma_min() -> float:
    return _lc_float("DECADIC_LC_GAMMA_MIN", DEFAULT_LC_GAMMA_MIN, 0.0, 0.9999)


def lc_gamma_max() -> float:
    return max(lc_gamma_min(), _lc_float("DECADIC_LC_GAMMA_MAX", DEFAULT_LC_GAMMA_MAX, 0.0, 0.9999))


def lc_gamma_step() -> float:
    return _lc_float("DECADIC_LC_GAMMA_STEP", DEFAULT_LC_GAMMA_STEP, 0.0)


def lc_gamma_rate_cycles() -> int:
    return max(1, int(_lc_float("DECADIC_LC_GAMMA_RATE_CYCLES", DEFAULT_LC_GAMMA_RATE_CYCLES, 1)))


def lc_gamma_trend_scale() -> float:
    return _lc_float("DECADIC_LC_GAMMA_TREND_SCALE", DEFAULT_LC_GAMMA_TREND_SCALE, 1e-6)


def lc_gate_surprise_gain() -> float:
    return _lc_float("DECADIC_LC_GATE_SURPRISE_GAIN", DEFAULT_LC_GATE_SURPRISE_GAIN, 0.0)


def lc_gate_uncertainty_gain() -> float:
    return _lc_float("DECADIC_LC_GATE_UNCERTAINTY_GAIN", DEFAULT_LC_GATE_UNCERTAINTY_GAIN, 0.0)


def lc_gate_max_bias() -> float:
    return _lc_float("DECADIC_LC_GATE_MAX_BIAS", DEFAULT_LC_GATE_MAX_BIAS, 0.0, 0.5)


# WS-EXPAND E1: cognitive map — pose estimation, experiential breadcrumb graph
# with MEASURED hop costs, and stall-gated waypoint routing. Conservative by
# design: the straight-line bearing stays the default; the planner only diverts
# a target's bearing after repeated measured stalls (evidence the direct route
# fails). Default ON (pose/graph accrue silently; behavior identical until a
# blockage is evidenced); tests pin OFF.
DEFAULT_COGNITIVE_MAP = True
DEFAULT_CMAP_BREADCRUMB_M = 1.5  # meters of measured travel per breadcrumb node
DEFAULT_CMAP_CONNECT_RADIUS_M = 3.0  # max anchor distance for plan endpoints
DEFAULT_CMAP_MAX_NODES = 512  # graph memory bound (oldest breadcrumbs evicted)
DEFAULT_CMAP_STALL_CYCLES = 150  # ~40 s at 4 cyc/s without progress = one strike
DEFAULT_CMAP_MIN_PROGRESS_M = 0.15  # approach that counts as progress
DEFAULT_CMAP_BLOCK_THRESHOLD = 2  # strikes before the planner may reroute
DEFAULT_CMAP_POSE_BLEND = 0.5  # complementary-filter gain toward observed pose
DEFAULT_CMAP_POS_SCALE_M = 20.0  # world meters mapping to +/-1.0 positional code


def cognitive_map_enabled() -> bool:
    return _env_bool("DECADIC_COGNITIVE_MAP", DEFAULT_COGNITIVE_MAP)


def cmap_breadcrumb_m() -> float:
    return _lc_float("DECADIC_CMAP_BREADCRUMB_M", DEFAULT_CMAP_BREADCRUMB_M, 0.1)


def cmap_connect_radius_m() -> float:
    return _lc_float("DECADIC_CMAP_CONNECT_RADIUS_M", DEFAULT_CMAP_CONNECT_RADIUS_M, 0.1)


def cmap_max_nodes() -> int:
    return max(8, int(_lc_float("DECADIC_CMAP_MAX_NODES", DEFAULT_CMAP_MAX_NODES, 8)))


def cmap_stall_cycles() -> int:
    return max(5, int(_lc_float("DECADIC_CMAP_STALL_CYCLES", DEFAULT_CMAP_STALL_CYCLES, 5)))


def cmap_min_progress_m() -> float:
    return _lc_float("DECADIC_CMAP_MIN_PROGRESS_M", DEFAULT_CMAP_MIN_PROGRESS_M, 0.001)


def cmap_block_threshold() -> int:
    return max(1, int(_lc_float("DECADIC_CMAP_BLOCK_THRESHOLD", DEFAULT_CMAP_BLOCK_THRESHOLD, 1)))


def cmap_pose_blend() -> float:
    return _lc_float("DECADIC_CMAP_POSE_BLEND", DEFAULT_CMAP_POSE_BLEND, 0.01, 1.0)


def cmap_pos_scale_m() -> float:
    return _lc_float("DECADIC_CMAP_POS_SCALE_M", DEFAULT_CMAP_POS_SCALE_M, 1.0)


# WS-EXPAND E1.6: online rollout action selection on the deliberate path.
# K sampled action variations, short interoceptive rollout truncated by the
# successor value, bounded bias toward the best. Effective bias additionally
# scales with the SF value ramp share, so a naive agent never plans (birth-
# identical); the search only runs on escalated cycles (refractory-bounded).
DEFAULT_PLANNER = True
DEFAULT_PLANNER_K = 7  # sampled candidates (+1 for the policy's own choice)
DEFAULT_PLANNER_HORIZON = 5  # short rollout steps (compounding-error control)
DEFAULT_PLANNER_SIGMA = 0.3  # candidate perturbation scale in action units
DEFAULT_PLANNER_BIAS_GAIN = 0.5  # fraction of the winning delta applied
DEFAULT_PLANNER_BIAS_MAX = 0.25  # L-inf clamp on the delta before gain


def planner_enabled() -> bool:
    return _env_bool("DECADIC_PLANNER", DEFAULT_PLANNER)


def planner_k() -> int:
    return max(0, int(_lc_float("DECADIC_PLANNER_K", DEFAULT_PLANNER_K, 0, 64)))


def planner_horizon() -> int:
    return max(1, int(_lc_float("DECADIC_PLANNER_HORIZON", DEFAULT_PLANNER_HORIZON, 1, 32)))


def planner_sigma() -> float:
    return _lc_float("DECADIC_PLANNER_SIGMA", DEFAULT_PLANNER_SIGMA, 0.01, 2.0)


def planner_bias_gain() -> float:
    return _lc_float("DECADIC_PLANNER_BIAS_GAIN", DEFAULT_PLANNER_BIAS_GAIN, 0.0, 1.0)


def planner_bias_max() -> float:
    return _lc_float("DECADIC_PLANNER_BIAS_MAX", DEFAULT_PLANNER_BIAS_MAX, 0.0, 1.0)


# WS-EXPAND E3: fine-motor error correction + per-actuator phase timing.
# E3.1 corrector: zero-init head adding a bounded correction to the PD targets,
# trained by feedback-error learning (realized per-joint tracking error as a
# supervised target — never a reward). E3.2 phase generator: earned rhythm —
# a zero-init head opens per-actuator oscillation amplitude/frequency where
# periodic drive pays. E3.3: the gate's threat fast-path bypasses the phase
# contribution (aperiodic recovery routes raw). All zero-init -> birth-
# identical; default ON; tests pin OFF.
DEFAULT_MOTOR_CORRECTOR = True
DEFAULT_MOTOR_CORRECTOR_GAIN = 0.3  # scale of the tanh-bounded correction
DEFAULT_MOTOR_CORRECTOR_FEL_K = 0.5  # tracking-error -> correction-target gain
DEFAULT_MOTOR_CORRECTOR_LOSS_WEIGHT = 0.5  # FEL supervision weight in the loss
DEFAULT_CPG = True
DEFAULT_CPG_AMP = 0.3  # max phase contribution per actuator (pre-tanh c)
DEFAULT_CPG_BASE_STEP = 0.35  # rad/cycle (~1.4 Hz at ~4 cyc/s); modulated ±50%


def motor_corrector_enabled() -> bool:
    return _env_bool("DECADIC_MOTOR_CORRECTOR", DEFAULT_MOTOR_CORRECTOR)


def motor_corrector_gain() -> float:
    return _lc_float("DECADIC_MOTOR_CORRECTOR_GAIN", DEFAULT_MOTOR_CORRECTOR_GAIN, 0.0, 1.0)


def motor_corrector_fel_k() -> float:
    return _lc_float("DECADIC_MOTOR_CORRECTOR_FEL_K", DEFAULT_MOTOR_CORRECTOR_FEL_K, 0.0, 5.0)


def motor_corrector_loss_weight() -> float:
    return _lc_float(
        "DECADIC_MOTOR_CORRECTOR_LOSS_WEIGHT", DEFAULT_MOTOR_CORRECTOR_LOSS_WEIGHT, 0.0
    )


def cpg_enabled() -> bool:
    return _env_bool("DECADIC_CPG", DEFAULT_CPG)


def cpg_amp() -> float:
    return _lc_float("DECADIC_CPG_AMP", DEFAULT_CPG_AMP, 0.0, 1.0)


def cpg_base_step() -> float:
    return _lc_float("DECADIC_CPG_BASE_STEP", DEFAULT_CPG_BASE_STEP, 0.0, 3.14)


# WS-EXPAND E4: cached (habit) vs deliberate dual control. Escalated cycles
# bank (input, action) teacher pairs; a small cached head distills them
# continually; skip cycles blend toward the cached action by a TRUST weight
# earned from distillation quality (0 at birth -> byte-identical; trust decays
# the moment the habit stops matching the teacher). Default ON; tests pin OFF.
DEFAULT_CACHED_POLICY = True
DEFAULT_CACHED_BUF = 512  # teacher-pair ring capacity
DEFAULT_CACHED_DISTILL_BATCH = 32  # most-recent pairs per distillation step
DEFAULT_CACHED_DISTILL_MIN = 64  # pairs banked before distillation starts
DEFAULT_CACHED_DISTILL_WEIGHT = 1.0  # distillation term weight in the loss
DEFAULT_CACHED_DISTILL_EMA_ALPHA = 0.05  # distill-loss EMA smoothing
DEFAULT_CACHED_TRUST_THRESHOLD = 0.01  # EMA below this starts earning trust
DEFAULT_CACHED_MAX_W = 1.0  # full habit drive at EMA 0 (skip cycles only)


def cached_policy_enabled() -> bool:
    return _env_bool("DECADIC_CACHED_POLICY", DEFAULT_CACHED_POLICY)


def cached_buf() -> int:
    return max(8, int(_lc_float("DECADIC_CACHED_BUF", DEFAULT_CACHED_BUF, 8)))


def cached_distill_batch() -> int:
    return max(1, int(_lc_float("DECADIC_CACHED_DISTILL_BATCH", DEFAULT_CACHED_DISTILL_BATCH, 1)))


def cached_distill_min() -> int:
    return max(1, int(_lc_float("DECADIC_CACHED_DISTILL_MIN", DEFAULT_CACHED_DISTILL_MIN, 1)))


def cached_distill_weight() -> float:
    return _lc_float("DECADIC_CACHED_DISTILL_WEIGHT", DEFAULT_CACHED_DISTILL_WEIGHT, 0.0)


def cached_distill_ema_alpha() -> float:
    return _lc_float("DECADIC_CACHED_DISTILL_EMA_ALPHA", DEFAULT_CACHED_DISTILL_EMA_ALPHA, 0.001, 1.0)


def cached_trust_threshold() -> float:
    return _lc_float("DECADIC_CACHED_TRUST_THRESHOLD", DEFAULT_CACHED_TRUST_THRESHOLD, 1e-6)


def cached_max_w() -> float:
    return _lc_float("DECADIC_CACHED_MAX_W", DEFAULT_CACHED_MAX_W, 0.0, 1.0)


# WS-EXPAND E5: aversive prediction (threat bearing channels on the goal
# vector; avoidance LEARNED through the ingress) + action veto (minimal
# uncertainty-weighted attenuation, never a hard zero). AVOID_URGENCY_K is the
# extinction-lite guardrail: threat channels scale by (1 - K*deficit), so a
# critically deprived agent re-tests a remembered threat instead of starving
# behind a stale belief.
DEFAULT_AVERSIVE_PREDICTION = True
DEFAULT_AVOID_URGENCY_K = 2.0
DEFAULT_ACTION_VETO = True
DEFAULT_VETO_MAX_ATTENUATION = 0.5  # never suppresses more than half (no hard zero)
DEFAULT_VETO_K = 10.0  # realized viability-drop -> supervision-target gain
DEFAULT_VETO_LOSS_WEIGHT = 0.5


def aversive_prediction_enabled() -> bool:
    return _env_bool("DECADIC_AVERSIVE_PREDICTION", DEFAULT_AVERSIVE_PREDICTION)


def avoid_urgency_k() -> float:
    return _lc_float("DECADIC_AVOID_URGENCY_K", DEFAULT_AVOID_URGENCY_K, 0.0)


def action_veto_enabled() -> bool:
    return _env_bool("DECADIC_ACTION_VETO", DEFAULT_ACTION_VETO)


def veto_max_attenuation() -> float:
    return _lc_float("DECADIC_VETO_MAX_ATTENUATION", DEFAULT_VETO_MAX_ATTENUATION, 0.0, 0.9)


def veto_k() -> float:
    return _lc_float("DECADIC_VETO_K", DEFAULT_VETO_K, 0.0)


def veto_loss_weight() -> float:
    return _lc_float("DECADIC_VETO_LOSS_WEIGHT", DEFAULT_VETO_LOSS_WEIGHT, 0.0)


# WS-EXPAND E6: per-slot input routing gate (identity at init; learned
# suppression floored so nothing is ever fully silenced; reopened by the E2.3
# surprise channel so gating can't blind the agent to newly-relevant percepts).
DEFAULT_INPUT_ROUTING = True
DEFAULT_SLOT_GATE_FLOOR = 0.1


def input_routing_enabled() -> bool:
    return _env_bool("DECADIC_INPUT_ROUTING", DEFAULT_INPUT_ROUTING)


def slot_gate_floor() -> float:
    return _lc_float("DECADIC_SLOT_GATE_FLOOR", DEFAULT_SLOT_GATE_FLOOR, 0.0, 1.0)


# WS-EXPAND E7: scheduled rest. Conservative defaults: first rest only after
# thousands of active cycles, bounded by wake time (value-drift guard), ~30 s
# of body idle per rest, aborted instantly by any threat.
DEFAULT_REST_CYCLE = True
DEFAULT_REST_LOAD_THRESHOLD = 4000.0
DEFAULT_REST_MIN_WAKE_CYCLES = 2000
DEFAULT_REST_CYCLES = 120
DEFAULT_REST_PC_LOAD_SCALE = 0.5
# WS-ATTN 6.2: rest triggers when tier backpressure (attn_pressure, ~0..3)
# stays at/above this. 0 disables the pressure trigger (load-only legacy).
DEFAULT_REST_PRESSURE_THRESHOLD = 2.0
# WS-SOAK: long soak TESTS revive the agent on death (to observe brain function
# past a single ~100-min lifespan) instead of leaving it dead. OFF by default --
# a normal run still dies for real; only the soak-test harness turns this on.
DEFAULT_SOAK_REVIVE = False
DEFAULT_SOAK_REVIVE_FRACTION = 0.8  # restore homeostasis to 80% of range


def rest_cycle_enabled() -> bool:
    return _env_bool("DECADIC_REST_CYCLE", DEFAULT_REST_CYCLE)


def rest_load_threshold() -> float:
    return _lc_float("DECADIC_REST_LOAD_THRESHOLD", DEFAULT_REST_LOAD_THRESHOLD, 10.0)


def rest_min_wake_cycles() -> int:
    return max(10, int(_lc_float("DECADIC_REST_MIN_WAKE_CYCLES", DEFAULT_REST_MIN_WAKE_CYCLES, 10)))


def rest_cycles() -> int:
    return max(1, int(_lc_float("DECADIC_REST_CYCLES", DEFAULT_REST_CYCLES, 1)))


def rest_pc_load_scale() -> float:
    return _lc_float("DECADIC_REST_PC_LOAD_SCALE", DEFAULT_REST_PC_LOAD_SCALE, 0.0)


def rest_pressure_threshold() -> float:
    return _lc_float("DECADIC_REST_PRESSURE_THRESHOLD", DEFAULT_REST_PRESSURE_THRESHOLD, 0.0)


def soak_revive_enabled() -> bool:
    """WS-SOAK: revive-on-death for long soak TESTS only (default off)."""
    return _env_bool("DECADIC_SOAK_REVIVE", DEFAULT_SOAK_REVIVE)


def soak_revive_fraction() -> float:
    return min(1.0, max(0.05, _lc_float("DECADIC_SOAK_REVIVE_FRACTION", DEFAULT_SOAK_REVIVE_FRACTION, 0.05)))


# WS-EXPAND E8.1 (TEST-FIRST): interoceptive embedding conditioning affect.
DEFAULT_INTEROCEPTIVE_HEAD = True


def interoceptive_head_enabled() -> bool:
    return _env_bool("DECADIC_INTEROCEPTIVE_HEAD", DEFAULT_INTEROCEPTIVE_HEAD)


# WS-EXPAND E8.2 (TEST-FIRST): valence-BLENDED replay salience — a bounded
# multiplier (1 .. 1+beta), never a replacement of the existing salience
# (blend-don't-replace was the evidence verdict).
DEFAULT_VALENCE_REPLAY = True
DEFAULT_VALENCE_REPLAY_BETA = 0.5


def valence_replay_enabled() -> bool:
    return _env_bool("DECADIC_VALENCE_REPLAY", DEFAULT_VALENCE_REPLAY)


def valence_replay_beta() -> float:
    return _lc_float("DECADIC_VALENCE_REPLAY_BETA", DEFAULT_VALENCE_REPLAY_BETA, 0.0, 4.0)


# WS-EXPAND E9: FSQ discrete abstraction bottleneck (side-channel; gradients
# never reach the shared trunk -> behavior byte-identical).
DEFAULT_SYMBOLS = True
DEFAULT_SYMBOL_LOSS_WEIGHT = 0.1
# WS-SYM 4.0: symbol feedback into the trunk. ON by default -- the discrete
# code the mind emits conditions its next deliberation (zero-init ingress, so
# birth-identical and learned). Meaning lives in the grounded binding, so it is
# drift-robust (docs/ws_symbol_integration_analysis.md).
DEFAULT_SYMBOL_FEEDBACK = True


def symbols_enabled() -> bool:
    return _env_bool("DECADIC_SYMBOLS", DEFAULT_SYMBOLS)


def symbols_feedback_enabled() -> bool:
    return _env_bool("DECADIC_SYMBOL_FEEDBACK", DEFAULT_SYMBOL_FEEDBACK)


# WS-SYM 3.3: recall-conditioned feedback -- prefer the code RECALLED for the
# focused entity over the agent's own previous code. ON by default.
def symbols_recall_feedback_enabled() -> bool:
    return _env_bool("DECADIC_SYMBOL_RECALL_FEEDBACK", True)


# WS-SYM 5.0: drift closed-loop. Sustained top-code churn (with grounding
# mature) freezes fsq_in (stops training the projection) so meanings stop
# wandering. ON by default; meaning is carried by the grounded binding.
def symbol_drift_freeze_enabled() -> bool:
    return _env_bool("DECADIC_SYMBOL_DRIFT_FREEZE", True)


def symbol_churn_threshold() -> float:
    # flip-rate above this (once mature) = drift -> freeze fsq_in.
    try:
        return max(0.0, float(os.environ.get("DECADIC_SYMBOL_CHURN_THRESHOLD", "0.20")))
    except (TypeError, ValueError):
        return 0.20


def symbol_freeze_min_updates() -> int:
    # grounding-maturity gate: no freeze until this many binding updates, so a
    # young agent (everything churns) is never frozen prematurely.
    try:
        return max(1, int(os.environ.get("DECADIC_SYMBOL_FREEZE_MIN_UPDATES", "500")))
    except (TypeError, ValueError):
        return 500


def symbol_loss_weight() -> float:
    return _lc_float("DECADIC_SYMBOL_LOSS_WEIGHT", DEFAULT_SYMBOL_LOSS_WEIGHT, 0.0)


# WS-EXPAND E10: other-agent modeling behind the adaptivity gate (models spawn
# only for entities whose movement defeats a ballistic prior; solo scenes carry
# zero models).
DEFAULT_OTHER_MODELING = True
DEFAULT_OTHER_ERR_THRESHOLD = 0.05  # meters of ballistic-prior error (EMA)
DEFAULT_OTHER_WARMUP_OBS = 32
DEFAULT_OTHER_EMA_ALPHA = 0.1
DEFAULT_OTHER_MAX_TRACKS = 32


def other_modeling_enabled() -> bool:
    return _env_bool("DECADIC_OTHER_MODELING", DEFAULT_OTHER_MODELING)


def other_err_threshold() -> float:
    return _lc_float("DECADIC_OTHER_ERR_THRESHOLD", DEFAULT_OTHER_ERR_THRESHOLD, 1e-4)


def other_warmup_obs() -> int:
    return max(2, int(_lc_float("DECADIC_OTHER_WARMUP_OBS", DEFAULT_OTHER_WARMUP_OBS, 2)))


def other_ema_alpha() -> float:
    return _lc_float("DECADIC_OTHER_EMA_ALPHA", DEFAULT_OTHER_EMA_ALPHA, 0.001, 1.0)


def other_max_tracks() -> int:
    return max(1, int(_lc_float("DECADIC_OTHER_MAX_TRACKS", DEFAULT_OTHER_MAX_TRACKS, 1)))


# WS-EXPAND E13: phase clock — INSTRUMENTATION ONLY (no behavioral coupling;
# the evidence review found no task benefit for phase-scheduled control).
DEFAULT_PHASE_CLOCK = True
DEFAULT_PHASE_SLOW_PERIOD = 64
DEFAULT_PHASE_FAST_PERIOD = 8


def phase_clock_enabled() -> bool:
    return _env_bool("DECADIC_PHASE_CLOCK", DEFAULT_PHASE_CLOCK)


def phase_slow_period() -> int:
    return max(2, int(_lc_float("DECADIC_PHASE_SLOW_PERIOD", DEFAULT_PHASE_SLOW_PERIOD, 2)))


def phase_fast_period() -> int:
    return max(2, int(_lc_float("DECADIC_PHASE_FAST_PERIOD", DEFAULT_PHASE_FAST_PERIOD, 2)))


# WS-IND I1: attention schema — predictive model of the system's own attention
# (gate outcome + ignition), trained on realized outcomes; prediction feeds
# back (zero-init ingress) and contributes a BOUNDED anticipatory gate bias
# sharing the E2 modulation cap. Default ON (zero-init -> birth-identical).
DEFAULT_ATTENTION_SCHEMA = True
DEFAULT_SCHEMA_LOSS_WEIGHT = 0.25
DEFAULT_SCHEMA_BIAS_GAIN = 0.05  # threshold drop at p(escalate)=1.0
DEFAULT_SCHEMA_BIAS_CAP = 0.05  # schema's own share; setter re-clamps the total


def attention_schema_enabled() -> bool:
    return _env_bool("DECADIC_ATTENTION_SCHEMA", DEFAULT_ATTENTION_SCHEMA)


def schema_loss_weight() -> float:
    return _lc_float("DECADIC_SCHEMA_LOSS_WEIGHT", DEFAULT_SCHEMA_LOSS_WEIGHT, 0.0)


def schema_bias_gain() -> float:
    return _lc_float("DECADIC_SCHEMA_BIAS_GAIN", DEFAULT_SCHEMA_BIAS_GAIN, 0.0, 0.15)


def schema_bias_cap() -> float:
    return _lc_float("DECADIC_SCHEMA_BIAS_CAP", DEFAULT_SCHEMA_BIAS_CAP, 0.0, 0.15)


# WS-IND I3: per-slot reality monitoring — relative reliability from per-slot
# noise EMAs, composed into the E6 routing gate (relevance x reliability).
DEFAULT_SLOT_RELIABILITY = True
DEFAULT_SLOT_REL_FAST_ALPHA = 0.3
DEFAULT_SLOT_REL_NOISE_ALPHA = 0.1
DEFAULT_SLOT_REL_FLOOR = 0.25
DEFAULT_SLOT_REL_WARMUP = 16


def slot_reliability_enabled() -> bool:
    return _env_bool("DECADIC_SLOT_RELIABILITY", DEFAULT_SLOT_RELIABILITY)


def slot_rel_fast_alpha() -> float:
    return _lc_float("DECADIC_SLOT_REL_FAST_ALPHA", DEFAULT_SLOT_REL_FAST_ALPHA, 0.01, 1.0)


def slot_rel_noise_alpha() -> float:
    return _lc_float("DECADIC_SLOT_REL_NOISE_ALPHA", DEFAULT_SLOT_REL_NOISE_ALPHA, 0.01, 1.0)


def slot_rel_floor() -> float:
    return _lc_float("DECADIC_SLOT_REL_FLOOR", DEFAULT_SLOT_REL_FLOOR, 0.0, 1.0)


def slot_rel_warmup() -> int:
    return max(1, int(_lc_float("DECADIC_SLOT_REL_WARMUP", DEFAULT_SLOT_REL_WARMUP, 1)))


# WS-IND I4: belief-update tempering — event-evidence writes onto WM slots are
# scaled by the source slot's perceptual confidence (never blocks a first
# observation; tempers, doesn't veto). weight 0 -> exact parity.
DEFAULT_BELIEF_TEMPER = True
DEFAULT_BELIEF_TEMPER_WEIGHT = 0.5  # evidence gain = 1 - w*(1 - confidence)


def belief_temper_enabled() -> bool:
    return _env_bool("DECADIC_BELIEF_TEMPER", DEFAULT_BELIEF_TEMPER)


def belief_temper_weight() -> float:
    return _lc_float("DECADIC_BELIEF_TEMPER_WEIGHT", DEFAULT_BELIEF_TEMPER_WEIGHT, 0.0, 1.0)


# WS-IND I2: sequential deliberation — on escalated cycles a no-grad DRAFT
# forward runs (recurrent state restored afterward, so the state advances
# exactly once per cycle) and its conclusion re-enters the final forward via a
# zero-init ingress. Compute cost is bounded by the Type-2 refractory. Keep
# only if the detour A/B says round 2 beats one-shot (plan I2.2).
DEFAULT_WS_SEQ = True


def ws_seq_enabled() -> bool:
    return _env_bool("DECADIC_WS_SEQ", DEFAULT_WS_SEQ)


# WS-IND I5: quality-space smoothness — a light temporal local-isometry loss
# on the FSQ projection (nearby latents -> nearby codes); trains fsq_in on the
# side (trunk stays detached, E9's parity guarantee unchanged).
DEFAULT_SYMBOL_SMOOTHNESS = True
DEFAULT_SYMBOL_SMOOTH_WEIGHT = 0.05


def symbol_smoothness_enabled() -> bool:
    return _env_bool("DECADIC_SYMBOL_SMOOTHNESS", DEFAULT_SYMBOL_SMOOTHNESS)


def symbol_smooth_weight() -> float:
    return _lc_float("DECADIC_SYMBOL_SMOOTH_WEIGHT", DEFAULT_SYMBOL_SMOOTH_WEIGHT, 0.0)


# WS-DEPTH D1: metacognitive calibration — predict own next prediction-error
# and P(drive improves | action), scored against realized outcomes (outcome as
# target). Zero-init heads -> birth-identical; calibration telemetry is the
# measurement instrument for behavioral self-awareness probes.
DEFAULT_METACOG_CALIBRATION = True
DEFAULT_METACOG_CAL_LOSS_WEIGHT = 0.25


def metacog_calibration_enabled() -> bool:
    return _env_bool("DECADIC_METACOG_CALIBRATION", DEFAULT_METACOG_CALIBRATION)


def metacog_cal_loss_weight() -> float:
    return _lc_float("DECADIC_METACOG_CAL_LOSS_WEIGHT", DEFAULT_METACOG_CAL_LOSS_WEIGHT, 0.0)


# WS-DEPTH P1 (stage A): recurrent percept refinement — a zero-init residual
# cell iterates on the fused percept before stage 1, trained by next-percept
# prediction (a percept-level forward model). Percept-key invariance is the
# standing guardrail: byte-identical at init, drift only under the ramp.
DEFAULT_PERCEPT_REFINE = True
DEFAULT_PERCEPT_REFINE_ITERS = 2
DEFAULT_PERCEPT_REFINE_LOSS_WEIGHT = 0.25


def percept_refine_enabled() -> bool:
    return _env_bool("DECADIC_PERCEPT_REFINE", DEFAULT_PERCEPT_REFINE)


def percept_refine_iters() -> int:
    return max(1, int(_lc_float("DECADIC_PERCEPT_REFINE_ITERS", DEFAULT_PERCEPT_REFINE_ITERS, 1, 4)))


def percept_refine_loss_weight() -> float:
    return _lc_float("DECADIC_PERCEPT_REFINE_LOSS_WEIGHT", DEFAULT_PERCEPT_REFINE_LOSS_WEIGHT, 0.0)


# WS-DEPTH P2 (partial): top-down perception governance — an OPT-IN cap on
# how much of the percept may be carried top-down, + the topdown_frac
# telemetry dial. Default 1.0 = INACTIVE: full top-down (gate -> 0) is the
# deliberately-built occlusion capability (perception reconstructed from the
# persistent mental image when the senses go dark — suite-pinned in
# test_persistent_mental_image), and the evidence concern is CHRONIC
# decoupling, which the telemetry watches and a soak can cap via env. The
# temporal generative head lands with WS-PERCEIVE proper.
DEFAULT_PERCEPT_TOPDOWN_CAP = 1.0  # 1.0 = no clamp (parity); <1.0 = governor


def percept_topdown_cap() -> float:
    return _lc_float("DECADIC_PERCEPT_TOPDOWN_CAP", DEFAULT_PERCEPT_TOPDOWN_CAP, 0.1, 1.0)


# WS-DEPTH D2: unified self-model as a workspace ignition CANDIDATE. Content
# is a deterministic packing of SELF_VEC (pure); salience = gain x
# interoceptive urgency x a birth ramp (0 at cycle 0 -> the self never wins at
# birth: parity). Deprivation should turn attention inward.
DEFAULT_SELF_CANDIDATE = True
DEFAULT_SELF_CANDIDATE_GAIN = 0.5
DEFAULT_SELF_CANDIDATE_RAMP_CYCLES = 2000


def self_candidate_enabled() -> bool:
    return _env_bool("DECADIC_SELF_CANDIDATE", DEFAULT_SELF_CANDIDATE)


def self_candidate_gain() -> float:
    return _lc_float("DECADIC_SELF_CANDIDATE_GAIN", DEFAULT_SELF_CANDIDATE_GAIN, 0.0)


def self_candidate_ramp_cycles() -> int:
    return max(1, int(_lc_float("DECADIC_SELF_CANDIDATE_RAMP_CYCLES", DEFAULT_SELF_CANDIDATE_RAMP_CYCLES, 1)))


# WS-DEPTH D3: sequential deliberation rounds (generalizes WS-IND I2's
# draft/commit). k-1 no-grad draft rounds, each feeding the accumulated draft
# back; the learned query chooser is define-only (logged, not controlling)
# until the detour A/B earns it.
DEFAULT_WS_SEQ_ROUNDS = 2


def ws_seq_rounds() -> int:
    return max(2, min(3, int(_lc_float("DECADIC_WS_SEQ_ROUNDS", DEFAULT_WS_SEQ_ROUNDS, 2, 3))))


# WS-EXPAND E10.4 (prerequisite): inverse dynamics — infer the action behind a
# proprio transition, trained supervised on the agent's OWN lived triples (the
# FEL buffers). The labeling model imitation-from-observation requires; the
# demonstrator-labeling step waits on percepts carrying the other's body pose.
DEFAULT_INVERSE_MODEL = True
DEFAULT_INVERSE_MODEL_LOSS_WEIGHT = 0.25


def inverse_model_enabled() -> bool:
    return _env_bool("DECADIC_INVERSE_MODEL", DEFAULT_INVERSE_MODEL)


def inverse_model_loss_weight() -> float:
    return _lc_float("DECADIC_INVERSE_MODEL_LOSS_WEIGHT", DEFAULT_INVERSE_MODEL_LOSS_WEIGHT, 0.0)


def sf_gamma() -> float:
    return min(0.9999, max(0.0, float(os.environ.get("DECADIC_SF_GAMMA", str(DEFAULT_SF_GAMMA)))))


def sf_lambda() -> float:
    return min(1.0, max(0.0, float(os.environ.get("DECADIC_SF_LAMBDA", str(DEFAULT_SF_LAMBDA)))))


def sf_loss_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_SF_LOSS_WEIGHT", str(DEFAULT_SF_LOSS_WEIGHT))))


def sf_value_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_SF_VALUE_WEIGHT", str(DEFAULT_SF_VALUE_WEIGHT))))


def sf_value_ramp_cycles() -> int:
    return max(1, int(os.environ.get("DECADIC_SF_VALUE_RAMP_CYCLES", str(DEFAULT_SF_VALUE_RAMP_CYCLES))))


def sf_value_weight_for_cycle(cycle: int) -> float:
    """Linear 0 -> sf_value_weight() ramp so behavior starts identical to today."""
    ramp = sf_value_ramp_cycles()
    frac = min(1.0, max(0.0, float(cycle) / float(ramp)))
    return sf_value_weight() * frac


# --- Hindsight relabeling (HER) ---------------------------------------------
# A goal episode that ends without achievement (abandoned / died) still taught
# the agent how to reach the state it DID end in. HER relabels such episodes with
# the terminal feature actually achieved as the goal and re-pushes them, turning
# near-misses into positive training signal -- the densest attack on the sparse
# consume-only reward, and the literal "the journey still taught me" mechanism.
DEFAULT_HER_ENABLED = True
DEFAULT_HER_RELABEL_K = 1  # relabeled copies pushed per failed episode (achieved-goal strategy)


def her_enabled() -> bool:
    return _env_bool("DECADIC_HER_ENABLED", DEFAULT_HER_ENABLED)


def her_relabel_k() -> int:
    return max(0, int(os.environ.get("DECADIC_HER_RELABEL_K", str(DEFAULT_HER_RELABEL_K))))


# --- Model-based imagined replay (Dreamer-lite) -----------------------------
# During consolidation, optionally roll out short imagined trajectories with the
# agent's own forward models from sampled real start states, and train the SF/
# value targets on them too ("expanding on it in quiet thought"). OFF by default:
# imagined value is only as good as the world model, so it is bounded (short
# horizon) and trust-weighted to limit hallucinated value.
DEFAULT_IMAGINATION_ENABLED = False
DEFAULT_IMAGINATION_HORIZON = 5  # imagined steps rolled out per sampled start state
DEFAULT_IMAGINATION_WEIGHT = 0.25  # weight of the imagined-rollout SF loss vs the real-replay loss


def imagination_enabled() -> bool:
    return _env_bool("DECADIC_IMAGINATION_ENABLED", DEFAULT_IMAGINATION_ENABLED)


def imagination_horizon() -> int:
    return max(1, int(os.environ.get("DECADIC_IMAGINATION_HORIZON", str(DEFAULT_IMAGINATION_HORIZON))))


def imagination_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_IMAGINATION_WEIGHT", str(DEFAULT_IMAGINATION_WEIGHT))))
