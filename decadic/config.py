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
DEFAULT_EFFORT_ENERGY_SCALE = 0.01
DEFAULT_WORK_ENERGY_SCALE = 0.02
DEFAULT_FATIGUE_RECOVERY_S = 8.0
DEFAULT_FATIGUE_PAIN_GAIN = 0.35
DEFAULT_STRAIN_PAIN_GAIN = 0.25
DEFAULT_EFFORT_MAX_ENERGY_DRAIN_PER_OBS = 0.08
DEFAULT_EFFORT_DRAIN_GRACE_MODE = "dojo_or_braced"


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


def type2_far_distance() -> float:
    try:
        v = float(os.environ.get("DECADIC_TYPE2_FAR_DISTANCE", str(DEFAULT_TYPE2_FAR_DISTANCE)))
    except (TypeError, ValueError):
        return DEFAULT_TYPE2_FAR_DISTANCE
    return min(1.0, max(0.0, v))


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
