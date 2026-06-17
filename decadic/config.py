"""Central constants for tensor/vector dimensions (Phase 1 stubs)."""

import os

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
DEFAULT_PARALLEL_SESSIONS = 5  # K: observations encoded per cycle (batched forward)
MAX_PARALLEL_SESSIONS = 16
DEFAULT_REVIVE_VIABILITY = 100.0  # viability restored by an admin revive

# Pooling of parallel-session encodes into the deliberative pass, and the
# persistent scene latent held in working memory. Env overrides:
# DECADIC_SESSION_RECENCY, DECADIC_WM_SCENE_ALPHA, DECADIC_WM_SCENE_BLEND.
DEFAULT_SESSION_RECENCY = 0.7  # gamma: weight decay per frame of age in the pooled percept
DEFAULT_WM_SCENE_ALPHA = 0.3  # EMA rate of the persisting scene latent (new evidence share)
DEFAULT_WM_SCENE_BLEND = 0.5  # scene-latent share of the attention vector vs entity hashes

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
DEFAULT_PLASTICITY_ALPHA = 0.1  # max magnitude of the plastic (Hebbian) overlay on weights
DEFAULT_PLASTICITY_ETA = 0.1  # Hebbian trace blend rate per cycle
DEFAULT_PLASTICITY_INSTABILITY_PCLOSS = 50.0  # pc-loss EMA above this auto-freezes plasticity

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
# detached (no cross-cycle BPTT).
DEFAULT_SELF_MODEL_FEEDBACK = False


def self_model_feedback_enabled() -> bool:
    return _env_bool("DECADIC_SELF_MODEL_FEEDBACK", DEFAULT_SELF_MODEL_FEEDBACK)


# Real global workspace (self-model program, Phase 2). When ON the post-hoc EMA
# blend of the working-memory attention summary into A is replaced by a
# capacity-limited winner-take-all competition with an ignition threshold: only a
# dominant coalition (share of the salience mass >= threshold) "ignites" and is
# globally broadcast (blended into A, fed back via the spine, boosts the episodic
# salience, and is described by the narrative). Below threshold there is no
# ignition: A holds its prior (nothing reaches global broadcast). Default OFF: the
# off-branch is the existing EMA blend, byte-identical to before. It is a live
# per-agent toggle (a pipeline branch, not an architecture change -> no rebuild).
DEFAULT_GWT_ENABLED = False


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
# the last committed moment (perception is held). 0 = off = today (the freshest
# percept is always "now"), byte-identical. A live per-agent setting.
DEFAULT_INTEGRATION_WINDOW_MS = 0.0


def integration_window_ms() -> float:
    return max(0.0, float(os.environ.get("DECADIC_INTEGRATION_WINDOW_MS", str(DEFAULT_INTEGRATION_WINDOW_MS))))


def integration_window_max_frames() -> int:
    return max(1, int(os.environ.get("DECADIC_INTEGRATION_WINDOW_MAX_FRAMES", "8")))


# Predictive affect (self-model program, Phase 4). When ON a small forward model
# predicts the next-step affective context (viability/pain/pleasure/priority) from
# the previous cycle's actual affect, and the predicted delta is added to the
# episodic proxy before it is projected into the stack -- so the agent perceives
# in light of how it expects to feel. Default OFF; the predictor's output layer is
# zero-init so on is byte-identical until it learns. Rebuilds the brain on toggle
# (the predictor is a stack submodule, checkpointed with the brain).
DEFAULT_PREDICTIVE_AFFECT = False


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
# agent models, not just an implicit process. Default OFF; the ingress is zero-init
# (byte-identical until learned). Rebuilds the brain on toggle (stack submodule).
DEFAULT_REPRESENTED_SELF = False


def represented_self_enabled() -> bool:
    return _env_bool("DECADIC_REPRESENTED_SELF", DEFAULT_REPRESENTED_SELF)


# Memory-efficient training path (self-model program, Phase 6 — hardware-gated).
# When ON the per-cycle training step uses (a) an 8-bit Adam optimizer when
# bitsandbytes is importable on CUDA (halving the optimizer-moment memory, the
# single largest training cost for the heavy presets) and (b) a bf16 autocast
# around the stack forward on CUDA (cutting activation memory). Both fall back
# silently to the fp32 path when unavailable (no bnb / CPU), and the flag defaults
# OFF, so the standard path is byte-identical. Aimed at fitting the 250m/500m/1b
# tiers on a single consumer GPU.
DEFAULT_MEMORY_EFFICIENT_TRAINING = False


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

# --- Tactile world model (full-body touch) ---------------------------------
# The body streams a soft, normalized per-part contact load for every touch
# sensor (feet, hands, shins, thighs, arms, torso, head, waist, butt). A tactile
# forward-model head predicts the NEXT per-part load from (state, action), so the
# brain forms tactile expectations and learns from prediction error which actions
# load which body part -- the per-limb credit-assignment signal for learning to
# push off. Touch has no innate setpoint, so this is PE-only (no preference term).
TACTILE_PRED_DIM = 16  # per-part loads the tactile world model predicts (= touch sensor count)
DEFAULT_AI_TACTILE_FWD_WEIGHT = 1.0  # weight of the tactile forward-model PE loss


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


def ai_intero_pref_weight() -> float:
    return max(0.0, float(os.environ.get("DECADIC_AI_INTERO_PREF_WEIGHT", str(DEFAULT_AI_INTERO_PREF_WEIGHT))))


def drive_priority_gain() -> float:
    """How strongly deprivation severity scales the drive-reduction loss weight."""
    return max(
        0.0, float(os.environ.get("DECADIC_DRIVE_PRIORITY_GAIN", str(DEFAULT_DRIVE_PRIORITY_GAIN)))
    )


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

# Data association of discovered proposals into working-memory object files.
DEFAULT_ASSOC_APPEARANCE_WEIGHT = 0.6  # appearance-cosine share of the match score (vs position)
DEFAULT_ASSOC_MATCH_THRESHOLD = 0.35  # min combined score to bind a proposal to an existing slot
DEFAULT_APPEARANCE_EMA = 0.5  # EMA rate of a slot's appearance fingerprint

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


def ltm_graph_enabled() -> bool:
    return _env_bool("DECADIC_LTM_GRAPH", DEFAULT_LTM_GRAPH_ENABLED)


def ltm_match_threshold() -> float:
    return float(os.environ.get("DECADIC_LTM_MATCH_THRESHOLD", str(DEFAULT_LTM_MATCH_THRESHOLD)))


def ltm_consolidate_min_seen() -> int:
    return max(1, int(os.environ.get("DECADIC_LTM_MIN_SEEN", str(DEFAULT_LTM_CONSOLIDATE_MIN_SEEN))))


def ltm_snapshot_limit() -> int:
    return max(1, int(os.environ.get("DECADIC_LTM_SNAPSHOT_LIMIT", str(DEFAULT_LTM_SNAPSHOT_LIMIT))))


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
DEFAULT_SF_GAMMA = 0.97  # discount on future features (horizon ~ 1/(1-gamma))
DEFAULT_SF_LAMBDA = 0.9  # eligibility-trace / lambda-return decay (credit smear over the journey)
DEFAULT_SF_LOSS_WEIGHT = 1.0  # weight of the SF TD(lambda) loss in the consolidator
DEFAULT_SF_VALUE_WEIGHT = 0.3  # MAX weight of the value-advantage policy-shaping term (post-ramp)
DEFAULT_SF_VALUE_RAMP_CYCLES = 2000  # cycles over which the shaping weight climbs 0 -> max


def sf_enabled() -> bool:
    return _env_bool("DECADIC_SF_ENABLED", DEFAULT_SF_ENABLED)


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
