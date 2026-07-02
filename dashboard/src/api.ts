/** Typed REST client for the Decadic server. */

export function httpBase(): string {
  const v = import.meta.env.VITE_DECADIC_HTTP;
  if (typeof v === "string" && v.trim().length > 0) return v.replace(/\/+$/, "");
  return "http://127.0.0.1:8765";
}

export type AgentStatus = "alive" | "dead";

export type AgentSummary = {
  agent_id: string;
  neural_enabled: boolean;
  cycles_completed: number;
  paused: boolean;
  status?: AgentStatus;
  died_at_cycle?: number | null;
  encoder_mode?: string | null;
  has_body?: boolean;
  preset?: string | null;
};

export type PlasticityConfig = {
  // True only when the stack was built with at least one of A/B/C enabled.
  available: boolean;
  plasticity_enabled: boolean;
  sparse_enabled: boolean;
  growth_enabled: boolean;
  // Live means/measurements; null when the matching mechanism is off or
  // not yet measured (e.g. plastic_alpha_mean() with no plastic overlay).
  plasticity_alpha: number | null;
  plasticity_alpha_effective?: number | null;
  sparse_density: number | null;
  awake_neurons: number;
  allocated_neurons: number;
  max_neurons: number;
};

export type CapacityConfig = {
  parallel_sessions: number;
  processing_mode?:
    | "serial_prefetch"
    | "stage_pipeline"
    | "persistent_parallel_perception"
    | "batching_observations"
    | string;
  stage_pipeline_enabled?: boolean;
  perceptual_processing_mode?: "persistent_parallel" | "batching_observations" | string;
  prefetch_queue_max?: number;
  prefetch_overload_policy?: string;
  ready_coalesce_policy?: string;
  working_memory_slots: number;
  working_memory_decay: number;
  // Manual assist-harness level; null = Auto (curriculum). Send -1 to clear.
  assist_override?: number | null;
  // Body support system: "guided" (assist-as-needed harness) | "legacy" (assist).
  curriculum_mode?: string;
  // Homeostasis: "metabolic" runs the wall-clock model; "immortal" pins reservoirs.
  viability_mode?: string;
  // Time-acceleration of the metabolic clock (1 = real 1:1 human timeline).
  metabolic_compression?: number;
  // "oracle" (graph handed by sim) vs "discovered" (graph from the agent's camera).
  perception_mode?: string;
  // Top-down predictive-perception loop (a core faculty; rebuilds the brain).
  perception_feedback?: boolean;
  // Self-state feedback spine: the previous cycle's self-report (A/C/E) shapes
  // the next cycle. Research faculty (default off); rebuilds the brain on toggle.
  self_model_feedback?: boolean;
  // Predictive affect: a forward model anticipates the next-step affect and colours
  // perception. Research faculty (default off); rebuilds the brain on toggle.
  predictive_affect?: boolean;
  // Represented self: interoception/affect/capability written onto the self-node +
  // fed back via the spine. Research faculty (default off); rebuilds on toggle.
  represented_self?: boolean;
  // Sensory encoder mode: "hf" (real frozen CLIP/Whisper) | "zeros" (synthetic).
  encoder_mode?: string;
  // Trainable anonymous scene dynamics head; optional but default-on in discovered mode.
  scene_dynamics_enabled?: boolean;
  // Read-only observation toggles (apply live; never feed cognition).
  cognition_trace?: boolean;
  probe_capture?: boolean;
  // Global-workspace competition (winner-take-all + ignition + broadcast) instead
  // of the working-memory EMA blend into A (applies live; default off).
  gwt_enabled?: boolean;
  // Temporal-integration window (ms): bind a span of percepts into one committed
  // "now" (applies live; 0 = off = freshest percept is always now).
  integration_window_ms?: number;
  // Write-behind episodic persistence: per-cycle SQLite write off the cognitive
  // lock (applies live; default on). No write is lost.
  episodic_async?: boolean;
  // Write-behind LTM consolidation: stage 10's WM->LTM commit off the cognitive
  // lock (applies live; default on). No consolidation is lost.
  ltm_async?: boolean;
  // Neuroplasticity (A/B/C) live state + caps.
  plasticity?: PlasticityConfig;
};

export type ActionRecord = {
  cycle: number;
  action: {
    type?: string;
    parameters?: { direction?: number[]; speed?: number; risk?: number };
  };
};

export type StateBusSnapshot = {
  A_state_of_mind: number[];
  B_emotion_physio: number[];
  B_pain_scalar: number;
  B_pleasure_scalar: number;
  C_narrative_emb: number[];
  C_narrative_text_stub: string;
  D_priority_scalar: number;
  D_priority_label: string;
  E_metacognition: number[];
  F_action_history: ActionRecord[];
  cycle_index: number;
};

export type EgoNode = {
  role: string;
  // kind: oracle entity kinds, or discovered "unknown" / "self_part" (a body part).
  id?: string;
  kind?: string;
  position?: number[];
  relative?: number[];
  standing?: boolean;
  moving?: boolean;
  control_mode?: string;
  salience?: number;
  last_seen_cycle?: number;
  // Discovered mode: egocentric bearing (azimuth, elevation) and learned agency.
  bearing?: number[];
  agency?: number;
};

export type EgoEdge = {
  source: string;
  target: string;
  // "agency" = the learned "this is mine" (self -> body part) relation.
  kind: "spatial" | "proximity" | "affective" | "context" | "agency";
  weight: number;
  distance?: number;
};

export type EgoGraph = {
  nodes: EgoNode[];
  edges: EgoEdge[];
};

// --- Long-term knowledge graph (the persistent, unbounded "hippocampal index")
// One permanent node per consolidated object, keyed by its learned appearance.
// No semantic labels are exposed: nodes are colored by appearance_hash (a 0-359
// hue derived from the appearance embedding) so distinct objects look distinct
// without being named.
export type LtmNode = {
  id: string;
  kind: string;
  salience: number;
  seen_count: number;
  last_cycle: number;
  affect: number;
  degree: number;
  // Deterministic 0-359 hue from the appearance embedding (node color).
  appearance_hash: number;
  property_beliefs?: LtmPropertyBelief[];
  unstable_property_count?: number;
  avg_property_confidence?: number;
};

export type LtmPropertyBelief = {
  property_key: string;
  value?: number | number[] | null;
  mean: number;
  variance: number;
  confidence: number;
  evidence_count: number;
  first_cycle: number;
  last_cycle: number;
  source: string;
  unstable: boolean;
};

export type LtmEdge = {
  source: string;
  target: string;
  kind: string;
  weight: number;
  count?: number;
  last_cycle?: number;
};

export type SemanticStats = {
  entities: number;
  events: number;
  relationships: number;
  correlations: number;
  conclusions: number;
  values: number;
};

export type LtmGraphSnapshot = {
  nodes: LtmNode[];
  edges: LtmEdge[];
  // Unbounded totals (the windowed nodes/edges above are a read-out cap only).
  total_nodes: number;
  total_edges: number;
  rendered_nodes?: number;
  rendered_edges?: number;
  snapshot_limit?: number;
  truncated_nodes?: boolean;
  truncated_edges?: boolean;
  edge_kind_counts?: Record<string, number>;
  edge_pair_counts?: Record<string, number>;
  total_property_beliefs?: number;
  unstable_property_count?: number;
  semantic?: SemanticStats;
};

export type WorkingMemorySlot = {
  entity_id: string;
  kind: string;
  position?: number[] | null;
  relative?: number[] | null;
  heading?: number | null;
  salience: number;
  affective_weight: number;
  audio_intensity?: number;
  last_event?: string | null;
  last_seen_cycle: number;
  seen_count: number;
  in_view: boolean;
  // Discovered mode: image-space centroid, egocentric bearing, agency score.
  uv?: number[] | null;
  bearing?: number[] | null;
  agency?: number | null;
  confidence?: number | null;
  kind_hint?: string | null;
  entity_role?: string | null;
  precision?: number | null;
  provisional?: boolean;
  evidence_count?: number;
  contradiction_pressure?: number;
  event_links?: string[];
  relationship_links?: string[];
  motion?: number[] | null;
  local_motion?: number | null;
  retina_contrast?: number | null;
  looming?: number | null;
  prediction_error?: number | null;
  prediction_uncertainty?: number | null;
  occlusion_age?: number;
  property_evidence?: Record<string, number | number[]>;
  scene_entity_id?: string | null;
};

export type WorkingMemorySnapshot = {
  capacity: number;
  decay: number;
  cycle: number;
  scene_latent_rms?: number | null;
  scene_preview?: number[] | null;
  slots: WorkingMemorySlot[];
};

export type DiscoverySnapshot = {
  updates: number;
  precision: number;
  recall: number;
  id_churn: number;
  id_stability: number;
  body_part_accuracy: number;
  last_detected: number;
  last_oracle: number;
  last_matched: number;
  last_body_parts_found: number;
  last_body_parts_truth: number;
};

export type ObjectFileSnapshot = {
  object_id?: string | null;
  idx: number;
  centroid_uv?: number[] | null;
  relative?: number[] | null;
  bearing?: number[] | null;
  appearance?: number[] | null;
  motion?: number[] | null;
  depth?: number | null;
  persistence: number;
  agency: number;
  kind_hint: string;
  entity_role?: string;
  provisional?: boolean;
  confidence: number;
  presence: number;
  spread?: number | null;
  mask_entropy?: number | null;
  flow?: number[] | null;
  local_motion?: number | null;
  retina_contrast?: number | null;
  looming?: number | null;
  property_evidence?: Record<string, number | number[]>;
};

export type DiscoveryHealth = {
  status: string;
  collapsed: boolean;
  reason: string;
  active_proposals: number;
  object_files: number;
  stable_tracked_objects: number;
  centroid_spread: number;
  appearance_cosine_mean?: number | null;
  appearance_cosine_max?: number | null;
  mask_entropy_mean?: number | null;
  stuff_count: number;
  body_candidate_count: number;
  looming_count: number;
  flow_confidence: number;
  low_confidence_count: number;
  ltm_write: string;
  perception_candidate_count?: number;
  perception_candidate_capacity?: number;
  scene_entities?: number;
  scene_focus_count?: number;
  wm_focus_capacity?: number;
  scene_entity_capacity?: number;
  attention_top?: AttentionTopSnapshot[];
  active_drive_deficits?: Record<string, number | string>;
};

export type PerceptionOrganSnapshot = {
  frame_seen: boolean;
  stale_frame: boolean;
  grid_size: number;
  flow_confidence: number;
  global_motion: number;
  local_motion_max: number;
  local_motion_mean: number;
  looming_count: number;
  stuff_count: number;
  body_candidate_count: number;
  foreground_count: number;
  checkpoint_status: string;
  bootstrap_proposal_count?: number;
  candidate_count?: number;
  candidate_capacity?: number;
};

export type AttentionTopSnapshot = {
  entity_id: string;
  attention_score: number;
  attention_reasons?: Record<string, number>;
  drive_match?: Record<string, number>;
};

export type RetinotopicMapSnapshot = {
  width: number;
  height: number;
  intensity: number[][];
  contrast: number[][];
  frame_delta: number[][];
};

export type LtmConsolidationStatus = {
  status: string;
  reason?: string;
  accepted_ids?: string[];
  identity_refresh?: boolean;
  property_update?: boolean;
  relationship_update?: boolean;
  relationship_updates_skipped?: number;
  semantic_update?: Partial<SemanticStats>;
  semantic_entities?: number;
  semantic_events?: number;
  semantic_relationships?: number;
  semantic_correlations?: number;
  semantic_conclusions?: number;
  semantic_values?: number;
  total_property_beliefs?: number;
  unstable_property_count?: number;
  avg_property_confidence?: number;
};

export type SceneEntitySnapshot = {
  entity_id: string;
  object_id?: string | null;
  kind_hint: string;
  entity_role?: string;
  provisional?: boolean;
  visible: boolean;
  occluded: boolean;
  occlusion_age: number;
  centroid_uv?: number[] | null;
  relative?: number[] | null;
  depth?: number | null;
  motion?: number[] | null;
  confidence: number;
  persistence: number;
  salience: number;
  attention_score?: number;
  attention_reasons?: Record<string, number>;
  drive_match?: Record<string, number>;
  agency: number;
  looming: number;
  local_motion: number;
  retina_contrast: number;
  predicted_centroid_uv?: number[] | null;
  predicted_relative?: number[] | null;
  prediction_visibility?: number | null;
  prediction_uncertainty?: number | null;
  prediction_error?: number | null;
  property_evidence?: Record<string, number | number[]>;
  first_cycle: number;
  last_seen_cycle: number;
  seen_count: number;
};

export type SceneRelationSnapshot = {
  src: string;
  dst: string;
  kind: string;
  confidence: number;
  last_cycle: number;
};

export type SceneWorkspaceSnapshot = {
  cycle: number;
  entity_count: number;
  visible_count: number;
  occluded_count: number;
  stable_count: number;
  stuff_count: number;
  body_candidate_count: number;
  duplicate_identity_count: number;
  prediction_unstable_count?: number;
  prediction_count?: number;
  reidentified_count?: number;
  prediction_assisted_count?: number;
  duplicate_prevention_count?: number;
  candidate_count?: number;
  focus_capacity?: number;
  active_drive_deficits?: Record<string, number | string>;
  attention_top?: AttentionTopSnapshot[];
  focus_ids: string[];
  prediction_error?: number | null;
  entities: SceneEntitySnapshot[];
  relations: SceneRelationSnapshot[];
};

export type SceneHealthSnapshot = {
  entity_count: number;
  visible_count: number;
  occluded_count: number;
  stable_count: number;
  stuff_count: number;
  body_candidate_count: number;
  duplicate_identity_count: number;
  focus_count: number;
  prediction_error?: number | null;
  prediction_unstable_count?: number;
  prediction_count?: number;
  reidentified_count?: number;
  prediction_assisted_count?: number;
  duplicate_prevention_count?: number;
  candidate_count?: number;
  attention_top?: AttentionTopSnapshot[];
  active_drive_deficits?: Record<string, number | string>;
};

export type FocusSnapshot = {
  ids: string[];
  entities: SceneEntitySnapshot[];
};

export type WorkspaceIgnitionSnapshot = {
  enabled: boolean;
  ignited: boolean;
  share: number;
  threshold: number;
  n_candidates: number;
  winners: number[];
  focus_ids?: string[];
  scene_entity_count?: number;
  scene_relation_count?: number;
};

export type ScenePredictionSnapshot = {
  enabled: boolean;
  dynamics_enabled?: boolean;
  model_active?: boolean;
  error?: number | null;
  loss?: number | null;
  uncertainty?: number | null;
  prediction_count?: number;
  reidentified_count?: number;
  prediction_assisted_count?: number;
  duplicate_prevention_count?: number;
  unstable_count?: number;
  target?: string;
};

export type PerceptualSnapshot = {
  last_timestamp_iso: string | null;
  vision_resolution: number[] | null;
  audio_duration_s?: number | null;
  audio_rms?: number | null;
  proprio_position: number[] | null;
  proprio_orientation: number[] | null;
  proprio_velocity: number[] | null;
  proprio_joints: number[] | null;
  proprio_contacts: number[] | null;
  current_action_observed: string | null;
  recent_events: Array<Record<string, unknown>>;
  integration_ticks: number;
  egocentric_nodes: EgoNode[];
  egocentric_edges?: EgoEdge[];
  egocentric_graph?: EgoGraph;
  working_memory?: WorkingMemorySnapshot;
  object_files?: ObjectFileSnapshot[];
  scene_workspace?: SceneWorkspaceSnapshot | null;
  scene_health?: SceneHealthSnapshot | null;
  focus?: FocusSnapshot | null;
  workspace_ignition?: WorkspaceIgnitionSnapshot | null;
  scene_prediction?: ScenePredictionSnapshot | null;
  discovery_health?: DiscoveryHealth | null;
  perception_organ?: PerceptionOrganSnapshot | null;
  retinotopic_map?: RetinotopicMapSnapshot | null;
  ltm_consolidation?: LtmConsolidationStatus | null;
  // Persistent, unbounded long-term relational graph (the hippocampal index).
  // Present whenever the long-term graph is enabled (default on).
  ltm_graph?: LtmGraphSnapshot | null;
  // "oracle" vs "discovered"; discovery is the eval report (discovered mode only).
  perception_mode?: string;
  discovery?: DiscoverySnapshot | null;
};

export type AgentState = {
  agent_id: string;
  neural_enabled: boolean;
  status?: AgentStatus;
  died_at_cycle?: number | null;
  paused?: boolean;
  capacity?: CapacityConfig;
  state_bus: StateBusSnapshot;
  perceptual: PerceptualSnapshot;
  viability: { value: number };
  metrics: Record<string, number | string | null>;
  vision_views?: string[];
  last_cycle_trace?: CycleTrace | null;
  // Cognitive trace: the human-readable "why" for the latest cycle.
  cognitive_trace?: CognitiveTrace | null;
};

// --- Cognitive trace ("why" monitoring) -----------------------------------

// One survival-objective driver: what the action does to a goal dimension.
export type IntentDriver = {
  goal: string;
  group: string;
  weight: number;
  predicted: number;
  preferred: number;
  current?: number | null;
  deviation: number;
  // >0 means the chosen action reduces deviation from the prior vs. standing still.
  action_delta: number;
  contribution: number;
};

export type SurpriseDim = {
  name: string;
  predicted: number;
  actual: number;
  residual: number;
};

export type SalientNode = {
  node_id?: string | null;
  kind?: string | null;
  salience?: number;
  affective_weight?: number;
};

export type Attribution = {
  target: string;
  channels: Record<string, number>;
  fractions: Record<string, number>;
  node?: SalientNode | null;
};

export type RecalledEpisode = {
  cycle?: number | null;
  similarity: number;
  salience: number;
  priority?: string | null;
  pain?: number | null;
  pleasure?: number | null;
  viability?: number | null;
  action_type?: string | null;
};

export type ProbeReadout = {
  predicted: number;
  kind: "regression" | "classification";
  best_latent: string;
  axis: number;
  score: number;
  score_kind: "r2" | "accuracy";
};

export type CounterfactualCandidate = {
  action: string;
  drive_cost?: number;
  intero_pred?: number[];
  proprio_pred?: number[];
  motor_rms?: number;
};

export type Counterfactuals = {
  candidates: CounterfactualCandidate[];
  objective: string;
};

export type CognitiveTrace = {
  cycle: number;
  intent: { summary: string; drivers: IntentDriver[] };
  self_surprise: {
    dims: SurpriseDim[];
    mean_abs_residual: number | null;
    summary: string;
  };
  affect: { pain: number; pleasure: number; risk: number; priority: string };
  recalled_episode?: RecalledEpisode | null;
  salient?: Attribution | null;
  counterfactuals?: Counterfactuals | null;
  probes?: Record<string, ProbeReadout> | null;
  narrative: string;
};

export type CognitiveHistoryRec = {
  cycle: number;
  intent: string;
  top_goal?: string | null;
  pain?: number | null;
  pleasure?: number | null;
  risk?: number | null;
  surprise?: number | null;
};

export type ExplainReport = {
  agent_id: string;
  cycle: number;
  trace: CognitiveTrace | null;
  history: CognitiveHistoryRec[];
  on_demand?: { counterfactuals?: Counterfactuals } | null;
};

export type StageTraceRec = {
  stage: number;
  name: string;
  payload: {
    timing_ms?: number | null;
    activity?: number | null;
    pc_part?: number | null;
    salience?: number | null;
    neural?: boolean;
    activations?: number[];
  };
};

export type CycleTrace = {
  cycle: number;
  stages: StageTraceRec[];
};

export type Metrics = {
  cycles_completed: number;
  approx_cycles_per_sec: number;
  neural_pc_loss_last: number;
  loss_total?: number;
  loss_dominant_term?: string;
  loss_dominant_fraction?: number;
  loss_terms?: Record<string, { raw?: number; weighted?: number }>;
  loss_canary_state?: string;
  loss_canary_reason?: string;
  loss_canary_pressure?: number;
  loss_canary_optimizer_action?: string;
  loss_canary_step_scale?: number;
  loss_canary_ema?: number | null;
  loss_canary_pc_ema?: number | null;
  loss_canary_slope_ema?: number | null;
  loss_canary_pc_slope_ema?: number | null;
  loss_canary_jump_ratio?: number;
  learning_rate: number;
  fast_path_hits: number;
  last_cycle_wall_ms: number;
  queue_depth: number;
  viability: number;
  priority_label: string;
  last_observation_iso: string | null;
  gpu_memory_max_allocated: number;
  paused: boolean;
  status?: AgentStatus;
  died_at_cycle?: number | null;
  parallel_sessions?: number;
  processing_mode?: string;
  stage_pipeline_enabled?: boolean;
  perceptual_processing_mode?: string;
  stage_pipeline_active_sessions?: number;
  stage_pipeline_ready_sessions?: number;
  stage_pipeline_committed_sessions?: number;
  stage_pipeline_committed_per_s?: number;
  stage_pipeline_dropped_sessions?: number;
  stage_pipeline_stale_sessions?: number;
  stage_pipeline_failed_sessions?: number;
  stage_pipeline_queue_depths?: Record<string, number>;
  stage_pipeline_inflight?: Record<string, number>;
  stage_pipeline_latency_ms?: Record<string, number>;
  stage_pipeline_recent_sessions?: Array<Record<string, unknown>>;
  stage_pipeline_selected_session?: Record<string, unknown> | null;
  stage_pipeline_arbitration_reason?: string | null;
  frames_received?: number;
  frames_prefetched?: number;
  frames_folded?: number;
  frames_deep_processed?: number;
  coalesced_sessions?: number;
  information_loss?: number;
  producer_overlap_ratio?: number;
  decode_on_consume_ms?: number;
  consume_wait_ms?: number;
  ready_queue_depth?: number;
  ready_coalesce_policy?: string;
  fold_lag_ms?: number;
  prefetch_queue_depth?: number;
  prefetch_queue_max?: number;
  prefetch_overload_policy?: string;
  prefetch_backpressure_events?: number;
  prefetch_backpressure_ms?: number;
  oldest_unfolded_age_ms?: number;
  prefetch_backpressure_warning?: boolean;
  oldest_unfolded_warning?: boolean;
  pipeline_sessions?: number;
  perception_queue_depth?: number;
  perception_inflight?: number;
  perception_ingest_hz?: number;
  perception_commit_hz?: number;
  frames_committed?: number;
  frames_dropped?: number;
  commit_lag_ms?: number;
  sample_age_ms?: number;
  batching_fallback?: boolean;
  working_memory_slots?: number;
  working_memory_decay?: number;
  encode_phase_ms?: number;
  encoder_mode?: string | null;
  preset?: string | null;
  forward_model_error?: number;
  tactile_pred_error?: number;
  assist_gain?: number;
  assist_override?: number | null;
  motor_babble_sigma?: number;
  motor_activity_rms?: number;
  motor_command?: number[];
  // Joint-brace guidance telemetry. rom_mean: mean per-joint range of motion
  // earned (0 welded -> 1 free). brace_engaged: mean brace tightness (1 welded
  // -> 0 free). joint_rom: per-hinge ROM fraction for the per-joint bars.
  rom_mean?: number;
  brace_engaged?: number;
  joint_rom?: number[];
  // Master on/off for the joint-brace orthosis (off -> native springs, free body).
  braces_enabled?: boolean;
  // Active joint-brace stance/posture and (for motion stances) its phase in [0, 1].
  stance?: string;
  stance_phase?: number;
  // Hold mode: the active movement is welded + looping until manually disabled.
  movement_hold?: boolean;
  // Manual scaffold safety: recenter/repose selected motion stance after a fall.
  manual_auto_reset?: boolean;
  foot_load_l?: number;
  foot_load_r?: number;
  hand_load_l?: number;
  hand_load_r?: number;
  teacher_support_active?: boolean;
  teacher_support_force?: number;
  teacher_support_torque?: number;
  teacher_drop_m?: number;
  teacher_target_drop_m?: number;
  teacher_height_error_m?: number;
  teacher_vertical_velocity?: number;
  teacher_support_mode?: string;
  caregiver_parent_present?: boolean;
  caregiver_missing_parent?: boolean;
  caregiver_kind?: string;
  caregiver_status?: string;
  caregiver_request_kind?: string;
  caregiver_last_offer_item?: string;
  caregiver_delivery_count?: number;
  caregiver_pending_request?: boolean;
  // Full-body touch: per-part contact loads (short name -> force/body weight).
  part_loads?: Record<string, number>;
  body_map?: {
    parts?: string[];
    contact_load?: number[];
    effort?: number[];
    work?: number[];
    strain?: number[];
    fatigue?: number[];
    pain?: number[];
  };
  effort?: {
    actuator_effort?: number[];
    actuator_work?: number[];
    joint_strain?: number[];
    joint_fatigue?: number[];
    effort_total?: number;
    work_total?: number;
    strain_total?: number;
    fatigue_total?: number;
    pain_total?: number;
    support_effort?: number;
  };
  effort_total?: number;
  work_total?: number;
  strain_total?: number;
  fatigue_total?: number;
  pain_total?: number;
  support_effort?: number;
  effort_energy_delta?: number;
  fatigue_pain?: number;
  strain_pain?: number;
  most_pained_part?: string;
  most_pained_part_pain?: number;
  net_energy_return?: number;
  effort_pred_error?: number;
  resource_relief_events?: number;
  // Locomotion / gait telemetry (eval-only; never read by cognition). Drives the
  // curriculum gates: cumulative path length, straight-line displacement from the
  // run origin, rolling fall-rate, foot-alternation regularity, consume count.
  distance_traveled?: number;
  net_displacement?: number;
  fall_rate?: number;
  gait_regularity?: number;
  consume_events?: number;
  // Anti-camping: RNG seed of the current life's resource scatter (-1 if off).
  resource_seed?: number;
  // Goal lifecycle (explicit latched intent for credit assignment): active goal
  // label or "none", idle|active status, cycles the goal has been open, episodes
  // closed, and the last close reason (achieved|abandoned|truncated|died).
  goal?: string;
  goal_status?: string;
  goal_dwell?: number;
  goal_episodes?: number;
  goal_last_outcome?: string;
  // Episodic replay timeline: closed goal episodes annotated with lambda-returns.
  episodes_closed?: number;
  episode_last_len?: number;
  episode_last_return?: number;
  // Successor-features value (Layer-2 incentive salience): predicted value of the
  // chosen action and the active (ramped 0->max) value-shaping weight.
  sf_value?: number;
  sf_value_weight?: number;
  // Hindsight relabeling: cumulative relabeled transitions and the last episode's.
  her_relabels?: number;
  her_last?: number;
  // Live curriculum overrides (null -> follow the process-env default).
  ai_intero_pref_weight?: number | null;
  drive_priority_gain?: number | null;
  drive_priority_gain_configured?: number;
  drive_priority_gain_effective?: number;
  hydration?: number;
  energy?: number;
  integrity?: number;
  stress?: number;
  viability_mode?: string;
  metabolic_compression?: number;
  time_to_death_s?: number | null;
  // Neuroplasticity (A/B/C) telemetry.
  plasticity_enabled?: boolean;
  sparse_enabled?: boolean;
  growth_enabled?: boolean;
  plasticity_alpha?: number | null;
  plasticity_alpha_configured?: number | null;
  plasticity_alpha_effective?: number | null;
  plasticity_guardian_state?: string;
  plasticity_guardian_action?: string;
  plasticity_pc_ema?: number | null;
  plasticity_pc_slope_ema?: number | null;
  plasticity_overlay_ratio_mean?: number | null;
  plasticity_overlay_ratio_max?: number | null;
  plasticity_freeze_count?: number;
  plasticity_thaw_count?: number;
  plasticity_warmup_blocked_reason?: string;
  plasticity_stable_cycles?: number;
  plasticity_thaw_eligible?: boolean;
  plasticity_thaw_cycles_remaining?: number;
  sparse_density?: number | null;
  awake_neurons?: number;
  allocated_neurons?: number;
  active_connections?: number;
  max_neurons?: number;
  rewire_events?: number;
  growth_events?: number;
  plasticity_frozen?: boolean;
  consolidation_sync_delta_mean?: number;
  consolidation_sync_delta_max?: number;
  consolidation_sync_moved_params?: number;
  consolidation_sync_reset_params?: number;
  // Perception feedback loop (top-down predictive perception) telemetry.
  perception_feedback?: boolean;
  precision_gate_mean?: number | null;
  perceptual_pred_error?: number | null;
  // Homeostatic drive-reduction telemetry (self-learned thirst/hunger seeking).
  homeostatic_drive?: boolean;
  intero_drive?: number | null;
  intero_pred_error?: number | null;
  // Discovered-perception telemetry (object discovery + body-self agency).
  perception_mode?: string;
  discovered_perception?: boolean;
  slots_present?: number;
  slot_recon_error?: number;
  discovered_objects?: number;
  scene_dynamics_enabled?: boolean;
  scene_dynamics_model_active?: boolean;
  scene_dynamics_loss?: number | null;
  scene_dynamics_uncertainty?: number | null;
  scene_dynamics_predictions?: number;
  scene_dynamics_matches?: number;
  self_parts?: number;
  agency_mean?: number;
  agency_loss?: number;
};

export type BrainLayer = {
  id: string;
  label: string;
  stage: number;
  units: number;
  params: number;
};

export type BrainFiber = { si: number; di: number; w: number };

export type BrainEdge = {
  src: string;
  dst: string;
  weight_count: number;
  w_rms: number;
  fibers: BrainFiber[];
};

export type BrainTopology = {
  layers: BrainLayer[];
  edges: BrainEdge[];
  totals: {
    neurons: number;
    connections: number;
    params: number;
    d_model: number;
    preset: string | null;
    // Present only when the stack has growable plastic blocks.
    awake_neurons?: number;
    allocated_neurons?: number;
    active_connections?: number;
  };
};

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(`${httpBase()}${path}`);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

export async function fetchAgents(): Promise<AgentSummary[]> {
  const r = await getJson<{ agents: AgentSummary[] }>("/agents");
  return r.agents;
}

export async function fetchState(agentId: string): Promise<AgentState> {
  const r = await getJson<{ payload: AgentState }>(`/agent/${agentId}/state`);
  return r.payload;
}

export async function fetchMetrics(agentId: string): Promise<Metrics> {
  const r = await getJson<{ metrics: Metrics }>(`/agent/${agentId}/metrics`);
  return r.metrics;
}

export function visionUrl(agentId: string, tick: number, camera?: string): string {
  const cam = camera && camera !== "egocentric" ? `&camera=${encodeURIComponent(camera)}` : "";
  return `${httpBase()}/agent/${agentId}/vision?t=${tick}${cam}`;
}

export function audioUrl(agentId: string, tick: number): string {
  return `${httpBase()}/agent/${agentId}/audio?t=${tick}`;
}

async function postJson(path: string): Promise<void> {
  const r = await fetch(`${httpBase()}${path}`, { method: "POST" });
  if (!r.ok) {
    let detail = `${path}: HTTP ${r.status}`;
    try {
      const body = (await r.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // keep status-only fallback
    }
    throw new Error(detail);
  }
}

export function pauseAgent(agentId: string): Promise<void> {
  return postJson(`/agent/${agentId}/pause`);
}

export function resumeAgent(agentId: string): Promise<void> {
  return postJson(`/agent/${agentId}/resume`);
}

export function resetAgent(agentId: string): Promise<void> {
  return postJson(`/agent/${agentId}/reset`);
}

export function openBodyViewer(agentId: string, open = true): Promise<void> {
  return postJson(`/agent/${agentId}/body/viewer?open=${open}`);
}

export function reviveAgent(agentId: string, restoreTo?: number): Promise<void> {
  const q = restoreTo != null ? `?restore_to=${restoreTo}` : "";
  return postJson(`/agent/${agentId}/revive${q}`);
}

// Provision the agent with water/food/medical support. mode "near" asks the body to place the
// (unlabeled) prop a step away so the agent must perceive and walk to it;
// mode "direct" asks the body to show the prop in the egocentric camera and
// move it toward the head until normal consumption fires. "admin" is the old
// instant reservoir top-up for explicit rescue/testing use.
export async function giveResource(
  agentId: string,
  resource: "water" | "food" | "medical_kit",
  mode: "near" | "direct" | "admin",
): Promise<void> {
  const r = await fetch(
    `${httpBase()}/agent/${agentId}/give?resource=${resource}&mode=${mode}`,
    { method: "POST" },
  );
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      const body = (await r.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // non-JSON error body; keep the status code message
    }
    throw new Error(detail);
  }
}

export type PlasticityKnobs = {
  plasticity_alpha?: number;
  sparse_density?: number;
  max_neurons?: number;
};

export async function configureAgent(
  agentId: string,
  cfg: Partial<CapacityConfig> & PlasticityKnobs,
): Promise<CapacityConfig> {
  const params = new URLSearchParams();
  if (cfg.parallel_sessions != null)
    params.set("parallel_sessions", String(cfg.parallel_sessions));
  if (cfg.processing_mode != null)
    params.set("processing_mode", String(cfg.processing_mode));
  if (cfg.stage_pipeline_enabled != null)
    params.set("stage_pipeline_enabled", cfg.stage_pipeline_enabled ? "1" : "0");
  if (cfg.perceptual_processing_mode != null)
    params.set("perceptual_processing_mode", String(cfg.perceptual_processing_mode));
  if (cfg.working_memory_slots != null)
    params.set("working_memory_slots", String(cfg.working_memory_slots));
  if (cfg.working_memory_decay != null)
    params.set("working_memory_decay", String(cfg.working_memory_decay));
  if (cfg.assist_override != null)
    params.set("assist_override", String(cfg.assist_override));
  if (cfg.curriculum_mode != null)
    params.set("curriculum_mode", cfg.curriculum_mode);
  if (cfg.viability_mode != null)
    params.set("viability_mode", String(cfg.viability_mode));
  if (cfg.metabolic_compression != null)
    params.set("metabolic_compression", String(cfg.metabolic_compression));
  if (cfg.perception_mode != null)
    params.set("perception_mode", String(cfg.perception_mode));
  if (cfg.perception_feedback != null)
    params.set("perception_feedback", cfg.perception_feedback ? "1" : "0");
  if (cfg.self_model_feedback != null)
    params.set("self_model_feedback", cfg.self_model_feedback ? "1" : "0");
  if (cfg.predictive_affect != null)
    params.set("predictive_affect", cfg.predictive_affect ? "1" : "0");
  if (cfg.represented_self != null)
    params.set("represented_self", cfg.represented_self ? "1" : "0");
  if (cfg.encoder_mode != null) params.set("encoder_mode", String(cfg.encoder_mode));
  if (cfg.cognition_trace != null)
    params.set("cognition_trace", cfg.cognition_trace ? "1" : "0");
  if (cfg.probe_capture != null)
    params.set("probe_capture", cfg.probe_capture ? "1" : "0");
  if (cfg.gwt_enabled != null)
    params.set("gwt_enabled", cfg.gwt_enabled ? "1" : "0");
  if (cfg.integration_window_ms != null)
    params.set("integration_window_ms", String(cfg.integration_window_ms));
  if (cfg.episodic_async != null)
    params.set("episodic_async", cfg.episodic_async ? "1" : "0");
  if (cfg.ltm_async != null)
    params.set("ltm_async", cfg.ltm_async ? "1" : "0");
  if (cfg.plasticity_alpha != null)
    params.set("plasticity_alpha", String(cfg.plasticity_alpha));
  if (cfg.sparse_density != null)
    params.set("sparse_density", String(cfg.sparse_density));
  if (cfg.max_neurons != null) params.set("max_neurons", String(cfg.max_neurons));
  const r = await fetch(`${httpBase()}/agent/${agentId}/config?${params.toString()}`, {
    method: "POST",
  });
  if (!r.ok) throw new Error(`config: HTTP ${r.status}`);
  return (await r.json()) as CapacityConfig;
}

export function recenterBody(agentId: string): Promise<void> {
  return postJson(`/agent/${agentId}/body/recenter`);
}

// Re-weld all joint braces (restart the ROM curriculum from fully welded).
export function resetBraces(agentId: string): Promise<void> {
  return postJson(`/agent/${agentId}/body/reset_braces`);
}

// Master on/off for the joint-brace orthosis. Off -> hinges relax to native
// springs and the brain alone holds the body up (it can fall); earned ROM is
// preserved and resumes when switched back on.
export function setBracesEnabled(agentId: string, enabled: boolean): Promise<void> {
  return postJson(`/agent/${agentId}/body/braces?enabled=${enabled}`);
}

export interface StanceInfo {
  name: string;
  label: string;
  motion: boolean;
}

// Selectable body stances. Mirrors the backend stance library
// (decadic/embodiment/stances.py, also served at GET /body/stances); kept here so
// the selector renders without an extra round-trip.
export const STANCES: StanceInfo[] = [
  { name: "stand", label: "Stand & balance", motion: false },
  { name: "all_fours", label: "Kneel on all fours", motion: false },
  { name: "kneel_left", label: "Kneel (twist left)", motion: false },
  { name: "kneel_right", label: "Kneel (twist right)", motion: false },
  { name: "kneel_upright", label: "Sit upright on knees", motion: false },
  { name: "crawl", label: "Crawl (motion)", motion: true },
  { name: "sit_to_stand", label: "Rise up (motion)", motion: true },
  { name: "kneel_to_stand", label: "Kneel to stand (motion)", motion: true },
];

// Re-pose the body into a stance without changing manual brace state.
export function setStance(agentId: string, name: string): Promise<void> {
  return postJson(`/agent/${agentId}/body/stance?name=${encodeURIComponent(name)}`);
}

// Hold the active movement: weld every joint brace (suspend the ROM curriculum --
// no range-of-motion release) and loop motion stances continuously, so the
// selected movement runs on repeat until disabled. Off resumes the ROM ratchet.
export function setMovementHold(agentId: string, enabled: boolean): Promise<void> {
  return postJson(`/agent/${agentId}/body/movement_hold?enabled=${enabled}`);
}

export function setManualAutoReset(agentId: string, enabled: boolean): Promise<void> {
  return postJson(`/agent/${agentId}/body/manual_auto_reset?enabled=${enabled}`);
}

// Freeze (or release) the parent NPC where it stands. The flag lives in the body
// process, so it resets to "walking" whenever the scenario restarts.
export function setParentPaused(agentId: string, paused: boolean): Promise<void> {
  return postJson(`/agent/${agentId}/body/npc?paused=${paused}`);
}

export async function createAgent(preset?: string): Promise<string> {
  const q = preset ? `?preset=${encodeURIComponent(preset)}` : "";
  const r = await fetch(`${httpBase()}/agent${q}`, { method: "POST" });
  if (!r.ok) throw new Error(`create: HTTP ${r.status}`);
  const body = (await r.json()) as { agent_id: string };
  return body.agent_id;
}

export async function deleteAgent(agentId: string): Promise<void> {
  const r = await fetch(`${httpBase()}/agent/${agentId}`, { method: "DELETE" });
  if (!r.ok) throw new Error(`delete: HTTP ${r.status}`);
}

// --- Saved Agents library (durable, separate from the backups/ checkpoints) ---

export type SavedAgent = {
  save_id: string;
  name: string;
  created_at: string;
  source_agent_id?: string | null;
  preset?: string | null;
  encoder_mode?: string | null;
  viability_mode?: string | null;
  viability?: number | null;
  cycle_index?: number | null;
  has_memory?: boolean;
  notes?: string | null;
};

export async function saveAgent(
  agentId: string,
  body: { name: string; notes?: string },
): Promise<SavedAgent> {
  const r = await fetch(`${httpBase()}/agent/${agentId}/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`save: HTTP ${r.status}`);
  return (await r.json()) as SavedAgent;
}

export async function listSavedAgents(): Promise<SavedAgent[]> {
  const r = await getJson<{ saves: SavedAgent[] }>("/saved-agents");
  return r.saves;
}

export async function loadSavedAgent(saveId: string): Promise<string> {
  const r = await fetch(`${httpBase()}/saved-agents/${saveId}/load`, { method: "POST" });
  if (!r.ok) throw new Error(`load: HTTP ${r.status}`);
  const body = (await r.json()) as { agent_id: string };
  return body.agent_id;
}

export async function deleteSavedAgent(saveId: string): Promise<void> {
  const r = await fetch(`${httpBase()}/saved-agents/${saveId}`, { method: "DELETE" });
  if (!r.ok) throw new Error(`delete save: HTTP ${r.status}`);
}

export function fetchBrainTopology(agentId: string): Promise<BrainTopology> {
  return getJson<BrainTopology>(`/agent/${agentId}/brain/topology`);
}

export type BrainLandscape = {
  ready: boolean;
  detail?: string;
  // Present only when ready === true.
  alphas?: number[];
  betas?: number[];
  z?: number[][]; // z[i][j] = loss at (alphas[i], betas[j])
  center_loss?: number;
  z_min?: number;
  z_max?: number;
  grid?: number;
  span?: number;
  batch?: number;
  cycle?: number;
  preset?: string | null;
  wall_ms?: number;
};

export function fetchBrainLandscape(agentId: string): Promise<BrainLandscape> {
  // 202 (warming up) returns { ready: false }; getJson tolerates it as JSON.
  return getJson<BrainLandscape>(`/agent/${agentId}/brain/landscape`);
}

export type DiscoveryReport = {
  agent_id: string;
  perception_mode: string;
  egocentric_graph: EgoGraph;
  working_memory: WorkingMemorySnapshot;
  object_files?: ObjectFileSnapshot[];
  scene_workspace?: SceneWorkspaceSnapshot | null;
  scene_health?: SceneHealthSnapshot | null;
  focus?: FocusSnapshot | null;
  workspace_ignition?: WorkspaceIgnitionSnapshot | null;
  scene_prediction?: ScenePredictionSnapshot | null;
  discovery_health?: DiscoveryHealth | null;
  perception_organ?: PerceptionOrganSnapshot | null;
  retinotopic_map?: RetinotopicMapSnapshot | null;
  ltm_consolidation?: LtmConsolidationStatus | null;
  discovery: DiscoverySnapshot | null;
  oracle_truth_count: number;
};

export function fetchDiscovery(agentId: string): Promise<DiscoveryReport> {
  return getJson<DiscoveryReport>(`/agent/${agentId}/discovery`);
}

export type EvalGate = {
  name: string;
  metric: string;
  op: string;
  threshold: number;
  mode?: string;
  fraction?: number;
};

export type EvalScenario = {
  scenario: string;
  description?: string;
  cycles: number;
  seeds: number[];
  agent_preset?: string | null;
  dojo_skill_id?: string | null;
  baseline?: string | null;
  poll_interval_s?: number;
  timeout_s?: number;
  gates: EvalGate[];
  body_required?: boolean;
  estimated_runtime_s?: number;
};

export type EvalReportSummary = {
  report_id: string;
  path: string;
  scenario: string;
  status: string;
  agent_id?: string | null;
  failures_count: number;
  failures?: string[];
  samples_path?: string;
  mtime?: number;
};

export type EvalReport = {
  scenario: string;
  status: string;
  agent_id?: string | null;
  seeds?: number[];
  health?: Record<string, unknown>;
  mechanical?: Record<string, unknown>;
  learning?: Record<string, unknown>;
  perception?: Record<string, unknown>;
  probes?: Record<string, unknown>;
  behavior?: Record<string, unknown>;
  baseline_comparison?: Record<string, unknown>;
  failures?: string[];
  samples_path?: string;
};

export type EvalStatus = {
  state: "idle" | "starting" | "running" | "stopping" | "completed" | "failed" | "cancelled" | string;
  job_id?: string | null;
  scenario?: string | null;
  agent_id?: string | null;
  started_at?: number | null;
  elapsed_s?: number;
  samples?: number;
  cycles?: number;
  target_cycles?: number;
  report_path?: string;
  samples_path?: string;
  error?: string;
  body_connected?: boolean;
  body_warning?: string;
};

export type EvalStartRequest = {
  scenario: string;
  cycles?: number | null;
  seeds?: number[] | null;
  preset?: string | null;
  dojo_skill_id?: string | null;
  poll_interval_s?: number | null;
  timeout_s?: number | null;
  agent_id?: string | null;
};

export async function fetchEvalScenarios(): Promise<EvalScenario[]> {
  const r = await getJson<{ scenarios: EvalScenario[] }>("/eval/scenarios");
  return r.scenarios;
}

export function fetchEvalScenario(id: string): Promise<EvalScenario> {
  return getJson<EvalScenario>(`/eval/scenarios/${encodeURIComponent(id)}`);
}

export async function fetchEvalReports(): Promise<EvalReportSummary[]> {
  const r = await getJson<{ reports: EvalReportSummary[] }>("/eval/reports");
  return r.reports;
}

export function fetchEvalReport(reportId: string): Promise<EvalReport> {
  return getJson<EvalReport>(`/eval/reports/${encodeURIComponent(reportId)}`);
}

export async function startEvalJob(req: EvalStartRequest): Promise<EvalStatus> {
  const r = await fetch(`${httpBase()}/eval/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`start eval: HTTP ${r.status}`);
  return (await r.json()) as EvalStatus;
}

export function fetchEvalStatus(): Promise<EvalStatus> {
  return getJson<EvalStatus>("/eval/status");
}

export async function stopEvalJob(): Promise<EvalStatus> {
  const r = await fetch(`${httpBase()}/eval/stop`, { method: "POST" });
  if (!r.ok) throw new Error(`stop eval: HTTP ${r.status}`);
  return (await r.json()) as EvalStatus;
}

export function fetchExplain(
  agentId: string,
  opts: { history?: number; counterfactuals?: boolean; attribution?: boolean } = {},
): Promise<ExplainReport> {
  const params = new URLSearchParams();
  if (opts.history != null) params.set("history", String(opts.history));
  if (opts.counterfactuals) params.set("counterfactuals", "1");
  if (opts.attribution) params.set("attribution", "1");
  const q = params.toString();
  return getJson<ExplainReport>(`/agent/${agentId}/explain${q ? `?${q}` : ""}`);
}

// Neuroplasticity + cognitive-faculty defaults applied to newly created agents.
export type AgentDefaults = {
  plasticity_enabled: boolean;
  sparse_enabled: boolean;
  growth_enabled: boolean;
  plasticity_alpha: number;
  sparse_density: number;
  max_neurons: number;
  growable_hidden_ceiling: number;
  // Core cognitive faculties (inherent; on by default).
  perception_feedback: boolean;
  // Self-state feedback spine (self-model program; research faculty, off by default).
  self_model_feedback?: boolean;
  // Predictive affect (self-model program; research faculty, off by default).
  predictive_affect?: boolean;
  // Represented self (self-model program; research faculty, off by default).
  represented_self?: boolean;
  perception_mode: string;
  encoder_mode: string;
};

export function fetchAgentDefaults(): Promise<AgentDefaults> {
  return getJson<AgentDefaults>("/settings/agent-defaults");
}

export async function setAgentDefaults(
  partial: Partial<AgentDefaults>,
): Promise<AgentDefaults> {
  const params = new URLSearchParams();
  if (partial.plasticity_enabled != null)
    params.set("plasticity_enabled", partial.plasticity_enabled ? "1" : "0");
  if (partial.sparse_enabled != null)
    params.set("sparse_enabled", partial.sparse_enabled ? "1" : "0");
  if (partial.growth_enabled != null)
    params.set("growth_enabled", partial.growth_enabled ? "1" : "0");
  if (partial.plasticity_alpha != null)
    params.set("plasticity_alpha", String(partial.plasticity_alpha));
  if (partial.sparse_density != null)
    params.set("sparse_density", String(partial.sparse_density));
  if (partial.max_neurons != null) params.set("max_neurons", String(partial.max_neurons));
  if (partial.growable_hidden_ceiling != null)
    params.set("growable_hidden_ceiling", String(partial.growable_hidden_ceiling));
  if (partial.perception_feedback != null)
    params.set("perception_feedback", partial.perception_feedback ? "1" : "0");
  if (partial.self_model_feedback != null)
    params.set("self_model_feedback", partial.self_model_feedback ? "1" : "0");
  if (partial.predictive_affect != null)
    params.set("predictive_affect", partial.predictive_affect ? "1" : "0");
  if (partial.represented_self != null)
    params.set("represented_self", partial.represented_self ? "1" : "0");
  if (partial.perception_mode != null)
    params.set("perception_mode", String(partial.perception_mode));
  if (partial.encoder_mode != null)
    params.set("encoder_mode", String(partial.encoder_mode));
  const r = await fetch(`${httpBase()}/settings/agent-defaults?${params.toString()}`, {
    method: "POST",
  });
  if (!r.ok) throw new Error(`agent-defaults: HTTP ${r.status}`);
  return (await r.json()) as AgentDefaults;
}

export function setPreset(agentId: string, preset: string): Promise<void> {
  return postJson(`/agent/${agentId}/preset?preset=${encodeURIComponent(preset)}`);
}

export type EnvironmentState = "stopped" | "running" | "paused" | "crashed";

export type EnvironmentStatus = {
  state: EnvironmentState;
  running: boolean;
  paused: boolean;
  agent_id: string | null;
  elements: string[];
  options: { vision?: boolean; audio?: boolean; braces?: boolean };
  pid: number | null;
  returncode: number | null;
  started_at: number | null;
  log_path: string | null;
  available_elements: string[];
};

export type EnvironmentStartRequest = {
  elements: string[];
  vision?: boolean;
  audio?: boolean;
  // Whether the manual joint-brace scaffold starts engaged.
  braces?: boolean;
  // Supersede a running body instead of erroring (used by "+ New agent").
  replace?: boolean;
  // Neural architecture preset for the fresh mind created with this body.
  preset?: string;
};

export function fetchEnvironment(): Promise<EnvironmentStatus> {
  return getJson<EnvironmentStatus>("/environment");
}

export async function startEnvironment(
  req: EnvironmentStartRequest,
): Promise<EnvironmentStatus> {
  const r = await fetch(`${httpBase()}/environment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vision: true, audio: false, ...req }),
  });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      const body = (await r.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // non-JSON error body; keep the status code message
    }
    throw new Error(detail);
  }
  return (await r.json()) as EnvironmentStatus;
}

async function postEnvironment(path: string): Promise<EnvironmentStatus> {
  const r = await fetch(`${httpBase()}${path}`, { method: "POST" });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as EnvironmentStatus;
}

export function pauseEnvironment(): Promise<EnvironmentStatus> {
  return postEnvironment("/environment/pause");
}

export function resumeEnvironment(): Promise<EnvironmentStatus> {
  return postEnvironment("/environment/resume");
}

export function stopEnvironment(): Promise<EnvironmentStatus> {
  return postEnvironment("/environment/stop");
}

export async function deleteEnvironment(): Promise<EnvironmentStatus> {
  const r = await fetch(`${httpBase()}/environment`, { method: "DELETE" });
  if (!r.ok) throw new Error(`delete environment: HTTP ${r.status}`);
  return (await r.json()) as EnvironmentStatus;
}

// --- Agent presets ---------------------------------------------------------
// A named scenario/body config the top-bar dropdown offers next to "+ New
// agent": which elements to spawn, which senses are on, whether the manual
// joint-brace scaffold starts engaged, and whether it is a disembodied mind. Server-defined
// (built-ins) plus user-saved.

export type AgentPreset = {
  id: string;
  name: string;
  elements: string[];
  vision: boolean;
  audio: boolean;
  braces: boolean;
  mind_only: boolean;
  builtin: boolean;
  created_at?: string | null;
};

export type CreateAgentPresetRequest = {
  name: string;
  elements: string[];
  vision: boolean;
  audio: boolean;
  braces: boolean;
  mind_only: boolean;
};

// The live, editable scenario/body config shared between the top-bar dropdown
// and the Environment tab. Camel-cased mindOnly to match React conventions;
// mapped to the snake_case API field when saved.
export type ScenarioDraft = {
  elements: string[];
  vision: boolean;
  audio: boolean;
  braces: boolean;
  mindOnly: boolean;
};

export async function listAgentPresets(): Promise<AgentPreset[]> {
  const r = await getJson<{ presets: AgentPreset[] }>("/agent-presets");
  return r.presets;
}

export async function createAgentPreset(
  body: CreateAgentPresetRequest,
): Promise<AgentPreset> {
  const r = await fetch(`${httpBase()}/agent-presets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      const b = (await r.json()) as { detail?: string };
      if (b.detail) detail = b.detail;
    } catch {
      // non-JSON error body; keep the status code message
    }
    throw new Error(detail);
  }
  return (await r.json()) as AgentPreset;
}

export async function deleteAgentPreset(presetId: string): Promise<void> {
  const r = await fetch(`${httpBase()}/agent-presets/${presetId}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(`delete preset: HTTP ${r.status}`);
}

// --- Skill gates -----------------------------------------------------------

// One promotion condition's live readout (mirrors training.gates.CriterionResult).
export type CriterionStatus = {
  label: string;
  key: string;
  comparator: "<=" | ">=" | "trend>=";
  threshold: number;
  value: number;
  satisfied: boolean;
  // UI hint in [0, 1] of how close the criterion is to opening.
  progress: number;
  unit: string;
};

export type GateStatus = {
  satisfied: boolean;
  progress: number;
  samples: number;
  enough_samples: boolean;
  criteria: CriterionStatus[];
};

// --- Skill Dojo -------------------------------------------------------------

export type SkillSource = "builtin" | "uploaded";

export type SkillCriterion = {
  key: string;
  comparator: "<=" | ">=" | "trend>=";
  threshold: number;
  label: string;
  unit?: string;
};

export type DojoSkillPhase = {
  index: number;
  name: string;
  description: string;
  teacher_weight: number;
  teacher_adaptation?: {
    enabled: boolean;
    min_weight: number;
    max_weight: number;
    rise_rate: number;
    fade_rate: number;
    danger_thresholds: Record<string, number>;
    stability_thresholds: Record<string, number>;
    stable_dwell_s: number;
    unstable_dwell_s: number;
    zero_required_for_graduation: boolean;
  } | null;
  config: Record<string, unknown>;
  body_commands: string[];
  periodic_body_commands: Array<{ command: string; period_s: number }>;
  min_dwell_s: number;
  timeout_s: number;
  max_attempts: number;
  auto_retry: boolean;
  reset_commands: string[];
  min_samples: number;
  demote_on_death: boolean;
  is_terminal: boolean;
  criteria: SkillCriterion[];
  failure_criteria: SkillCriterion[];
  failure_min_samples: number;
};

export type DojoSkill = {
  skill_id: string;
  version: string;
  name: string;
  description: string;
  target_behavior: string;
  teacher: string;
  source: SkillSource;
  builtin: boolean;
  required_sensors: string[];
  checkpoint_on_graduate: boolean;
  caregiver_enabled: boolean;
  caregiver_threshold: number;
  warnings?: string[];
  phases: DojoSkillPhase[];
};

export type DojoHistoryRec = {
  phase: number;
  name: string;
  event: string;
  at: string;
};

export type DojoStatus = {
  state: "stopped" | "running" | "paused" | "graduated" | "failed" | "error";
  running: boolean;
  paused: boolean;
  agent_id: string | null;
  skill_id: string | null;
  skill_name: string | null;
  phase_index: number | null;
  phase_name: string | null;
  phase_description: string | null;
  phase_count: number;
  teacher_weight: number;
  teacher_assist: number;
  teacher_min: number;
  teacher_max: number;
  teacher_rise_rate: number;
  teacher_fade_rate: number;
  stable_dwell_s: number;
  unstable_dwell_s: number;
  assist_reason: string;
  teacher_origin: "self" | "dagger" | "demo";
  objective_confidence: number;
  confidence_reason: string;
  confidence_dwell_s: number;
  teacher_live: boolean;
  teacher_support_active: boolean;
  teacher_support_force: number;
  teacher_support_torque: number;
  teacher_drop_m: number;
  teacher_target_drop_m: number;
  teacher_height_error_m: number;
  teacher_vertical_velocity: number;
  teacher_support_mode: string;
  caregiver_enabled: boolean;
  caregiver_status: string;
  caregiver_kind?: string | null;
  caregiver_need: string;
  caregiver_threshold: number;
  caregiver_trigger_reservoir: string | null;
  caregiver_request_kind: string | null;
  caregiver_last_offer_cycle: number | null;
  caregiver_last_offer_item: string | null;
  caregiver_missing_parent: boolean;
  caregiver_refractory_s: number;
  caregiver_delivery_count: number;
  caregiver_pending: boolean;
  hydration: number;
  energy: number;
  integrity: number;
  distance_traveled: number;
  net_displacement: number;
  consume_events: number;
  resource_relief_events: number;
  fall_rate: number;
  stance_phase: number;
  samples: number;
  gate: GateStatus | null;
  failure: CriterionStatus | null;
  attempt_index: number;
  attempt_failures: number;
  attempt_elapsed_s: number;
  attempt_timeout_s: number;
  max_attempts: number;
  auto_retry: boolean;
  last_attempt_outcome: string | null;
  failure_reason: string | null;
  manual_scaffold_active: boolean;
  history: DojoHistoryRec[];
  report_path: string | null;
  started_at: number | null;
  poll_interval_s: number;
  error: string | null;
};

export async function fetchDojoSkills(): Promise<DojoSkill[]> {
  const r = await getJson<{ skills: DojoSkill[] }>("/dojo/skills");
  return r.skills;
}

export function fetchDojoStatus(): Promise<DojoStatus> {
  return getJson<DojoStatus>("/dojo/status");
}

async function postDojo(path: string, body?: unknown): Promise<DojoStatus> {
  const init: RequestInit = { method: "POST" };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  const r = await fetch(`${httpBase()}${path}`, init);
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: string };
      if (j.detail) detail = j.detail;
    } catch {
      // non-JSON error body; keep the status code message
    }
    throw new Error(detail);
  }
  return (await r.json()) as DojoStatus;
}

export function startDojo(req: {
  agent_id: string;
  skill_id: string;
  auto_retry?: boolean;
  max_attempts?: number;
  timeout_multiplier?: number;
}): Promise<DojoStatus> {
  return postDojo("/dojo/start", req);
}

export function pauseDojo(): Promise<DojoStatus> {
  return postDojo("/dojo/pause");
}

export function resumeDojo(): Promise<DojoStatus> {
  return postDojo("/dojo/resume");
}

export function stopDojo(): Promise<DojoStatus> {
  return postDojo("/dojo/stop");
}

export function setDojoPhase(index: number): Promise<DojoStatus> {
  return postDojo("/dojo/phase", { index });
}

export async function uploadDojoSkill(skill: unknown): Promise<DojoSkill> {
  const r = await fetch(`${httpBase()}/dojo/skills/upload`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(skill),
  });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: string };
      if (j.detail) detail = j.detail;
    } catch {
      // non-JSON error body; keep the status code message
    }
    throw new Error(detail);
  }
  return (await r.json()) as DojoSkill;
}

export async function deleteDojoSkill(skillId: string): Promise<void> {
  const r = await fetch(`${httpBase()}/dojo/skills/${encodeURIComponent(skillId)}`, {
    method: "DELETE",
  });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: string };
      if (j.detail) detail = j.detail;
    } catch {
      // non-JSON error body; keep the status code message
    }
    throw new Error(detail);
  }
}
