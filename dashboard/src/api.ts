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
  sparse_density: number | null;
  awake_neurons: number;
  allocated_neurons: number;
  max_neurons: number;
};

export type CapacityConfig = {
  parallel_sessions: number;
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
};

export type LtmEdge = {
  source: string;
  target: string;
  kind: string;
  weight: number;
};

export type LtmGraphSnapshot = {
  nodes: LtmNode[];
  edges: LtmEdge[];
  // Unbounded totals (the windowed nodes/edges above are a read-out cap only).
  total_nodes: number;
  total_edges: number;
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
  foot_load_l?: number;
  foot_load_r?: number;
  hand_load_l?: number;
  hand_load_r?: number;
  // Full-body touch: per-part contact loads (short name -> force/body weight).
  part_loads?: Record<string, number>;
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
  sparse_density?: number | null;
  awake_neurons?: number;
  allocated_neurons?: number;
  active_connections?: number;
  max_neurons?: number;
  rewire_events?: number;
  growth_events?: number;
  plasticity_frozen?: boolean;
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
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
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

// Provision the agent with water/food. mode "near" asks the body to place the
// (unlabeled) prop a step away so the agent must perceive and walk to it;
// mode "direct" is an admin top-up that credits the reservoir immediately.
export async function giveResource(
  agentId: string,
  resource: "water" | "food",
  mode: "near" | "direct",
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

// Selectable joint-brace stances. Mirrors the backend stance library
// (decadic/embodiment/stances.py, also served at GET /body/stances); kept here so
// the selector renders without an extra round-trip.
export const STANCES: StanceInfo[] = [
  { name: "stand", label: "Stand & balance", motion: false },
  { name: "all_fours", label: "Kneel on all fours", motion: false },
  { name: "kneel_left", label: "Kneel (twist left)", motion: false },
  { name: "kneel_right", label: "Kneel (twist right)", motion: false },
  { name: "crawl", label: "Crawl (motion)", motion: true },
  { name: "sit_to_stand", label: "Rise up (motion)", motion: true },
];

// Re-pose the body into a stance and restart that stance's ROM curriculum (the
// body re-poses into the start pose and re-welds every joint brace).
export function setStance(agentId: string, name: string): Promise<void> {
  return postJson(`/agent/${agentId}/body/stance?name=${encodeURIComponent(name)}`);
}

// Hold the active movement: weld every joint brace (suspend the ROM curriculum --
// no range-of-motion release) and loop motion stances continuously, so the
// selected movement runs on repeat until disabled. Off resumes the ROM ratchet.
export function setMovementHold(agentId: string, enabled: boolean): Promise<void> {
  return postJson(`/agent/${agentId}/body/movement_hold?enabled=${enabled}`);
}

// Freeze (or release) the parent NPC where it stands. The flag lives in the body
// process, so it resets to "walking" whenever the scenario restarts.
export function setParentPaused(agentId: string, paused: boolean): Promise<void> {
  return postJson(`/agent/${agentId}/body/npc?paused=${paused}`);
}

export async function createAgent(): Promise<string> {
  const r = await fetch(`${httpBase()}/agent`, { method: "POST" });
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
  discovery: DiscoverySnapshot | null;
  oracle_truth_count: number;
};

export function fetchDiscovery(agentId: string): Promise<DiscoveryReport> {
  return getJson<DiscoveryReport>(`/agent/${agentId}/discovery`);
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
  // Whether the joint-brace orthosis starts engaged (off -> free body).
  braces?: boolean;
  // Supersede a running body instead of erroring (used by "+ New agent").
  replace?: boolean;
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
// agent": which elements to spawn, which senses are on, whether the joint
// braces start engaged, and whether it is a disembodied mind. Server-defined
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

// --- Walking curriculum ----------------------------------------------------

export type CurriculumState = "stopped" | "running" | "paused" | "error";

// One promotion condition's live readout (mirrors gates.CriterionResult).
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

export type CurriculumHistoryRec = {
  phase: number;
  name: string;
  // "promoted" | "demoted" | "graduated"
  event: string;
  at: string;
  cycles: number | null;
};

export type CurriculumStatus = {
  state: CurriculumState;
  running: boolean;
  paused: boolean;
  graduated: boolean;
  agent_id: string | null;
  phase_index: number | null;
  phase_name: string | null;
  phase_description: string | null;
  phase_count: number;
  is_terminal: boolean;
  min_dwell_s: number | null;
  dwell_s: number;
  window_size: number;
  satisfier: { enabled: boolean; resources: string[]; period_s: number | null };
  gate: GateStatus | null;
  history: CurriculumHistoryRec[];
  started_at: number | null;
  poll_interval_s: number;
  error: string | null;
  log_path: string | null;
};

export type CurriculumStartRequest = {
  agent_id: string;
  include_affective?: boolean;
  overrides?: Record<string, unknown> | null;
};

export function fetchCurriculum(): Promise<CurriculumStatus> {
  return getJson<CurriculumStatus>("/curriculum");
}

async function postCurriculum(
  path: string,
  body?: unknown,
): Promise<CurriculumStatus> {
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
  return (await r.json()) as CurriculumStatus;
}

export function startCurriculum(
  req: CurriculumStartRequest,
): Promise<CurriculumStatus> {
  return postCurriculum("/curriculum/start", req);
}

export function pauseCurriculum(): Promise<CurriculumStatus> {
  return postCurriculum("/curriculum/pause");
}

export function resumeCurriculum(): Promise<CurriculumStatus> {
  return postCurriculum("/curriculum/resume");
}

export function stopCurriculum(): Promise<CurriculumStatus> {
  return postCurriculum("/curriculum/stop");
}

export function setCurriculumPhase(index: number): Promise<CurriculumStatus> {
  return postCurriculum("/curriculum/phase", { index });
}
