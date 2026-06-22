import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  deleteDojoSkill,
  fetchDojoSkills,
  fetchEnvironment,
  fetchDojoStatus,
  pauseDojo,
  resumeDojo,
  resetBraces,
  setBracesEnabled,
  setDojoPhase,
  setMovementHold,
  setStance,
  startDojo,
  startEnvironment,
  stopDojo,
  STANCES,
  uploadDojoSkill,
  type CriterionStatus,
  type DojoSkill,
  type DojoStatus,
  type EnvironmentStatus,
  type Metrics,
  type AgentState,
} from "../api";
import { heavyPresetWarning } from "../neuralPresets";
import { usePolling } from "../usePolling";
import Info from "./Info";

const DOJO_SCENE = ["house", "food", "water", "npc"];

type BannerKind = "info" | "warning" | "error" | "success";

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
}

function fmtVal(c: CriterionStatus): string {
  const txt = Math.abs(c.value) >= 100 ? c.value.toFixed(0) : c.value.toFixed(3);
  return `${txt}${c.unit ? ` ${c.unit}` : ""}`;
}

function Banner(props: { kind: BannerKind; children: ReactNode }) {
  return <div className={`dojo-banner ${props.kind}`}>{props.children}</div>;
}

function Meter(props: { label: string; value: number; detail?: string; tone?: "assist" | "confidence" | "rom" }) {
  const pct = clamp01(props.value) * 100;
  return (
    <div className="dojo-meter">
      <div className="dojo-meter-head">
        <span>{props.label}</span>
        <span>{props.detail ?? `${pct.toFixed(0)}%`}</span>
      </div>
      <div className="dojo-meter-track">
        <div className={`dojo-meter-fill ${props.tone ?? "assist"}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function ToggleSwitch(props: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  onChange: () => void;
  hint?: React.ReactNode;
}) {
  return (
    <button
      type="button"
      className={`dojo-toggle ${props.checked ? "on" : ""}`}
      disabled={props.disabled}
      onClick={props.onChange}
      aria-pressed={props.checked}
    >
      <span className="dojo-toggle-copy">
        <span>{props.label}</span>
        {props.hint && <small>{props.hint}</small>}
      </span>
      <span className="dojo-switch" aria-hidden="true">
        <span />
      </span>
    </button>
  );
}

function StageHeader(props: { index: number; title: string; meta?: string; active?: boolean }) {
  return (
    <div className={`dojo-stage-head ${props.active ? "active" : ""}`}>
      <span className="dojo-stage-num">{props.index}</span>
      <div>
        <h3>{props.title}</h3>
        {props.meta && <p>{props.meta}</p>}
      </div>
    </div>
  );
}

function DojoStatusHeader(props: { status: DojoStatus | null; selected: DojoSkill | null; running: boolean }) {
  const st = props.status;
  const state = st?.state ?? "stopped";
  const timeoutLeft =
    st && st.attempt_timeout_s > 0
      ? `${Math.max(0, st.attempt_timeout_s - st.attempt_elapsed_s).toFixed(0)}s left`
      : "timeout off";
  return (
    <div className={`dojo-status-band ${state}`}>
      <div className="dojo-status-main">
        <span className="dojo-status-dot" />
        <b>{state}</b>
        {st?.agent_id && <span>Agent {st.agent_id.slice(0, 8)}</span>}
        <span>{st?.skill_name ?? props.selected?.name ?? "No skill selected"}</span>
        {st?.phase_index != null && (
          <span>
            Phase {st.phase_index}/{Math.max(0, st.phase_count - 1)}
          </span>
        )}
      </div>
      <div className="dojo-status-metrics">
        {props.running && st ? (
          <>
            <span>assist {st.teacher_assist.toFixed(2)}</span>
            <span>confidence {(st.objective_confidence * 100).toFixed(0)}%</span>
            <span>attempt {st.attempt_index}</span>
            <span>{timeoutLeft}</span>
          </>
        ) : (
          <span>Ready to configure a training run</span>
        )}
      </div>
      {st?.error && <span className="ctrl-error">{st.error}</span>}
    </div>
  );
}

function WorkflowSteps(props: { running: boolean; state: string }) {
  const active = props.running ? 3 : props.state === "graduated" || props.state === "failed" ? 4 : 1;
  const steps = ["Skill", "Configure", "Train", "Review"];
  return (
    <div className="dojo-workflow" aria-label="Skill Dojo workflow">
      {steps.map((label, i) => {
        const idx = i + 1;
        return (
          <div key={label} className={`dojo-workflow-step ${idx === active ? "active" : ""} ${idx < active ? "done" : ""}`}>
            <span>{idx}</span>
            <b>{label}</b>
          </div>
        );
      })}
    </div>
  );
}

function SkillLibraryPanel(props: {
  skills: DojoSkill[];
  selected: DojoSkill | null;
  running: boolean;
  busy: boolean;
  uploadMessage: { kind: BannerKind; text: string } | null;
  fileInput: React.MutableRefObject<HTMLInputElement | null>;
  onSelect: (id: string) => void;
  onUpload: (file: File | null) => void;
  onDelete: () => void;
}) {
  return (
    <section className="dojo-stage">
      <StageHeader
        index={1}
        title="Skill Library"
        meta="Choose a built-in skill or import a validated JSON skill."
        active={!props.running}
      />
      <div className="dojo-library">
        <div className="dojo-skill-list">
          {props.skills.map((skill) => (
            <button
              key={skill.skill_id}
              type="button"
              className={`dojo-skill-row ${props.selected?.skill_id === skill.skill_id ? "selected" : ""}`}
              disabled={props.busy || props.running}
              onClick={() => props.onSelect(skill.skill_id)}
            >
              <span className="dojo-skill-name">{skill.name}</span>
              <span>{skill.source}</span>
              <span>v{skill.version}</span>
              <span>{skill.phases.length} phases</span>
            </button>
          ))}
        </div>
        <div className="dojo-skill-detail">
          {props.selected ? (
            <>
              <div className="dojo-detail-head">
                <span className="cur-badge">{props.selected.source}</span>
                <b>{props.selected.name}</b>
              </div>
              <p>{props.selected.description || props.selected.target_behavior}</p>
              <div className="dojo-kv-grid">
                <span>Teacher</span>
                <b>{props.selected.teacher}</b>
                <span>Version</span>
                <b>{props.selected.version}</b>
                <span>Terminal phase</span>
                <b>{props.selected.phases.some((p) => p.is_terminal) ? "yes" : "no"}</b>
                <span>Checkpoint</span>
                <b>{props.selected.checkpoint_on_graduate ? "on graduation" : "off"}</b>
                <span>Caregiver</span>
                <b>
                  {props.selected.caregiver_enabled
                    ? `enabled below ${props.selected.caregiver_threshold.toFixed(0)}%`
                    : "off"}
                </b>
              </div>
              {props.selected.warnings?.length ? (
                <Banner kind="warning">{props.selected.warnings.join(" ")}</Banner>
              ) : (
                <Banner kind="success">Skill validation is clean.</Banner>
              )}
            </>
          ) : (
            <div className="empty">No skills loaded.</div>
          )}
          {props.uploadMessage && <Banner kind={props.uploadMessage.kind}>{props.uploadMessage.text}</Banner>}
          <div className="dojo-actions">
            <input
              ref={props.fileInput}
              type="file"
              accept="application/json,.json"
              style={{ display: "none" }}
              onChange={(e) => props.onUpload(e.target.files?.[0] ?? null)}
            />
            <button className="btn" disabled={props.busy || props.running} onClick={() => props.fileInput.current?.click()}>
              Upload JSON Skill
            </button>
            {props.selected && !props.selected.builtin && (
              <button className="btn reset" disabled={props.busy || props.running} onClick={props.onDelete}>
                Delete Uploaded Skill
              </button>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function RunSetupPanel(props: {
  selected: DojoSkill | null;
  running: boolean;
  busy: boolean;
  hasAgent: boolean;
  autoRetry: boolean;
  maxAttempts: number;
  timeoutMultiplier: number;
  onAutoRetry: (v: boolean) => void;
  onMaxAttempts: (v: number) => void;
  onTimeoutMultiplier: (v: number) => void;
  onStart: () => void;
}) {
  return (
    <section className="dojo-stage">
      <StageHeader
        index={2}
        title="Run Setup"
        meta="Set retry policy before starting a supervised training run."
        active={!props.running}
      />
      <div className="dojo-setup-grid">
        <ToggleSwitch
          label="Auto retry"
          checked={props.autoRetry}
          disabled={props.busy || props.running}
          onChange={() => props.onAutoRetry(!props.autoRetry)}
          hint="Restart failed or timed-out attempts."
        />
        <label className="dojo-number-field">
          <span>Max attempts</span>
          <input
            type="number"
            min={1}
            max={100}
            value={props.maxAttempts}
            disabled={props.busy || props.running}
            onChange={(e) => props.onMaxAttempts(Math.max(1, Number(e.target.value) || 1))}
          />
        </label>
        <label className="dojo-number-field">
          <span>Timeout multiplier</span>
          <input
            type="number"
            min={0.1}
            max={10}
            step={0.1}
            value={props.timeoutMultiplier}
            disabled={props.busy || props.running}
            onChange={(e) => props.onTimeoutMultiplier(Math.max(0.1, Number(e.target.value) || 1))}
          />
        </label>
      </div>
      <div className="dojo-setup-footer">
        <div>
          <b>{props.selected?.name ?? "No skill selected"}</b>
          <span>{props.hasAgent ? "Use current agent" : "Create training environment"}</span>
        </div>
        <button className="btn start dojo-primary-action" disabled={props.busy || props.running || !props.selected} onClick={props.onStart}>
          Start Training Run
        </button>
      </div>
    </section>
  );
}

function PhaseTimeline(props: {
  skill: DojoSkill | null;
  status: DojoStatus;
  busy: boolean;
  onJump: (i: number) => void;
}) {
  const current = props.status.phase_index ?? 0;
  const phases = props.skill?.phases ?? [];
  if (!phases.length) return null;
  return (
    <div className="dojo-phase-timeline">
      {phases.map((phase) => {
        const phaseState =
          phase.index === current ? "active" : phase.index < current ? "passed" : phase.is_terminal ? "terminal" : "pending";
        const adaptation = phase.teacher_adaptation?.enabled
          ? `${phase.teacher_adaptation.min_weight.toFixed(2)}-${phase.teacher_adaptation.max_weight.toFixed(2)}`
          : phase.teacher_weight.toFixed(2);
        return (
          <button
            key={phase.index}
            type="button"
            className={`dojo-phase-node ${phaseState}`}
            disabled={props.busy}
            title={`Manual phase jump to ${phase.name}`}
            onClick={() => props.onJump(phase.index)}
          >
            <span>Phase {phase.index}</span>
            <b>{phase.name}</b>
            <small>teacher {adaptation}</small>
            <small>{phase.timeout_s ? `${phase.timeout_s}s timeout` : "no timeout"}</small>
          </button>
        );
      })}
    </div>
  );
}

function TeacherAssistPanel(props: { status: DojoStatus }) {
  const s = props.status;
  return (
    <section className="dojo-subpanel">
      <h4>Teacher Support</h4>
      <Meter
        label="Live teacher assist"
        value={s.teacher_assist}
        detail={`${s.teacher_assist.toFixed(2)} ${s.teacher_live ? "live" : "replay"} (${s.teacher_origin})`}
      />
      <Meter
        label="Objective confidence"
        value={s.objective_confidence}
        detail={`${(s.objective_confidence * 100).toFixed(0)}%`}
        tone="confidence"
      />
      <div className="dojo-kv-grid compact">
        <span>Assist reason</span>
        <b>{s.assist_reason || "idle"}</b>
        <span>Confidence</span>
        <b>{s.confidence_reason || "idle"}</b>
        <span>Support</span>
        <b>
          {s.teacher_support_active ? "active" : "off"} F{s.teacher_support_force.toFixed(2)} T
          {s.teacher_support_torque.toFixed(1)}
        </b>
        <span>Mode</span>
        <b>{s.teacher_support_mode || "off"}</b>
        <span>Drop</span>
        <b>
          {s.teacher_drop_m.toFixed(2)}m / target {s.teacher_target_drop_m.toFixed(2)}m
        </b>
        <span>Height error</span>
        <b>{s.teacher_height_error_m.toFixed(2)}m</b>
        <span>Down velocity</span>
        <b>{s.teacher_vertical_velocity.toFixed(2)}m/s</b>
        <span>Stable dwell</span>
        <b>{s.stable_dwell_s.toFixed(1)}s</b>
      </div>
    </section>
  );
}

function AttemptPanel(props: { status: DojoStatus }) {
  const s = props.status;
  const timeoutLeft =
    s.attempt_timeout_s > 0 ? Math.max(0, s.attempt_timeout_s - s.attempt_elapsed_s).toFixed(0) : "off";
  const failed = s.state === "failed" || s.last_attempt_outcome === "failed" || s.last_attempt_outcome === "timeout";
  return (
    <section className="dojo-subpanel">
      <h4>Attempt Lifecycle</h4>
      <div className="dojo-attempt-grid">
        <div>
          <span>Attempt</span>
          <b>{s.attempt_index}</b>
        </div>
        <div>
          <span>Failures</span>
          <b>
            {s.attempt_failures}/{s.max_attempts}
          </b>
        </div>
        <div>
          <span>Timeout</span>
          <b>{timeoutLeft === "off" ? "off" : `${timeoutLeft}s left`}</b>
        </div>
        <div>
          <span>Retry</span>
          <b>{s.auto_retry ? "on" : "off"}</b>
        </div>
      </div>
      {(s.failure_reason || s.last_attempt_outcome) && (
        <Banner kind={failed ? "error" : "info"}>
          Attempt {s.last_attempt_outcome ?? "running"}
          {s.failure_reason ? `: ${s.failure_reason}` : ""}
        </Banner>
      )}
    </section>
  );
}

function CaregiverPanel(props: { status: DojoStatus }) {
  const s = props.status;
  const activeNeed = s.caregiver_need && s.caregiver_need !== "none";
  const hydration = s.hydration ?? 100;
  const energy = s.energy ?? 100;
  const integrity = s.integrity ?? 100;
  const threshold = s.caregiver_threshold ?? 80;
  return (
    <section className="dojo-subpanel">
      <div className="dojo-subpanel-head">
        <h4>Caregiver Scaffold</h4>
        <span>{s.caregiver_enabled ? s.caregiver_status || "monitoring" : "off"}</span>
      </div>
      <div className="dojo-kv-grid compact">
        <span>Reservoirs</span>
        <b>
          H{hydration.toFixed(0)} E{energy.toFixed(0)} I{integrity.toFixed(0)}
        </b>
        <span>Trigger</span>
        <b>{s.caregiver_enabled ? `< ${threshold.toFixed(0)}%` : "disabled"}</b>
        <span>Need</span>
        <b>{activeNeed ? s.caregiver_need : "none"}</b>
        <span>Request</span>
        <b>{s.caregiver_request_kind ?? "none"}</b>
        <span>Last offer</span>
        <b>{s.caregiver_last_offer_item ?? "none"}</b>
        <span>Deliveries</span>
        <b>{s.caregiver_delivery_count}</b>
      </div>
      {s.caregiver_missing_parent && (
        <Banner kind="error">Caregiver parent is missing. Embodied skill graduation is blocked.</Banner>
      )}
      {s.caregiver_pending && !s.caregiver_missing_parent && (
        <Banner kind="info">Caregiver request pending through visible parent delivery.</Banner>
      )}
    </section>
  );
}

function GateChecklist(props: { status: DojoStatus }) {
  const gate = props.status.gate;
  const failure = props.status.failure;
  return (
    <section className="dojo-subpanel">
      <div className="dojo-subpanel-head">
        <h4>Gate Checklist</h4>
        <span>
          {props.status.manual_scaffold_active
            ? "manual scaffold active"
            : gate?.satisfied
              ? "open"
              : gate?.enough_samples
                ? "holding"
                : `warming (${gate?.samples ?? 0})`}
        </span>
      </div>
      <div className="dojo-gate-list">
        {gate?.criteria?.length ? (
          gate.criteria.map((c) => (
            <div key={c.key} className={`dojo-gate-row ${c.satisfied ? "ok" : ""}`}>
              <span>{c.satisfied ? "pass" : "wait"}</span>
              <b>{c.label}</b>
              <small>
                {fmtVal(c)} {c.comparator} {c.threshold}
              </small>
              <Meter label="" value={c.progress} tone={c.satisfied ? "confidence" : "assist"} />
            </div>
          ))
        ) : (
          <div className="empty">No gate samples yet.</div>
        )}
      </div>
      {failure && (
        <Banner kind="error">
          Fail-fast: {failure.label} ({fmtVal(failure)} {failure.comparator} {failure.threshold})
        </Banner>
      )}
      {props.status.manual_scaffold_active && (
        <Banner kind="warning">Manual scaffold is active. Turn braces and movement hold off to allow phase graduation.</Banner>
      )}
    </section>
  );
}

function LiveTrainingPanel(props: {
  status: DojoStatus | null;
  selected: DojoSkill | null;
  running: boolean;
  paused: boolean;
  busy: boolean;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  onJump: (i: number) => void;
}) {
  if (!props.running || !props.status) {
    return (
      <section className="dojo-stage">
        <StageHeader index={3} title="Live Training" meta="Training metrics appear after a run starts." />
        <div className="empty">No active Skill Dojo run.</div>
      </section>
    );
  }
  const s = props.status;
  return (
    <section className="dojo-stage active">
      <StageHeader index={3} title="Live Training" meta="Current phase, teacher assist, gates, and attempt lifecycle." active />
      <PhaseTimeline skill={props.selected} status={s} busy={props.busy} onJump={props.onJump} />
      <div className="dojo-current-phase">
        <div>
          <span className="cur-badge">Phase {s.phase_index ?? "-"}</span>
          <h3>{s.phase_name ?? "Training phase"}</h3>
          {s.phase_description && <p>{s.phase_description}</p>}
        </div>
        <div className="dojo-run-controls">
          <button className="btn" disabled={props.busy || props.paused} onClick={props.onPause}>
            Pause
          </button>
          <button className="btn start" disabled={props.busy || !props.paused} onClick={props.onResume}>
            Resume
          </button>
          <button className="btn reset" disabled={props.busy} onClick={props.onStop}>
            Stop
          </button>
        </div>
      </div>
      <div className="dojo-live-grid">
        <TeacherAssistPanel status={s} />
        <AttemptPanel status={s} />
        <CaregiverPanel status={s} />
      </div>
      <div className="dojo-live-warning">
        <Banner kind="info">Graduation requires zero live teacher support.</Banner>
      </div>
      <GateChecklist status={s} />
    </section>
  );
}

function ReviewPanel(props: { status: DojoStatus | null }) {
  const s = props.status;
  const isFinal = s?.state === "graduated" || s?.state === "failed";
  return (
    <section className="dojo-stage">
      <StageHeader
        index={4}
        title="Review"
        meta={isFinal ? "Run outcome and report artifact." : "Reports become useful after a run finishes or records events."}
        active={isFinal}
      />
      {s?.state === "graduated" && <Banner kind="success">Skill graduated. Final evaluation completed without live teacher support.</Banner>}
      {s?.state === "failed" && <Banner kind="error">Skill run failed: {s.failure_reason ?? "retry budget exhausted"}</Banner>}
      {s?.report_path ? (
        <div className="dojo-report-path">
          <span>Report</span>
          <b>{s.report_path}</b>
        </div>
      ) : (
        <div className="empty">No report path yet.</div>
      )}
    </section>
  );
}

function ManualScaffoldPanel(props: {
  agentId: string | null;
  metrics: Metrics | null;
  running: boolean;
  disabled: boolean;
}) {
  const m = props.metrics;
  const bracesEnabled = m?.braces_enabled ?? false;
  const movementHold = m?.movement_hold ?? false;
  const romMean = m?.rom_mean ?? 0;
  const braceEngaged = m?.brace_engaged ?? 0;
  const stance = m?.stance ?? "stand";
  const stancePhase = m?.stance_phase ?? 0;
  const activeStance = STANCES.find((s) => s.name === stance);
  const staticStances = STANCES.filter((s) => !s.motion);
  const motionStances = STANCES.filter((s) => s.motion);
  const disabled = props.disabled || !props.agentId;
  const contaminated = props.running && (bracesEnabled || movementHold);
  const loads = [
    ["left foot", m?.foot_load_l],
    ["right foot", m?.foot_load_r],
    ["left hand", m?.hand_load_l],
    ["right hand", m?.hand_load_r],
  ];
  const call = (fn: Promise<void>) => {
    fn.catch(() => {});
  };
  const stanceButton = (s: (typeof STANCES)[number]) => (
    <button
      key={s.name}
      type="button"
      className={`dojo-stance-btn ${stance === s.name ? "active" : ""}`}
      disabled={disabled}
      onClick={() => props.agentId && call(setStance(props.agentId, s.name))}
    >
      {s.label}
    </button>
  );
  return (
    <aside className="dojo-context-panel">
      <div className="dojo-context-head">
        <h3>Manual Body Scaffold</h3>
        <span>manual only</span>
      </div>
      <p className="dojo-context-copy">
        Braces and hold are operator tools for setup/debugging. Skill Dojo teacher support is separate.
      </p>
      {contaminated && <Banner kind="warning">Manual scaffold blocks phase graduation while training is running.</Banner>}
      <ToggleSwitch
        label="Joint braces"
        checked={bracesEnabled}
        disabled={disabled}
        onChange={() => props.agentId && call(setBracesEnabled(props.agentId, !bracesEnabled))}
        hint={bracesEnabled ? "manual orthosis on" : "free body"}
      />
      <ToggleSwitch
        label="Movement hold"
        checked={movementHold}
        disabled={disabled || !bracesEnabled}
        onChange={() => props.agentId && call(setMovementHold(props.agentId, !movementHold))}
        hint="locks/replays stance motion"
      />
      <button className="btn dojo-full-btn" disabled={disabled || !bracesEnabled} onClick={() => props.agentId && call(resetBraces(props.agentId))}>
        Reset Brace ROM
      </button>
      <div className="dojo-stance-group">
        <div className="dojo-context-label">
          Static posture
          <Info tip="Selecting a stance re-poses the body. It does not turn braces on or reset ROM." />
        </div>
        <div className="dojo-stance-grid">{staticStances.map(stanceButton)}</div>
      </div>
      <div className="dojo-stance-group">
        <div className="dojo-context-label">Motion stance</div>
        <div className="dojo-stance-grid">{motionStances.map(stanceButton)}</div>
      </div>
      <Meter
        label="Brace ROM release"
        value={romMean}
        detail={`${(romMean * 100).toFixed(0)}% ROM / ${(braceEngaged * 100).toFixed(0)}% braced`}
        tone="rom"
      />
      <div className="dojo-kv-grid compact">
        <span>Stance</span>
        <b>
          {activeStance?.label ?? stance}
          {activeStance?.motion ? ` (${(stancePhase * 100).toFixed(0)}%)` : ""}
        </b>
        {loads.map(([label, value]) => (
          <Fragment key={label}>
            <span>{label}</span>
            <b>{typeof value === "number" ? value.toFixed(2) : "-"}</b>
          </Fragment>
        ))}
      </div>
    </aside>
  );
}

export default function SkillDojoPanel(props: {
  agentId: string | null;
  metrics: Metrics | null;
  state?: AgentState | null;
  creationPreset: string;
  onStarted?: (agentId: string) => void;
}) {
  const { data: status } = usePolling<DojoStatus>(fetchDojoStatus, 1000);
  const { data: polledSkills } = usePolling<DojoSkill[]>(fetchDojoSkills, 5000);
  const { data: env } = usePolling<EnvironmentStatus>(fetchEnvironment, 1500);
  const [skills, setSkills] = useState<DojoSkill[]>([]);
  const [selectedId, setSelectedId] = useState("stand_and_recover");
  const [autoRetry, setAutoRetry] = useState(true);
  const [maxAttempts, setMaxAttempts] = useState(5);
  const [timeoutMultiplier, setTimeoutMultiplier] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadMessage, setUploadMessage] = useState<{ kind: BannerKind; text: string } | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (polledSkills) setSkills(polledSkills);
  }, [polledSkills]);

  useEffect(() => {
    if (skills.length && !skills.some((s) => s.skill_id === selectedId)) {
      setSelectedId(skills[0].skill_id);
    }
  }, [skills, selectedId]);

  const selected = useMemo(
    () => skills.find((s) => s.skill_id === selectedId) ?? skills[0] ?? null,
    [skills, selectedId],
  );
  const running = status?.running ?? false;
  const paused = status?.paused ?? false;
  const state = status?.state ?? "stopped";
  const hasAgent = !!(env?.agent_id ?? props.agentId);
  const perceptionHealth = props.state?.perceptual.discovery_health;

  const refreshSkills = async (selectId?: string) => {
    const next = await fetchDojoSkills();
    setSkills(next);
    if (selectId) setSelectedId(selectId);
  };

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      await refreshSkills();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onUpload = async (file: File | null) => {
    if (!file) return;
    await run(async () => {
      setUploadMessage(null);
      const text = await file.text();
      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch {
        setUploadMessage({ kind: "error", text: "Skill file is not valid JSON." });
        throw new Error("Skill file is not valid JSON");
      }
      const uploaded = await uploadDojoSkill(parsed);
      await refreshSkills(uploaded.skill_id);
      const warn = uploaded.warnings?.length ? ` Warnings: ${uploaded.warnings.join(" ")}` : "";
      setUploadMessage({
        kind: uploaded.warnings?.length ? "warning" : "success",
        text: `Imported ${uploaded.name} v${uploaded.version}.${warn}`,
      });
    });
    if (fileInput.current) fileInput.current.value = "";
  };

  const onDelete = () =>
    run(async () => {
      if (!selected || selected.builtin) return;
      if (!window.confirm(`Delete uploaded skill "${selected.name}"?`)) return;
      await deleteDojoSkill(selected.skill_id);
      const next = await fetchDojoSkills();
      setSkills(next);
      setSelectedId(next[0]?.skill_id ?? "stand_and_recover");
      setUploadMessage({ kind: "info", text: `Deleted ${selected.name}.` });
    });

  const onStart = () =>
    run(async () => {
      let agentId = env?.agent_id ?? props.agentId ?? null;
      if (!agentId || !env?.running) {
        const heavyWarning = heavyPresetWarning(props.creationPreset);
        if (
          heavyWarning &&
          !window.confirm(`Create a fresh agent with the "${props.creationPreset}" architecture?` + heavyWarning)
        ) {
          return;
        }
        const started = await startEnvironment({
          elements: DOJO_SCENE,
          vision: true,
          audio: false,
          braces: false,
          replace: !!agentId,
          preset: props.creationPreset,
        });
        agentId = started.agent_id;
        if (agentId) props.onStarted?.(agentId);
      }
      if (!agentId) throw new Error("No agent to train");
      if (!selected) throw new Error("No skill selected");
      await startDojo({
        agent_id: agentId,
        skill_id: selected.skill_id,
        auto_retry: autoRetry,
        max_attempts: maxAttempts,
        timeout_multiplier: timeoutMultiplier,
      });
    });

  return (
    <div className="panel cur-panel dojo-panel">
      <h2>
        Skill Dojo
        <Info tip="Reusable skill training around the Decadic loop. Teacher support trains and protects attempts; final graduation requires autonomous performance." />
      </h2>
      <DojoStatusHeader status={status ?? null} selected={selected} running={running} />
      <WorkflowSteps running={running} state={state} />
      {error && <Banner kind="error">{error}</Banner>}
      {perceptionHealth && perceptionHealth.status !== "healthy" && (
        <Banner kind={perceptionHealth.collapsed ? "error" : "warning"}>
          Perception health: {perceptionHealth.reason}. Object-dependent skill results are not trustworthy until object files recover.
        </Banner>
      )}
      <div className="dojo-workspace">
        <main className="dojo-main-flow">
          {!running && (
            <>
              <SkillLibraryPanel
                skills={skills}
                selected={selected}
                running={running}
                busy={busy}
                uploadMessage={uploadMessage}
                fileInput={fileInput}
                onSelect={setSelectedId}
                onUpload={(file) => void onUpload(file)}
                onDelete={onDelete}
              />
              <RunSetupPanel
                selected={selected}
                running={running}
                busy={busy}
                hasAgent={hasAgent}
                autoRetry={autoRetry}
                maxAttempts={maxAttempts}
                timeoutMultiplier={timeoutMultiplier}
                onAutoRetry={setAutoRetry}
                onMaxAttempts={setMaxAttempts}
                onTimeoutMultiplier={setTimeoutMultiplier}
                onStart={onStart}
              />
            </>
          )}
          <LiveTrainingPanel
            status={status ?? null}
            selected={selected}
            running={running}
            paused={paused}
            busy={busy}
            onPause={() => void run(pauseDojo)}
            onResume={() => void run(resumeDojo)}
            onStop={() => void run(stopDojo)}
            onJump={(i) => void run(() => setDojoPhase(i))}
          />
          <ReviewPanel status={status ?? null} />
        </main>
        <ManualScaffoldPanel agentId={props.agentId} metrics={props.metrics} running={running} disabled={false} />
      </div>
      {!props.agentId && (
        <p className="cur-hint">Start a skill run to create a training environment, or select an existing agent.</p>
      )}
    </div>
  );
}
