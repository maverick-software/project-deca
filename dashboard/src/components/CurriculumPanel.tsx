import { useState } from "react";
import {
  fetchCurriculum,
  fetchEnvironment,
  pauseCurriculum,
  resumeCurriculum,
  setCurriculumPhase,
  startCurriculum,
  startEnvironment,
  stopCurriculum,
  type CriterionStatus,
  type CurriculumStatus,
  type EnvironmentStatus,
} from "../api";
import { usePolling } from "../usePolling";
import Info from "./Info";

// The calm fixed scene the curriculum runs in: shelter + food + water.
const CURRICULUM_SCENE = ["house", "food", "water"];

function fmtVal(c: CriterionStatus): string {
  const v = c.value;
  const txt = Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3);
  return `${txt}${c.unit ? ` ${c.unit}` : ""}`;
}

/** One promotion criterion as a labelled progress bar. */
function GateBar(props: { c: CriterionStatus }) {
  const { c } = props;
  const pct = Math.max(0, Math.min(1, c.progress)) * 100;
  return (
    <div className="cur-gate">
      <div className="strip-label" style={{ marginBottom: 2, fontSize: 11 }}>
        <span>
          {c.satisfied ? "\u2714 " : ""}
          {c.label}
        </span>
        <span style={{ color: "#7c8499" }}>
          {fmtVal(c)} {c.comparator} {c.threshold}
        </span>
      </div>
      <div className="cur-meter">
        <div
          className={`cur-meter-fill ${c.satisfied ? "ok" : ""}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

/** Phase stepper: a dot per phase, the current one highlighted. */
function PhaseSteps(props: {
  count: number;
  current: number | null;
  onJump: (i: number) => void;
  disabled: boolean;
}) {
  const items = Array.from({ length: props.count }, (_, i) => i);
  return (
    <div className="cur-steps">
      {items.map((i) => (
        <button
          key={i}
          className={`cur-step ${i === props.current ? "on" : ""}`}
          disabled={props.disabled}
          title={`Jump to phase ${i} (manual override)`}
          onClick={() => props.onJump(i)}
        >
          {i}
        </button>
      ))}
    </div>
  );
}

/**
 * Walking-curriculum control panel. The curriculum is the "parent that shapes
 * the world and reads gates" - it never adds a reward or touches the loss. This
 * panel enables/disables it, shows the active phase + per-gate progress, the
 * phase history, and a manual phase override for experiments.
 */
export default function CurriculumPanel(props: {
  agentId: string | null;
  onStarted?: (agentId: string) => void;
}) {
  const { data: cur } = usePolling<CurriculumStatus>(fetchCurriculum, 1000);
  const { data: env } = usePolling<EnvironmentStatus>(fetchEnvironment, 1500);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [includeAffective, setIncludeAffective] = useState(false);

  const running = cur?.running ?? false;
  const paused = cur?.paused ?? false;
  const state = cur?.state ?? "stopped";
  const envAgent = env?.agent_id ?? null;
  const envRunning = env?.running ?? false;

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // Start a fresh training run: bind to a running body if there is one, else
  // spin up the fixed curriculum scene first, then start the curriculum.
  const onStartRun = () =>
    run(async () => {
      let agentId = envAgent ?? props.agentId ?? null;
      if (!agentId || !envRunning) {
        const started = await startEnvironment({
          elements: CURRICULUM_SCENE,
          vision: true,
          audio: false,
          replace: !!agentId,
        });
        agentId = started.agent_id;
      }
      if (!agentId) throw new Error("No agent to train (environment did not start)");
      await startCurriculum({ agent_id: agentId, include_affective: includeAffective });
      props.onStarted?.(agentId);
    });

  const onAttach = () =>
    run(async () => {
      const agentId = envAgent ?? props.agentId;
      if (!agentId) throw new Error("No running agent to attach to");
      await startCurriculum({ agent_id: agentId, include_affective: includeAffective });
      props.onStarted?.(agentId);
    });

  const gate = cur?.gate ?? null;
  const dwellPct =
    cur?.min_dwell_s && cur.min_dwell_s > 0
      ? Math.min(1, (cur.dwell_s ?? 0) / cur.min_dwell_s) * 100
      : 0;

  return (
    <div className="panel cur-panel">
      <h2>
        Walking Curriculum
        <Info tip="A faithful developmental trainer: it shapes the world (places food/water a step ahead) and reads observational gates to advance phases. It only retunes the agent's existing self-supervised drives via live config - it NEVER adds a reward or a term to the loss. Walking emerges from the agent's own predictive-coding + homeostatic-drive machinery." />
      </h2>

      <div className={`cur-status cur-${state}`}>
        <span className="cur-dot" />
        <b>{cur?.graduated ? "graduated" : state}</b>
        {cur?.agent_id && (
          <span className="env-meta">agent {cur.agent_id.slice(0, 8)}</span>
        )}
        {running && cur?.phase_index != null && (
          <span className="env-meta">
            phase {cur.phase_index}/{Math.max(0, (cur.phase_count ?? 1) - 1)}
          </span>
        )}
        {cur?.error && <span className="ctrl-error">{cur.error}</span>}
      </div>

      {running && cur && (
        <>
          <div className="cur-phase">
            <div className="cur-phase-head">
              <span className="cur-badge">{cur.phase_name}</span>
              {cur.is_terminal && <span className="cur-badge terminal">terminal</span>}
              <PhaseSteps
                count={cur.phase_count}
                current={cur.phase_index}
                onJump={(i) => void run(() => setCurriculumPhase(i))}
                disabled={busy}
              />
            </div>
            {cur.phase_description && (
              <p className="cur-desc">{cur.phase_description}</p>
            )}
          </div>

          <div className="strip-label" style={{ marginTop: 6 }}>
            <span>
              Minimum dwell
              <Info tip="Each phase requires a minimum time before it can promote, so a gate must hold (not just blip) before advancing." />
            </span>
            <span style={{ color: "#7c8499" }}>
              {Math.round(cur.dwell_s)}s / {cur.min_dwell_s ?? 0}s
            </span>
          </div>
          <div className="cur-meter">
            <div className="cur-meter-fill dwell" style={{ width: `${dwellPct}%` }} />
          </div>

          <div className="cur-section-label">
            Promotion gate
            <Info tip="Every criterion must hold over a rolling window of eval-only metrics before the agent advances to the next phase. Demotion (stepping back a phase) happens on death so the agent re-consolidates an earlier skill." />
            <span className="cur-gate-state">
              {gate?.satisfied
                ? "open"
                : gate?.enough_samples
                  ? "holding…"
                  : `warming up (${gate?.samples ?? 0})`}
            </span>
          </div>
          <div className="cur-gates">
            {gate?.criteria?.length ? (
              gate.criteria.map((c) => <GateBar key={c.key} c={c} />)
            ) : (
              <div className="empty">No gate (terminal phase).</div>
            )}
          </div>

          {cur.satisfier?.enabled && (
            <div className="cur-satisfier">
              Satisfier: placing {cur.satisfier.resources.join(" + ")} every{" "}
              {cur.satisfier.period_s}s (a step ahead — the agent must walk to it).
            </div>
          )}
        </>
      )}

      <div className="env-controls">
        {!running ? (
          <>
            <button
              className="btn start"
              disabled={busy}
              title="Start (or attach to) the fixed curriculum scene and begin training"
              onClick={onStartRun}
            >
              &#9654; Start training run
            </button>
            {envRunning && envAgent && (
              <button
                className="btn"
                disabled={busy}
                title={`Attach the curriculum to the running agent ${envAgent.slice(0, 8)}`}
                onClick={onAttach}
              >
                Attach to running agent
              </button>
            )}
            <label className="env-chip" style={{ marginLeft: 4 }}>
              <input
                type="checkbox"
                checked={includeAffective}
                disabled={busy}
                onChange={(e) => setIncludeAffective(e.target.checked)}
              />
              Include affective phase (needs a threat)
            </label>
          </>
        ) : (
          <>
            {paused ? (
              <button
                className="btn start"
                disabled={busy}
                onClick={() => void run(resumeCurriculum)}
              >
                &#9654; Resume
              </button>
            ) : (
              <button
                className="btn stop"
                disabled={busy}
                onClick={() => void run(pauseCurriculum)}
              >
                &#10073;&#10073; Pause
              </button>
            )}
            <button
              className="btn reset"
              disabled={busy}
              title="Stop the curriculum (the agent and its learned weights are kept)"
              onClick={() => void run(stopCurriculum)}
            >
              &#9632; Stop
            </button>
          </>
        )}
        {busy && <span className="strip-label">working…</span>}
        {error && <span className="ctrl-error">{error}</span>}
      </div>

      {cur?.history?.length ? (
        <>
          <div className="cur-section-label" style={{ marginTop: 14 }}>
            Phase history
          </div>
          <div className="cur-history">
            {cur.history
              .slice()
              .reverse()
              .map((h, i) => (
                <div key={`${h.at}-${i}`} className={`cur-hist-row ${h.event}`}>
                  <span className="cur-hist-event">{h.event}</span>
                  <span className="cur-hist-phase">
                    phase {h.phase} · {h.name}
                  </span>
                  <span className="cur-hist-at">{h.at.slice(11, 19)}</span>
                </div>
              ))}
          </div>
        </>
      ) : null}

      {!running && (
        <p className="cur-hint">
          Training runs in a single fixed scene (house + food + water). Phases shift
          only the agent's own drive/exploration knobs and where satisfiers appear —
          the body never restarts, so learned weights and earned joint ROM carry
          across the whole run.
        </p>
      )}
    </div>
  );
}
