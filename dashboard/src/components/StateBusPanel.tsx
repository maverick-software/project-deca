import type { ActionRecord, StateBusSnapshot } from "../api";
import Info from "./Info";

/** Signed value → blue (negative) / dark (zero) / red (positive). */
function heatColor(v: number, maxAbs: number): string {
  if (maxAbs < 1e-9) return "#1d2230";
  const n = Math.max(-1, Math.min(1, v / maxAbs));
  const intensity = Math.round(Math.abs(n) * 70 + 12);
  return n >= 0
    ? `hsl(355 75% ${intensity}%)`
    : `hsl(215 75% ${intensity}%)`;
}

function HeatStrip(props: { label: string; values: (number | null)[]; tip: string }) {
  // The server can serialize NaN/None as null (e.g. an unstable neural state);
  // coerce to finite numbers so a single bad cell can't crash the whole panel.
  const values = props.values.map((v) =>
    typeof v === "number" && Number.isFinite(v) ? v : 0,
  );
  const maxAbs = values.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  return (
    <div>
      <div className="strip-label">
        <span>
          {props.label}
          <Info tip={props.tip} />
        </span>
        <span>{values.length}d · max |{maxAbs.toFixed(2)}|</span>
      </div>
      <div className="heatstrip">
        {values.map((v, i) => (
          <div key={i} style={{ backgroundColor: heatColor(v, maxAbs) }} title={v.toFixed(4)} />
        ))}
      </div>
    </div>
  );
}

function fmtDir(a: ActionRecord): string {
  const p = a.action.parameters;
  if (!p?.direction) return a.action.type ?? "?";
  const d = p.direction
    .map((x) => (typeof x === "number" && Number.isFinite(x) ? x : 0).toFixed(2))
    .join(", ");
  return `${a.action.type} dir=[${d}] v=${(p.speed ?? 0).toFixed(2)}`;
}

export default function StateBusPanel(props: { bus: StateBusSnapshot }) {
  const { bus } = props;
  const actions = [...bus.F_action_history].reverse().slice(0, 24);
  return (
    <div className="panel span-7">
      <h2>
        State Bus (A–F)
        <Info tip="The agent's persistent inner state — six elements (A–F) that survive between cycles and are read/written by every stage. This is what makes it a continuous mind rather than a stateless input-output function." />
      </h2>
      <div className="heat-legend">
        <span>−</span>
        <div className="heat-gradient" />
        <span>+</span>
        <span>each cell = one dimension, scaled to that strip's max · hover a cell for its value</span>
      </div>
      <HeatStrip
        label="A — State of Mind"
        values={bus.A_state_of_mind}
        tip="The agent's overall cognitive state: a latent vector rewritten every cycle by the neural stack. Think of it as 'what it is like to be the agent right now'. Shifting patterns = a changing mental state; a frozen pattern = a stuck one."
      />
      <HeatStrip
        label="B — Emotion / Physiology"
        values={bus.B_emotion_physio}
        tip="Affect vector — the felt-body dimension of the state. The pain and pleasure scalars shown in Vitals live alongside this vector and color every stage's processing."
      />
      <HeatStrip
        label="C — Internal Narrative"
        values={bus.C_narrative_emb}
        tip="Embedding of the agent's running story about itself and its situation — what happened, what it's doing, what it expects. Feeds memory storage and recall."
      />
      <HeatStrip
        label="E — Metacognition"
        values={bus.E_metacognition}
        tip="Self-monitoring vector: confidence, error tracking, 'how well am I thinking?'. Lets the agent notice when its own predictions are unreliable."
      />

      <div className="strip-label" style={{ marginTop: 10, marginBottom: 4 }}>
        <span>
          F — Action history (latest first)
          <Info tip="Element F: the actions emitted by stage 10 each cycle. 'move dir=[x, y, z] v=speed' is a locomotion command sent to the body — the direction vector steers the pelvis, v is the requested speed." />
        </span>
        <span>cycle {bus.cycle_index}</span>
      </div>
      <div className="actions">
        {actions.length === 0 && <div className="empty">no actions yet</div>}
        {actions.map((a, i) => (
          <div className="action-row" key={`${a.cycle}-${i}`}>
            <span className="cyc">#{a.cycle}</span>
            <span>{fmtDir(a)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
