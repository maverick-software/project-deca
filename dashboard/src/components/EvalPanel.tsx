import type { AgentState, Metrics } from "../api";
import Info from "./Info";

function Bar(props: { value: number; label: string; tip: string }) {
  const pct = Math.max(0, Math.min(1, props.value)) * 100;
  return (
    <div className="eval-bar">
      <div className="eval-bar-head">
        <span>
          {props.label}
          <Info tip={props.tip} />
        </span>
        <b>{props.value.toFixed(2)}</b>
      </div>
      <div className="eval-track">
        <div className="eval-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function EvalPanel(props: { state: AgentState; metrics: Metrics | null }) {
  const p = props.state.perceptual;
  const mode = p.perception_mode ?? "oracle";
  const d = p.discovery;

  if (mode !== "discovered" || !d) {
    return (
      <div className="panel span-5">
        <h2>
          Discovery Evaluation
          <Info tip="How well the agent's discovered graph matches the (eval-only) oracle ground truth: detection precision/recall, identity stability, and body-part agency accuracy. The oracle is never fed to cognition — it is used only to score." />
        </h2>
        <div className="empty">
          Available in <b>discovered</b> perception mode. The simulator's entity list is held
          back from cognition and used only to score what the agent has recovered on its own.
        </div>
      </div>
    );
  }

  return (
    <div className="panel span-5">
      <h2>
        Discovery Evaluation
        <Info tip="Discovered graph vs. eval-only oracle truth (the simulator's entity list, never fed to cognition). Greedy direction matching: a detection counts when its egocentric bearing lines up with a real entity. Body-part accuracy compares slots flagged 'mine' against the real hands/feet." />
      </h2>

      <div className="strip-label">
        <span>{d.updates} evals · {d.last_detected} detected · {d.last_oracle} truth · {d.last_matched} matched</span>
        <span>eval-only</span>
      </div>

      <Bar
        value={d.precision}
        label="precision"
        tip="Of the objects the agent discovered, the fraction that line up with a real entity (by egocentric direction). Rises as the presence head learns to suppress empty slots."
      />
      <Bar
        value={d.recall}
        label="recall"
        tip="Of the real entities in view, the fraction the agent discovered. Rises as slot attention captures more objects."
      />
      <Bar
        value={d.id_stability}
        label="id stability"
        tip="1 − id churn: how consistently a tracked object keeps its coined id across frames (object permanence / re-identification quality)."
      />
      <Bar
        value={d.body_part_accuracy}
        label="body-part accuracy"
        tip="Of the agent's discovered body parts ('mine'), the fraction whose direction matches a real limb (hand/foot world position, eval-only). Rises as the agency head learns the efference↔motion contingency."
      />

      <div className="cap-readouts">
        <div>
          <span className="cap-k">objects</span>
          <span className="cap-v">{props.metrics?.discovered_objects ?? 0}</span>
        </div>
        <div>
          <span className="cap-k">body parts</span>
          <span className="cap-v">{props.metrics?.self_parts ?? 0}</span>
        </div>
        <div>
          <span className="cap-k">slot recon</span>
          <span className="cap-v">{(props.metrics?.slot_recon_error ?? 0).toFixed(3)}</span>
        </div>
        <div>
          <span className="cap-k">agency μ</span>
          <span className="cap-v">{(props.metrics?.agency_mean ?? 0).toFixed(3)}</span>
        </div>
      </div>
    </div>
  );
}
