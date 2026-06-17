import { useState } from "react";
import {
  fetchExplain,
  type AgentState,
  type Counterfactuals,
  type ExplainReport,
  type IntentDriver,
} from "../api";
import { usePolling } from "../usePolling";
import Info from "./Info";

function fmt(n: number | null | undefined, d = 3): string {
  return n == null || Number.isNaN(n) ? "—" : n.toFixed(d);
}

function driverColor(d: IntentDriver): string {
  // Green when the chosen action is predicted to help (reduce deviation),
  // amber when it doesn't move the needle, relative to standing still.
  return d.action_delta > 1e-4
    ? "rgba(80,200,140,0.9)"
    : d.action_delta < -1e-4
      ? "rgba(220,120,110,0.9)"
      : "rgba(150,160,180,0.85)";
}

function IntentBars(props: { drivers: IntentDriver[] }) {
  const drivers = props.drivers ?? [];
  if (!drivers.length) return <div className="empty">No active survival objective this cycle.</div>;
  const max = Math.max(...drivers.map((d) => Math.abs(d.contribution)), 1e-6);
  return (
    <div className="cog-bars">
      {drivers.map((d) => (
        <div className="cog-bar-row" key={`${d.group}:${d.goal}`}>
          <span className="cog-bar-label" title={`${d.group} · weight ${fmt(d.weight, 2)}`}>
            {d.goal}
          </span>
          <div className="cog-bar-track">
            <div
              className="cog-bar-fill"
              style={{
                width: `${Math.max(2, (Math.abs(d.contribution) / max) * 100)}%`,
                background: driverColor(d),
              }}
            />
          </div>
          <span className="cog-bar-val" title="predicted → preferred (action vs. standing still)">
            {fmt(d.predicted, 2)}→{fmt(d.preferred, 2)}{" "}
            <em className={d.action_delta > 0 ? "good" : d.action_delta < 0 ? "bad" : ""}>
              {d.action_delta > 0 ? "+" : ""}
              {fmt(d.action_delta, 3)}
            </em>
          </span>
        </div>
      ))}
    </div>
  );
}

function Sparkline(props: { values: (number | null | undefined)[]; color: string }) {
  const vals = props.values.map((v) => (v == null || Number.isNaN(v) ? 0 : v));
  if (vals.length < 2) return null;
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const span = hi - lo || 1;
  const W = 160;
  const H = 28;
  const pts = vals
    .map((v, i) => `${(i / (vals.length - 1)) * W},${H - ((v - lo) / span) * H}`)
    .join(" ");
  return (
    <svg className="cog-spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <polyline points={pts} fill="none" stroke={props.color} strokeWidth={1.5} />
    </svg>
  );
}

function CounterfactualTable(props: { cf: Counterfactuals }) {
  const rows = props.cf.candidates ?? [];
  return (
    <div className="cog-cf">
      <div className="strip-label">
        <span>decision landscape</span>
        <span>{props.cf.objective}</span>
      </div>
      <table className="cog-table">
        <thead>
          <tr>
            <th>action</th>
            <th>drive cost</th>
            <th>motor rms</th>
            <th>predicted reservoirs</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr key={c.action} className={c.action === "emitted" ? "row-part" : ""}>
              <td>{c.action === "emitted" ? "chosen ▶" : c.action}</td>
              <td>{fmt(c.drive_cost, 4)}</td>
              <td>{fmt(c.motor_rms, 3)}</td>
              <td>{(c.intero_pred ?? []).map((v) => v.toFixed(2)).join(" / ") || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function CognitionPanel(props: { agentId: string; state: AgentState }) {
  const { agentId, state } = props;
  const trace = state.cognitive_trace ?? null;
  const [showCf, setShowCf] = useState(false);

  // History (compact per-cycle summaries) for the temporal sparklines.
  const { data: explain } = usePolling<ExplainReport | null>(
    () => (agentId ? fetchExplain(agentId, { history: 120, counterfactuals: showCf }) : Promise.resolve(null)),
    1000,
    [agentId, showCf],
  );
  const history = explain?.history ?? [];
  const cf = explain?.on_demand?.counterfactuals ?? null;

  if (!trace) {
    return (
      <div className="panel span-7">
        <h2>
          Cognition / Why
          <Info tip="A read-only translation of the agent's neural state into why it acted: survival-intent decomposition, input attribution, self-model surprise, episodic grounding, and (optionally) latent probes + a narrative." />
        </h2>
        <div className="empty">
          No cognitive trace yet. The agent needs a streaming body and{" "}
          <code>DECADIC_COGNITION_TRACE</code> enabled (it is on by default). Once cycles run with
          a body, this panel explains each decision.
        </div>
      </div>
    );
  }

  const a = trace.affect;
  const salient = trace.salient ?? null;
  const fr = salient?.fractions ?? null;
  const node = salient?.node ?? null;
  const ep = trace.recalled_episode ?? null;
  const probes = trace.probes ?? null;
  const surprise = trace.self_surprise;

  return (
    <div className="panel span-7">
      <h2>
        Cognition / Why <span className="badge">cycle {trace.cycle}</span>
        <Info tip="Tier A (the agent's own objective/world model) is high-trust; Tier B probes are correlational read-outs labeled with quality (R²/acc); Tier C narrative is a gloss. This measures functional/structural correlates (reportable, global-workspace access), not phenomenal experience." />
      </h2>

      {trace.narrative && <div className="cog-narrative">“{trace.narrative}”</div>}

      <div className="cog-grid">
        <section className="cog-section">
          <div className="strip-label">
            <span>survival intent</span>
            <span title="the agent's own free-energy objective">{trace.intent.summary}</span>
          </div>
          <IntentBars drivers={trace.intent.drivers} />
        </section>

        <section className="cog-section">
          <div className="strip-label">
            <span>affect</span>
            <span>{a.priority}</span>
          </div>
          <div className="cog-affect">
            <span className="cog-chip" title="felt pain (bounded [0,1])">pain {fmt(a.pain, 2)}</span>
            <span className="cog-chip" title="felt pleasure">pleasure {fmt(a.pleasure, 2)}</span>
            <span className="cog-chip" title="risk estimate (P)">risk {fmt(a.risk, 2)}</span>
          </div>
        </section>

        <section className="cog-section">
          <div className="strip-label">
            <span>attribution</span>
            <span title="d|motor|/d(input); sampled">
              {salient ? salient.target : "(sampled every N cycles)"}
            </span>
          </div>
          {fr ? (
            <div className="cog-attr">
              {Object.entries(fr).map(([k, v]) => (
                <div className="cog-bar-row" key={k}>
                  <span className="cog-bar-label">{k}</span>
                  <div className="cog-bar-track">
                    <div
                      className="cog-bar-fill"
                      style={{ width: `${Math.max(2, v * 100)}%`, background: "rgba(120,150,230,0.9)" }}
                    />
                  </div>
                  <span className="cog-bar-val">{(v * 100).toFixed(0)}%</span>
                </div>
              ))}
              {node?.node_id && (
                <div className="cog-note">
                  most-attended node: <b>{node.node_id}</b>
                  {node.kind ? ` (${node.kind})` : ""}
                </div>
              )}
            </div>
          ) : (
            <div className="empty">No attribution this cycle (sampled).</div>
          )}
        </section>

        <section className="cog-section">
          <div className="strip-label">
            <span>self-model surprise</span>
            <span>{surprise.summary}</span>
          </div>
          <div className="cog-affect">
            {(surprise.dims ?? []).slice(0, 5).map((d) => (
              <span className="cog-chip" key={d.name} title={`predicted ${fmt(d.predicted, 2)} vs actual ${fmt(d.actual, 2)}`}>
                {d.name} {fmt(d.residual, 2)}
              </span>
            ))}
          </div>
        </section>

        <section className="cog-section">
          <div className="strip-label">
            <span>episodic grounding</span>
            <span>{ep ? `sim ${fmt(ep.similarity, 2)}` : "—"}</span>
          </div>
          {ep ? (
            <div className="cog-note">
              Resembles cycle <b>{ep.cycle}</b>
              {ep.priority ? ` · priority ${ep.priority}` : ""}
              {ep.pain != null ? ` · pain ${fmt(ep.pain, 2)}` : ""}
              {ep.action_type ? ` · ${ep.action_type}` : ""}
            </div>
          ) : (
            <div className="empty">No similar episode recalled.</div>
          )}
        </section>

        {probes && Object.keys(probes).length > 0 && (
          <section className="cog-section">
            <div className="strip-label">
              <span>interpretability probes</span>
              <span title="linear decode of latents from eval-only truth">read-only</span>
            </div>
            <table className="cog-table">
              <thead>
                <tr>
                  <th>variable</th>
                  <th>predicted</th>
                  <th>latent · axis</th>
                  <th>quality</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(probes).map(([name, p]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{fmt(p.predicted, 3)}</td>
                    <td>
                      {p.best_latent} · #{p.axis}
                    </td>
                    <td>
                      {p.score_kind} {fmt(p.score, 2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        <section className="cog-section cog-wide">
          <div className="strip-label">
            <span>temporal trace</span>
            <span>{history.length} cycles</span>
          </div>
          <div className="cog-sparks">
            <div className="cog-spark-row">
              <span>pain</span>
              <Sparkline values={history.map((h) => h.pain)} color="rgba(220,120,110,0.95)" />
            </div>
            <div className="cog-spark-row">
              <span>risk</span>
              <Sparkline values={history.map((h) => h.risk)} color="rgba(230,180,90,0.95)" />
            </div>
            <div className="cog-spark-row">
              <span>surprise</span>
              <Sparkline values={history.map((h) => h.surprise)} color="rgba(120,150,230,0.95)" />
            </div>
          </div>
        </section>

        <section className="cog-section cog-wide">
          <div className="strip-label">
            <span>counterfactuals</span>
            <button className="cog-btn" onClick={() => setShowCf((v) => !v)}>
              {showCf ? "hide" : "preview alternatives"}
            </button>
          </div>
          {showCf ? (
            cf ? (
              <CounterfactualTable cf={cf} />
            ) : (
              <div className="empty">Computing the decision landscape…</div>
            )
          ) : (
            <div className="empty">
              Preview what the agent's world model predicts for alternative motor commands.
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
