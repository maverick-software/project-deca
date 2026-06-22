import type { AgentState } from "../api";
import Info from "./Info";

const BOX = 200;

function agencyColor(a: number | null | undefined): string {
  if (a == null) return "rgba(150,160,180,0.7)";
  // Violet for high agency ("mine"), grey-blue for low.
  const m = Math.max(0, Math.min(1, (a + 0.2) / 0.6));
  return `rgba(${Math.round(150 + 50 * m)},${Math.round(160 - 30 * m)},${Math.round(180 + 70 * m)},0.95)`;
}

export default function DiscoveryPanel(props: { state: AgentState }) {
  const p = props.state.perceptual;
  const mode = p.perception_mode ?? "oracle";
  const wm = p.working_memory;
  const slots = wm?.slots ?? [];
  const inView = slots.filter((s) => s.in_view);
  const parts = slots.filter((s) => s.kind === "self_part");
  const health = p.discovery_health;
  const ltm = p.ltm_consolidation;
  const organ = p.perception_organ;
  const scene = p.scene_workspace;
  const scenePred = p.scene_prediction;

  if (mode !== "discovered") {
    return (
      <div className="panel span-5">
        <h2>
          Object Discovery
          <Info tip="The agent's coined object files: anonymous objects discovered from its own camera via slot attention, re-identified across frames by appearance + motion, with a learned agency ('mine') score for body parts." />
        </h2>
        <div className="empty">
          This agent is in <b>oracle</b> perception mode — the world graph is handed to it by
          the simulator. Switch to <b>discovered</b> mode (Agent Settings) and reset to let
          the graph emerge from the agent's own perception.
        </div>
      </div>
    );
  }

  return (
    <div className="panel span-5">
      <h2>
        Object Discovery
        <Info tip="Coined object files discovered from the egocentric camera (slot attention), re-identified across frames by appearance + predicted image position. 'mine' = a slot whose motion the agent has learned to command (agency); these become body parts. Nothing here comes from the simulator." />
      </h2>

      <div className="strip-label">
        <span>{slots.length} objects · {inView.length} in view · {parts.length} body parts</span>
        <span>{health ? `${health.status} / LTM ${health.ltm_write}` : "image space"}</span>
      </div>

      {health && (
        <div className={`health-strip ${health.collapsed ? "bad" : health.status === "healthy" ? "ok" : "warn"}`}>
          <span>{health.reason}</span>
          <span>{health.object_files} object files</span>
          <span>spread {health.centroid_spread.toFixed(3)}</span>
          <span>
            cosine {health.appearance_cosine_mean != null ? health.appearance_cosine_mean.toFixed(3) : "n/a"}
          </span>
          <span>flow {health.flow_confidence.toFixed(3)}</span>
          <span>loom {health.looming_count}</span>
          <span>stuff {health.stuff_count}</span>
          <span>body {health.body_candidate_count}</span>
          <span>{ltm?.status ?? health.ltm_write}</span>
        </div>
      )}
      {organ && (
        <div className={`health-strip ${organ.stale_frame ? "warn" : "ok"}`}>
          <span>{organ.checkpoint_status}</span>
          <span>global {organ.global_motion.toFixed(3)}</span>
          <span>local {organ.local_motion_max.toFixed(3)}</span>
          <span>foreground {organ.foreground_count}</span>
        </div>
      )}
      {scene && (
        <div className={`health-strip ${scene.duplicate_identity_count > 0 ? "warn" : "ok"}`}>
          <span>scene {scene.entity_count} entities</span>
          <span>{scene.visible_count} visible</span>
          <span>{scene.occluded_count} occluded</span>
          <span>{scene.stable_count} stable</span>
          <span>{scene.focus_ids.length} focused</span>
          <span>{scene.relations.length} relations</span>
          <span>
            scene PE {scene.prediction_error != null ? scene.prediction_error.toFixed(3) : "n/a"}
          </span>
          <span>{scene.reidentified_count ?? 0} re-id</span>
        </div>
      )}
      {scenePred && (
        <div className={`health-strip ${(scenePred.unstable_count ?? 0) > 0 ? "warn" : "ok"}`}>
          <span>{scenePred.model_active ? "learned dynamics" : "constant-velocity fallback"}</span>
          <span>{scenePred.prediction_count ?? 0} predicted</span>
          <span>{scenePred.reidentified_count ?? 0} matched</span>
          <span>
            loss {scenePred.loss != null ? scenePred.loss.toFixed(4) : "n/a"}
          </span>
          <span>
            unc {scenePred.uncertainty != null ? scenePred.uncertainty.toFixed(3) : "n/a"}
          </span>
          <span>{scenePred.unstable_count ?? 0} unstable</span>
        </div>
      )}

      <svg
        className="discovery-svg"
        viewBox={`0 0 ${BOX} ${BOX}`}
        preserveAspectRatio="xMidYMid meet"
      >
        <rect x={0} y={0} width={BOX} height={BOX} className="disc-frame" />
        {slots.map((s, i) => {
          const uv = s.uv;
          if (!uv || uv.length < 2) return null;
          const cx = uv[0] * BOX;
          const cy = uv[1] * BOX;
          const r = 4 + 8 * Math.min(1, s.salience);
          const isPart = s.kind === "self_part";
          return (
            <g key={s.entity_id ?? i}>
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill={agencyColor(s.agency)}
                stroke={isPart ? "rgba(200,130,255,1)" : "rgba(0,0,0,0.4)"}
                strokeWidth={isPart ? 2.5 : 1}
                opacity={s.in_view ? 0.95 : 0.4}
              >
                <title>
                  {s.entity_id} · {s.kind}
                  {s.agency != null ? ` · agency ${s.agency.toFixed(3)}` : ""}
                </title>
              </circle>
              <text x={cx} y={cy - r - 2} textAnchor="middle" className="graph-label">
                {s.entity_id?.replace("obj-", "#") ?? ""}
              </text>
            </g>
          );
        })}
      </svg>

      <table className="disc-table">
        <thead>
          <tr>
            <th>object</th>
            <th>kind</th>
            <th>seen</th>
            <th>salience</th>
            <th>conf</th>
            <th>motion</th>
            <th>agency</th>
          </tr>
        </thead>
        <tbody>
          {slots.slice(0, 12).map((s) => (
            <tr key={s.entity_id} className={s.kind === "self_part" ? "row-part" : ""}>
              <td>{s.entity_id}</td>
              <td>{s.kind === "self_part" ? "mine ✋" : s.kind}</td>
              <td>{s.seen_count}{s.in_view ? " ●" : ""}</td>
              <td>{s.salience.toFixed(2)}</td>
              <td>{s.confidence != null ? s.confidence.toFixed(2) : "n/a"}</td>
              <td>{s.local_motion != null ? s.local_motion.toFixed(2) : "n/a"}</td>
              <td>{s.agency != null ? s.agency.toFixed(3) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
