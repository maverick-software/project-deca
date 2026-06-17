import type { Deployment } from "../../vastApi";

function fmtElapsed(s: number): string {
  const sec = Math.max(0, Math.floor(s));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const ss = sec % 60;
  return h > 0 ? `${h}h ${m}m` : `${m}m ${ss}s`;
}

/** Active instance summary + lifecycle controls (watch / stop / destroy). */
export default function ActiveDeployment(props: {
  deployment: Deployment;
  busy: boolean;
  onWatch: () => void;
  onStop: () => void;
  onDestroy: () => void;
}) {
  const { deployment: d } = props;
  return (
    <div className="panel" style={{ display: "grid", gap: 10 }}>
      <h3 style={{ margin: 0 }}>
        Active deployment{" "}
        {d.ready ? (
          <span style={{ color: "#3fb950" }}>● ready</span>
        ) : (
          <span style={{ color: "#58a6ff" }}>● {d.phase}</span>
        )}
      </h3>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
          gap: 10,
          fontSize: 13,
        }}
      >
        <Metric label="Instance" value={d.instance_id ?? "-"} />
        <Metric label="$ / hr" value={d.dph != null ? `$${d.dph.toFixed(3)}` : "-"} />
        <Metric label="Elapsed" value={fmtElapsed(d.elapsed_s)} />
        <Metric
          label="Est. cost"
          value={d.est_cost_usd != null ? `$${d.est_cost_usd.toFixed(3)}` : "-"}
        />
        <Metric label="Preset" value={d.preset ?? "-"} />
        <Metric label="Scene" value={d.scene ?? "-"} />
        <Metric label="Agent" value={d.agent_id ? d.agent_id.slice(0, 8) : "-"} />
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button
          onClick={props.onWatch}
          disabled={!d.ready}
          title="Jump to Overview; the panels show the remote agent"
        >
          Watch agent
        </button>
        <button onClick={props.onStop} disabled={props.busy} title="Pause billing, keep disk">
          Stop
        </button>
        <button
          onClick={props.onDestroy}
          disabled={props.busy}
          style={{ borderColor: "#f85149", color: "#f85149" }}
          title="Checkpoint, copy back, then destroy the instance"
        >
          Destroy
        </button>
      </div>

      <div style={{ fontSize: 11, opacity: 0.6 }}>
        Billing runs while the instance is up. Stop pauses GPU billing (disk still
        charged); Destroy terminates it entirely.
      </div>
    </div>
  );
}

function Metric(props: { label: string; value: string | number }) {
  return (
    <div>
      <div style={{ fontSize: 11, opacity: 0.6 }}>{props.label}</div>
      <div style={{ fontWeight: 600 }}>{props.value}</div>
    </div>
  );
}
