import type { Deployment } from "../../vastApi";

const PHASE_LABELS: Record<string, string> = {
  creating: "Rent instance",
  waiting: "Wait for boot",
  uploading: "Upload code",
  installing: "Install + prewarm",
  serving: "Start brain",
  tunneling: "Open tunnel",
  starting_agent: "Spawn agent",
  ready: "Ready",
};

/** Live provisioning stepper + streaming remote log. */
export default function DeployProgress(props: { deployment: Deployment }) {
  const { deployment: d } = props;
  const order = d.phase_order;
  const ci = d.phase === "ready" ? order.length : order.indexOf(d.phase);
  const failed = d.phase === "error";

  return (
    <div className="panel" style={{ display: "grid", gap: 10 }}>
      <h3 style={{ margin: 0 }}>Provisioning</h3>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {order.map((p, i) => {
          const done = i < ci;
          const active = i === ci && !failed;
          const color = done
            ? "#3fb950"
            : active
              ? "#58a6ff"
              : failed && i === ci
                ? "#f85149"
                : "rgba(255,255,255,0.25)";
          return (
            <span
              key={p}
              style={{
                fontSize: 12,
                padding: "3px 8px",
                borderRadius: 12,
                border: `1px solid ${color}`,
                color,
                whiteSpace: "nowrap",
              }}
            >
              {done ? "✓ " : active ? "● " : ""}
              {PHASE_LABELS[p] ?? p}
            </span>
          );
        })}
      </div>

      {failed && d.error && (
        <div className="error-banner" style={{ margin: 0 }}>
          {d.error}
        </div>
      )}

      <pre
        style={{
          margin: 0,
          maxHeight: 220,
          overflowY: "auto",
          background: "rgba(0,0,0,0.35)",
          padding: 10,
          borderRadius: 6,
          fontSize: 12,
          lineHeight: 1.4,
          whiteSpace: "pre-wrap",
        }}
      >
        {d.log.length ? d.log.join("\n") : "Waiting for the first log line..."}
      </pre>
    </div>
  );
}
