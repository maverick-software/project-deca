import { useEffect, useState } from "react";
import type { CapacityConfig, Metrics } from "../api";
import { configureAgent, giveResource } from "../api";
import Info from "./Info";

function Gauge(props: { value: number; max: number; color: string }) {
  const pct = Math.max(0, Math.min(100, (props.value / props.max) * 100));
  return (
    <div className="gauge-track">
      <div className="gauge-fill" style={{ width: `${pct}%`, backgroundColor: props.color }} />
    </div>
  );
}

function reservoirColor(v: number): string {
  return v > 60 ? "#4fd683" : v > 30 ? "#ffb74a" : "#ff5a6e";
}

function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds)) return "\u221e"; // infinity
  if (seconds <= 0) return "0s";
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m`;
  return `${Math.round(seconds)}s`;
}

// Compression slider is exponential: exponent 0..4 -> 1x .. 10000x.
function expToCompression(exp: number): number {
  return Math.round(Math.pow(10, exp));
}
function compressionToExp(c: number): number {
  return Math.max(0, Math.min(4, Math.log10(Math.max(1, c))));
}

export default function HomeostasisPanel(props: {
  agentId: string;
  metrics: Metrics | null;
  capacity?: CapacityConfig;
}) {
  const { agentId, metrics } = props;
  const mode = props.capacity?.viability_mode ?? metrics?.viability_mode ?? "metabolic";
  const compression =
    props.capacity?.metabolic_compression ?? metrics?.metabolic_compression ?? 1;

  const hydration = metrics?.hydration ?? 100;
  const energy = metrics?.energy ?? 100;
  const integrity = metrics?.integrity ?? 100;
  const stress = metrics?.stress ?? 0;
  const viability = metrics?.viability ?? Math.min(hydration, energy, integrity);
  const ttd = metrics?.time_to_death_s;

  const [exp, setExp] = useState<number>(compressionToExp(compression));
  const [busy, setBusy] = useState(false);
  const [provBusy, setProvBusy] = useState(false);
  const [provError, setProvError] = useState<string | null>(null);

  // Re-sync compression slider when the agent changes (not every poll).
  useEffect(() => {
    setExp(compressionToExp(compression));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentId]);

  const commit = async (cfg: Partial<CapacityConfig>) => {
    setBusy(true);
    try {
      await configureAgent(agentId, cfg);
    } catch {
      // server unreachable; keep optimistic local value
    } finally {
      setBusy(false);
    }
  };

  const give = async (resource: "water" | "food", giveMode: "near" | "direct") => {
    setProvBusy(true);
    setProvError(null);
    try {
      await giveResource(agentId, resource, giveMode);
    } catch (e) {
      setProvError(e instanceof Error ? e.message : String(e));
    } finally {
      setProvBusy(false);
    }
  };

  const metabolic = mode === "metabolic";

  return (
    <div className="panel span-4">
      <h2>
        Homeostasis
        <Info tip="The body's survival reservoirs. Viability is the minimum of hydration, energy, and integrity. In Metabolic mode they drain on a real human timeline (thirst ~3 days, hunger ~3 weeks) and damage cuts integrity; Immortal mode pins everything at full so you can watch learning without death." />
      </h2>

      <div className="mode-toggle">
        <button
          className={`btn ${metabolic ? "start" : ""}`}
          disabled={busy || metabolic}
          onClick={() => void commit({ viability_mode: "metabolic" })}
          title="Run the full wall-clock metabolic model (the body can die)."
        >
          Metabolic
        </button>
        <button
          className={`btn ${!metabolic ? "start" : ""}`}
          disabled={busy || !metabolic}
          onClick={() => void commit({ viability_mode: "immortal" })}
          title="Pin all reservoirs at full and disable death (long learning runs)."
        >
          Immortal
        </button>
      </div>

      <div className="strip-label" style={{ marginTop: 14 }}>
        <span>
          Provisions
          <Info tip="Give the agent water or food two ways. 'Nearby' drops the (unlabeled) item a step in front of it, so it must perceive and walk over to consume it. 'Direct' shows the item in the egocentric camera and moves it toward the head until normal consumption fires, preserving the object-to-relief learning path." />
        </span>
      </div>
      {(["water", "food"] as const).map((resource) => (
        <div
          key={resource}
          style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6 }}
        >
          <span style={{ flex: "0 0 64px", textTransform: "capitalize" }}>{resource}</span>
          <button
            className="btn"
            style={{ flex: 1 }}
            disabled={provBusy || !agentId}
            onClick={() => void give(resource, "near")}
            title={`Place ${resource} a step in front of the agent — it must see and walk to it (preserves self-learned seeking; needs a running scenario with ${resource}).`}
          >
            Place nearby
          </button>
          <button
            className="btn"
            style={{ flex: 1 }}
            disabled={provBusy || !agentId}
            onClick={() => void give(resource, "direct")}
            title={`Show ${resource} in the agent's view and deliver it toward the head until normal consumption gives ${resource === "water" ? "hydration" : "energy"} relief.`}
          >
            Give directly
          </button>
        </div>
      ))}
      {provBusy && <div className="strip-label">working…</div>}
      {provError && <div className="ctrl-error">{provError}</div>}

      <div className="strip-label" style={{ marginTop: 12 }}>
        <span>
          Viability (min)
          <Info tip="The lowest of the three reservoirs. Hitting 0 in Metabolic mode means the agent is no longer viable and the mind freezes." />
        </span>
        <span>{viability.toFixed(1)} / 100</span>
      </div>
      <Gauge value={viability} max={100} color={reservoirColor(viability)} />

      <div className="strip-label" style={{ marginTop: 12 }}>
        <span>
          Hydration
          <Info tip="Water reserve. Drains fastest (thirst kills in ~3 days). Refilled by drinking water glasses." />
        </span>
        <span>{hydration.toFixed(1)}</span>
      </div>
      <Gauge value={hydration} max={100} color={reservoirColor(hydration)} />

      <div className="strip-label" style={{ marginTop: 12 }}>
        <span>
          Energy
          <Info tip="Caloric reserve. Drains slowly (starvation in ~3 weeks). Refilled by eating food." />
        </span>
        <span>{energy.toFixed(1)}</span>
      </div>
      <Gauge value={energy} max={100} color={reservoirColor(energy)} />

      <div className="strip-label" style={{ marginTop: 12 }}>
        <span>
          Integrity
          <Info tip="Tissue health. Cut by collisions/falls; heals slowly over days, but only while fed and hydrated. Heavy damage is lethal." />
        </span>
        <span>{integrity.toFixed(1)}</span>
      </div>
      <Gauge value={integrity} max={100} color={reservoirColor(integrity)} />

      <div className="strip-label" style={{ marginTop: 12 }}>
        <span>
          Stress
          <Info tip="Blends pain, lingering threat, and prediction error. Multiplies depletion and slows healing — like cortisol raising metabolic burn." />
        </span>
        <span>{stress.toFixed(2)}</span>
      </div>
      <Gauge value={stress * 100} max={100} color="#ff5a6e" />

      <div className="statrow" style={{ marginTop: 14 }}>
        <span className="k">
          Est. time to death
          <Info tip="Projected time until the fastest-draining reservoir empties at the current stress and compression. Infinite in Immortal mode." />
        </span>
        <span className="v">{metabolic ? fmtDuration(ttd) : "\u221e"}</span>
      </div>

      <label className="cap-row" style={{ marginTop: 10 }}>
        <span>
          Time compression
          <Info tip="Accelerates the metabolic clock for testing. 1x is the real human timeline; higher values fast-forward thirst, hunger, and tissue healing alike (thirst/hunger keep their ~7:1 ratio)." />
        </span>
        <input
          type="range"
          min={0}
          max={4}
          step={0.25}
          value={exp}
          disabled={!metabolic}
          onChange={(e) => setExp(Number(e.target.value))}
          onPointerUp={() => void commit({ metabolic_compression: expToCompression(exp) })}
          onKeyUp={() => void commit({ metabolic_compression: expToCompression(exp) })}
        />
        <b>{expToCompression(exp)}x</b>
      </label>
      {busy && <div className="strip-label">applying…</div>}
    </div>
  );
}
