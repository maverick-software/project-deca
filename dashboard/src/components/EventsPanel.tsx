import type { PerceptualSnapshot } from "../api";
import Info from "./Info";

export default function EventsPanel(props: { perceptual: PerceptualSnapshot }) {
  const events = [...props.perceptual.recent_events].reverse();
  return (
    <div className="panel span-5">
      <h2>
        Recent Events
        <Info tip="Discrete things that happened to the body, as reported in observations. Damaging events (collisions, falls) hit a viability fast path immediately — they hurt before the next full cycle even runs." />
      </h2>
      <div className="event-legend">
        <span className="chip">
          <span className="dot" style={{ background: "var(--pain)" }} />
          collision — hard impact on a touch sensor
        </span>
        <span className="chip">
          <span className="dot" style={{ background: "var(--warn)" }} />
          fall — torso dropped below standing height
        </span>
        <span className="chip">
          <span className="dot" style={{ background: "#ffd84a" }} />
          threat_near — predator within range
        </span>
        <span className="chip">
          <span className="dot" style={{ background: "var(--pleasure)" }} />
          food — morsel eaten (energy credit)
        </span>
        <span className="chip">
          <span className="dot" style={{ background: "#5aa9ff" }} />
          water — glass drunk (hydration credit)
        </span>
        <span className="chip">number = intensity (0–1)</span>
      </div>
      <div className="events">
        {events.length === 0 && <div className="empty">no events observed</div>}
        {events.map((e, i) => {
          const type = String(e.type ?? "event");
          const intensity = typeof e.intensity === "number" ? e.intensity : null;
          const source = e.source ?? e.region_name ?? "";
          return (
            <div className="event-row" key={i}>
              <span className={`event-type ${type}`}>{type}</span>
              <span className="event-src">{String(source)}</span>
              {intensity !== null && <span>{intensity.toFixed(2)}</span>}
            </div>
          );
        })}
      </div>
      <div className="statrow" style={{ marginTop: 10 }}>
        <span className="k">
          Last observation
          <Info tip="Time the server last received sensory data from the body. If this stops advancing, the body adapter has disconnected." />
        </span>
        <span className="v">
          {props.perceptual.last_timestamp_iso
            ? props.perceptual.last_timestamp_iso.slice(11, 19)
            : "—"}
        </span>
      </div>
      <div className="statrow">
        <span className="k">
          Integration ticks
          <Info tip="Number of observations fused into the perceptual state since start/reset. Usually runs faster than cycles — the body senses more often than the mind deliberates." />
        </span>
        <span className="v">{props.perceptual.integration_ticks}</span>
      </div>
    </div>
  );
}
