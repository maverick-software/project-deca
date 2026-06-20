import type { Metrics } from "../api";
import Info from "./Info";

/** Full-body touch order (head -> toe) with friendly labels for the contact map. */
const PART_ORDER: { key: string; label: string }[] = [
  { key: "head", label: "Head" },
  { key: "torso", label: "Torso" },
  { key: "waist", label: "Waist" },
  { key: "butt", label: "Pelvis" },
  { key: "left_uarm", label: "L upper arm" },
  { key: "right_uarm", label: "R upper arm" },
  { key: "left_larm", label: "L forearm" },
  { key: "right_larm", label: "R forearm" },
  { key: "left_hand", label: "L hand" },
  { key: "right_hand", label: "R hand" },
  { key: "left_thigh", label: "L thigh" },
  { key: "right_thigh", label: "R thigh" },
  { key: "left_shin", label: "L shin" },
  { key: "right_shin", label: "R shin" },
  { key: "left_foot", label: "L foot" },
  { key: "right_foot", label: "R foot" },
];

/** Hinge order matching the body's joint_rom vector (head -> toe, right -> left). */
const JOINT_ORDER: string[] = [
  "abd_z", "abd_y", "abd_x",
  "R hip_x", "R hip_z", "R hip_y", "R knee", "R ankle_y", "R ankle_x",
  "L hip_x", "L hip_z", "L hip_y", "L knee", "L ankle_y", "L ankle_x",
  "R shldr1", "R shldr2", "R elbow",
  "L shldr1", "L shldr2", "L elbow",
];

/** Horizontal meter (0..1) with an optional "milestone" tick (e.g. earned cap). */
function Meter(props: {
  value: number;
  max?: number;
  color: string;
  marker?: number;
}) {
  const max = props.max ?? 1;
  const pct = Math.max(0, Math.min(1, props.value / max)) * 100;
  return (
    <div
      style={{
        position: "relative",
        height: 10,
        borderRadius: 5,
        background: "#1a2030",
        border: "1px solid var(--panel-border)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          height: "100%",
          width: `${pct}%`,
          background: props.color,
          transition: "width 0.4s ease",
        }}
      />
      {props.marker != null && (
        <div
          style={{
            position: "absolute",
            top: -2,
            bottom: -2,
            left: `${Math.max(0, Math.min(1, props.marker / max)) * 100}%`,
            width: 2,
            background: "#cdd6ec",
          }}
          title={`milestone ${props.marker.toFixed(2)}`}
        />
      )}
    </div>
  );
}

export default function LocomotionPanel(props: { metrics: Metrics | null }) {
  const m = props.metrics;
  const romMean = m?.rom_mean ?? 0;
  const braceEngaged = m?.brace_engaged ?? 0;
  const jointRom = m?.joint_rom ?? [];
  const loadL = m?.foot_load_l ?? 0;
  const loadR = m?.foot_load_r ?? 0;
  const handL = m?.hand_load_l ?? 0;
  const handR = m?.hand_load_r ?? 0;
  const partLoads = m?.part_loads ?? {};
  const tactilePE = m?.tactile_pred_error;
  const fmt = (x: number | undefined, d = 2) => (x != null ? x.toFixed(d) : "—");

  return (
    <div className="panel span-7">
      <h2>
        Locomotion / Joint Braces
        <Info tip="The joint-brace guidance system. Every hinge is braced toward an upright standing pose by a stiff internal joint spring (no external force ever touches the body, so the feet keep full weight and there is no glide). Each joint starts welded and earns range of motion only as the brain's per-joint forward-model error falls — so freedom of movement is earned by the brain learning to predict its own body." />
      </h2>

      <div className="strip-label">
        <span>
          Range of motion earned
          <Info tip="Mean per-joint ROM across all hinges (0% = fully welded into the stand pose, 100% = native/free). It ratchets open monotonically as each joint's prediction error stays low." />
        </span>
        <span>{(romMean * 100).toFixed(0)}%</span>
      </div>
      <Meter value={romMean} color="linear-gradient(90deg,#3b82e0,#5fd08a)" />

      <div className="strip-label" style={{ marginTop: 6 }}>
        <span style={{ fontSize: 11 }}>
          Brace engaged (mean tightness) — 100% welded → 0% free
        </span>
        <span style={{ fontSize: 11 }}>{(braceEngaged * 100).toFixed(0)}%</span>
      </div>

      <div className="strip-label" style={{ marginTop: 12 }}>
        <span>
          Per-joint range of motion
          <Info tip="ROM earned per hinge (0 welded → 1 free). A joint widens only once its own proprioceptive forward-model error has stayed low long enough — so well-predicted joints free up first." />
        </span>
        <span style={{ fontSize: 11, color: "#7c8499" }}>0 welded → 1 free</span>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: "5px 14px",
          marginTop: 8,
        }}
      >
        {JOINT_ORDER.map((label, i) => {
          const v = jointRom[i] ?? 0;
          return (
            <div key={label}>
              <div className="strip-label" style={{ marginBottom: 2, fontSize: 10 }}>
                <span>{label}</span>
                <span>{(v * 100).toFixed(0)}%</span>
              </div>
              <Meter value={v} color="linear-gradient(90deg,#e0a23b,#5fd08a)" />
            </div>
          );
        })}
      </div>

      <div style={{ display: "flex", gap: 16, marginTop: 14 }}>
        <div style={{ flex: 1 }}>
          <div className="strip-label">
            <span>
              Left foot load
              <Info tip="Downward force on the left sole as a fraction of body weight. ~0.5 when standing evenly on both feet; near 0 means that foot is unloaded (off the ground). The braces never unload the feet, so a standing body always reads ~1.0 total." />
            </span>
            <span>{fmt(loadL)}</span>
          </div>
          <Meter value={loadL} max={1.2} color="#5fd08a" />
        </div>
        <div style={{ flex: 1 }}>
          <div className="strip-label">
            <span>
              Right foot load
              <Info tip="Downward force on the right sole as a fraction of body weight. ~0.5 when standing evenly on both feet; near 0 means that foot is unloaded (off the ground). The braces never unload the feet, so a standing body always reads ~1.0 total." />
            </span>
            <span>{fmt(loadR)}</span>
          </div>
          <Meter value={loadR} max={1.2} color="#5fd08a" />
        </div>
      </div>

      <div style={{ display: "flex", gap: 16, marginTop: 14 }}>
        <div style={{ flex: 1 }}>
          <div className="strip-label">
            <span>
              Left hand load
              <Info tip="Downward force on the left hand as a fraction of body weight. Positive when the palm presses the ground or a prop — e.g. bracing or pushing the torso up during a stand-up." />
            </span>
            <span>{fmt(handL)}</span>
          </div>
          <Meter value={handL} max={1.2} color="#3b82e0" />
        </div>
        <div style={{ flex: 1 }}>
          <div className="strip-label">
            <span>
              Right hand load
              <Info tip="Downward force on the right hand as a fraction of body weight. Positive when the palm presses the ground or a prop — e.g. bracing or pushing the torso up during a stand-up." />
            </span>
            <span>{fmt(handR)}</span>
          </div>
          <Meter value={handR} max={1.2} color="#3b82e0" />
        </div>
      </div>

      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginTop: 14 }}>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">
            Total limb load
            <Info tip="All four limbs (feet + hands) load as a fraction of body weight. ~1.0 means the body is fully bearing its own weight through its limbs, whether via the legs, the arms, or both. Since no external force ever supports the body, this is always real ground reaction." />
          </span>
          <span className="v">{fmt(loadL + loadR + handL + handR)}</span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">
            Tactile pred. error
            <Info tip="Mean-squared error of the tactile world model: how surprised the brain is by the per-part contact loads it actually felt vs. what it predicted from its last action. Falls as the brain learns which actions load which body part — the per-limb credit-assignment signal for learning to push off." />
          </span>
          <span className="v">{fmt(tactilePE, 4)}</span>
        </div>
      </div>

      <div className="strip-label" style={{ marginTop: 16 }}>
        <span>
          Full-body contact map
          <Info tip="Soft per-part contact load (force / body weight) for every touch sensor on the body — head to toe. Live in ALL support modes: this is the body's sense of touch, fed into perception so the brain can learn which body part is bearing weight or pushing off. Bars near 0 are not touching anything." />
        </span>
        <span style={{ fontSize: 11, color: "#7c8499" }}>force / body weight</span>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "6px 16px",
          marginTop: 8,
        }}
      >
        {PART_ORDER.map(({ key, label }) => {
          const v = partLoads[key] ?? 0;
          return (
            <div key={key}>
              <div
                className="strip-label"
                style={{ marginBottom: 2, fontSize: 11 }}
              >
                <span>{label}</span>
                <span>{fmt(v)}</span>
              </div>
              <Meter
                value={v}
                max={1.2}
                color="linear-gradient(90deg,#3b82e0,#5fd08a)"
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
