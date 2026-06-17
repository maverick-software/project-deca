import type { LocalCheckpoint } from "../../vastApi";

export type RunConfigValue = {
  preset: string;
  encoder: string;
  whisper_model: string;
  scene: string;
  disk: number;
  restore_agent: string | null;
};

const PRESETS = ["tiny", "medium", "full", "xl"];
const SCENES: Record<string, string> = {
  none: "Mind only (no body)",
  bear: "House + bear (threat avoidance)",
  food: "House + food + water (foraging)",
};
const WHISPERS = ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"];

/** Controlled deploy-config form (preset / encoder / scene / disk / restore). */
export default function RunConfig(props: {
  value: RunConfigValue;
  checkpoints: LocalCheckpoint[];
  onChange: (patch: Partial<RunConfigValue>) => void;
}) {
  const { value, checkpoints } = props;
  return (
    <div className="panel" style={{ display: "grid", gap: 10 }}>
      <h3 style={{ margin: 0 }}>Run configuration</h3>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
        <label style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>Brain preset</span>
          <select
            value={value.preset}
            onChange={(e) => props.onChange({ preset: e.target.value })}
          >
            {PRESETS.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>

        <label style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>Encoders</span>
          <select
            value={value.encoder}
            onChange={(e) => props.onChange({ encoder: e.target.value })}
          >
            <option value="hf">hf (real CLIP + Whisper)</option>
            <option value="zeros">zeros (synthetic)</option>
          </select>
        </label>

        {value.encoder === "hf" && (
          <label style={{ display: "grid", gap: 2 }}>
            <span style={{ fontSize: 12, opacity: 0.8 }}>Whisper model</span>
            <select
              value={value.whisper_model}
              onChange={(e) => props.onChange({ whisper_model: e.target.value })}
            >
              {WHISPERS.map((w) => (
                <option key={w} value={w}>
                  {w.replace("openai/", "")}
                </option>
              ))}
            </select>
          </label>
        )}

        <label style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>Scene</span>
          <select
            value={value.scene}
            onChange={(e) => props.onChange({ scene: e.target.value })}
          >
            {Object.entries(SCENES).map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
        </label>

        <label style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>Disk (GB)</span>
          <input
            type="number"
            min={20}
            value={value.disk}
            onChange={(e) => props.onChange({ disk: Number(e.target.value) })}
            style={{ width: 80 }}
          />
        </label>

        <label style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>Agent</span>
          <select
            value={value.restore_agent ?? ""}
            onChange={(e) => props.onChange({ restore_agent: e.target.value || null })}
            title="Ship a saved local checkpoint and restore it, or start fresh"
          >
            <option value="">Fresh agent (from birth)</option>
            {checkpoints.map((c) => (
              <option key={c.agent_id} value={c.agent_id}>
                restore {c.agent_id.slice(0, 8)}
                {c.has_brain ? " (+brain)" : ""}
              </option>
            ))}
          </select>
        </label>
      </div>
      {value.restore_agent && (
        <div style={{ fontSize: 11, opacity: 0.65 }}>
          Restoring ships a mind-only agent (no scene); the brain preset follows the
          checkpoint.
        </div>
      )}
    </div>
  );
}
