import { useEffect, useState } from "react";
import { listAgentPresets, type AgentPreset } from "../../api";
import { DB_LOSS_INFO, DISK_INFO, ENCODER_INFO, SCENE_INFO, WHISPER_MODEL_INFO } from "../../explainers";
import { HEAVY_NEURAL_PRESETS, NEURAL_PRESET_INFO, NEURAL_PRESET_LABELS, heavyPresetWarning } from "../../neuralPresets";
import type { LocalCheckpoint } from "../../vastApi";
import Info from "../Info";

export type RunConfigValue = {
  preset: string;
  encoder: string;
  whisper_model: string;
  scene: string;
  disk: number;
  restore_agent: string | null;
};

const WHISPERS = ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"];

/**
 * Controlled deploy-config form (preset / encoder / scene / disk / restore).
 *
 * Scene and brain-preset options are fetched/imported from the same sources
 * the rest of the dashboard uses (/agent-presets, decadic/nn/config.py via
 * neuralPresets.ts) instead of a separate hardcoded list, so this panel can't
 * drift out of sync with the "+ New agent" dropdown again.
 */
export default function RunConfig(props: {
  value: RunConfigValue;
  checkpoints: LocalCheckpoint[];
  onChange: (patch: Partial<RunConfigValue>) => void;
}) {
  const { value, checkpoints } = props;
  const [presets, setPresets] = useState<AgentPreset[] | null>(null);
  const [presetsError, setPresetsError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const p = await listAgentPresets();
        if (alive) setPresets(p);
      } catch (e) {
        if (alive) setPresetsError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const heavyWarning = heavyPresetWarning(value.preset);

  return (
    <div className="panel" style={{ display: "grid", gap: 10 }}>
      <h3 style={{ margin: 0 }}>Run configuration</h3>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
        <label style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>
            Brain preset
            <Info tip={`Size of the rented agent's brain. ${NEURAL_PRESET_INFO}`} />
          </span>
          <select
            value={value.preset}
            onChange={(e) => props.onChange({ preset: e.target.value })}
          >
            {Object.entries(NEURAL_PRESET_LABELS).map(([p, label]) => (
              <option key={p} value={p}>
                {label}
              </option>
            ))}
          </select>
        </label>

        <label style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>
            Encoders
            <Info tip={ENCODER_INFO} />
          </span>
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
            <span style={{ fontSize: 12, opacity: 0.8 }}>
              Whisper model
              <Info tip={WHISPER_MODEL_INFO} />
            </span>
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
          <span style={{ fontSize: 12, opacity: 0.8 }}>
            Scene
            <Info tip={SCENE_INFO} />
          </span>
          <select
            value={value.scene}
            onChange={(e) => props.onChange({ scene: e.target.value })}
            disabled={!presets}
          >
            {!presets && <option value={value.scene}>Loading scenes...</option>}
            {presets?.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>

        <label style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>
            Disk (GB)
            <Info tip={DISK_INFO} />
          </span>
          <input
            type="number"
            min={20}
            value={value.disk}
            onChange={(e) => props.onChange({ disk: Number(e.target.value) })}
            style={{ width: 80 }}
          />
        </label>

        <label style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>
            Agent
            <Info
              tip={`Ship a saved local checkpoint and restore it, or start fresh. ${DB_LOSS_INFO}`}
            />
          </span>
          <select
            value={value.restore_agent ?? ""}
            onChange={(e) => props.onChange({ restore_agent: e.target.value || null })}
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

      {presetsError && (
        <div style={{ fontSize: 11, color: "#f85149" }}>
          Could not load scene presets ({presetsError}) - scene selection unavailable until
          this loads.
        </div>
      )}

      {HEAVY_NEURAL_PRESETS.has(value.preset) && (
        <div style={{ fontSize: 11, color: "#f0883e", whiteSpace: "pre-wrap" }}>
          {heavyWarning.trim()}
        </div>
      )}

      {value.restore_agent && (
        <div style={{ fontSize: 11, opacity: 0.65 }}>
          Restoring ships a mind-only agent (no scene); the brain preset follows the
          checkpoint.
        </div>
      )}
    </div>
  );
}
