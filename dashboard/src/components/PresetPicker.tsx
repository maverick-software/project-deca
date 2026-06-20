import { useEffect, useState } from "react";
import { setPreset } from "../api";
import Info from "./Info";
import {
  NEURAL_PRESET_INFO,
  NEURAL_PRESET_LABELS,
  heavyPresetWarning,
} from "../neuralPresets";

/** Network-size selector; switching rebuilds the selected brain from scratch. */
export default function PresetPicker(props: {
  agentId: string;
  preset: string | null | undefined;
}) {
  const { agentId } = props;
  const [busy, setBusy] = useState(false);
  // Optimistic value so the select flips instantly; cleared once polling agrees.
  const [optimistic, setOptimistic] = useState<string | null>(null);
  const current = optimistic ?? props.preset ?? "full";

  useEffect(() => {
    setOptimistic(null);
  }, [props.preset, agentId]);

  const onChange = async (next: string) => {
    if (next === current) return;
    const heavyWarning = heavyPresetWarning(next);
    const ok = window.confirm(
      `Switch this neural net to the "${next}" architecture?\n\n` +
        "This rebuilds the brain from scratch — weights, state, and memory are wiped " +
        "(different architectures cannot share weights)." +
        heavyWarning,
    );
    if (!ok) return;
    setBusy(true);
    setOptimistic(next);
    try {
      await setPreset(agentId, next);
    } catch {
      setOptimistic(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <span className="preset-picker">
      <select
        value={current}
        disabled={busy}
        title="Neural network size for this agent"
        onChange={(e) => void onChange(e.target.value)}
      >
        {Object.entries(NEURAL_PRESET_LABELS).map(([id, label]) => (
          <option key={id} value={id}>
            {label}
          </option>
        ))}
      </select>
      <Info tip={`${NEURAL_PRESET_INFO} Switching rebuilds the brain with fresh weights (architectures are incompatible), like a reset.`} />
    </span>
  );
}
