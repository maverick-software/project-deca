import { useEffect, useState } from "react";
import { setPreset } from "../api";
import Info from "./Info";

// Ordered small -> large; counts are measured baseline weight-connections
// (faculties off) and match decadic/nn/config.py + tests/test_neural_presets.py.
const PRESET_LABELS: Record<string, string> = {
  tiny: "tiny (~0.76M conn)",
  "2_5m": "2.5m (~2.4M conn)",
  "5m": "5m (~4.8M conn)",
  medium: "medium (~8.4M conn)",
  "10m": "10m (~10M conn)",
  full: "full (~25M conn)",
  xl: "xl (~51M conn)",
  xxl: "xxl (~75M conn)",
  ultra: "ultra (~100M conn)",
  "250m": "250m (~249M conn)",
  "500m": "500m (~494M conn)",
  "1b": "1b (~976M conn)",
};

// Define-only tiers: they build and run, but training them every cycle in fp32
// Adam exhausts a single consumer GPU. Mirror of config.HEAVY_PRESETS.
const HEAVY_PRESETS = new Set(["250m", "500m", "1b"]);

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
    const heavyWarning = HEAVY_PRESETS.has(next)
      ? `\n\nWARNING: "${next}" is a HEAVY tier (~250M-1B connections). It trains every ` +
        "cognitive cycle in fp32, so it needs a large-VRAM GPU and will slow cycles a lot " +
        "(watch Cycle wall ms) or run out of memory on consumer cards. Define-only: there " +
        "is no mixed-precision / sharded training path yet."
      : "";
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
        {Object.entries(PRESET_LABELS).map(([id, label]) => (
          <option key={id} value={id}>
            {label}
          </option>
        ))}
      </select>
      <Info tip="Size of this agent's trainable cognitive stack (baseline weight-connections): tiny ~0.76M, 2.5m ~2.4M, 5m ~4.8M, medium ~8.4M, 10m ~10M, full ~25M, xl ~51M, xxl ~75M, ultra ~100M, 250m ~249M, 500m ~494M, 1b ~976M. Switching rebuilds the brain with fresh weights (architectures are incompatible), like a reset. Larger brains cost more per cycle and more GPU memory, so watch 'Cycle wall ms' after switching (a slow cycle can stall body commands). The 250m/500m/1b tiers are define-only: they run, but training them every cycle in fp32 will exhaust a single consumer GPU." />
    </span>
  );
}
