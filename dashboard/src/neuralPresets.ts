// Ordered small -> large. Labels show TOTAL trainable parameters (the size the
// running model actually reports as `trainable_params`, i.e. cognitive stack +
// the shared frozen HF sensory encoders), and the scalable cognitive-stack size
// in parens. Only the stack scales per preset and drives per-cycle compute; the
// ~177M sensory encoder is fixed overhead shared by every tier.
//   total = stack(all params) + encoder(~177M, mode=hf)   -- see decadic/nn/bundle.py
// full/xl/ultra totals are MEASURED from live bundle logs (205.1M / 230.2M /
// 287.3M); the rest are computed (stack + ~177M encoder). Stack "conn" counts
// (dim>=2 matrices) match decadic/nn/config.py + tests/test_neural_presets.py.
export const NEURAL_PRESET_LABELS: Record<string, string> = {
  tiny: "tiny (~178M params)",
  "2_5m": "2.5m (~180M params)",
  "5m": "5m (~182M params)",
  medium: "medium (~186M params)",
  "10m": "10m (~188M params)",
  full: "full (~205M params)",
  xl: "xl (~230M params)",
  xxl: "xxl (~260M params)",
  ultra: "ultra (~287M params)",
  "250m": "250m (~451M params)",
  "500m": "500m (~721M params)",
  "1b": "1b (~1.25B params)",
};

// Define-only tiers: they build and run, but training them every cycle in fp32
// Adam exhausts a single consumer GPU. Mirror of config.HEAVY_PRESETS.
export const HEAVY_NEURAL_PRESETS = new Set(["250m", "500m", "1b"]);

export const NEURAL_PRESET_INFO =
  "Total trainable parameters for this agent = the scalable cognitive stack + the shared frozen HF sensory encoders (~177M, identical across every tier). The number in parens is just the cognitive stack, which is what actually scales per preset and drives per-cycle compute and training memory (the encoder runs once per frame regardless). Totals: tiny ~178M, 2.5m ~180M, 5m ~182M, medium ~186M, 10m ~188M, full ~205M, xl ~230M, xxl ~260M, ultra ~287M, 250m ~451M, 500m ~721M, 1b ~1.25B. full/xl/ultra are measured from live builds; the rest add the ~177M encoder to the stack. Larger stacks cost more per cycle and more GPU memory, so watch 'Cycle wall ms' and VRAM after creating/switching. The 250m/500m/1b tiers are define-only: they run, but training them every cycle in fp32 will exhaust a single consumer GPU.";

export function heavyPresetWarning(preset: string): string {
  if (!HEAVY_NEURAL_PRESETS.has(preset)) return "";
  return (
    `\n\nWARNING: "${preset}" is a HEAVY tier (~450M-1.25B total params). It trains every ` +
    "cognitive cycle in fp32, so it needs a large-VRAM GPU and will slow cycles a lot " +
    "(watch Cycle wall ms) or run out of memory on consumer cards. Define-only: there " +
    "is no mixed-precision / sharded training path yet."
  );
}
