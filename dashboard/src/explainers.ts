/**
 * Shared explanatory copy for config choices that show up in more than one
 * panel (Deploy/GPU tab, CognitionTogglesPanel). One copy of each string so
 * the explanation can't drift between the two places a setting is offered.
 */

export const ENCODER_INFO =
  "hf: real frozen CLIP + Whisper (downloads ~1 GB on first run; gives the patch tokens " +
  "discovered perception needs). zeros: cheap synthetic fallback (fast, but discovered " +
  "perception is inert).";

export const WHISPER_MODEL_INFO =
  "Which Whisper checkpoint transcribes audio for the hf encoder. Only matters when " +
  "Encoder = hf AND the scene has audio sensing on (none of the 6 built-in scenes do " +
  "today, so this is currently inert either way). tiny/base/small trade download size, " +
  "VRAM, and transcription latency for accuracy - tiny is fastest and least accurate, " +
  "small is the most accurate of the three offered here.";

export const SCENE_INFO =
  "Which world elements this agent's body spawns with (house, food, water, a predator, " +
  "other NPCs, etc.) - the same preset list as the '+ New agent' dropdown on the main " +
  "dashboard, so a scene here means the same thing it does there. 'Mind only' skips the " +
  "body entirely.";

export const DISK_INFO =
  "GB of persistent disk on the rented instance (billed alongside GPU time, at a lower " +
  "rate; keeps billing while the instance is Stopped, only stops when Destroyed). Rough " +
  "budget: ~6-10 GB base image + deps, +1-2 GB for the hf encoder weights (CLIP + " +
  "Whisper, downloaded once), + the brain checkpoint - measured 0.7-1.9 GB locally for " +
  "full/xl-tier agents, scales up with preset size (250m/500m/1b are define-only tiers " +
  "and would need considerably more - untested, this is an estimate not a measurement), " +
  "+ episodic memory growth over the run (measured 4 KB to ~80 MB locally depending on " +
  "session length - budget more for long soaks), + logs. The 40 GB default has headroom " +
  "through xl; go bigger for ultra and above or very long runs.";

export const DB_LOSS_INFO =
  "Episodic memory (the SQLite database of everything this agent has recalled) lives " +
  "only on the rented instance's disk. Destroying the instance deletes it permanently - " +
  "only the neural weights (checkpoint + brain.pt) are copied back locally. There is no " +
  "sync of episodic memory today.";
