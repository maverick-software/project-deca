/** Typed REST client for the Vast.ai GPU deployment control plane (/vast/*). */

import { httpBase } from "./api";

export type VastDefaults = {
  gpu_name: string;
  num_gpus: number;
  max_dph: number;
  min_gpu_ram: number;
  verified: boolean;
  disk: number;
  image: string;
  preset: string;
  encoder: string;
  whisper_model: string;
  scene: string;
};

export type VastSettings = {
  has_api_key: boolean;
  api_key_masked: string;
  has_ssh_key_path: boolean;
  ssh_key_path_masked: string;
  defaults: VastDefaults;
  config_path: string;
  cli_available: boolean;
};

export type VastAccount = {
  balance: number | null;
  credit: number | null;
  email: string | null;
  id: number | null;
};

export type VastOffer = {
  id: number;
  gpu_name: string | null;
  num_gpus: number | null;
  // Per-GPU VRAM in GB (Vast reports MB in --raw; the server converts to GB).
  gpu_ram_gb: number | null;
  // Host RAM in GB.
  cpu_ram_gb: number | null;
  dph_total: number | null;
  dlperf: number | null;
  dlperf_per_usd: number | null;
  cuda_max_good: number | null;
  geolocation: string | null;
  reliability: number | null;
  verified: boolean;
};

export type DeploymentPhase =
  | "idle"
  | "creating"
  | "waiting"
  | "uploading"
  | "installing"
  | "serving"
  | "tunneling"
  | "starting_agent"
  | "ready"
  | "error"
  | "stopped"
  | "destroying";

export type Deployment = {
  phase: DeploymentPhase;
  phase_order: string[];
  busy: boolean;
  active: boolean;
  instance_id: number | null;
  ssh_host: string | null;
  ssh_port: number | null;
  dph: number | null;
  elapsed_s: number;
  est_cost_usd: number | null;
  agent_id: string | null;
  scene: string | null;
  preset: string | null;
  error: string | null;
  ready: boolean;
  log: string[];
};

export type LocalCheckpoint = { agent_id: string; has_brain: boolean };

export type FsEntry = { name: string; path: string; is_dir: boolean };
export type FsListing = { path: string; parent: string | null; entries: FsEntry[] };

export type GpuName = {
  // Human name as Vast reports it (may contain spaces, e.g. "RTX 4090").
  name: string;
  // Query token for gpu_name= filters (underscored, e.g. "RTX_4090").
  value: string;
  // Live count of rentable offers for this model.
  count: number;
};

export type VastSettingsUpdate = {
  api_key?: string;
  clear_api_key?: boolean;
  ssh_key_path?: string;
  clear_ssh_key_path?: boolean;
  defaults?: Partial<VastDefaults>;
};

export type DeployBody = {
  offer_id: number;
  preset?: string;
  encoder?: string;
  whisper_model?: string;
  scene?: string;
  disk?: number;
  image?: string;
  restore_agent?: string | null;
};

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(`${httpBase()}${path}`);
  if (!r.ok) throw new Error(await errorDetail(r));
  return (await r.json()) as T;
}

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(`${httpBase()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await errorDetail(r));
  return (await r.json()) as T;
}

async function errorDetail(r: Response): Promise<string> {
  try {
    const body = (await r.json()) as { detail?: string };
    if (body?.detail) return body.detail;
  } catch {
    // non-JSON body; fall through to status text
  }
  return `HTTP ${r.status}`;
}

export function fetchVastSettings(): Promise<VastSettings> {
  return getJson<VastSettings>("/vast/settings");
}

export function saveVastSettings(update: VastSettingsUpdate): Promise<VastSettings> {
  return postJson<VastSettings>("/vast/settings", update);
}

export function fetchVastAccount(): Promise<VastAccount> {
  return getJson<VastAccount>("/vast/account");
}

export function searchOffers(params: {
  gpu_name?: string;
  num_gpus?: number;
  max_dph?: number;
  min_gpu_ram?: number;
  verified?: boolean;
  limit?: number;
}): Promise<{ query: string; offers: VastOffer[] }> {
  const q = new URLSearchParams();
  if (params.gpu_name) q.set("gpu_name", params.gpu_name);
  if (params.num_gpus != null) q.set("num_gpus", String(params.num_gpus));
  if (params.max_dph != null) q.set("max_dph", String(params.max_dph));
  if (params.min_gpu_ram != null) q.set("min_gpu_ram", String(params.min_gpu_ram));
  if (params.verified != null) q.set("verified", params.verified ? "true" : "false");
  if (params.limit != null) q.set("limit", String(params.limit));
  return getJson(`/vast/offers?${q.toString()}`);
}

export function fetchLocalCheckpoints(): Promise<{ checkpoints: LocalCheckpoint[] }> {
  return getJson("/vast/local-checkpoints");
}

export function browseFs(path?: string): Promise<FsListing> {
  const q = path ? `?path=${encodeURIComponent(path)}` : "";
  return getJson(`/vast/browse-fs${q}`);
}

export function fetchGpuNames(): Promise<{ gpu_names: GpuName[] }> {
  return getJson("/vast/gpu-names");
}

export function startDeploy(body: DeployBody): Promise<Deployment> {
  return postJson<Deployment>("/vast/deploy", body);
}

export function fetchDeployment(): Promise<Deployment> {
  return getJson<Deployment>("/vast/deployment");
}

export function stopDeployment(): Promise<Deployment> {
  return postJson<Deployment>("/vast/deployment/stop");
}

export function destroyDeployment(): Promise<Deployment> {
  return postJson<Deployment>("/vast/deployment/destroy");
}
