import { useCallback, useEffect, useState } from "react";
import {
  fetchGpuNames,
  searchOffers,
  type GpuName,
  type VastDefaults,
  type VastOffer,
} from "../../vastApi";

const num = (v: number | null, digits = 2) => (v == null ? "-" : v.toFixed(digits));

// VRAM is already in GB from the server; drop a trailing .0 (24 GB, 22.5 GB).
const fmtGb = (v: number | null) =>
  v == null ? "-" : `${Number.isInteger(v) ? v : v.toFixed(1)} GB`;

const fmtMbps = (v: number | null) => (v == null ? "-" : `${Math.round(v).toLocaleString()} Mbps`);

// Standard "at least" VRAM tiers. Charles's list (10/12/16/24/32/64/96/128/240)
// plus 48/80, common real datacenter-card sizes (A6000/L40, A100/H100) that
// would otherwise fall through the gaps - trim if unwanted.
const VRAM_TIERS = [10, 12, 16, 24, 32, 48, 64, 80, 96, 128, 240];

/** GPU offer search: filter bar + a Vast-style card per offer. Rent hands the offer id to the parent. */
export default function GpuSearch(props: {
  defaults: VastDefaults;
  disabled: boolean;
  renting: boolean;
  onRent: (offer: VastOffer) => void;
}) {
  const { defaults } = props;
  const [gpu, setGpu] = useState(defaults.gpu_name);
  const [minRam, setMinRam] = useState(defaults.min_gpu_ram);
  const [verified, setVerified] = useState(defaults.verified);
  const [offers, setOffers] = useState<VastOffer[]>([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gpuNames, setGpuNames] = useState<GpuName[]>([]);
  const [namesLoading, setNamesLoading] = useState(false);
  const [namesError, setNamesError] = useState<string | null>(null);

  const loadNames = useCallback(async () => {
    setNamesLoading(true);
    setNamesError(null);
    try {
      const res = await fetchGpuNames();
      setGpuNames(res.gpu_names);
    } catch (e) {
      setNamesError(e instanceof Error ? e.message : String(e));
    } finally {
      setNamesLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadNames();
  }, [loadNames]);

  // Use the live dropdown unless the model list failed to load (then fall back
  // to a free-text box so search still works).
  const useDropdown = !namesError;
  const knownValues = gpuNames.map((g) => g.value);
  const showCurrentAsCustom = gpu.length > 0 && !knownValues.includes(gpu);

  const run = async () => {
    setSearching(true);
    setError(null);
    try {
      const res = await searchOffers({
        gpu_name: gpu.trim() || undefined,
        min_gpu_ram: minRam || undefined,
        verified,
      });
      setOffers(res.offers);
      if (res.offers.length === 0) setError("No matching offers; widen the filters.");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="panel" style={{ display: "grid", gap: 10 }}>
      <h3 style={{ margin: 0 }}>Find a GPU</h3>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "end" }}>
        <label style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>
            GPU model{namesLoading ? " (loading...)" : ""}
          </span>
          {useDropdown ? (
            <span style={{ display: "flex", gap: 4, alignItems: "center" }}>
              <select
                value={gpu}
                onChange={(e) => setGpu(e.target.value)}
                disabled={namesLoading && gpuNames.length === 0}
                style={{ minWidth: 170 }}
                title="Rentable GPU models (live availability count)"
              >
                <option value="">Any GPU</option>
                {showCurrentAsCustom && (
                  <option value={gpu}>{gpu.replace(/_/g, " ")}</option>
                )}
                {gpuNames.map((g) => (
                  <option key={g.value} value={g.value}>
                    {g.name} ({g.count})
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={() => void loadNames()}
                disabled={namesLoading}
                title="Refresh GPU list"
                style={{ padding: "2px 6px" }}
              >
                ↻
              </button>
            </span>
          ) : (
            <input
              value={gpu}
              onChange={(e) => setGpu(e.target.value)}
              placeholder="RTX_4090"
              style={{ width: 120 }}
              title={`Model list unavailable (${namesError}); type a model, e.g. RTX_4090`}
            />
          )}
        </label>

        <label style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>VRAM</span>
          <select
            value={minRam || ""}
            onChange={(e) => setMinRam(e.target.value ? Number(e.target.value) : 0)}
            title="Minimum per-GPU VRAM. Real cards land on odd sizes (24, 48, 80, 141 GB) so this is an 'at least' filter, not exact."
          >
            <option value="">Any</option>
            {VRAM_TIERS.map((gb) => (
              <option key={gb} value={gb}>
                At least {gb} GB
              </option>
            ))}
          </select>
        </label>

        <label style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 13 }}>
          <input
            type="checkbox"
            checked={verified}
            onChange={(e) => setVerified(e.target.checked)}
          />
          Verified only
        </label>
        <button onClick={() => void run()} disabled={searching || props.disabled}>
          {searching ? "Searching..." : "Search"}
        </button>
      </div>

      {error && <div className="error-banner" style={{ margin: 0 }}>{error}</div>}

      {offers.length > 0 && (
        <div style={{ display: "grid", gap: 8 }}>
          {offers.map((o) => (
            <OfferCard key={o.id} offer={o} renting={props.renting || props.disabled} onRent={props.onRent} />
          ))}
        </div>
      )}
    </div>
  );
}

function OfferCard(props: { offer: VastOffer; renting: boolean; onRent: (o: VastOffer) => void }) {
  const o = props.offer;
  const count = o.num_gpus ?? 1;
  return (
    <div
      style={{
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 6,
        padding: "10px 12px",
        display: "grid",
        gap: 6,
        fontSize: 13,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
        <strong style={{ fontSize: 14 }}>
          {count}x {o.gpu_name ?? "?"}
        </strong>
        {o.verified && (
          <span style={{ fontSize: 11, color: "#3fb950" }}>verified</span>
        )}
        {o.is_datacenter != null && (
          <span style={{ fontSize: 11, opacity: 0.7 }}>
            {o.is_datacenter ? "datacenter" : "community"}
          </span>
        )}
        {o.days_remaining != null && (
          <span style={{ fontSize: 11, opacity: 0.7 }}>max ~{Math.round(o.days_remaining)}d</span>
        )}
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontWeight: 600 }}>${num(o.dph_total, 3)}/hr</span>
          <button onClick={() => props.onRent(o)} disabled={props.renting} title={`Rent offer ${o.id}`}>
            Rent
          </button>
        </span>
      </div>

      <div style={{ opacity: 0.85 }}>
        {num(o.total_flops, 1)} TFLOPS &middot; {fmtGb(o.gpu_ram_gb)} VRAM
        {o.gpu_mem_bw_gbps != null && <> &middot; {num(o.gpu_mem_bw_gbps, 0)} GB/s mem</>}
      </div>

      <div style={{ opacity: 0.7, fontSize: 12 }}>
        {o.cpu_name ?? "CPU ?"}
        {o.cpu_cores != null && ` (${o.cpu_cores} cores)`}
        {" · "}
        {"↑"}
        {fmtMbps(o.inet_up_mbps)} {"↓"}
        {fmtMbps(o.inet_down_mbps)}
        {o.direct_port_count != null && <> &middot; {o.direct_port_count} ports</>}
      </div>

      <div style={{ opacity: 0.7, fontSize: 12 }}>
        DLPerf {num(o.dlperf, 1)} &middot; DLPerf/$ {num(o.dlperf_per_usd, 1)} &middot; Reliability{" "}
        {o.reliability != null ? `${(o.reliability * 100).toFixed(1)}%` : "-"}
        {o.geolocation && <> &middot; {o.geolocation}</>}
      </div>
    </div>
  );
}
