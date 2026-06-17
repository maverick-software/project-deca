import { useCallback, useEffect, useState } from "react";
import {
  fetchGpuNames,
  searchOffers,
  type GpuName,
  type VastDefaults,
  type VastOffer,
} from "../../vastApi";

const num = (v: number | null, digits = 2) =>
  v == null ? "-" : v.toFixed(digits);

// VRAM is already in GB from the server; drop a trailing .0 (24 GB, 22.5 GB).
const fmtGb = (v: number | null) =>
  v == null ? "-" : `${Number.isInteger(v) ? v : v.toFixed(1)} GB`;

/** GPU offer search with a results table; Rent hands the offer id to the parent. */
export default function GpuSearch(props: {
  defaults: VastDefaults;
  disabled: boolean;
  renting: boolean;
  onRent: (offer: VastOffer) => void;
}) {
  const { defaults } = props;
  const [gpu, setGpu] = useState(defaults.gpu_name);
  const [numGpus, setNumGpus] = useState(defaults.num_gpus);
  const [maxDph, setMaxDph] = useState(defaults.max_dph);
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
        num_gpus: numGpus,
        max_dph: maxDph || undefined,
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
          <span style={{ fontSize: 12, opacity: 0.8 }}># GPUs</span>
          <input
            type="number"
            min={1}
            value={numGpus}
            onChange={(e) => setNumGpus(Number(e.target.value))}
            style={{ width: 70 }}
          />
        </label>
        <label style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>Max $/hr</span>
          <input
            type="number"
            step={0.05}
            min={0}
            value={maxDph}
            onChange={(e) => setMaxDph(Number(e.target.value))}
            style={{ width: 80 }}
          />
        </label>
        <label style={{ display: "grid", gap: 2 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>Min GPU RAM (GB)</span>
          <input
            type="number"
            min={0}
            value={minRam}
            onChange={(e) => setMinRam(Number(e.target.value))}
            style={{ width: 80 }}
          />
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
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ textAlign: "left", opacity: 0.7 }}>
                <th>GPU</th>
                <th>#</th>
                <th>VRAM</th>
                <th>$/hr</th>
                <th>dlperf</th>
                <th>dlperf/$</th>
                <th>Location</th>
                <th>Rel.</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {offers.map((o) => (
                <tr key={o.id} style={{ borderTop: "1px solid rgba(255,255,255,0.08)" }}>
                  <td>{o.gpu_name ?? "?"}</td>
                  <td>{o.num_gpus ?? "?"}</td>
                  <td>{fmtGb(o.gpu_ram_gb)}</td>
                  <td>${num(o.dph_total, 3)}</td>
                  <td>{num(o.dlperf, 1)}</td>
                  <td>{num(o.dlperf_per_usd, 1)}</td>
                  <td>{o.geolocation ?? "?"}</td>
                  <td>{num(o.reliability, 2)}</td>
                  <td>
                    <button
                      onClick={() => props.onRent(o)}
                      disabled={props.renting || props.disabled}
                      title={`Rent offer ${o.id}`}
                    >
                      Rent
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
