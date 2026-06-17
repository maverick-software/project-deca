import { useEffect, useState } from "react";
import { usePolling } from "../usePolling";
import {
  destroyDeployment,
  fetchDeployment,
  fetchLocalCheckpoints,
  fetchVastSettings,
  startDeploy,
  stopDeployment,
  type LocalCheckpoint,
  type VastOffer,
  type VastSettings,
} from "../vastApi";
import ActiveDeployment from "./vast/ActiveDeployment";
import DeployProgress from "./vast/DeployProgress";
import GpuSearch from "./vast/GpuSearch";
import RunConfig, { type RunConfigValue } from "./vast/RunConfig";
import VastCredentials from "./vast/VastCredentials";

/** Deploy / GPU tab: store key, search a GPU, rent, watch, stop/destroy. */
export default function DeploymentPanel(props: { onWatchAgent: () => void }) {
  const [settings, setSettings] = useState<VastSettings | null>(null);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [checkpoints, setCheckpoints] = useState<LocalCheckpoint[]>([]);
  const [run, setRun] = useState<RunConfigValue | null>(null);
  const [renting, setRenting] = useState(false);
  const [actionBusy, setActionBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const { data: deployment } = usePolling(fetchDeployment, 1500);

  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const s = await fetchVastSettings();
        if (!alive) return;
        setSettings(s);
        setRun((prev) =>
          prev ?? {
            preset: s.defaults.preset,
            encoder: s.defaults.encoder,
            whisper_model: s.defaults.whisper_model,
            scene: s.defaults.scene,
            disk: s.defaults.disk,
            restore_agent: null,
          },
        );
      } catch (e) {
        if (alive) setSettingsError(e instanceof Error ? e.message : String(e));
      }
      try {
        const c = await fetchLocalCheckpoints();
        if (alive) setCheckpoints(c.checkpoints);
      } catch {
        // checkpoints are optional; ignore
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const onRent = async (offer: VastOffer) => {
    if (!run) return;
    setRenting(true);
    setActionError(null);
    try {
      await startDeploy({
        offer_id: offer.id,
        preset: run.preset,
        encoder: run.encoder,
        whisper_model: run.whisper_model,
        scene: run.scene,
        disk: run.disk,
        restore_agent: run.restore_agent,
      });
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setRenting(false);
    }
  };

  const onStop = async () => {
    setActionBusy(true);
    setActionError(null);
    try {
      await stopDeployment();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setActionBusy(false);
    }
  };

  const onDestroy = async () => {
    if (!window.confirm("Destroy the rented instance? This stops all billing.")) return;
    setActionBusy(true);
    setActionError(null);
    try {
      await destroyDeployment();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setActionBusy(false);
    }
  };

  if (settingsError) {
    return (
      <div className="grid solo">
        <div className="error-banner">Could not load Vast settings: {settingsError}</div>
      </div>
    );
  }
  if (!settings || !run) {
    return <div className="grid solo"><div className="panel">Loading...</div></div>;
  }

  const phase = deployment?.phase ?? "idle";
  const instanceId = deployment?.instance_id ?? null;
  const credsReady = settings.has_api_key && settings.cli_available;
  const showProgress = !!deployment && phase !== "idle";
  const showActive = instanceId != null || !!deployment?.active;
  const showSearch =
    credsReady && (phase === "idle" || (phase === "error" && instanceId == null));

  return (
    <div className="grid solo" style={{ display: "grid", gap: 12 }}>
      <VastCredentials settings={settings} onSaved={setSettings} />

      {actionError && <div className="error-banner">{actionError}</div>}

      {showProgress && deployment && <DeployProgress deployment={deployment} />}

      {showActive && deployment && (
        <ActiveDeployment
          deployment={deployment}
          busy={actionBusy}
          onWatch={props.onWatchAgent}
          onStop={() => void onStop()}
          onDestroy={() => void onDestroy()}
        />
      )}

      {showSearch && (
        <>
          <RunConfig
            value={run}
            checkpoints={checkpoints}
            onChange={(patch) => setRun((r) => (r ? { ...r, ...patch } : r))}
          />
          <GpuSearch
            defaults={settings.defaults}
            disabled={!credsReady || renting}
            renting={renting}
            onRent={(o) => void onRent(o)}
          />
        </>
      )}

      {!credsReady && (
        <div className="panel" style={{ opacity: 0.8 }}>
          Add your API key above (and make sure the <code>vastai</code> CLI is installed
          on the server) to search for GPUs and rent one.
        </div>
      )}
    </div>
  );
}
