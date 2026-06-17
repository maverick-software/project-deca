import { useEffect, useState } from "react";
import type { AgentDefaults, CapacityConfig, Metrics } from "../api";
import { configureAgent, fetchAgentDefaults, setAgentDefaults } from "../api";
import CognitionTogglesPanel from "./CognitionTogglesPanel";
import Info from "./Info";

function fmtBytes(n: number): string {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(1)} ${u[i]}`;
}

export default function CapacityPanel(props: {
  agentId: string | null;
  metrics: Metrics | null;
  capacity?: CapacityConfig;
}) {
  const { agentId, metrics } = props;
  const k = props.capacity?.parallel_sessions ?? metrics?.parallel_sessions ?? 1;
  const s = props.capacity?.working_memory_slots ?? metrics?.working_memory_slots ?? 12;
  const decay = props.capacity?.working_memory_decay ?? metrics?.working_memory_decay ?? 0.9;

  const plast = props.capacity?.plasticity;
  const [local, setLocal] = useState<CapacityConfig>({
    parallel_sessions: k,
    working_memory_slots: s,
    working_memory_decay: decay,
  });
  const [alpha, setAlpha] = useState<number>(plast?.plasticity_alpha ?? 0.1);
  const [density, setDensity] = useState<number>(plast?.sparse_density ?? 0.5);
  const [maxN, setMaxN] = useState<number>(plast?.max_neurons ?? 0);
  const [busy, setBusy] = useState(false);

  // New-agent neuroplasticity defaults (stored server-side on the registry).
  const [defaults, setDefaults] = useState<AgentDefaults | null>(null);
  const [defBusy, setDefBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    fetchAgentDefaults()
      .then((d) => {
        if (alive) setDefaults(d);
      })
      .catch(() => {
        /* server unreachable */
      });
    return () => {
      alive = false;
    };
  }, []);

  const commitDefaults = async (partial: Partial<AgentDefaults>) => {
    setDefBusy(true);
    setDefaults((d) => (d ? { ...d, ...partial } : d)); // optimistic
    try {
      const applied = await setAgentDefaults(partial);
      setDefaults(applied);
    } catch {
      // server unreachable; keep optimistic value
    } finally {
      setDefBusy(false);
    }
  };

  // Re-sync from server when the agent changes (not on every poll, to avoid fighting drags).
  useEffect(() => {
    setLocal({
      parallel_sessions: k,
      working_memory_slots: s,
      working_memory_decay: decay,
    });
    if (plast) {
      setAlpha(plast.plasticity_alpha ?? 0.1);
      setDensity(plast.sparse_density ?? 0.5);
      setMaxN(plast.max_neurons ?? 0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentId]);

  const commit = async (next: CapacityConfig) => {
    if (!agentId) return;
    setBusy(true);
    try {
      const applied = await configureAgent(agentId, next);
      setLocal(applied);
    } catch {
      // server unreachable; keep optimistic local value
    } finally {
      setBusy(false);
    }
  };

  const commitPlasticity = async (knobs: {
    plasticity_alpha?: number;
    sparse_density?: number;
    max_neurons?: number;
  }) => {
    if (!agentId) return;
    setBusy(true);
    try {
      await configureAgent(agentId, knobs);
    } catch {
      // server unreachable; keep optimistic local value
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel span-5">
      <h2>
        Agent Settings
        <Info tip="Defaults applied to newly created agents, plus live throughput and memory knobs for the currently selected agent." />
      </h2>

      <div className="defaults-section">
        <h3>
          New agent defaults
          <Info tip="Neuroplasticity is opt-in. These switches decide which mechanisms a brand-new agent is built with. They do NOT change agents that already exist — create a new agent (or delete + recreate) to apply them." />
          {defBusy && <span className="strip-label">saving…</span>}
        </h3>

        <label className="cap-row toggle-row">
          <span>
            A · Hebbian plasticity
            <Info tip="A fast, neuromodulated overlay on the weights: connections between co-active neurons strengthen, gated by pain/pleasure. Adds short-term adaptability on top of gradient learning." />
          </span>
          <input
            type="checkbox"
            checked={defaults?.plasticity_enabled ?? false}
            disabled={!defaults}
            onChange={(e) => void commitDefaults({ plasticity_enabled: e.target.checked })}
          />
        </label>

        <label className="cap-row toggle-row">
          <span>
            B · Dynamic sparse training
            <Info tip="Keeps only a fraction of possible connections active, periodically pruning the weakest and growing new ones where gradients are largest — structural rewiring at a fixed connection budget." />
          </span>
          <input
            type="checkbox"
            checked={defaults?.sparse_enabled ?? false}
            disabled={!defaults}
            onChange={(e) => void commitDefaults({ sparse_enabled: e.target.checked })}
          />
        </label>

        {defaults?.sparse_enabled && (
          <label className="cap-row">
            <span>
              · connection density
              <Info tip="Fraction of possible connections kept active in the plastic blocks. Lower = sparser and cheaper; 1.0 = fully dense." />
            </span>
            <input
              type="range"
              min={0.05}
              max={1}
              step={0.05}
              value={defaults?.sparse_density ?? 0.5}
              onChange={(e) =>
                setDefaults((d) =>
                  d ? { ...d, sparse_density: Number(e.target.value) } : d,
                )
              }
              onPointerUp={() =>
                void commitDefaults({ sparse_density: defaults?.sparse_density })
              }
              onKeyUp={() =>
                void commitDefaults({ sparse_density: defaults?.sparse_density })
              }
            />
            <b>{(defaults?.sparse_density ?? 0.5).toFixed(2)}</b>
          </label>
        )}

        <label className="cap-row toggle-row">
          <span>
            C · Neuron growth
            <Info tip="Lets the brain wake dormant neurons (add capacity) while prediction error stays high, up to the cap below. New neurons wake function-preserving, then learn." />
          </span>
          <input
            type="checkbox"
            checked={defaults?.growth_enabled ?? false}
            disabled={!defaults}
            onChange={(e) => void commitDefaults({ growth_enabled: e.target.checked })}
          />
        </label>

        {defaults?.growth_enabled && (
          <label className="cap-row">
            <span>
              · grow up to N
              <Info tip="Per-block cap on awake hidden neurons for the new agent. The growth controller wakes dormant neurons toward this cap." />
            </span>
            <input
              type="range"
              min={8}
              max={512}
              step={8}
              value={defaults?.max_neurons ?? 256}
              onChange={(e) =>
                setDefaults((d) => (d ? { ...d, max_neurons: Number(e.target.value) } : d))
              }
              onPointerUp={() => void commitDefaults({ max_neurons: defaults?.max_neurons })}
              onKeyUp={() => void commitDefaults({ max_neurons: defaults?.max_neurons })}
            />
            <b>{defaults?.max_neurons ?? 256}</b>
          </label>
        )}

        <div className="strip-label">Applies to the next agent you create.</div>
      </div>

      <CognitionTogglesPanel
        agentId={agentId}
        capacity={props.capacity}
        metrics={metrics}
        defaults={defaults}
        onCommitDefaults={(p) => void commitDefaults(p)}
      />

      {!agentId && (
        <div className="cap-empty">Select an agent to view its live capacity controls.</div>
      )}

      {agentId && (
      <>
      <h3 className="cap-subhead">
        Workspace capacity
        <Info tip="Throughput and memory knobs for the global workspace. K = parallel sessions: how many buffered observations are encoded together each cycle (the brain's effective parallelism). S = working-memory slots: how many entities can be held at once. Decay: how fast an unseen entity's salience fades — higher means longer object permanence." />
      </h3>

      <label className="cap-row">
        <span>
          K · parallel sessions
          <Info tip="Observations encoded per cycle in one batched, no-grad pass. Higher K = more percepts integrated per heartbeat, bounded by GPU memory." />
        </span>
        <input
          type="range"
          min={1}
          max={16}
          step={1}
          value={local.parallel_sessions}
          onChange={(e) =>
            setLocal({ ...local, parallel_sessions: Number(e.target.value) })
          }
          onPointerUp={() => void commit(local)}
          onKeyUp={() => void commit(local)}
        />
        <b>{local.parallel_sessions}</b>
      </label>

      <label className="cap-row">
        <span>
          S · memory slots
          <Info tip="Maximum entities held in working memory. Beyond this, the lowest-salience slot is evicted." />
        </span>
        <input
          type="range"
          min={1}
          max={48}
          step={1}
          value={local.working_memory_slots}
          onChange={(e) =>
            setLocal({ ...local, working_memory_slots: Number(e.target.value) })
          }
          onPointerUp={() => void commit(local)}
          onKeyUp={() => void commit(local)}
        />
        <b>{local.working_memory_slots}</b>
      </label>

      <label className="cap-row">
        <span>
          Salience decay
          <Info tip="Per-integration multiplier on slot salience. 0.90 fades a memory to ~5% after ~28 unseen cycles; closer to 1.0 means objects persist far longer out of view." />
        </span>
        <input
          type="range"
          min={0.5}
          max={0.99}
          step={0.01}
          value={local.working_memory_decay}
          onChange={(e) =>
            setLocal({ ...local, working_memory_decay: Number(e.target.value) })
          }
          onPointerUp={() => void commit(local)}
          onKeyUp={() => void commit(local)}
        />
        <b>{local.working_memory_decay.toFixed(2)}</b>
      </label>

      <div className="cap-readouts">
        <div>
          <span className="cap-k">cycles/s</span>
          <span className="cap-v">{(metrics?.approx_cycles_per_sec ?? 0).toFixed(1)}</span>
        </div>
        <div>
          <span className="cap-k">live sessions</span>
          <span className="cap-v">{metrics?.parallel_sessions ?? 0}</span>
        </div>
        <div>
          <span className="cap-k">active slots</span>
          <span className="cap-v">{metrics?.working_memory_slots ?? 0}</span>
        </div>
        <div>
          <span className="cap-k">encode</span>
          <span className="cap-v">{(metrics?.encode_phase_ms ?? 0).toFixed(2)} ms</span>
        </div>
        <div>
          <span className="cap-k">GPU mem</span>
          <span className="cap-v">{fmtBytes(metrics?.gpu_memory_max_allocated ?? 0)}</span>
        </div>
      </div>

      {metrics?.perception_feedback && (
        <div className="cap-readouts">
          <div>
            <span className="cap-k">
              perceptual precision
              <Info tip="Mean precision gate: how much the percept trusts bottom-up senses (→1) vs. the learned top-down prediction from history (→0). It self-tunes — the network learns this balance from the data, it is not set by hand." />
            </span>
            <span className="cap-v">{(metrics?.precision_gate_mean ?? 0).toFixed(2)}</span>
          </div>
          <div>
            <span className="cap-k">
              percept pred-err
              <Info tip="Perceptual prediction error: how well top-down history predicts the bottom-up percept. Trends down as the loop learns to anticipate perception." />
            </span>
            <span className="cap-v">{(metrics?.perceptual_pred_error ?? 0).toFixed(3)}</span>
          </div>
        </div>
      )}

      {metrics?.homeostatic_drive && (
        <div className="cap-readouts">
          <div>
            <span className="cap-k">
              homeostatic drive
              <Info tip="Root survival motivation (always on with a body): innate deprivation pain when a reservoir sits below its comfort setpoint. Rises as the agent goes without; it is felt urgency, not a goal pointing at any object. Uprightness is learned only as instrumental to avoiding integrity loss and pain." />
            </span>
            <span className="cap-v">{(metrics?.intero_drive ?? 0).toFixed(2)}</span>
          </div>
          <div>
            <span className="cap-k">
              intero pred-err
              <Info tip="Interoceptive world-model error: how well the brain predicts its own next reservoir levels from (state, action). Trends down as it learns its body's dynamics — the substrate that lets acting reduce predicted drive." />
            </span>
            <span className="cap-v">{(metrics?.intero_pred_error ?? 0).toFixed(3)}</span>
          </div>
        </div>
      )}

      {plast?.available && (
        <div className="plast-section">
          <h3>
            Neuroplasticity
            <Info tip="Three opt-in mechanisms that let the network change its own wiring while it learns (enabled at server start). A: Hebbian plasticity adds a fast, neuromodulated overlay on the weights. B: dynamic sparse training prunes weak connections and grows new ones. C: neuron growth wakes dormant neurons up to a cap, so the brain literally adds capacity over time." />
            {metrics?.plasticity_frozen && (
              <span className="badge paused" title="Instability guard tripped; plastic updates are frozen.">
                frozen
              </span>
            )}
          </h3>

          {plast.plasticity_enabled && (
            <label className="cap-row">
              <span>
                A · plastic strength
                <Info tip="Magnitude (alpha) of the Hebbian overlay added to each weight: W_eff = W + alpha·Hebb. 0 disables the overlay; higher values let recent correlated activity sway the response more strongly." />
              </span>
              <input
                type="range"
                min={0}
                max={0.5}
                step={0.01}
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                onPointerUp={() => void commitPlasticity({ plasticity_alpha: alpha })}
                onKeyUp={() => void commitPlasticity({ plasticity_alpha: alpha })}
              />
              <b>{alpha.toFixed(2)}</b>
            </label>
          )}

          {plast.sparse_enabled && (
            <label className="cap-row">
              <span>
                B · connection density
                <Info tip="Fraction of possible connections kept active in the plastic blocks. Lower = sparser, cheaper, and more like real cortex; 1.0 = fully dense. Changing this re-seeds the masks and periodically rewires (prune weakest, grow highest-gradient)." />
              </span>
              <input
                type="range"
                min={0.05}
                max={1}
                step={0.05}
                value={density}
                onChange={(e) => setDensity(Number(e.target.value))}
                onPointerUp={() => void commitPlasticity({ sparse_density: density })}
                onKeyUp={() => void commitPlasticity({ sparse_density: density })}
              />
              <b>{density.toFixed(2)}</b>
            </label>
          )}

          {plast.growth_enabled && (
            <label className="cap-row">
              <span>
                C · grow up to N
                <Info tip="Per-block cap on awake hidden neurons. The growth controller wakes dormant neurons toward this cap while prediction error stays high (organic growth). Lowering it puts neurons back to sleep immediately. New neurons wake function-preserving (silent), then learn." />
              </span>
              <input
                type="range"
                min={8}
                max={512}
                step={8}
                value={maxN}
                onChange={(e) => setMaxN(Number(e.target.value))}
                onPointerUp={() => void commitPlasticity({ max_neurons: maxN })}
                onKeyUp={() => void commitPlasticity({ max_neurons: maxN })}
              />
              <b>{maxN}</b>
            </label>
          )}

          <div className="cap-readouts">
            <div>
              <span className="cap-k">awake</span>
              <span className="cap-v">
                {metrics?.awake_neurons ?? plast.awake_neurons} / {metrics?.allocated_neurons ?? plast.allocated_neurons}
              </span>
            </div>
            <div>
              <span className="cap-k">density</span>
              <span className="cap-v">{(metrics?.sparse_density ?? plast.sparse_density ?? 1).toFixed(2)}</span>
            </div>
            <div>
              <span className="cap-k">alpha</span>
              <span className="cap-v">{(metrics?.plasticity_alpha ?? plast.plasticity_alpha ?? 0).toFixed(3)}</span>
            </div>
            <div>
              <span className="cap-k">grow / rewire</span>
              <span className="cap-v">
                {metrics?.growth_events ?? 0} / {metrics?.rewire_events ?? 0}
              </span>
            </div>
          </div>
        </div>
      )}

      {busy && <div className="strip-label">applying…</div>}
      </>
      )}
    </div>
  );
}
