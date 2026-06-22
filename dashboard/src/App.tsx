import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createAgentPreset,
  fetchAgents,
  fetchMetrics,
  fetchState,
  listAgentPresets,
  type AgentPreset,
  type ScenarioDraft,
} from "./api";
import { useHistory, usePolling } from "./usePolling";
import AgentPicker from "./components/AgentPicker";
import AgentAdmin from "./components/AgentAdmin";
import AgentControls from "./components/AgentControls";
import VitalsPanel from "./components/VitalsPanel";
import HomeostasisPanel from "./components/HomeostasisPanel";
import CyclePanel from "./components/CyclePanel";
import StateBusPanel from "./components/StateBusPanel";
import PerceptionCard from "./components/PerceptionCard";
import EventsPanel from "./components/EventsPanel";
import CycleWheelPanel from "./components/CycleWheelPanel";
import GraphPanel from "./components/GraphPanel";
import LongTermMemoryPanel from "./components/LongTermMemoryPanel";
import DiscoveryPanel from "./components/DiscoveryPanel";
import EvalPanel from "./components/EvalPanel";
import CognitionPanel from "./components/CognitionPanel";
import CapacityPanel from "./components/CapacityPanel";
import MotorPanel from "./components/MotorPanel";
import LocomotionPanel from "./components/LocomotionPanel";
import BrainMapPanel from "./components/BrainMapPanel";
import LandscapePanel from "./components/LandscapePanel";
import EnvironmentPanel from "./components/EnvironmentPanel";
import SkillDojoPanel from "./components/SkillDojoPanel";
import DeploymentPanel from "./components/DeploymentPanel";
import SavedAgentsPanel from "./components/SavedAgentsPanel";
import PresetPicker from "./components/PresetPicker";
import TabBar, { type TabDef } from "./components/TabBar";
import VitalsStrip from "./components/VitalsStrip";
import ErrorBoundary from "./components/ErrorBoundary";

const TABS: TabDef[] = [
  { id: "overview", label: "Overview" },
  { id: "environment", label: "Environment" },
  { id: "dojo", label: "Skill Dojo" },
  { id: "graph", label: "Self-Indexed Graph" },
  { id: "discovery", label: "Discovery" },
  { id: "cognition", label: "Cognition / Why" },
  { id: "motor", label: "Motor / Active Inference" },
  { id: "brain", label: "Brain Map" },
  { id: "landscape", label: "Loss Landscape" },
  { id: "events", label: "Events + State Bus" },
  { id: "capacity", label: "Agent Settings" },
  { id: "deploy", label: "Deploy / GPU" },
  { id: "library", label: "Saved Agents" },
];

const TAB_STORAGE_KEY = "decadic.activeTab";
const DEFAULT_CREATION_PRESET = "tiny";

// A calm homeostasis body by default until presets load from the server.
const DEFAULT_DRAFT: ScenarioDraft = {
  elements: ["house", "food", "water"],
  vision: true,
  audio: false,
  braces: false,
  mindOnly: false,
};

function presetToDraft(p: AgentPreset): ScenarioDraft {
  return {
    elements: [...p.elements],
    vision: p.vision,
    audio: p.audio,
    braces: p.braces,
    mindOnly: p.mind_only,
  };
}

export default function App() {
  const [agentId, setAgentId] = useState<string | null>(null);
  // A just-created agent won't appear in the polled list for up to 2s; don't
  // let the fallback below clobber its selection in that window.
  const justCreated = useRef<{ id: string; at: number } | null>(null);

  const [activeTab, setActiveTab] = useState<string>(() => {
    const saved = localStorage.getItem(TAB_STORAGE_KEY);
    return saved && TABS.some((t) => t.id === saved) ? saved : "overview";
  });
  const selectTab = (id: string) => {
    setActiveTab(id);
    localStorage.setItem(TAB_STORAGE_KEY, id);
  };

  // Agent presets + the shared "draft" scenario the top-bar dropdown and the
  // Environment tab both edit. Picking a preset loads it into the draft; editing
  // the draft in the Environment tab marks it Custom (selectedPresetId = null).
  const [presets, setPresets] = useState<AgentPreset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(null);
  const [draft, setDraft] = useState<ScenarioDraft>(DEFAULT_DRAFT);
  const [creationPreset, setCreationPreset] = useState(DEFAULT_CREATION_PRESET);

  const refreshPresets = useCallback(async (): Promise<AgentPreset[]> => {
    const list = await listAgentPresets();
    setPresets(list);
    return list;
  }, []);

  useEffect(() => {
    void refreshPresets().catch(() => {});
  }, [refreshPresets]);

  // On first load (no selection yet), reflect the first preset into the draft so
  // the Environment tab mirrors the top-bar dropdown.
  useEffect(() => {
    if (selectedPresetId == null && presets.length > 0 && draft === DEFAULT_DRAFT) {
      setSelectedPresetId(presets[0].id);
      setDraft(presetToDraft(presets[0]));
    }
  }, [presets, selectedPresetId, draft]);

  const onSelectPreset = useCallback(
    (id: string) => {
      const p = presets.find((x) => x.id === id);
      if (!p) return;
      setSelectedPresetId(id);
      setDraft(presetToDraft(p));
    },
    [presets],
  );

  const updateDraft = useCallback((patch: Partial<ScenarioDraft>) => {
    setDraft((d) => ({ ...d, ...patch }));
    // Any manual edit diverges from a named preset -> Custom.
    setSelectedPresetId(null);
  }, []);

  const savePreset = useCallback(
    async (name: string): Promise<AgentPreset> => {
      const created = await createAgentPreset({
        name,
        elements: draft.elements,
        vision: draft.vision,
        audio: draft.audio,
        braces: draft.braces,
        mind_only: draft.mindOnly,
      });
      await refreshPresets();
      setSelectedPresetId(created.id);
      return created;
    },
    [draft, refreshPresets],
  );

  const { data: agents } = usePolling(fetchAgents, 2000);

  useEffect(() => {
    if (!agents) return;
    if (agentId && agents.some((a) => a.agent_id === agentId)) return;
    const jc = justCreated.current;
    if (jc && jc.id === agentId && Date.now() - jc.at < 10000) return;
    setAgentId(agents.length > 0 ? agents[0].agent_id : null);
  }, [agents, agentId]);

  const { data: state, error: stateError } = usePolling(
    () => (agentId ? fetchState(agentId) : Promise.resolve(null)),
    700,
    [agentId],
  );
  const { data: metrics } = usePolling(
    () => (agentId ? fetchMetrics(agentId) : Promise.resolve(null)),
    700,
    [agentId],
  );

  const sample = useMemo(() => {
    if (!metrics) return null;
    return {
      t: metrics.cycles_completed,
      pcLoss: metrics.neural_pc_loss_last,
      viability: metrics.viability,
      pain: state?.state_bus.B_pain_scalar ?? 0,
      pleasure: state?.state_bus.B_pleasure_scalar ?? 0,
      fwdErr: metrics.forward_model_error,
      assistGain: metrics.assist_gain,
    };
  }, [metrics, state]);
  const history = useHistory(sample);

  const summary = agents?.find((a) => a.agent_id === agentId);
  const status = metrics?.status ?? state?.status ?? summary?.status ?? "alive";
  const diedAt = metrics?.died_at_cycle ?? state?.died_at_cycle ?? summary?.died_at_cycle ?? null;
  const dead = status === "dead";

  return (
    <div className="app">
      <div className="topbar">
        <h1>Decadic — Live Cognition</h1>
        <span className="sub">Decadic Cycle Cognitive Architecture</span>
        {metrics?.encoder_mode != null && (
          <span
            className={`badge encoder ${metrics.encoder_mode === "hf" ? "hf" : ""}`}
            title={
              metrics.encoder_mode === "hf"
                ? "Pretrained encoders active: CLIP vision + Whisper audio enter the forward pass"
                : "Zero encoders: only proprioception reaches the network (set DECADIC_ENCODER_MODE=hf)"
            }
          >
            encoders: {String(metrics.encoder_mode)}
          </span>
        )}
        {agentId && state && <VitalsStrip state={state} />}
        <div style={{ flex: 1 }} />
        {agentId && (
          <PresetPicker agentId={agentId} preset={metrics?.preset ?? summary?.preset} />
        )}
        {agentId && (
          <AgentControls
            agentId={agentId}
            status={status}
            paused={metrics?.paused ?? summary?.paused ?? false}
          />
        )}
        <AgentPicker agents={agents ?? []} selected={agentId} onSelect={setAgentId} />
        <AgentAdmin
          selected={agentId}
          presets={presets}
          selectedPresetId={selectedPresetId}
          draft={draft}
          creationPreset={creationPreset}
          onSelectPreset={onSelectPreset}
          onCreationPresetChange={setCreationPreset}
          onCreated={(id) => {
            justCreated.current = { id, at: Date.now() };
            setAgentId(id);
          }}
          onDeleted={() => setAgentId(null)}
        />
      </div>

      {stateError && agentId && (
        <div className="error-banner">Server unreachable or agent gone: {stateError}</div>
      )}

      {agentId && dead && (
        <div className="death-banner">
          This mind has died (viability reached 0
          {diedAt != null ? ` at cycle ${diedAt}` : ""}). Its weights are frozen and a
          tombstone checkpoint was saved. Use <b>Revive</b> to restore the same mind, or{" "}
          <b>Reincarnate</b> for a fresh one.
        </div>
      )}

      <TabBar tabs={TABS} active={activeTab} onSelect={selectTab} />

      {activeTab === "environment" && (
        <div className="grid solo">
          <EnvironmentPanel
            draft={draft}
            updateDraft={updateDraft}
            selectedPreset={presets.find((p) => p.id === selectedPresetId) ?? null}
            creationPreset={creationPreset}
            onSavePreset={savePreset}
            onStarted={(id) => {
              justCreated.current = { id, at: Date.now() };
              setAgentId(id);
            }}
          />
        </div>
      )}

      {activeTab === "dojo" && (
        <div className="grid solo">
          <ErrorBoundary label="Skill Dojo" resetKey={agentId ?? "none"}>
            <SkillDojoPanel
              agentId={agentId}
              metrics={metrics}
              state={state}
              creationPreset={creationPreset}
              onStarted={(id) => {
                justCreated.current = { id, at: Date.now() };
                setAgentId(id);
              }}
            />
          </ErrorBoundary>
        </div>
      )}

      {activeTab === "deploy" && (
        <DeploymentPanel onWatchAgent={() => selectTab("overview")} />
      )}

      {activeTab === "library" && (
        <div className="grid solo">
          <SavedAgentsPanel
            onLoaded={(id) => {
              justCreated.current = { id, at: Date.now() };
              setAgentId(id);
              selectTab("overview");
            }}
          />
        </div>
      )}

      {activeTab !== "environment" &&
        activeTab !== "dojo" &&
        activeTab !== "capacity" &&
        activeTab !== "deploy" &&
        activeTab !== "library" &&
        !agentId && (
        <div className="panel">
          <div className="empty">
            No agents running. Click <b>+ New agent</b> above to spawn an embodied mind (a
            MuJoCo body with vision) - pick a scene or <b>Mind only</b> from the dropdown next
            to it. The <b>Environment</b> tab composes a fuller scenario, or connect a body from
            a terminal, e.g.{" "}
            <code>python scripts/mujoco_decadic_adapter.py --steps 0 --vision --view</code>
          </div>
        </div>
      )}

      {activeTab === "capacity" && (
        <div className="grid solo">
          <ErrorBoundary label="Agent Settings" resetKey={metrics?.cycles_completed}>
            <CapacityPanel agentId={agentId} metrics={metrics} capacity={state?.capacity} />
          </ErrorBoundary>
        </div>
      )}

      {activeTab !== "environment" && activeTab !== "dojo" && agentId && state && (
        <>
          {activeTab === "overview" && (
            <div className="overview">
              <div className="overview-vitals">
                <ErrorBoundary label="Vitals" resetKey={state.state_bus.cycle_index}>
                  <VitalsPanel state={state} metrics={metrics} />
                </ErrorBoundary>
                <ErrorBoundary label="Homeostasis" resetKey={state.state_bus.cycle_index}>
                  <HomeostasisPanel
                    agentId={agentId}
                    metrics={metrics}
                    capacity={state.capacity}
                  />
                </ErrorBoundary>
              </div>
              <div className="overview-main">
                <ErrorBoundary label="Perception" resetKey={state.state_bus.cycle_index}>
                  <PerceptionCard agentId={agentId} state={state} />
                </ErrorBoundary>
                <div className="overview-stack">
                  <ErrorBoundary label="Decadic Cycle" resetKey={state.state_bus.cycle_index}>
                    <CycleWheelPanel trace={state.last_cycle_trace} />
                  </ErrorBoundary>
                  <ErrorBoundary label="Cognitive Cycle" resetKey={state.state_bus.cycle_index}>
                    <CyclePanel metrics={metrics} history={history} />
                  </ErrorBoundary>
                </div>
              </div>
            </div>
          )}

          {activeTab === "graph" && (
            <div className="grid">
              <ErrorBoundary label="Self-Indexed Graph" resetKey={state.state_bus.cycle_index}>
                <GraphPanel state={state} />
              </ErrorBoundary>
              <ErrorBoundary label="Long-term Memory" resetKey={state.state_bus.cycle_index}>
                <LongTermMemoryPanel state={state} />
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "discovery" && (
            <div className="grid">
              <ErrorBoundary label="Object Discovery" resetKey={state.state_bus.cycle_index}>
                <DiscoveryPanel state={state} />
              </ErrorBoundary>
              <ErrorBoundary label="Discovery Evaluation" resetKey={state.state_bus.cycle_index}>
                <EvalPanel state={state} metrics={metrics} />
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "cognition" && (
            <div className="grid solo">
              <ErrorBoundary label="Cognition / Why" resetKey={state.state_bus.cycle_index}>
                <CognitionPanel agentId={agentId} state={state} />
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "motor" && (
            <div className="grid">
              <ErrorBoundary label="Motor / Active Inference" resetKey={metrics?.cycles_completed}>
                <MotorPanel metrics={metrics} history={history} agentId={agentId} />
              </ErrorBoundary>
              <ErrorBoundary label="Locomotion" resetKey={metrics?.cycles_completed}>
                <LocomotionPanel metrics={metrics} />
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "brain" && (
            <div className="grid solo">
              <ErrorBoundary label="Brain Map" resetKey={state.state_bus.cycle_index}>
                <BrainMapPanel agentId={agentId} state={state} metrics={metrics} />
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "landscape" && (
            <div className="grid solo">
              <ErrorBoundary label="Loss Landscape" resetKey={state.state_bus.cycle_index}>
                <LandscapePanel agentId={agentId} />
              </ErrorBoundary>
            </div>
          )}

          {activeTab === "events" && (
            <div className="grid">
              <ErrorBoundary label="Recent Events" resetKey={state.state_bus.cycle_index}>
                <EventsPanel perceptual={state.perceptual} />
              </ErrorBoundary>
              <ErrorBoundary label="State Bus" resetKey={state.state_bus.cycle_index}>
                <StateBusPanel bus={state.state_bus} />
              </ErrorBoundary>
            </div>
          )}
        </>
      )}
    </div>
  );
}
