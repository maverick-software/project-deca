import { useEffect, useMemo, useState } from "react";
import {
  fetchEvalReport,
  fetchEvalReports,
  fetchEvalScenarios,
  fetchEvalStatus,
  startEvalJob,
  stopEvalJob,
  type EvalReport,
  type EvalReportSummary,
  type EvalScenario,
} from "../api";
import { usePolling } from "../usePolling";

function fmtTime(s?: number | null): string {
  const v = Number(s ?? 0);
  if (!Number.isFinite(v) || v <= 0) return "0s";
  if (v < 90) return `${v.toFixed(0)}s`;
  return `${(v / 60).toFixed(1)}m`;
}

function JsonSummary(props: { title: string; value?: Record<string, unknown> }) {
  const value = props.value || {};
  const entries = Object.entries(value).slice(0, 8);
  return (
    <div className="eval-summary-box">
      <h3>{props.title}</h3>
      {entries.length === 0 ? (
        <span className="muted">No data</span>
      ) : (
        entries.map(([k, v]) => (
          <div key={k} className="eval-kv">
            <span>{k}</span>
            <b>{typeof v === "object" && v !== null ? JSON.stringify(v).slice(0, 90) : String(v)}</b>
          </div>
        ))
      )}
    </div>
  );
}

export default function TrainingEvalPanel(props: { agentId: string | null }) {
  const [scenarios, setScenarios] = useState<EvalScenario[]>([]);
  const [reports, setReports] = useState<EvalReportSummary[]>([]);
  const [selectedScenario, setSelectedScenario] = useState("health_smoke");
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);
  const [report, setReport] = useState<EvalReport | null>(null);
  const [cycles, setCycles] = useState("");
  const [preset, setPreset] = useState("");
  const [dojoSkill, setDojoSkill] = useState("");
  const [useCurrentAgent, setUseCurrentAgent] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const { data: status } = usePolling(fetchEvalStatus, 2000);

  const scenario = useMemo(
    () => scenarios.find((s) => s.scenario === selectedScenario) ?? scenarios[0] ?? null,
    [scenarios, selectedScenario],
  );

  const refreshReports = async () => {
    const list = await fetchEvalReports();
    setReports(list);
    if (!selectedReportId && list.length > 0) setSelectedReportId(list[0].report_id);
  };

  useEffect(() => {
    void fetchEvalScenarios()
      .then((rows) => {
        setScenarios(rows);
        if (rows.length > 0) setSelectedScenario((cur) => rows.some((r) => r.scenario === cur) ? cur : rows[0].scenario);
      })
      .catch((e) => setError(String(e)));
    void refreshReports().catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selectedReportId) {
      setReport(null);
      return;
    }
    void fetchEvalReport(selectedReportId)
      .then(setReport)
      .catch((e) => setError(String(e)));
  }, [selectedReportId]);

  useEffect(() => {
    if (scenario) {
      setCycles(String(scenario.cycles ?? ""));
      setPreset(scenario.agent_preset ?? "");
      setDojoSkill(scenario.dojo_skill_id ?? "");
    }
  }, [scenario?.scenario]);

  const running = status?.state === "starting" || status?.state === "running" || status?.state === "stopping";

  const launch = async () => {
    if (!scenario) return;
    if (!confirming) {
      setConfirming(true);
      return;
    }
    setBusy(true);
    setError("");
    try {
      await startEvalJob({
        scenario: scenario.scenario,
        cycles: cycles.trim() ? Number(cycles) : null,
        preset: preset.trim() || null,
        dojo_skill_id: dojoSkill.trim() || null,
        agent_id: useCurrentAgent ? props.agentId : null,
      });
      setConfirming(false);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    try {
      await stopEvalJob();
      await refreshReports();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel training-eval-panel">
      <div className="panel-head">
        <div>
          <h2>Training Eval</h2>
          <p>Run explicit learning and survival evaluations, then inspect reports without keeping an agent alive.</p>
        </div>
        <button type="button" className="mini" onClick={() => void refreshReports()}>
          Refresh reports
        </button>
      </div>

      {error && <div className="dojo-banner error">{error}</div>}

      <div className={`eval-status-band ${status?.state ?? "idle"}`}>
        <div>
          <b>{status?.state ?? "idle"}</b>
          <span>{status?.scenario ?? "No eval running"}</span>
          {status?.agent_id && <span>Agent {status.agent_id.slice(0, 8)}</span>}
        </div>
        <div>
          <span>{status?.cycles ?? 0}/{status?.target_cycles ?? 0} cycles</span>
          <span>{status?.samples ?? 0} samples</span>
          <span>{fmtTime(status?.elapsed_s)}</span>
          {status?.body_connected ? <span>body connected</span> : <span className="warn">no body</span>}
        </div>
        {status?.error && <small>{status.error}</small>}
      </div>

      <div className="eval-layout">
        <section className="eval-column">
          <h3>Scenarios</h3>
          <div className="eval-scenario-list">
            {scenarios.map((s) => (
              <button
                key={s.scenario}
                type="button"
                className={`eval-card ${s.scenario === selectedScenario ? "selected" : ""}`}
                onClick={() => setSelectedScenario(s.scenario)}
              >
                <b>{s.scenario}</b>
                <span>{s.description}</span>
                <small>{s.cycles} cycles - timeout {fmtTime(s.timeout_s)} {s.body_required ? "- body likely required" : ""}</small>
              </button>
            ))}
          </div>
          {scenario && (
            <div className="eval-gates">
              <h3>Gates</h3>
              {scenario.gates.map((g) => (
                <div key={`${g.name}-${g.metric}`} className="eval-gate">
                  <span>{g.name}</span>
                  <b>{g.metric} {g.op} {g.threshold}</b>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="eval-column">
          <h3>Run</h3>
          <label className="eval-field">
            <span>Scenario</span>
            <select value={selectedScenario} onChange={(e) => setSelectedScenario(e.target.value)}>
              {scenarios.map((s) => <option key={s.scenario} value={s.scenario}>{s.scenario}</option>)}
            </select>
          </label>
          <label className="eval-field">
            <span>Cycles</span>
            <input value={cycles} onChange={(e) => setCycles(e.target.value)} />
          </label>
          <label className="eval-field">
            <span>Preset</span>
            <input value={preset} onChange={(e) => setPreset(e.target.value)} placeholder="default" />
          </label>
          <label className="eval-field">
            <span>Dojo skill</span>
            <input value={dojoSkill} onChange={(e) => setDojoSkill(e.target.value)} placeholder="none" />
          </label>
          <label className="eval-check">
            <input
              type="checkbox"
              checked={useCurrentAgent}
              disabled={!props.agentId}
              onChange={(e) => setUseCurrentAgent(e.target.checked)}
            />
            <span>Use selected agent {props.agentId ? props.agentId.slice(0, 8) : "(none selected)"}</span>
          </label>
          {confirming && scenario && (
            <div className="dojo-banner warning">
              Confirm launch: {scenario.scenario}, {cycles || scenario.cycles} cycles, creates {useCurrentAgent ? "no new agent" : "a new agent"}
              {dojoSkill.trim() ? `, starts dojo ${dojoSkill}` : ""}.
            </div>
          )}
          <div className="eval-actions">
            <button type="button" disabled={busy || running || !scenario} onClick={() => void launch()}>
              {confirming ? "Confirm start" : "Start eval"}
            </button>
            <button type="button" disabled={busy || !running} onClick={() => void stop()}>
              Stop
            </button>
          </div>
          {status?.report_path && <p className="muted">Latest report: {status.report_path}</p>}
        </section>

        <section className="eval-column eval-reports">
          <h3>Reports</h3>
          {reports.length === 0 ? (
            <div className="empty">No reports yet.</div>
          ) : (
            <div className="eval-report-list">
              {reports.map((r) => (
                <button
                  type="button"
                  key={r.report_id}
                  className={`eval-report-row ${selectedReportId === r.report_id ? "selected" : ""}`}
                  onClick={() => setSelectedReportId(r.report_id)}
                >
                  <b className={r.status === "pass" ? "pass" : "fail"}>{r.status}</b>
                  <span>{r.scenario}</span>
                  <small>{r.failures_count} failures</small>
                </button>
              ))}
            </div>
          )}
          {report && (
            <div className="eval-report-detail">
              <h3>{report.scenario} - {report.status}</h3>
              {report.failures && report.failures.length > 0 && (
                <div className="dojo-banner error">
                  {report.failures.map((f) => <div key={f}>{f}</div>)}
                </div>
              )}
              <div className="eval-summary-grid">
                <JsonSummary title="Health" value={report.health} />
                <JsonSummary title="Learning" value={report.learning} />
                <JsonSummary title="Perception" value={report.perception} />
                <JsonSummary title="Behavior" value={report.behavior} />
              </div>
              {report.samples_path && <p className="muted">Samples: {report.samples_path}</p>}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
