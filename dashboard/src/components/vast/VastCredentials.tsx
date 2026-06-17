import { useState } from "react";
import {
  fetchVastAccount,
  saveVastSettings,
  type VastAccount,
  type VastSettings,
} from "../../vastApi";

/** Store + mask the Vast.ai API key and SSH public-key path; show balance. */
export default function VastCredentials(props: {
  settings: VastSettings;
  onSaved: (s: VastSettings) => void;
}) {
  const { settings } = props;
  const [keyInput, setKeyInput] = useState("");
  const [sshPath, setSshPath] = useState(settings.ssh_key_path || "");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [account, setAccount] = useState<VastAccount | null>(null);

  const save = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const next = await saveVastSettings({
        api_key: keyInput.trim() || undefined,
        ssh_key_path: sshPath.trim(),
      });
      props.onSaved(next);
      setKeyInput("");
      setMsg("Saved.");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const clearKey = async () => {
    if (!window.confirm("Remove the stored Vast.ai API key from this machine?")) return;
    setBusy(true);
    setMsg(null);
    try {
      props.onSaved(await saveVastSettings({ clear_api_key: true }));
      setAccount(null);
      setMsg("Key removed.");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const checkAccount = async () => {
    setBusy(true);
    setMsg(null);
    try {
      setAccount(await fetchVastAccount());
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel" style={{ display: "grid", gap: 10 }}>
      <h3 style={{ margin: 0 }}>Vast.ai credentials</h3>

      {!settings.cli_available && (
        <div className="error-banner" style={{ margin: 0 }}>
          The <code>vastai</code> CLI was not found on the server. Install it where the
          Decadic server runs: <code>pip install vastai</code>, then restart the server.
        </div>
      )}

      <div style={{ fontSize: 13, opacity: 0.85 }}>
        {settings.has_api_key ? (
          <>
            Key stored: <b>{settings.api_key_masked}</b>{" "}
            <button onClick={() => void clearKey()} disabled={busy}>
              Remove
            </button>{" "}
            <button onClick={() => void checkAccount()} disabled={busy}>
              Check account
            </button>
          </>
        ) : (
          <>No API key stored yet. Paste one from cloud.vast.ai/manage-keys.</>
        )}
      </div>

      <label style={{ display: "grid", gap: 4 }}>
        <span style={{ fontSize: 12, opacity: 0.8 }}>
          API key {settings.has_api_key ? "(enter to replace)" : ""}
        </span>
        <input
          type="password"
          placeholder="vast.ai API key"
          value={keyInput}
          onChange={(e) => setKeyInput(e.target.value)}
          autoComplete="off"
          style={{ width: "100%" }}
        />
      </label>

      <label style={{ display: "grid", gap: 4 }}>
        <span style={{ fontSize: 12, opacity: 0.8 }}>
          SSH public/private key path (optional; used for the tunnel + remote exec)
        </span>
        <input
          type="text"
          placeholder="~/.ssh/id_ed25519"
          value={sshPath}
          onChange={(e) => setSshPath(e.target.value)}
          style={{ width: "100%" }}
        />
      </label>

      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button onClick={() => void save()} disabled={busy}>
          Save
        </button>
        {account && (
          <span style={{ fontSize: 13 }}>
            Balance: <b>${account.balance?.toFixed?.(2) ?? account.balance ?? "?"}</b>
            {account.email ? ` (${account.email})` : ""}
          </span>
        )}
        {msg && <span style={{ fontSize: 12, opacity: 0.8 }}>{msg}</span>}
      </div>

      <div style={{ fontSize: 11, opacity: 0.6 }}>
        Stored locally at <code>{settings.config_path}</code> (never sent anywhere but
        Vast.ai). Register your SSH key with Vast.ai before renting.
      </div>
    </div>
  );
}
