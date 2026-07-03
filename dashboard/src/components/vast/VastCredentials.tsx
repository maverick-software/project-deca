import { useState } from "react";
import {
  fetchVastAccount,
  saveVastSettings,
  type VastAccount,
  type VastSettings,
} from "../../vastApi";
import FileBrowserModal from "./FileBrowserModal";

/**
 * Store + mask the Vast.ai API key and SSH key path; show balance.
 *
 * Neither secret is ever shown in the clear once saved: a connected key
 * renders as a masked value with a Disconnect action (no input box), and
 * adding a new one only surfaces an input while nothing is stored. Same
 * pattern for the SSH key path, since a raw path can leak the local OS
 * username / directory layout.
 */
export default function VastCredentials(props: {
  settings: VastSettings;
  onSaved: (s: VastSettings) => void;
}) {
  const { settings } = props;
  const [keyInput, setKeyInput] = useState("");
  const [sshInput, setSshInput] = useState("");
  const [editingSsh, setEditingSsh] = useState(false);
  const [browsing, setBrowsing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [account, setAccount] = useState<VastAccount | null>(null);

  const saveKey = async () => {
    if (!keyInput.trim()) return;
    setBusy(true);
    setMsg(null);
    try {
      const next = await saveVastSettings({ api_key: keyInput.trim() });
      props.onSaved(next);
      setKeyInput("");
      setMsg("Key saved.");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const disconnectKey = async () => {
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

  const saveSsh = async () => {
    if (!sshInput.trim()) return;
    setBusy(true);
    setMsg(null);
    try {
      const next = await saveVastSettings({ ssh_key_path: sshInput.trim() });
      props.onSaved(next);
      setSshInput("");
      setEditingSsh(false);
      setMsg("SSH key path saved.");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const removeSsh = async () => {
    if (!window.confirm("Remove the stored SSH key path?")) return;
    setBusy(true);
    setMsg(null);
    try {
      props.onSaved(await saveVastSettings({ clear_ssh_key_path: true }));
      setMsg("SSH key path removed.");
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

      {/* --- API key --- */}
      {settings.has_api_key ? (
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 13, opacity: 0.85 }}>
            API key connected: <b>{settings.api_key_masked}</b>
          </span>
          <button onClick={() => void checkAccount()} disabled={busy}>
            Check account
          </button>
          <button onClick={() => void disconnectKey()} disabled={busy}>
            Disconnect
          </button>
          {account && (
            <span style={{ fontSize: 13 }}>
              Balance: <b>${account.balance?.toFixed?.(2) ?? account.balance ?? "?"}</b>
              {account.email ? ` (${account.email})` : ""}
            </span>
          )}
        </div>
      ) : (
        <label style={{ display: "grid", gap: 4 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>
            API key - paste one from cloud.vast.ai/manage-keys
          </span>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              type="password"
              placeholder="vast.ai API key"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              autoComplete="off"
              style={{ flex: 1 }}
            />
            <button onClick={() => void saveKey()} disabled={busy || !keyInput.trim()}>
              Connect
            </button>
          </div>
        </label>
      )}

      {/* --- SSH key path --- */}
      {settings.has_ssh_key_path && !editingSsh ? (
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 13, opacity: 0.85 }}>
            SSH key: <b>{settings.ssh_key_path_masked}</b>
          </span>
          <button onClick={() => setEditingSsh(true)} disabled={busy}>
            Change
          </button>
          <button onClick={() => void removeSsh()} disabled={busy}>
            Remove
          </button>
        </div>
      ) : (
        <label style={{ display: "grid", gap: 4 }}>
          <span style={{ fontSize: 12, opacity: 0.8 }}>
            SSH key path (optional; used for the tunnel + remote exec)
          </span>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              type="text"
              placeholder="~/.ssh/id_ed25519"
              value={sshInput}
              onChange={(e) => setSshInput(e.target.value)}
              autoComplete="off"
              style={{ flex: 1 }}
            />
            <button onClick={() => setBrowsing(true)} disabled={busy}>
              Browse...
            </button>
            <button onClick={() => void saveSsh()} disabled={busy || !sshInput.trim()}>
              Save
            </button>
            {settings.has_ssh_key_path && (
              <button
                onClick={() => {
                  setEditingSsh(false);
                  setSshInput("");
                }}
                disabled={busy}
              >
                Cancel
              </button>
            )}
          </div>
        </label>
      )}

      {msg && <div style={{ fontSize: 12, opacity: 0.8 }}>{msg}</div>}

      <div style={{ fontSize: 11, opacity: 0.6 }}>
        Stored locally at <code>{settings.config_path}</code> (never sent anywhere but
        Vast.ai). Register your SSH key with Vast.ai before renting.
      </div>

      {browsing && (
        <FileBrowserModal
          onPick={(path) => {
            setSshInput(path);
            setBrowsing(false);
          }}
          onClose={() => setBrowsing(false)}
        />
      )}
    </div>
  );
}
