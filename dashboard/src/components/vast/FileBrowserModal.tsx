import { useEffect, useState } from "react";
import { browseFs, type FsListing } from "../../vastApi";

/**
 * Server-side folder browser for picking the SSH key file.
 *
 * A browser tab can't hand back a real OS filesystem path from a native file
 * dialog (that's blocked for security), but the Decadic server is 127.0.0.1
 * only - it runs on this same machine - so it can walk the filesystem itself
 * and hand the picked path to the frontend. This gets the same "click
 * through folders, pick a file" experience without exposing anything beyond
 * what the operator's own server can already see.
 */
export default function FileBrowserModal(props: {
  initialPath?: string;
  onPick: (path: string) => void;
  onClose: () => void;
}) {
  const [listing, setListing] = useState<FsListing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = async (path?: string) => {
    setLoading(true);
    setError(null);
    try {
      setListing(await browseFs(path));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load(props.initialPath);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.6)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
      onClick={props.onClose}
    >
      <div
        className="panel"
        style={{
          width: 480,
          maxHeight: "70vh",
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h4 style={{ margin: 0 }}>Select SSH key file</h4>
          <button onClick={props.onClose}>Close</button>
        </div>

        <div style={{ fontSize: 11, opacity: 0.6, wordBreak: "break-all" }}>
          {listing?.path ?? "Loading..."}
        </div>

        {error && <div className="error-banner">{error}</div>}

        <div
          style={{
            overflowY: "auto",
            flex: 1,
            border: "1px solid rgba(255,255,255,0.15)",
            borderRadius: 4,
            minHeight: 200,
          }}
        >
          {listing?.parent && (
            <div
              style={{ padding: "6px 10px", cursor: "pointer", opacity: 0.8 }}
              onClick={() => void load(listing.parent!)}
            >
              .. (up one level)
            </div>
          )}
          {listing?.entries.map((e) => (
            <div
              key={e.path}
              style={{ padding: "6px 10px", cursor: "pointer" }}
              onClick={() => (e.is_dir ? void load(e.path) : props.onPick(e.path))}
            >
              {e.is_dir ? "[dir]  " : "[file] "}
              {e.name}
            </div>
          ))}
          {loading && <div style={{ padding: 10, opacity: 0.7 }}>Loading...</div>}
          {!loading && listing && listing.entries.length === 0 && (
            <div style={{ padding: 10, opacity: 0.7 }}>Empty folder.</div>
          )}
        </div>

        <div style={{ fontSize: 11, opacity: 0.6 }}>
          Click a folder to open it, click a file to select it as the SSH key path.
        </div>
      </div>
    </div>
  );
}
