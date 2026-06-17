# ai.md — deploy/vast

Quick orientation for future edits in this area.

## Purpose
UI-driven Vast.ai GPU deployment. The local FastAPI server is the control plane;
this folder holds the two bash scripts it runs on the rented box over ssh. There
is no standalone CLI for the operator — the dashboard's **Deploy / GPU** tab is
the entry point.

## Files
- `setup_remote.sh` — remote one-time provisioning (system libs + python deps +
  encoder prewarm). Idempotent. Reads env: `ENCODER`, `WHISPER_MODEL`.
- `run_remote.sh` — launches the brain server in a detached tmux session with
  CUDA + EGL env. Reads env: `PRESET`, `ENCODER`, `WHISPER_MODEL`. Generates
  `_run_server.sh` (gitignored) on the box to avoid tmux quoting issues.
- `README.md` — operator + architecture docs.

## Where the logic lives (not here)
- `decadic/api/vast/settings_store.py` — key + defaults (0600 JSON at
  `~/.decadic/vast.json`).
- `decadic/api/vast/cli.py` — `vastai` CLI wrappers + ssh exec/tunnel helpers.
- `decadic/api/vast/controller.py` — deployment state machine + background task.
- `decadic/api/vast/proxy.py` — reverse proxy (added INNER; CORS stays OUTER).
- `decadic/api/vast/routes.py` — `/vast/*` endpoints (mounted in `create_app`).
- `dashboard/src/vastApi.ts` + `dashboard/src/components/DeploymentPanel.tsx`
  (+ `components/vast/*`) — the UI.

## Gotchas
- The scene/body is NOT launched by `run_remote.sh`; the controller calls
  `POST /environment` through the tunnel so the supervisor's adapter inherits
  `MUJOCO_GL=egl` from the server process.
- Keep `--no-deps` on the project install so the image's CUDA torch is preserved.
- Scripts MUST stay LF-terminated (they run under bash on Linux).
- Files in this package each stay < 500 lines (house rule).
