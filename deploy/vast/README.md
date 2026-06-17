# Vast.ai GPU deployment (UI-driven)

Rent a cloud GPU, ship the Decadic brain + headless MuJoCo body to it, and watch
the agent learn from the dashboard — no terminal required. Everything is driven
from the **Deploy / GPU** tab; this folder only holds the two scripts the local
server runs on the rented box over ssh.

## How it works

```
Browser (localhost:5173)
   -> Dashboard "Deploy / GPU" tab
   -> Local server :8765  (control plane + reverse proxy)
        - stores the API key (~/.decadic/vast.json, 0600)
        - vastai search / create / destroy
        - ssh: upload code, run setup_remote.sh + run_remote.sh
        - ssh -L <free>:localhost:8765  (tunnel)
        - proxies /agents, /agent/*, /environment* -> tunnel
   -> Vast.ai GPU box
        uvicorn :8765 (CUDA)  +  MuJoCo body (EGL, headless)  +  CLIP/Whisper
```

While a deployment is `ready`, the local server transparently forwards agent
reads/writes to the remote, so the existing Overview / Brain / Motor panels show
the **remote** agent. WebSockets are untouched (the body connects to the brain
locally on the box).

## Prerequisites (one-time, on the machine running the Decadic server)

1. The `vastai` CLI — **bundled**: it is a project dependency, so `pip install -e .`
   installs it. The control plane finds it even when its Scripts dir is not on
   PATH (it runs the entry point via the server's own interpreter), so no manual
   install or PATH edit is needed. Override the binary with `DECADIC_VASTAI_BIN`
   if you want a specific one.
2. An OpenSSH client on `PATH` (`ssh`; ships with Windows 10/11 and Linux/macOS).
3. A Vast.ai account + API key (cloud.vast.ai → Account → API key).
4. An SSH key registered with Vast.ai **before** renting (Account → SSH keys).
   Point the dashboard at its path (e.g. `~/.ssh/id_ed25519`).

> Already running the server when you added this feature? Restart the server
> window so it picks up the bundled `vastai` (the running process cached the old
> "not found" state at startup).

## Using it

1. Open the dashboard → **Deploy / GPU**.
2. Paste your API key, set your SSH key path, **Save**. (Optionally **Check
   account** for your balance.)
3. **Run configuration**: brain preset, encoders (hf = real CLIP+Whisper),
   scene, disk. Optionally restore a local checkpoint (mind-only).
4. **Find a GPU**: filter (model / count / $hr / RAM / verified), **Search**,
   then **Rent** a row.
5. Watch the provisioning stepper + live log. When it reaches **ready**, click
   **Watch agent** — the Overview panels render the remote agent (PC-loss should
   trend down once encoders warm up).
6. **Stop** pauses GPU billing (keeps disk); **Destroy** checkpoints the agent,
   copies it back, then terminates the instance (stops all billing).

## The scripts (run by `decadic/api/vast/controller.py` over ssh)

- `setup_remote.sh` — apt EGL/MuJoCo libs; `pip install -e . --no-deps` then the
  non-torch runtime deps (preserves the image's CUDA torch); prewarms CLIP +
  Whisper when `ENCODER=hf`. Reads optional env `ENCODER`, `WHISPER_MODEL`.
- `run_remote.sh` — starts `uvicorn decadic.api.app:app` on `0.0.0.0:8765` inside
  a detached `tmux` session named `decadic`, with `DECADIC_DEVICE=cuda` and
  `MUJOCO_GL=egl` so the body the supervisor spawns renders headless. Reads
  optional env `PRESET`, `ENCODER`, `WHISPER_MODEL`. The scene/body is started
  afterwards via `POST /environment` through the tunnel.

Default image: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`.

## Cost & safety notes

- Provisioning auto-destroys the instance if it enters a bad status
  (`exited`/`offline`/`unknown`) or if any step fails — so a broken rent does not
  keep billing.
- The API key is stored 0600, masked in API responses, and never logged.
- The control endpoints are unauthenticated like the rest of the app; keep the
  server bound to `127.0.0.1`.
