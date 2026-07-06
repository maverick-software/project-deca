# Gap Analysis — Remote Deploy & Connect-and-Watch (Vast.ai)

**Date:** 2026-07-06 · **Goal:** from the web GUI, rent a GPU box, deploy the
Deca stack onto it, and watch the agent learn live in the same dashboard — with
a one-time, self-service SSH setup.

## TL;DR

The deploy system already exists and is coherent end to end. A single missing
capability — the **SSH key lifecycle** (generate a keypair and register its
public half with Vast) — blocks the entire chain, because every provisioning
step authenticates over SSH. Close that one gap and the existing pipeline,
tunnel, proxy, and dashboard light up. Everything else here is preflight,
verification, and polish.

---

## 1. Desired end state

1. Operator opens the dashboard, enters their Vast API key, and clicks **one**
   button to set up SSH (generate + register) if they haven't already.
2. Operator searches offers (host-quality floors already applied), picks one,
   sets a run config, and clicks **Deploy**.
3. The box is provisioned automatically; a tunnel opens; the dashboard's normal
   agent panels begin rendering the **remote** agent — metrics, vision, the
   MuJoCo world — with no URL change.
4. Optionally, the remote body streams its spectator camera to YouTube (already
   built; just needs a stream key in the run config).
5. One button tears the deployment down and stops billing.

---

## 2. What already exists (inventory)

Most of the target is built. Verified in-repo:

| Capability | Where | State |
|---|---|---|
| Deploy state machine: create → wait_running → upload → install → serve → tunnel → start_agent → ready; **auto-destroy on failure** | `decadic/api/vast/controller.py` `_provision` (220–247) | Complete |
| Vast CLI wrapper: `create_instance`, `show_instance`, `ssh_url`, `copy`, `destroy_instance`, `stop_instance`, `search_offers`, `show_user` | `decadic/api/vast/cli.py` | Complete |
| Raw SSH exec + `ssh -N -L` tunnel | `cli.py` `ssh_exec` (301), `open_tunnel` (325) | Complete |
| Reverse proxy: forwards `/agents`, `/agent/*`, `/environment*` to the tunnel so existing panels show the remote agent | `decadic/api/vast/proxy.py` | Complete |
| REST surface: `/vast/settings`, `/vast/account`, `/vast/offers`, `/vast/deploy`, `/vast/deployment(/destroy)`, `/vast/gpu-names`, `/vast/browse-fs` | `decadic/api/vast/routes.py` | Complete |
| Settings store: API key + **ssh_key_path** + deploy defaults, 0600, masked reads | `decadic/api/vast/settings_store.py` | Complete |
| Remote provisioning scripts (EGL + ffmpeg libs; server in tmux with `MUJOCO_GL=egl`) | `deploy/vast/setup_remote.sh`, `run_remote.sh` | Complete |
| Dashboard: credentials, GPU search, run config, deploy progress, active deployment, SSH-key file picker | `dashboard/src/components/vast/*`, `DeploymentPanel.tsx`, `vastApi.ts` | Complete |
| Host-quality offer floors (reliability/bandwidth/CPU/disk) | `routes.py` `_build_offer_filter` | Complete (2026-07-06) |
| Live spectator stream (RTMP → YouTube), deployment-agnostic | `decadic/embodiment/stream_publisher.py` + adapter | Complete |

**Implication:** the request to "build out tunneling, API, deploy" is ~90%
already met. The work is to finish the one missing lifecycle and verify the
loop, not to build a deploy system.

---

## 3. Gaps (delta to the end state)

### G1 — SSH key lifecycle *(critical blocker)*
The settings store holds a path to a **private** key and the controller uses it
for every SSH/scp/tunnel call (`controller.py` 305, 333, 410). But nothing:
- **generates** a keypair if the operator has none, and
- **registers the public key with Vast**, which is what makes a rented instance
  authorize the connection.

Vast attaches SSH keys at the account or instance level; the CLI provides
(confirmed against docs.vast.ai):
`vastai create ssh-key "<public_key>"`, `vastai show ssh-keys`,
`vastai attach ssh <instance_id> "<public_key>"`, `delete/update ssh-key`.

Without registration, `create_instance --ssh` succeeds, the box boots, and then
`_step_upload`'s SSH connection is refused (`Permission denied (publickey)`),
the deploy fails, and the instance auto-destroys — presenting exactly as "I
rented a box but nothing reaches my GUI." **This is the blocker.**

### G2 — SSH toolchain preflight (Windows)
Provisioning shells out to `ssh` (and Vast `copy` uses scp/rsync). On Windows
these come from the OpenSSH client optional feature. If absent, deploy fails
opaquely. No preflight checks for them.

### G3 — Deploy preflight surfaced in the GUI
Today the only gate is `has_api_key` + `cli_available` (`vast_e2e_test.py`
preflight; the GUI credentials panel). There is no check that (a) an SSH key is
configured, (b) that key is **registered with Vast**, (c) `ssh` exists. The
deploy therefore fails mid-flight instead of being blocked up front with an
actionable reason.

### G4 — Connect-and-watch verification
The plumbing to render the remote agent exists (proxy + dashboard panels), but
it hasn't been validated end to end on a live deployment: remote metrics,
remote vision frames, environment views, and the live-window path. The YouTube
stream on the box needs only a stream-key field in the run config to be wired
through (env passthrough already added in `run_remote.sh`).

### G5 — Key security & UX polish
Enforce 0600 on generated keys (POSIX) / document the Windows ACL story; show a
key fingerprint (never the private key); a "Generate & register" button; clear
copy about what is stored where.

### G6 — Adopt an externally-rented instance *(optional, lower priority)*
The controller only manages instances it created. A box rented by hand in the
Vast console (e.g. 44002020) can't be adopted into the tunnel/proxy flow. Nice
to have; not required for the primary GUI-driven path.

---

## 4. Implementation plan

Dependency-ordered milestones. G1 is the unblock; do M0–M2 first.

**M0 — Preflight (read-only, ship first).** Add a `/vast/preflight` check (and
surface in the credentials panel): `ssh`/`ssh-keygen` present, SSH key
configured, key registered with Vast (`show ssh-keys` reconcile), API key, CLI.
Each returns pass/fail + a one-line fix. *Accept:* the panel blocks Deploy with
named reasons; no deploy attempt can reach SSH without these green.

**M1 — Key generation + registration backend.** CLI wrappers
`create_ssh_key(pub)`, `show_ssh_keys()`, `attach_ssh(instance_id, pub)`. A
`SshKeyManager`: if no key configured, generate `id_ed25519` under
`~/.decadic/ssh/` (0600), store the private path; read the `.pub`; register it
with Vast idempotently (skip if already in `show ssh-keys`). *Accept:* unit
tests for generate/idempotent-register/reconcile; `show ssh-keys` contains the
fingerprint after one call; re-run is a no-op.

**M2 — Wire registration into deploy.** In `_step_create` (or a new
`_step_ensure_ssh` before it): ensure the configured key is account-registered,
and `attach ssh <new_instance_id> <pub>` right after create for a per-instance
guarantee. *Accept:* a fresh deploy SSHes into the box on the first try; the
`Permission denied` failure mode is gone.

**M3 — GUI: generate + status + gating.** In `VastCredentials.tsx` /
`DeploymentPanel.tsx`: a "Set up SSH (generate & register)" button, a
registered/not-registered badge with fingerprint, and Deploy disabled until
preflight is green. *Accept:* a first-time operator with only an API key can go
from zero to a green preflight without touching a terminal.

**M4 — Connect-and-watch verification + stream field.** Validate on a live
deployment that the proxied dashboard shows remote metrics + vision + views;
add a stream-key field to `RunConfig.tsx` that flows into the deploy env
(`DECADIC_YT_STREAM_KEY`) so the remote body streams. *Accept:* a screen-recorded
run showing the remote agent's cycles advancing and its world rendering in the
local dashboard; (optional) the YouTube stream live from the box.

**M5 — E2E hardening.** Extend `vast_e2e_test.py` preflight to assert
ssh-key-registered; add a `check` for it. *Accept:* a full green e2e including
the SSH-registration gate, rent→deploy→watch→destroy.

**(Deferred) M6 — Adopt existing instance.** An `adopt <instance_id>` path that
attaches the key, opens the tunnel, and marks the deployment ready without
creating a new box. Only if the manual-rent workflow matters.

---

## 5. Work breakdown

- **M0.1** `has_ssh_binary()` / `has_keygen()` probes (shutil.which). *Test:* mocked PATH.
- **M0.2** `/vast/preflight` route aggregating api_key, cli, ssh_bin, ssh_key_present, ssh_key_registered. *Test:* each false path returns its reason.
- **M0.3** Credentials-panel preflight strip; Deploy button gated on all-green.
- **M1.1** `cli.create_ssh_key`, `cli.show_ssh_keys`, `cli.attach_ssh`. *Test:* arg construction + JSON parse.
- **M1.2** `SshKeyManager.ensure_local_key()` → generate ed25519 if absent, 0600, store path. *Test:* generate + reuse.
- **M1.3** `SshKeyManager.ensure_registered()` → read `.pub`, reconcile against `show ssh-keys`, `create ssh-key` if missing (idempotent). *Test:* first-call registers, second is no-op.
- **M2.1** `_step_ensure_ssh` before `_step_create`; `attach ssh` after create. *Test:* controller step order; attach called with new id.
- **M2.2** Clear error mapping: an SSH-auth failure at upload points the operator back to M0 preflight.
- **M3.1** "Generate & register SSH" button + `POST /vast/ssh/setup`. *Test:* returns fingerprint, not the private key.
- **M3.2** Registered badge + fingerprint + gating in `DeploymentPanel`.
- **M4.1** Live verification checklist (metrics, vision, views through the proxy).
- **M4.2** `RunConfig` stream-key field → deploy env → `run_remote.sh` passthrough (already present).
- **M5.1** `vast_e2e_test` `ssh_registered` preflight check.

---

## 6. Risks & notes

- **Registration propagation.** Account-level `create ssh-key` may take a moment
  to propagate; the per-instance `attach ssh` in M2 is the reliable guarantee —
  do both.
- **Windows perms.** `os.chmod(0600)` is a no-op on Windows; rely on the user
  profile ACL and document it. Never print or return the private key.
- **scp/rsync.** Vast `copy` needs a working scp/rsync over the same key; covered
  by the M0 ssh-binary probe but worth a live check in M4.
- **Host-key churn.** Already handled (`StrictHostKeyChecking=no`,
  `UserKnownHostsFile=/dev/null`) in `cli.py`.
- **Billing.** Idle rented boxes bill; the failure path already auto-destroys,
  and the persistent (success) path is intentionally kept alive for watching —
  the GUI must keep the destroy button prominent.

---

## 7. Bottom line

One real feature stands between you and a GUI-driven, watchable remote agent:
**self-service SSH key generation + Vast registration** (G1 → M0–M2). It's a
small, well-bounded backend addition plus a button. The deploy pipeline,
tunnel, proxy, dashboard, offer filtering, and live stream are already in place.
