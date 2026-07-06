# Implementation Plan & WBS — Remote Deploy Connect (Vast.ai SSH lifecycle)

**Companion:** `vast_connect_gap_analysis.md`. Dependency-ordered. ⚙ = needs a
live Vast account / rented box to validate. **Unblock = M0→M2** (the SSH key
lifecycle); M3–M5 are GUI, verification, and hardening.

**Guiding principle:** do not rebuild the deploy system — it exists. Add the one
missing capability (generate + register an SSH key with Vast) and gate the flow
on a preflight so failures surface *before* renting, not mid-deploy.

---

## Interfaces to add (frozen surface)

New backend module `decadic/api/vast/ssh_keys.py`:

```python
class SshKeyManager:
    def __init__(self, store: VastSettingsStore, cli: VastCli) -> None: ...
    def ensure_local_key(self) -> str:            # returns private-key path; generates ed25519 if none
    def public_key(self) -> str | None:           # contents of "<path>.pub", or None
    def fingerprint(self) -> str | None:          # ssh-keygen -lf, safe to show in UI
    async def is_registered(self) -> bool:        # pub key present in `vastai show ssh-keys`
    async def ensure_registered(self) -> dict:    # idempotent create ssh-key; returns {registered, fingerprint}
    async def attach_to_instance(self, instance_id: int) -> None
```

New CLI wrappers in `decadic/api/vast/cli.py`:

```python
async def create_ssh_key(self, public_key: str) -> None       # vastai create ssh-key "<pub>" -y
async def show_ssh_keys(self) -> list[dict]                    # vastai show ssh-keys --raw
async def attach_ssh(self, instance_id: int, public_key: str) -> None  # vastai attach ssh <id> "<pub>"
```

New REST (in `routes.py`):

```
GET  /vast/preflight        -> {api_key, cli, ssh_binary, ssh_key_present, ssh_key_registered, ready, reasons[]}
POST /vast/ssh/setup        -> generate (if needed) + register; returns {fingerprint, registered} (never the private key)
```

No changes to the frozen deploy phase contract other than inserting one step.

---

## M0 — Preflight (read-only; ship first)

Blocks Deploy with actionable reasons instead of failing after billing starts.

- **M0.1 SSH toolchain probes.** `ssh_keys._which_ssh()`, `_which_keygen()` via
  `shutil.which`. *Accept:* unit test with a monkeypatched PATH toggles each.
- **M0.2 `GET /vast/preflight`.** Aggregate: `has_api_key` (store),
  `cli_available` (cli), `ssh_binary` (M0.1), `ssh_key_present`
  (`store.has_ssh_key_path()`), `ssh_key_registered`
  (`SshKeyManager.is_registered()`), plus a `ready` bool and a `reasons` list of
  one-line fixes for each false. *Accept:* `tests/test_vast_preflight.py` — each
  false branch yields its reason; all-true ⇒ `ready=True`.
- **M0.3 GUI preflight strip.** `VastCredentials.tsx` renders the five checks as
  pass/fail chips; `DeploymentPanel` disables **Deploy** unless `ready`. *Accept:*
  Deploy is un-clickable until green; each red chip shows its fix text.

*Milestone accept:* no deploy attempt can reach the SSH step without a green
preflight.

---

## M1 — Key generation + registration (backend)

- **M1.1 CLI wrappers.** `create_ssh_key`, `show_ssh_keys`, `attach_ssh` (see
  surface). Reuse `_run_json` / `_run` + `_key_args`; `-y` on create. *Accept:*
  `tests/test_vast_ssh_keys.py` asserts argv construction + JSON parse against a
  fake `_run`.
- **M1.2 `ensure_local_key()`.** If `store.get_ssh_key_path()` is empty or the
  file is missing, generate `~/.decadic/ssh/id_ed25519` via
  `ssh-keygen -t ed25519 -N "" -f <path>`, `chmod 0600` (POSIX; no-op Windows),
  store the path. Idempotent: an existing valid key is reused. *Accept:* generate
  creates both halves at 0600; second call is a no-op and returns the same path.
- **M1.3 `ensure_registered()`.** Read `<path>.pub`; compare against
  `show_ssh_keys()` (match on key body / fingerprint); if absent call
  `create_ssh_key(pub)`. Return `{registered: True, fingerprint}`. *Accept:*
  first call registers (create invoked once); second call is a no-op (create not
  invoked); reconcile matches on fingerprint not raw string whitespace.
- **M1.4 `POST /vast/ssh/setup`.** Calls `ensure_local_key()` then
  `ensure_registered()`; returns `{fingerprint, registered}`. **Never returns the
  private key.** *Accept:* route test asserts response has no private material;
  masked path only.

*Milestone accept:* from only an API key, one backend call yields a registered
key whose fingerprint appears in `show ssh-keys`.

---

## M2 — Wire registration into deploy

- **M2.1 `_step_ensure_ssh`.** New first step in `controller._provision`, before
  `_step_create`: `SshKeyManager.ensure_local_key()` +
  `ensure_registered()`; on failure raise `VastCliError` with the M0 fix text.
  *Accept:* `tests/test_vast_controller_scene.py`-style step-order test shows
  ensure_ssh runs before create.
- **M2.2 Per-instance attach.** In `_step_create`, immediately after
  `create_instance` returns the id, call `cli.attach_ssh(iid, pub)` (per-instance
  guarantee independent of account-key propagation lag). *Accept:* attach called
  with the new id and the configured pub key.
- **M2.3 Error mapping.** An SSH-auth failure during `_step_upload` is caught and
  re-raised as "SSH not authorized — run SSH setup (see preflight)", pointing
  back to M0. *Accept:* simulated `Permission denied` upload surfaces the mapped
  message, not a raw stderr dump.

*Milestone accept:* ⚙ a fresh deploy SSHes into the box on the first attempt; the
`Permission denied (publickey)` failure mode is eliminated.

---

## M3 — GUI: one-button SSH setup + status

- **M3.1 "Set up SSH" action.** Button in `VastCredentials.tsx` → `POST
  /vast/ssh/setup`; on success show a **registered** badge + fingerprint.
  *Accept:* first-time operator with only an API key reaches green preflight
  without a terminal.
- **M3.2 Status + gating.** Registered/not badge; `vastApi.ts` gains
  `getPreflight()` and `setupSsh()`; Deploy stays disabled until `ready`.
  *Accept:* badge reflects live state; Deploy enables exactly when preflight goes
  green.
- **M3.3 Copy/explainer.** Short `explainers.ts` entry: what's stored where, that
  the private key never leaves the machine, and that registration is one-time.

*Milestone accept:* zero-to-green SSH setup is a single click.

---

## M4 — Connect-and-watch verification (⚙)

- **M4.1 Live loop check.** On a real deployment, confirm the proxied dashboard
  renders remote `/agent/{id}/metrics`, vision frames, and environment views;
  record a short clip as the artifact. *Accept:* remote `cycles_completed`
  advances in the local dashboard; the MuJoCo world renders.
- **M4.2 Stream-key field.** Add an optional stream-key input to `RunConfig.tsx`
  that flows into the deploy env as `DECADIC_YT_STREAM_KEY` (the `run_remote.sh`
  passthrough already exists), so the remote body can go live. *Accept:* setting
  it makes the box publish to YouTube; leaving it blank is a no-op.
- **M4.3 Runbook.** Append a "deploy & watch from the GUI" section to
  `deploy/vast/README.md` (or a new `docs/` note).

---

## M5 — E2E hardening

- **M5.1 Preflight check in the harness.** `vast_e2e_test.py` gains an
  `ssh_registered` PASS/FAIL check in its preflight block (fail fast, no rent).
  *Accept:* run with an unregistered key fails at preflight and rents nothing.
- **M5.2 Full green run.** ⚙ rent → deploy → remote model runs → watch → destroy,
  including the SSH gate. *Accept:* `VAST_E2E: PASS` with the new check present.

---

## (Deferred) M6 — Adopt an externally-rented instance

`POST /vast/adopt {instance_id}`: attach the key, open the tunnel, mark ready —
no new rental. Only if the manual-console workflow matters. *Accept:* a
hand-rented box becomes watchable in the dashboard.

---

## Test plan

- **New:** `tests/test_vast_ssh_keys.py` (M1.1–M1.3), `tests/test_vast_preflight.py`
  (M0.2), controller step-order + attach (extend `test_vast_controller_scene.py`
  for M2.1–M2.2), route no-leak test for `/vast/ssh/setup` (M1.4).
- **Discipline:** all SSH/CLI calls mocked at the `_run`/`_run_json` seam — no
  network, no real keygen writes outside `tmp_path`. Live paths (⚙) are covered
  only by the e2e (M5.2) and M4.1.
- **Parity:** `/vast/*` stays local-only (proxy already excludes it); no change to
  the agent/environment contract.

## Config / secrets

- Private key: `~/.decadic/ssh/id_ed25519` (0600), path in the existing settings
  store. Public key registered with Vast. **Never logged, never returned by any
  route.** Reuse the store's masking discipline.
- No new env flags required; the stream-key path reuses `DECADIC_YT_STREAM_KEY`.

## Dependency graph

```
M0.1 -> M0.2 -> M0.3
M1.1 -> M1.2 -> M1.3 -> M1.4
(M0.2, M1.3) -> M2.1 -> M2.2 -> M2.3
(M1.4, M0.3) -> M3.1 -> M3.2 -> M3.3
M2 -> M4.1 ; M4.2 (parallel) ; M4.3
M2 -> M5.1 -> M5.2
M2 -> [M6, deferred]
```

## Sequencing

Land **M0 + M1 + M2** together — that is the entire unblock and is fully
unit-testable offline. Ship **M3** next (turns it into one click). **M4/M5** are
the live validation once a deploy actually connects. **M6** only if needed.

## Effort shape (not calendar)

M0 small · M1 small–medium (keygen + reconcile) · M2 small · M3 small (one
button + badge + gate) · M4 verification-only · M5 one check + a live run. The
critical path is M1.2/M1.3 (correct idempotent registration); everything else is
plumbing already patterned in the codebase.
