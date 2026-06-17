"""Vast.ai GPU deployment control plane (UI-driven).

Lets the dashboard store a Vast.ai API key, search for GPU offers, rent an
instance, ship the codebase, run the Decadic brain + headless MuJoCo body on
CUDA, tunnel the remote API back, and reverse-proxy agent traffic so the
existing panels show the remote agent learning live -- all without a terminal.

Modules:
- ``settings_store``: persistent, perm-restricted key + deploy defaults.
- ``cli``: async wrappers around the ``vastai`` CLI + ssh helpers.
- ``controller``: deployment state machine + background provisioning task.
- ``proxy``: reverse proxy to the tunneled remote when a deployment is active.
- ``routes``: the ``/vast/*`` FastAPI routes (mounted by ``decadic.api.app``).
"""

from __future__ import annotations

__all__ = ["settings_store", "cli", "controller", "proxy", "routes"]
