"""Interpretability probes: decode the opaque State Bus latents into readable
variables. Supervision is strictly eval-only (oracle/eval truth), and probes are
applied purely read-only - their weights never enter the cognitive optimizer."""
