"""Falsification + integration-measurement metrics for the self-model program.

These are diagnostics, not part of the live cognitive cycle: every phase that
adds a feedback pathway (self-state spine, global workspace, temporal window,
predictive affect) must *prove* it raises integration rather than merely
relabelling outputs. The perturbational-complexity proxy lives here so the
before/after harness can score off-vs-on builds.
"""

from decadic.metrics.integration import (
    IntegrationResult,
    SELF_STATE_KEYS,
    integration_delta,
    lz76_complexity,
    perturbational_complexity,
    self_state_vector,
)

__all__ = [
    "IntegrationResult",
    "SELF_STATE_KEYS",
    "integration_delta",
    "lz76_complexity",
    "perturbational_complexity",
    "self_state_vector",
]
