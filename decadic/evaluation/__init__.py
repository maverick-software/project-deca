"""Training evaluation harness.

Evaluation code is read-only with respect to the Decadic cognition loop. It
consumes metrics, discovery reports, dojo status, and probe banks to produce
repeatable learning-health and competence reports.
"""

from decadic.evaluation.runner import build_report, load_eval_spec
from decadic.evaluation.types import EvalReport, EvalSample, EvalSpec

__all__ = ["EvalReport", "EvalSample", "EvalSpec", "build_report", "load_eval_spec"]

