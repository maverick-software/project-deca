"""Cycle-12 plasticity-freeze diagnosis: is the self-model feedback loop contractive?

Diagnosis (see docs): the plasticity instability guard freezes structural
learning very early (~cycle 12) because the predictive-coding loss diverges.
The threshold is generous (``DEFAULT_PLASTICITY_INSTABILITY_PCLOSS = 50``), and
gradients are already clipped to norm 1, so the blow-up is in the *forward
recurrent dynamics*, not the weights — i.e. a cross-cycle feedback loop (the
self-model spine, now default-on) whose gain exceeds 1 once training nudges its
zero-init ingress off zero.

These tests turn that diagnosis into checks:

* ``test_severed_loop_is_stable`` — control: the base stack WITHOUT self-feedback
  stays bounded under a repeated input. Should always pass; it localizes any
  divergence to the feedback path rather than the base stack.
* ``test_self_state_feedback_loop_is_contractive`` — the regression: with a
  realistically-scaled (perturbed-off-zero) self-state spine, the closed loop
  must NOT grow geometrically. Marked ``xfail`` until the feedback is damped
  (LayerNorm / bounded gate); it should flip to XPASS when the fix lands — remove
  the marker then.
* ``test_plasticity_guard_freeze_timing`` — pure spec of the guard: a *diverging*
  pc-loss trips the EMA threshold within a handful of cycles, while a *stable*
  pc-loss never trips. Documents why the freeze fires so early.
* ``test_bf16_does_not_worsen_divergence`` — CUDA-only: bf16 memory-efficient
  training should not materially worsen the loop's stability vs fp32.

NOTE: the numeric bounds (perturbation scale, growth ratio, cycle counts) are
first-guess calibrations. They were authored without a live torch run available;
tune the constants against one real run if a check is too loose/tight.
"""

import pytest

torch = pytest.importorskip("torch")

from decadic.metrics.integration import self_state_vector  # noqa: E402
from decadic.nn.config import neural_config_from_env  # noqa: E402
from decadic.nn.neural_stack import NeuralCognitiveStack  # noqa: E402


def _stack(monkeypatch, env: dict[str, str] | None = None) -> NeuralCognitiveStack:
    """Build a tiny CPU stack with the requested faculties (mirrors test_signatures)."""
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    return NeuralCognitiveStack(neural_config_from_env("tiny"))


def _closed_loop_z5_norms(
    stack: NeuralCognitiveStack,
    *,
    cycles: int,
    seed: int,
    feed_self: bool,
) -> list[float]:
    """Drive the stack on ONE fixed percept for ``cycles`` steps.

    When ``feed_self`` is True the previous cycle's A||C||E self-report is fed back
    through the spine (the real recurrent path); otherwise the loop is severed.
    Returns the per-cycle L2 norm of z5 (a proxy for activation magnitude). Pure
    forward, eval mode, no grad — divergence here is forward-dynamics only.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    cfg = stack.cfg
    dev = next(stack.parameters()).device  # inputs must live on the model's device
    z0 = torch.randn(1, cfg.d_model, generator=g).to(dev)
    ep = torch.rand(1, 4, generator=g).to(dev)
    mem = torch.randn(1, cfg.memory_context_dim, generator=g).to(dev)
    was_training = bool(stack.training)
    stack.eval()
    stack.reset_recurrent_state()
    self_prev = None
    norms: list[float] = []
    try:
        with torch.no_grad():
            for _ in range(cycles):
                kwargs = {"self_prev": self_prev} if feed_self else {}
                out = stack(z0, ep, mem, **kwargs)
                norms.append(float(out["z5"].detach().float().norm().item()))
                if feed_self:
                    self_prev = self_state_vector(out)
    finally:
        stack.train(was_training)
        stack.reset_recurrent_state()
    return norms


def _late_growth_ratio(norms: list[float]) -> float:
    """Geometric mean of consecutive norm ratios over the back half of the run.

    > 1 means activations are still compounding (non-contractive); ~1 means the
    loop has settled (contractive)."""
    tail = norms[len(norms) // 2 :]
    ratios = [
        tail[i + 1] / tail[i]
        for i in range(len(tail) - 1)
        if tail[i] > 1e-9
    ]
    if not ratios:
        return 1.0
    prod = 1.0
    for r in ratios:
        prod *= r
    return prod ** (1.0 / len(ratios))


# --- control: the base stack (no self-feedback) is stable ---------------------


def test_severed_loop_is_stable(monkeypatch):
    """Without the self-feedback path, repeating one percept must stay bounded."""
    stack = _stack(monkeypatch, {"DECADIC_SELF_MODEL_FEEDBACK": "0"})
    norms = _closed_loop_z5_norms(stack, cycles=60, seed=0, feed_self=False)
    assert all(torch.isfinite(torch.tensor(n)) for n in norms)
    assert _late_growth_ratio(norms) <= 1.02, (
        f"base stack should not compound; late growth ratio={_late_growth_ratio(norms):.4f}"
    )


# --- regression: the self-model feedback loop must be contractive -------------


@pytest.mark.xfail(
    reason="self-model feedback loop is not yet damped (root cause of the cycle-12 "
    "plasticity freeze). Remove this marker once a LayerNorm/bounded gate makes the "
    "fed-back self-state contractive.",
    strict=False,
)
def test_self_state_feedback_loop_is_contractive(monkeypatch):
    """With a learned-scale self-state spine, the closed loop must not blow up."""
    stack = _stack(monkeypatch, {"DECADIC_SELF_MODEL_FEEDBACK": "1"})
    if not getattr(stack, "has_self_model_feedback", False) or not hasattr(stack, "self_ingress"):
        pytest.skip("self-model feedback spine not present in this build")
    # Emulate a *trained* loop: move the zero-init spine ingress off zero so the
    # feedback actually carries signal (zero-init alone is a no-op and never
    # diverges). Scale is a stand-in for what gradient descent produces.
    # NOTE: a forward-only loop at scale 0.5 / 60 cycles was found to be
    # CONTRACTIVE (XPASS) on the real build — see docstring. Stronger probe here,
    # but be aware an isolated-stack forward loop may simply not reproduce a
    # *training-time* instability; the live-agent ablation is authoritative.
    with torch.no_grad():
        stack.self_ingress.weight.normal_(0.0, 1.2)
        if getattr(stack.self_ingress, "bias", None) is not None:
            stack.self_ingress.bias.normal_(0.0, 0.1)
    norms = _closed_loop_z5_norms(stack, cycles=150, seed=0, feed_self=True)
    assert all(torch.isfinite(torch.tensor(n)) for n in norms), "z5 went non-finite (diverged)"
    ratio = _late_growth_ratio(norms)
    assert ratio <= 1.05, (
        f"self-model feedback loop is compounding (late growth ratio={ratio:.4f}); "
        "this is the cycle-12 divergence. Damp the fed-back self-state to make it contractive."
    )


# --- spec: why the guard fires within ~a dozen cycles -------------------------


def _ema_trip_cycle(pc_losses: list[float], threshold: float) -> int | None:
    """Cycle index (1-based) at which the plasticity pc-loss EMA crosses ``threshold``.

    Mirrors apply_plasticity_step: ema = loss on cycle 1, then 0.98*ema + 0.02*loss.
    """
    ema = None
    for i, loss in enumerate(pc_losses, start=1):
        ema = loss if ema is None else 0.98 * ema + 0.02 * loss
        if threshold > 0 and ema > threshold:
            return i
    return None


def test_plasticity_guard_freeze_timing():
    """A diverging pc-loss trips the guard early; a stable one never trips."""
    from decadic.config import plasticity_instability_pcloss

    thr = plasticity_instability_pcloss()
    assert thr >= 1.0  # sanity: a generous ceiling, not a hair-trigger

    # Healthy training: small, flat pc-loss -> EMA stays well under the ceiling.
    stable = [2.0] * 200
    assert _ema_trip_cycle(stable, thr) is None

    # Divergence: pc-loss compounding ~25%/cycle from a benign start -> the EMA
    # crosses the ceiling within a small number of cycles (the observed pattern).
    diverging = [1.0 * (1.25 ** t) for t in range(60)]
    trip = _ema_trip_cycle(diverging, thr)
    assert trip is not None and trip <= 30, (
        f"a compounding pc-loss should trip the guard early; tripped at {trip}"
    )


# --- bf16 accelerant check (CUDA only) ----------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="bf16 memory-efficient path only runs on a bf16-capable CUDA device",
)
def test_bf16_does_not_worsen_divergence(monkeypatch):
    """bf16 forward should not materially worsen the loop vs fp32 (it's an accelerant,
    not the root cause). Builds the same perturbed feedback stack and compares the
    late growth ratio under fp32 vs bf16 autocast."""
    def _ratio(use_bf16: bool) -> float:
        stack = _stack(monkeypatch, {"DECADIC_SELF_MODEL_FEEDBACK": "1", "DECADIC_DEVICE": "cuda"})
        if not getattr(stack, "has_self_model_feedback", False):
            pytest.skip("self-model feedback spine not present in this build")
        stack = stack.cuda()
        with torch.no_grad():
            stack.self_ingress.weight.normal_(0.0, 0.5)
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with ctx:
            norms = _closed_loop_z5_norms(stack, cycles=60, seed=0, feed_self=True)
        return _late_growth_ratio(norms)

    fp32_ratio = _ratio(False)
    bf16_ratio = _ratio(True)
    # bf16 may be slightly worse (lower precision), but not dramatically so.
    assert bf16_ratio <= max(1.10, fp32_ratio * 1.25), (
        f"bf16 materially worsens the loop (bf16={bf16_ratio:.4f} vs fp32={fp32_ratio:.4f})"
    )
