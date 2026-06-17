"""Narrative synthesis - turn the structured Cognitive Trace into prose.

Tier C of the monitoring stack. ``template`` mode (default) renders deterministic
English directly from the structured record - faithful, dependency-free, and the
same every time for the same trace. ``lm`` mode (opt-in via DECADIC_NARRATIVE_MODE)
asks a small frozen language model to gloss the same record; it is explicitly a
non-authoritative paraphrase and falls back to the template if the model is
unavailable. ``off`` produces nothing.

The output is written to ``state_bus.narrative_text_stub`` and into the trace's
``narrative`` field. It is a read-out only and never feeds back into cognition.
"""

from __future__ import annotations

from typing import Any


def _affect_phrase(affect: dict[str, Any]) -> str:
    pain = float(affect.get("pain") or 0.0)
    pleasure = float(affect.get("pleasure") or 0.0)
    risk = float(affect.get("risk") or 0.0)
    if pain >= 0.66:
        feel = "in strong distress"
    elif pain >= 0.33:
        feel = "uncomfortable"
    elif pain > 0.1:
        feel = "mildly uneasy"
    elif pleasure >= 0.33:
        feel = "content"
    else:
        feel = "calm"
    stance = "wary" if risk < 0.42 else "exploring"
    return f"{feel} and {stance}"


def _template(trace: dict[str, Any]) -> str:
    parts: list[str] = []

    intent = (trace.get("intent") or {}).get("summary") or ""
    if intent:
        parts.append(intent[0].upper() + intent[1:] + ".")

    salient = trace.get("salient") or {}
    node = salient.get("node") if isinstance(salient, dict) else None
    if isinstance(node, dict) and node.get("node_id"):
        kind = node.get("kind")
        label = f"{node['node_id']}" + (f" ({kind})" if kind else "")
        parts.append(f"Attending to {label}.")
    fractions = salient.get("fractions") if isinstance(salient, dict) else None
    if isinstance(fractions, dict) and fractions:
        driver = max(fractions, key=lambda k: fractions[k])
        if fractions[driver] >= 0.5:
            parts.append(f"The choice was driven mostly by {driver}.")

    affect = trace.get("affect") or {}
    parts.append("Feeling " + _affect_phrase(affect) + ".")

    # Self-model spine: when the self-state feedback loop is closed, the report is
    # OF the fed-back content -- describe how steadily the agent's sense of self
    # is carried forward (continuity = cosine of this cycle's self-report against
    # the one it was conditioned on).
    self_model = trace.get("self_model") or {}
    if isinstance(self_model, dict) and self_model.get("active"):
        cont = self_model.get("continuity")
        if isinstance(cont, (int, float)):
            steadiness = (
                "steady" if cont >= 0.6 else "shifting" if cont >= 0.2 else "unsettled"
            )
            parts.append(
                f"Its sense of self is {steadiness}, carried forward from the last moment."
            )
        else:
            parts.append("Carrying its prior state of mind forward (self-model engaged).")

    surprise = trace.get("self_surprise") or {}
    mar = surprise.get("mean_abs_residual")
    if isinstance(mar, (int, float)) and mar >= 0.25:
        dims = surprise.get("dims") or []
        worst = dims[0]["name"] if dims else "the body"
        parts.append(f"The body moved unexpectedly ({worst} off by {dims[0]['residual'] if dims else mar}).")

    ep = trace.get("recalled_episode")
    if isinstance(ep, dict) and ep.get("cycle") is not None:
        bits = [f"resembles cycle {ep['cycle']}"]
        if ep.get("priority"):
            bits.append(f"priority {ep['priority']}")
        pain = ep.get("pain")
        if isinstance(pain, (int, float)):
            bits.append(f"pain {round(float(pain), 2)}")
        parts.append("This " + ", ".join(bits) + ".")

    probes = trace.get("probes")
    if isinstance(probes, dict) and probes:
        good = [
            (t, v) for t, v in probes.items()
            if isinstance(v, dict) and float(v.get("score") or 0.0) >= 0.5
        ]
        if good:
            t, v = max(good, key=lambda kv: float(kv[1].get("score") or 0.0))
            parts.append(
                f"Probe read-out: {v.get('best_latent')} encodes {t} "
                f"({v.get('score_kind')} {v.get('score')})."
            )

    return " ".join(parts).strip()


def _lm_render(trace: dict[str, Any]) -> str:
    """Optional frozen-LM gloss. Best-effort; falls back to the template on any
    failure. Explicitly a non-authoritative paraphrase of the structured record."""
    try:  # pragma: no cover - exercised only when transformers is installed
        from transformers import pipeline  # type: ignore

        global _LM_PIPE
        try:
            _LM_PIPE  # type: ignore[used-before-def]
        except NameError:
            _LM_PIPE = pipeline("text2text-generation", model="google/flan-t5-small")  # type: ignore
        base = _template(trace)
        prompt = (
            "Rephrase this agent self-report as one short, plain sentence "
            "(do not add facts): " + base
        )
        out = _LM_PIPE(prompt, max_new_tokens=48)  # type: ignore
        text = (out[0].get("generated_text") or "").strip() if out else ""
        return text or base
    except Exception:
        return ""


def render(trace: dict[str, Any] | None, mode: str = "template") -> str:
    """Render the trace to prose for the given mode (off | template | lm)."""
    if not trace or mode == "off":
        return ""
    if mode == "lm":
        return _lm_render(trace) or _template(trace)
    return _template(trace)
