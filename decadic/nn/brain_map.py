"""Topology export of the NeuralCognitiveStack for the dashboard Brain Map.

Pure read-only walk over the stack's modules: emits one node per pipeline
block (with unit/parameter counts) and the edges between them, including the
top-k strongest individual weights so the client can draw real "fibers".
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from decadic.nn.plastic import PlasticSparseGrowableMLP

# Hard cap on individually-exported weights across the whole map; everything
# else is summarized per-edge (count + RMS). Keeps the payload ~100 KB.
MAX_FIBERS_TOTAL = 1200


def _first_weight(module: nn.Module) -> torch.Tensor | None:
    """A 2D (out x in) weight matrix representing the block's input mapping."""
    if isinstance(module, PlasticSparseGrowableMLP):
        # Effective first-layer mapping: pruned/dormant connections read as 0.
        with torch.no_grad():
            return module.l1_weight * module.mask1 * module.awake.unsqueeze(1)
    if isinstance(module, nn.Linear):
        return module.weight
    if isinstance(module, nn.GRUCell):
        return module.weight_ih[: module.hidden_size]
    if isinstance(module, nn.LSTMCell):
        return module.weight_ih[: module.hidden_size]
    if isinstance(module, nn.TransformerEncoder):
        first = module.layers[0]
        d = first.self_attn.embed_dim
        return first.self_attn.in_proj_weight[:d]
    if isinstance(module, nn.Sequential):
        for sub in module:
            if isinstance(sub, nn.Linear):
                return sub.weight
    return None


def _block_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def brain_topology(stack: nn.Module, *, preset: str | None = None) -> dict[str, Any]:
    """Layers, edges, and totals of a NeuralCognitiveStack (JSON-shaped)."""
    cfg = stack.cfg
    d = cfg.d_model

    # (id, attribute, label, decadic stage, output units)
    layer_defs: list[tuple[str, str, str, int, int]] = [
        ("ingress", "ingress", "Sensory ingress", 1, d),
        ("stage1", "stage1", "Perception MLP", 1, d),
        ("stage2", "stage2", f"Experience framing ×{cfg.transformer_layers}", 2, d),
        ("epi_proj", "epi_proj", "Episodic proxy", 3, d),
        ("mem_proj", "mem_proj", "Memory context", 3, d),
        ("stage3", "stage3", "Memory fusion", 3, d),
        ("risk_mlp", "risk_mlp", "Risk-utility", 4, d),
        ("stage5_enc", "stage5_enc", f"Narrative encoder ×{cfg.encoder_decoder_layers}", 5, d),
        (
            "stage5_dec",
            "stage5_dec",
            f"Strategy decoder ×{max(1, cfg.encoder_decoder_layers // 2)}",
            8,
            d,
        ),
        ("gru_cell", "gru_cell", "Emotion GRU", 6, cfg.gru_hidden),
        ("emotion_head", "emotion_head", "Emotion head", 6, cfg.emotion_out),
        ("lstm_cell", "lstm_cell", "State-of-mind LSTM", 7, cfg.lstm_hidden),
        ("state_mind_head", "state_mind_head", "State-of-mind head", 7, cfg.state_mind_out),
        ("narrative_head", "narrative_head", "Narrative head", 8, cfg.narrative_out),
        ("metacog_head", "metacog_head", "Metacognition head", 8, cfg.metacog_out),
        ("policy", "policy", "Policy", 9, 4),
        ("pc_heads", "pc_heads", "Predictive-coding heads ×4", 10, d),
    ]

    layers: list[dict[str, Any]] = []
    modules: dict[str, nn.Module] = {}
    for lid, attr, label, stage, units in layer_defs:
        mod = getattr(stack, attr)
        modules[lid] = mod
        layers.append(
            {
                "id": lid,
                "label": label,
                "stage": stage,
                "units": int(units),
                "params": int(_block_params(mod)),
            }
        )

    # (src, dst, col_start, col_end) — which input columns of dst's first
    # weight matrix the src block feeds (cat order from forward()).
    edge_defs: list[tuple[str, str, int, int]] = [
        ("ingress", "stage1", 0, d),
        ("stage1", "stage2", 0, d),
        ("stage2", "stage3", 0, d),
        ("epi_proj", "stage3", d, 2 * d),
        ("mem_proj", "stage3", 2 * d, 3 * d),
        ("stage3", "risk_mlp", 0, d),
        ("risk_mlp", "stage5_enc", 0, d),
        ("stage5_enc", "stage5_dec", 0, d),
        ("stage5_dec", "gru_cell", 0, d),
        ("risk_mlp", "gru_cell", d, 2 * d),
        ("gru_cell", "emotion_head", 0, cfg.gru_hidden),
        ("stage5_dec", "lstm_cell", 0, d),
        ("emotion_head", "lstm_cell", d, d + cfg.emotion_out),
        ("lstm_cell", "state_mind_head", 0, cfg.lstm_hidden),
        ("stage5_dec", "narrative_head", 0, d),
        ("stage5_enc", "metacog_head", 0, d),
        ("lstm_cell", "policy", 0, cfg.lstm_hidden),
        ("state_mind_head", "policy", cfg.lstm_hidden, cfg.lstm_hidden + cfg.state_mind_out),
        ("stage2", "pc_heads", 0, d),
    ]

    units_by_id = {layer["id"]: layer["units"] for layer in layers}
    per_edge_k = max(8, MAX_FIBERS_TOTAL // len(edge_defs))
    edges: list[dict[str, Any]] = []
    with torch.no_grad():
        for src, dst, c0, c1 in edge_defs:
            w = _first_weight(modules[dst])
            if w is None:
                continue
            c1 = min(c1, w.shape[1])
            if c0 >= c1:
                continue
            sub = w[:, c0:c1].detach().float()
            k = min(per_edge_k, sub.numel())
            flat = sub.flatten()
            _, top_idx = torch.topk(flat.abs(), k)
            signed = flat[top_idx].cpu().tolist()
            cols = sub.shape[1]
            # si/di are indices into the source/destination *clusters*; blocks
            # whose first weight is an internal hidden layer (e.g. policy) get
            # their rows folded onto the displayed output units.
            src_units = max(1, units_by_id[src])
            dst_units = max(1, units_by_id[dst])
            fibers = [
                {
                    "si": int((i % cols) % src_units),
                    "di": int((i // cols) % dst_units),
                    "w": round(float(v), 5),
                }
                for i, v in zip(top_idx.cpu().tolist(), signed)
            ]
            edges.append(
                {
                    "src": src,
                    "dst": dst,
                    "weight_count": int(sub.numel()),
                    "w_rms": round(float(sub.pow(2).mean().sqrt()), 6),
                    "fibers": fibers,
                }
            )

    weights = sum(p.numel() for p in stack.parameters() if p.dim() >= 2)
    total_params = sum(p.numel() for p in stack.parameters())
    # True unit count: every Linear/recurrent output across the whole stack,
    # including hidden layers inside blocks (clusters display block outputs).
    neurons = 0
    for mod in stack.modules():
        if isinstance(mod, PlasticSparseGrowableMLP):
            # Count only awake hidden neurons (dormant ones aren't "grown" yet)
            # plus the block's output units.
            neurons += mod.awake_count() + mod.out_features
        elif isinstance(mod, nn.Linear):
            neurons += mod.out_features
        elif isinstance(mod, (nn.GRUCell, nn.LSTMCell)):
            neurons += mod.hidden_size
    totals: dict[str, Any] = {
        "neurons": int(neurons),
        "connections": int(weights),
        "params": int(total_params),
        "d_model": int(d),
        "preset": preset,
    }
    if getattr(stack, "has_plastic", False):
        totals["awake_neurons"] = int(stack.awake_neurons())
        totals["allocated_neurons"] = int(stack.allocated_neurons())
        totals["active_connections"] = int(stack.active_connections())
    return {
        "layers": layers,
        "edges": edges,
        "totals": totals,
    }
