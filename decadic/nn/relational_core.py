"""WS5-M3.1: the relational core -- where relations become computable.

A deliberately small pre-norm transformer over the cycle's token set:
[K WM slot tokens ; k recalled-episode tokens ; 1 interoceptive token].
Attention between tokens is the mechanism that computes entity-entity (and
entity-memory, entity-body) relations; the masked-mean pooled summary
augments the stage-4 risk input through a zero-init ingress in the stack.

Notes:
- dropout=0 everywhere: the parity culture requires determinism.
- Permutation-equivariant by design: "wolf behind rock" differs from "rock
  behind wolf" in the tokens' CONTENT (spatial features), not their order.
- Sizing (layers/heads/width) is env-tunable pending M3.2's measured cycle
  cost on the full preset; defaults are the small end (2 layers, 2 heads).
"""

from __future__ import annotations

import os

import torch
from torch import nn

from decadic.memory.embeddings import EMBEDDING_DIM
from decadic.state.working_memory import SLOT_TENSOR_DIM

INTERO_DIM = 4  # the episodic affect proxy the forward already receives


def relational_layers() -> int:
    return max(1, int(os.environ.get("DECADIC_RELATIONAL_LAYERS", "2")))


def relational_heads() -> int:
    return max(1, int(os.environ.get("DECADIC_RELATIONAL_HEADS", "2")))


class RelationalCore(nn.Module):
    def __init__(self, d_rel: int = 32) -> None:
        super().__init__()
        heads = relational_heads()
        # d_rel must divide by heads; round up to the next multiple.
        self.d_rel = int(d_rel + (-d_rel) % heads)
        self.proj_slot = nn.Linear(SLOT_TENSOR_DIM, self.d_rel)
        self.proj_mem = nn.Linear(EMBEDDING_DIM, self.d_rel)
        self.proj_intero = nn.Linear(INTERO_DIM, self.d_rel)
        self.type_emb = nn.Parameter(torch.zeros(3, self.d_rel))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_rel,
            nhead=heads,
            dim_feedforward=2 * self.d_rel,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: the fast-path optimization is unused
        # with norm_first anyway (torch warns), and our token counts are
        # single-digit -- determinism over micro-optimization.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=relational_layers(), enable_nested_tensor=False
        )

    def forward(
        self,
        slots: torch.Tensor | None,  # (K, SLOT_TENSOR_DIM) or None
        slots_mask: torch.Tensor | None,  # (K,) bool
        mem_tokens: torch.Tensor | None,  # (k, EMBEDDING_DIM) or None
        mem_mask: torch.Tensor | None,  # (k,) bool
        intero: torch.Tensor,  # (1, INTERO_DIM)
    ) -> torch.Tensor:  # (1, d_rel) pooled relational summary
        dev = intero.device
        dt = intero.dtype
        toks = []
        valid = []
        if slots is not None and slots_mask is not None:
            s = slots.to(device=dev, dtype=dt).reshape(-1, SLOT_TENSOR_DIM)
            toks.append(self.proj_slot(s) + self.type_emb[0])
            valid.append(slots_mask.to(dev).reshape(-1).bool())
        if mem_tokens is not None and mem_mask is not None:
            t = mem_tokens.to(device=dev, dtype=dt).reshape(-1, EMBEDDING_DIM)
            toks.append(self.proj_mem(t) + self.type_emb[1])
            valid.append(mem_mask.to(dev).reshape(-1).bool())
        # The interoceptive token is always present: relations are computed
        # in light of how the body currently is (never an empty token set).
        toks.append(self.proj_intero(intero.reshape(1, INTERO_DIM)) + self.type_emb[2])
        valid.append(torch.ones(1, dtype=torch.bool, device=dev))

        x = torch.cat(toks, dim=0).unsqueeze(0)  # (1, T, d_rel)
        m = torch.cat(valid, dim=0)  # (T,)
        enc = self.encoder(x, src_key_padding_mask=(~m).unsqueeze(0))  # (1, T, d)
        # Masked mean pool over valid tokens only.
        mw = m.to(dt).reshape(1, -1, 1)
        summed = (enc * mw).sum(dim=1)
        return summed / mw.sum(dim=1).clamp_min(1.0)
