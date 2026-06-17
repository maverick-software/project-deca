"""Object-centric Slot Attention over the egocentric feature map.

This is how the agent *discovers* objects from its own camera instead of being
handed an oracle entity list. Frozen CLIP gives a grid of patch features; Slot
Attention (Locatello et al., 2020) lets K slots compete to explain those
features, so each slot binds to a coherent region (an object proposal). A small
spatial-broadcast decoder reconstructs the feature map from the slots, giving a
self-supervised objective (DINOSAUR-style feature reconstruction - no pixel
labels, no oracle) and per-slot soft masks we read out as image-space position.

Built only in discovered perception mode; absent (and serialized away) otherwise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """K slots iteratively attend over N patch features; decoded back for self-supervision."""

    def __init__(
        self,
        *,
        in_dim: int,
        n_patches: int,
        k: int = 7,
        slot_dim: int = 64,
        iters: int = 3,
        mlp_hidden: int = 128,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_patches = n_patches
        self.grid = max(1, int(round(math.sqrt(n_patches))))
        self.k = k
        self.slot_dim = slot_dim
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.norm_in = nn.LayerNorm(in_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_mlp = nn.LayerNorm(slot_dim)

        self.to_k = nn.Linear(in_dim, slot_dim)
        self.to_v = nn.Linear(in_dim, slot_dim)
        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, slot_dim),
        )

        # Learned slot initialization (shared mean/log-sigma; slots are sampled
        # i.i.d. so the K of them are exchangeable, as Slot Attention requires).
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.1)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Presence: how strongly a slot claims real content (vs empty background).
        self.presence_head = nn.Linear(slot_dim, 1)

        # Spatial-broadcast decoder: each slot is broadcast across the grid (plus a
        # learned positional code) and decoded to a per-position feature + alpha;
        # alphas are softmaxed across slots to mix the reconstruction.
        self.pos_emb = nn.Parameter(torch.randn(1, 1, n_patches, slot_dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, in_dim + 1),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """features: [B, N, in_dim] patch tokens. Returns slots, masks, presence, recon."""
        b, n, _ = features.shape
        f = self.norm_in(features)
        k = self.to_k(f)
        v = self.to_v(f)

        mu = self.slot_mu.expand(b, self.k, -1)
        sigma = F.softplus(self.slot_log_sigma).expand(b, self.k, -1)
        slots = mu + sigma * torch.randn_like(mu)

        attn = torch.zeros(b, self.k, n, device=features.device, dtype=features.dtype)
        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))
            # dots: [B, N, K] -> softmax over slots (slots compete per position).
            dots = torch.einsum("bnd,bkd->bnk", k, q) * self.scale
            attn_pos = dots.softmax(dim=-1) + self.eps  # [B, N, K]
            # Normalize over positions so each slot's updates are a weighted mean.
            weights = attn_pos / attn_pos.sum(dim=1, keepdim=True)
            updates = torch.einsum("bnk,bnd->bkd", weights, v)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(b, self.k, self.slot_dim)
            slots = slots + self.mlp(self.norm_pre_mlp(slots))
            attn = attn_pos.permute(0, 2, 1)  # [B, K, N]

        presence = torch.sigmoid(self.presence_head(slots)).squeeze(-1)  # [B, K]

        # Decode: broadcast each slot across the grid, add positional code.
        slots_b = slots.unsqueeze(2).expand(b, self.k, n, self.slot_dim) + self.pos_emb
        dec = self.decoder(slots_b)  # [B, K, N, in_dim+1]
        feat_k = dec[..., : self.in_dim]  # [B, K, N, in_dim]
        alpha = dec[..., self.in_dim :]  # [B, K, N, 1]
        masks = alpha.softmax(dim=1)  # mix across slots -> [B, K, N, 1]
        recon = (feat_k * masks).sum(dim=1)  # [B, N, in_dim]
        recon_masks = masks.squeeze(-1)  # [B, K, N]

        return {
            "slots": slots,  # [B, K, slot_dim]
            "attn": attn,  # [B, K, N] attention masks
            "masks": recon_masks,  # [B, K, N] reconstruction assignment
            "presence": presence,  # [B, K]
            "recon": recon,  # [B, N, in_dim]
        }

    def centroids(self, masks: torch.Tensor) -> torch.Tensor:
        """Image-space (u, v) centroid in [0, 1] per slot from a [B, K, N] mask.

        u is horizontal (column), v vertical (row); origin top-left. Returns
        [B, K, 3] = (u, v, spread) where spread is the mask's normalized extent.
        """
        b, kk, n = masks.shape
        g = self.grid
        device = masks.device
        rows = (torch.arange(n, device=device) // g).float() + 0.5
        cols = (torch.arange(n, device=device) % g).float() + 0.5
        u = cols / g
        v = rows / g
        w = masks / (masks.sum(dim=-1, keepdim=True) + self.eps)  # [B, K, N]
        cu = (w * u).sum(dim=-1)  # [B, K]
        cv = (w * v).sum(dim=-1)
        var = (w * ((u - cu.unsqueeze(-1)) ** 2 + (v - cv.unsqueeze(-1)) ** 2)).sum(dim=-1)
        spread = var.clamp_min(0.0).sqrt()
        return torch.stack([cu, cv, spread], dim=-1)  # [B, K, 3]
