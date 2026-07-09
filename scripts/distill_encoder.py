"""WS-DEPTH P3.1 — encoder liberation, stage 1: offline distillation (REAL).

Distills the frozen inherited vision path into a trainable student on the
agent's OWN recorded frames, so experience can eventually reshape perception.
Offline job (GPU + a frame archive from probe captures / vision dumps); never
on the hot path. The SWAP (P3.2, DECADIC_ENCODER_MODE=student) remains gated
on the go/no-go printed at the end + the standing percept-key invariance
canary. Rollback is one env var.

Teacher: the project's own FrozenSensoryEncoders forward() over each frame
(base64-wrapped exactly like a live observation, so the student learns the
embedding the brain actually consumes). Student: a small conv net -> the same
embedding width. Go/no-go: held-out cosine similarity (student vs teacher) —
the proxy for "the agent's memories stay findable through the student's eyes."

Usage (rig session):
  .venv\\Scripts\\python.exe scripts\\distill_encoder.py --frames <dir_of_pngs>
      --out saved_agents\\student_encoder.pt [--epochs 10] [--device cuda]
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="P3.1 offline encoder distillation")
    ap.add_argument("--frames", required=True, help="directory of PNG/JPG frames (the agent's own life)")
    ap.add_argument("--out", required=True, help="output path for the student state_dict")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--go_threshold", type=float, default=0.98, help="held-out cosine for GO")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    from decadic.nn.frozen_encoders import FrozenSensoryEncoders

    frames = sorted(
        p for p in Path(args.frames).glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if len(frames) < 32:
        print(f"[distill] need >=32 frames, found {len(frames)} in {args.frames}")
        return 2
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    teacher = FrozenSensoryEncoders().to(device).eval()

    def teacher_embed(path: Path) -> "torch.Tensor | None":
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        with torch.no_grad():
            out = teacher({"vision_b64": b64})
        return out.detach().reshape(-1) if out is not None else None

    print(f"[distill] embedding {len(frames)} frames through the teacher...")
    embeds, keep = [], []
    for p in frames:
        e = teacher_embed(p)
        if e is not None and torch.isfinite(e).all():
            embeds.append(e.cpu())
            keep.append(p)
    if len(embeds) < 32:
        print("[distill] teacher produced too few valid embeddings")
        return 2
    dim = embeds[0].numel()

    # Student: small conv trunk -> teacher-width embedding. Deliberately
    # modest — P3's point is a RESHAPEABLE substrate, not a bigger CLIP.
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("[distill] pillow required (pip install pillow)")
        return 2

    def load_img(p: Path) -> "torch.Tensor":
        img = Image.open(p).convert("RGB").resize((224, 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    student = nn.Sequential(
        nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
        nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.GELU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
        nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GELU(),
        nn.AdaptiveAvgPool2d(4), nn.Flatten(),
        nn.Linear(128 * 16, 512), nn.GELU(), nn.Linear(512, dim),
    ).to(device)

    n_hold = max(8, int(len(keep) * args.holdout))
    train_idx = list(range(len(keep) - n_hold))
    hold_idx = list(range(len(keep) - n_hold, len(keep)))
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    import random as rnd

    rng = rnd.Random(7)
    for epoch in range(args.epochs):
        rng.shuffle(train_idx)
        tot, nb = 0.0, 0
        for i in range(0, len(train_idx), args.batch):
            batch = train_idx[i : i + args.batch]
            x = torch.stack([load_img(keep[j]) for j in batch]).to(device)
            t = torch.stack([embeds[j] for j in batch]).to(device)
            opt.zero_grad()
            pred = student(x)
            # KD: cosine + MSE — direction matters most for retrieval keys.
            loss = (1.0 - torch.nn.functional.cosine_similarity(pred, t).mean()) + 0.1 * torch.nn.functional.mse_loss(pred, t)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        print(f"[distill] epoch {epoch + 1}/{args.epochs} loss {tot / max(1, nb):.5f}")

    with torch.no_grad():
        x = torch.stack([load_img(keep[j]) for j in hold_idx]).to(device)
        t = torch.stack([embeds[j] for j in hold_idx]).to(device)
        cos = float(torch.nn.functional.cosine_similarity(student(x), t).mean())
    torch.save({"student": student.state_dict(), "dim": dim, "holdout_cos": cos}, args.out)
    verdict = "GO" if cos >= args.go_threshold else "NO-GO"
    print(
        f"[distill] held-out cosine {cos:.4f} (threshold {args.go_threshold}) -> {verdict}\n"
        f"[distill] saved {args.out}. P3.2 swap additionally requires the "
        f"percept-key invariance canary green on a live probe."
    )
    return 0 if cos >= args.go_threshold else 1


if __name__ == "__main__":
    sys.exit(main())
