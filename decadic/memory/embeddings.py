"""Fixed-size vectors for episodic similarity and retrieval context (Phase 3)."""

from __future__ import annotations

import numpy as np

from decadic.state.state_bus import StateBus

# Episode fingerprint layout: internal-state vectors (narrative/emotion/metacog,
# plus z5 for stored rows) followed by a fixed-width *perceptual key*. The
# perceptual tail lets memories be recalled by sensory likeness (Loop 2): a
# never-seen bear, sitting near a cat in the learned percept space, retrieves the
# cat's affect-laden episode. The key is a parameter-free compression of the
# learned percept z0, so the similarity reflects CLIP/z0's *learned* geometry,
# not any hand-coded feature. When the perception-feedback loop is off, callers
# pass key=None and the tail is zeros: cosine ranking is identical to the pure
# internal-state embedding (appending equal zeros to both sides is a no-op), so
# parity holds. Pre-upgrade rows of the old length are skipped by search_similar.
PERCEPT_KEY_DIM = 16
EMBEDDING_DIM = 64 + PERCEPT_KEY_DIM


def _pad(vec: np.ndarray, target: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size >= target:
        return v[:target].astype(np.float32, copy=False)
    out = np.zeros(target, dtype=np.float32)
    out[: v.size] = v
    return out


def perceptual_key(percept: np.ndarray | None, dim: int = PERCEPT_KEY_DIM) -> np.ndarray:
    """Parameter-free L2-normalized compression of a percept latent (z0) to ``dim``.

    The (variable-length, preset-dependent) percept is folded into ``dim`` buckets
    by chunk means, then unit-normalized so its contribution to cosine similarity
    is bounded and preset-independent. Returns zeros for a missing/empty percept.
    """
    out = np.zeros(dim, dtype=np.float32)
    if percept is None:
        return out
    v = np.asarray(percept, dtype=np.float32).reshape(-1)
    n = v.size
    if n == 0 or dim <= 0:
        return out
    counts = np.zeros(dim, dtype=np.float32)
    idx = np.minimum(dim - 1, (np.arange(n) * dim) // n)
    np.add.at(out, idx, v)
    np.add.at(counts, idx, 1.0)
    out = np.where(counts > 0, out / np.maximum(counts, 1.0), 0.0).astype(np.float32)
    norm = float(np.linalg.norm(out))
    if norm > 1e-8:
        out = out / norm
    return out


def query_vector_from_state_bus(sb: StateBus, percept: np.ndarray | None = None) -> np.ndarray:
    """Pre-forward retrieval query from prior-cycle bus vectors + current percept key."""
    n = sb.narrative_emb.astype(np.float32, copy=False)
    e = sb.emotion_physio.astype(np.float32, copy=False)
    m = sb.metacognition.astype(np.float32, copy=False)
    parts = [_pad(n, 24), _pad(e, 24), _pad(m, 16), perceptual_key(percept)]
    return np.concatenate(parts)


def episode_embedding_from_cycle(
    sb: StateBus, z5: np.ndarray | None, percept: np.ndarray | None = None
) -> np.ndarray:
    """Post-forward episode fingerprint (stored with each episodic row)."""
    z5 = np.asarray(z5, dtype=np.float32).reshape(-1)
    z5h = _pad(z5, 16)
    n = sb.narrative_emb.astype(np.float32, copy=False)
    e = sb.emotion_physio.astype(np.float32, copy=False)
    m = sb.metacognition.astype(np.float32, copy=False)
    parts = [_pad(n, 16), _pad(e, 16), _pad(m, 16), z5h, perceptual_key(percept)]
    return np.concatenate(parts)
