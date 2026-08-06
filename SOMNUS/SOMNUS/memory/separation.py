"""Dentate-gyrus pattern separation.

Two similar-but-distinct incidents must not collide in the episodic store.
We decorrelate on write via a fixed random projection followed by top-k
sparsification -- the standard computational account of dentate gyrus, and
about fifteen lines of code.
"""

from __future__ import annotations

import numpy as np

_SEED = 20260806


class PatternSeparator:
    """Fixed random projection + k-winners-take-all sparsification."""

    def __init__(self, dimension: int, sparsity: float = 0.05, expansion: int = 4) -> None:
        """Sparse EXPANSION then competition.

        A same-size projection plus k-winners leaves near-identical inputs
        near-identical -- the winner set does not change. Dentate gyrus
        decorrelates because it expands into a much larger, sparser population,
        where small input differences flip which units win.
        """
        self.dimension = dimension
        self.expanded = dimension * expansion
        self.k = max(1, int(self.expanded * sparsity))
        rng = np.random.default_rng(_SEED)
        self._projection = rng.standard_normal((self.expanded, dimension)) / np.sqrt(dimension)

    def separate(self, vector: list[float]) -> list[float]:
        v = np.asarray(vector, dtype=float)
        if v.shape[0] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {v.shape[0]}")

        projected = self._projection @ v
        out = np.zeros_like(projected)
        winners = np.argpartition(np.abs(projected), -self.k)[-self.k :]
        out[winners] = projected[winners]

        norm = np.linalg.norm(out)
        if norm > 0:
            out /= norm
        return out.tolist()
