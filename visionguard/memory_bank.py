from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MAX_GREEDY_CANDIDATES = 20_000
MAX_GREEDY_DIMENSIONS = 128


@dataclass
class MemoryBank:
    features: np.ndarray
    sample_ratio: float = 0.01
    random_seed: int = 42
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float32)
        if self.features.ndim != 2 or self.features.shape[0] == 0:
            raise ValueError("Memory bank features must be a non-empty 2D array")
        if not np.isfinite(self.features).all():
            raise ValueError("Memory bank features must all be finite")
        if not 0 < self.sample_ratio <= 1:
            raise ValueError("sample_ratio must be in the interval (0, 1]")
        self.rng = np.random.default_rng(self.random_seed)

    def coreset_subsample(self, ratio: Optional[float] = None) -> np.ndarray:
        """Greedily cover a seeded candidate pool; return the original feature vectors.

        Large banks are capped to a deterministic random candidate pool and projected
        to at most 128 dimensions for selection. This bounds coreset-selection cost;
        nearest-neighbour inference still uses the full original feature vectors.
        """
        ratio = self.sample_ratio if ratio is None else ratio
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in the interval (0, 1]")
        target_count = min(
            self.features.shape[0],
            max(1, int(np.ceil(self.features.shape[0] * ratio))),
        )
        if target_count == self.features.shape[0]:
            return self.features.copy()

        if self.features.shape[0] > MAX_GREEDY_CANDIDATES:
            candidate_indices = self.rng.choice(
                self.features.shape[0],
                size=max(MAX_GREEDY_CANDIDATES, target_count),
                replace=False,
            )
        else:
            candidate_indices = np.arange(self.features.shape[0])
        candidates = self.features[candidate_indices]
        count = min(target_count, candidates.shape[0])
        if candidates.shape[1] > MAX_GREEDY_DIMENSIONS:
            projection = self.rng.standard_normal(
                (candidates.shape[1], MAX_GREEDY_DIMENSIONS),
                dtype=np.float32,
            ) / np.sqrt(MAX_GREEDY_DIMENSIONS)
            selection_features = candidates @ projection
        else:
            selection_features = candidates

        selected = np.empty(count, dtype=np.int64)
        selected_mask = np.zeros(candidates.shape[0], dtype=bool)
        selected[0] = self.rng.integers(candidates.shape[0])
        selected_mask[selected[0]] = True
        nearest_squared = np.full(candidates.shape[0], np.inf, dtype=np.float32)

        for index in range(count):
            center = selection_features[selected[index]]
            for start in range(0, candidates.shape[0], 4096):
                stop = min(start + 4096, candidates.shape[0])
                delta = selection_features[start:stop] - center
                squared = np.einsum("ij,ij->i", delta, delta)
                nearest_squared[start:stop] = np.minimum(nearest_squared[start:stop], squared)
            nearest_squared[selected_mask] = -1
            if index + 1 < count:
                selected[index + 1] = int(np.argmax(nearest_squared))
                selected_mask[selected[index + 1]] = True
        return candidates[selected]

    def nearest_distances(
        self,
        query: np.ndarray,
        query_chunk_size: int = 256,
        bank_chunk_size: int = 2048,
    ) -> np.ndarray:
        """Return exact nearest-neighbour distances using bounded distance tiles."""
        if query.ndim == 1:
            query = query[None, :]
        query = np.asarray(query, dtype=np.float32)
        if query.ndim != 2 or query.shape[1] != self.features.shape[1]:
            raise ValueError("Query features must be 2D and match the memory-bank feature dimension")
        if query_chunk_size < 1 or bank_chunk_size < 1:
            raise ValueError("Chunk sizes must be positive")

        nearest = np.full(query.shape[0], np.inf, dtype=np.float32)
        for q_start in range(0, query.shape[0], query_chunk_size):
            q = query[q_start:q_start + query_chunk_size]
            q_squared = np.einsum("ij,ij->i", q, q)[:, None]
            tile_min = np.full(q.shape[0], np.inf, dtype=np.float32)
            for b_start in range(0, self.features.shape[0], bank_chunk_size):
                bank = self.features[b_start:b_start + bank_chunk_size]
                bank_squared = np.einsum("ij,ij->i", bank, bank)[None, :]
                squared = q_squared + bank_squared - 2 * q @ bank.T
                tile_min = np.minimum(tile_min, np.maximum(squared, 0).min(axis=1))
            nearest[q_start:q_start + q.shape[0]] = np.sqrt(tile_min)
        return nearest

    def image_score(self, query: np.ndarray) -> float:
        dists = self.nearest_distances(query)
        return float(np.max(dists)) if dists.size else 0.0
