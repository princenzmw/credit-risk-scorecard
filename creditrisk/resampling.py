"""Minimal SMOTE so the resampling experiment needs no extra dependency.

SMOTE creates synthetic minority examples by picking a minority point, one of its k nearest minority
neighbours, and a random point on the line between them.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_oversample(X, y, k: int = 5, seed: int = 42):
    """Return (X_balanced, y_balanced) with the minority class (y == 1) oversampled to parity."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    minority = X[y == 1]
    n_new = int((y == 0).sum() - (y == 1).sum())
    if n_new <= 0 or len(minority) <= k:
        return X, y
    rng = np.random.default_rng(seed)
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(minority).kneighbors(
        minority, return_distance=False)[:, 1:]
    base = rng.integers(0, len(minority), n_new)
    chosen = neighbours[base, rng.integers(0, k, n_new)]
    gap = rng.random((n_new, 1))
    synthetic = minority[base] + gap * (minority[chosen] - minority[base])
    return np.vstack([X, synthetic]), np.concatenate([y, np.ones(n_new, dtype=int)])
