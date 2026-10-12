"""Evaluation helpers for imbalanced credit-risk models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from . import config as C


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov statistic: max separation between default / non-default score CDFs."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_score) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "base_rate": float(np.mean(y_true)),
    }


def calibration_table(y_true, y_score, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed default rate per score decile."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    out = df.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean"))
    return out.reset_index()


def confusion_at(y_true, y_score, threshold: float) -> dict:
    """Counts when scores >= threshold are declined (flagged as default)."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "precision": precision,
        "decline_rate": (tp + fp) / len(y),
    }


def cost_threshold(y_true, y_score, cost_fn: float = C.COST_FN, cost_fp: float = C.COST_FP,
                   grid: np.ndarray | None = None) -> tuple[float, float]:
    """Return (threshold, cost) minimising cost_fn*FN + cost_fp*FP on the given data."""
    grid = np.linspace(0.01, 0.80, 80) if grid is None else grid
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_score)
    best_t, best_c = float(grid[0]), float("inf")
    for t in grid:
        pred = p >= t
        fn = int(((~pred) & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_c:
            best_t, best_c = float(t), float(cost)
    return best_t, best_c


def assign_band(p) -> pd.Series:
    """Map probabilities of default to the configured risk bands."""
    return pd.cut(pd.Series(np.asarray(p, dtype=float)), bins=C.PD_BAND_EDGES,
                  labels=C.PD_BAND_LABELS, right=False, include_lowest=True)


def band_table(y_true, y_score) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true), "band": assign_band(y_score).values})
    out = df.groupby("band", observed=False).agg(loans=("y", "size"), observed_default_rate=("y", "mean"))
    return out.reset_index()
