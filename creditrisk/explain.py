"""Model-agnostic local explanations by occlusion.

For each raw input we replace its value with the training reference (median) and
measure how much the predicted probability of default changes. Positive values
push the risk UP for this applicant. It ignores interactions, so treat it as a
sensitivity view, not an exact decomposition (use SHAP for that).
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def reference_values(train_df: pd.DataFrame) -> dict:
    """Training medians of the raw features (NaNs ignored)."""
    return {c: float(train_df[c].median()) for c in C.RAW_FEATURES}


def occlusion_contributions(model, row: pd.DataFrame, reference: dict) -> pd.Series:
    """Return per-feature change in PD when that feature is set to its reference."""
    base = float(model.predict_proba(row[C.RAW_FEATURES])[0, 1])
    deltas = {}
    for col in C.RAW_FEATURES:
        alt = row[C.RAW_FEATURES].copy()
        alt[col] = reference[col]
        deltas[col] = base - float(model.predict_proba(alt)[0, 1])
    return pd.Series(deltas).sort_values(key=abs, ascending=False)
