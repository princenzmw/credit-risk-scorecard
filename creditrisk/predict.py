"""Load saved artefacts and score applicants."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from . import config as C
from .data import validate_schema
from .metrics import assign_band


def load_artifacts(models_dir: str | Path = C.MODELS_DIR) -> dict:
    models_dir = Path(models_dir)
    return {
        "model": joblib.load(models_dir / "model.joblib"),
        "threshold": json.loads((models_dir / "threshold.json").read_text()),
        "reference": json.loads((models_dir / "reference.json").read_text()),
        "metrics": json.loads((models_dir / "metrics.json").read_text()),
    }


def score_dataframe(model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return probability of default, risk band and decision for each row."""
    validate_schema(df)
    pd_hat = model.predict_proba(df[C.RAW_FEATURES])[:, 1]
    out = pd.DataFrame({"pd": pd_hat})
    out["band"] = assign_band(pd_hat).astype(str).values
    out["decision"] = ["Decline / review" if p >= threshold else "Approve" for p in pd_hat]
    return out
