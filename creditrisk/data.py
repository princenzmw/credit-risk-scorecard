"""Loading, validating and splitting the raw data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def validate_schema(df: pd.DataFrame, require_target: bool = False) -> None:
    """Raise ValueError if required columns are missing."""
    required = list(C.RAW_FEATURES) + ([C.TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_raw(path: str | Path = C.DATA_PATH) -> pd.DataFrame:
    """Read cs-training.csv, drop the unnamed index column, validate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for download instructions."
        )
    df = pd.read_csv(path)
    if C.INDEX_COL in df.columns:
        df = df.drop(columns=[C.INDEX_COL])
    validate_schema(df, require_target=True)
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, seed: int = C.RANDOM_STATE):
    """Stratified train / validation / test split (60/20/20)."""
    train_val, test = train_test_split(
        df, test_size=C.TEST_SIZE, random_state=seed, stratify=df[C.TARGET]
    )
    rel_val = C.VAL_SIZE / (1.0 - C.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=seed, stratify=train_val[C.TARGET]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def xy(df: pd.DataFrame):
    """Return (features, target)."""
    return df[C.RAW_FEATURES].copy(), df[C.TARGET].astype(int)
