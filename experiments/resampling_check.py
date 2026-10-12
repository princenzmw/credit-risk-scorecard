"""Does rebalancing help? Compare no resampling, class weights and (optionally) SMOTE.

Run from the repo root:  python experiments/resampling_check.py
SMOTE needs `pip install imbalanced-learn`; if it is missing those rows are skipped,
so only report SMOTE results you actually produced on your machine.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from creditrisk import config as C  # noqa: E402
from creditrisk.data import load_raw, split_data, xy  # noqa: E402
from creditrisk.features import CreditFeatureBuilder  # noqa: E402
from creditrisk.metrics import evaluate  # noqa: E402
from creditrisk.models import make_gbm, make_logistic  # noqa: E402


def _smote_variants():
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError:
        print("imbalanced-learn not installed: skipping SMOTE rows.")
        return {}
    smote = SMOTE(random_state=C.RANDOM_STATE)
    return {
        "Logistic + SMOTE": ImbPipeline([
            ("features", CreditFeatureBuilder()), ("scale", StandardScaler()), ("smote", smote),
            ("clf", LogisticRegression(max_iter=2000, random_state=C.RANDOM_STATE))]),
        "Gradient boosting + SMOTE": ImbPipeline([
            ("features", CreditFeatureBuilder()), ("smote", smote),
            ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=15,
                                                    min_samples_leaf=50, l2_regularization=1.0,
                                                    random_state=C.RANDOM_STATE))]),
    }


def main() -> None:
    train, val, _ = split_data(load_raw())
    X_tr, y_tr = xy(train)
    X_va, y_va = xy(val)
    variants = {
        "Logistic, no resampling": make_logistic(None),
        "Logistic, class_weight=balanced": make_logistic("balanced"),
        "Gradient boosting, no resampling": make_gbm(None),
        "Gradient boosting, class_weight=balanced": make_gbm("balanced"),
    }
    variants.update(_smote_variants())
    rows = []
    for name, model in variants.items():
        model.fit(X_tr, y_tr)
        ev = evaluate(y_va, model.predict_proba(X_va)[:, 1])
        rows.append({"variant": name, **{k: round(v, 4) for k, v in ev.items() if k != "base_rate"}})
        print(rows[-1])
    out = pd.DataFrame(rows)
    path = C.REPORTS_DIR / "resampling_check.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
