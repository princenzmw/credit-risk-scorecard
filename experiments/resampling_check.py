"""Does rebalancing help? Compare no resampling, class weights and SMOTE on the validation set.

Run from the repo root:  python experiments/resampling_check.py
SMOTE uses the small implementation in creditrisk/resampling.py. If imbalanced-learn is installed
(pip install imbalanced-learn) its SMOTE is run too as a cross-check.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from creditrisk import config as C  # noqa: E402
from creditrisk.data import load_raw, split_data, xy  # noqa: E402
from creditrisk.features import CreditFeatureBuilder  # noqa: E402
from creditrisk.metrics import evaluate  # noqa: E402
from creditrisk.models import make_gbm, make_logistic  # noqa: E402
from creditrisk.resampling import smote_oversample  # noqa: E402


def fit_with_smote(estimator, scale: bool, X_tr: pd.DataFrame, y_tr: pd.Series) -> Pipeline:
    """Build features on the training data, oversample the minority class, fit, return a scorer."""
    builder = CreditFeatureBuilder().fit(X_tr)
    feats = builder.transform(X_tr)
    steps = [("features", builder)]
    values = feats.to_numpy()
    if scale:
        scaler = StandardScaler().fit(feats)
        values = scaler.transform(feats)
        steps.append(("scale", scaler))
    balanced_x, balanced_y = smote_oversample(values, y_tr.to_numpy())
    if not scale:
        balanced_x = pd.DataFrame(balanced_x, columns=feats.columns)
    estimator.fit(balanced_x, balanced_y)
    steps.append(("clf", estimator))
    return Pipeline(steps)


def _imblearn_variants():
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError:
        print("imbalanced-learn not installed: skipping the cross-check rows.")
        return {}
    smote = SMOTE(random_state=C.RANDOM_STATE)
    return {
        "Logistic, SMOTE (imbalanced-learn)": ImbPipeline([
            ("features", CreditFeatureBuilder()), ("scale", StandardScaler()), ("smote", smote),
            ("clf", LogisticRegression(max_iter=2000, random_state=C.RANDOM_STATE))]),
        "Gradient boosting, SMOTE (imbalanced-learn)": ImbPipeline([
            ("features", CreditFeatureBuilder()), ("smote", smote),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=50,
                l2_regularization=1.0, random_state=C.RANDOM_STATE))]),
    }


def main() -> None:
    train, val, _ = split_data(load_raw())
    X_tr, y_tr = xy(train)
    X_va, y_va = xy(val)
    lr_est = clone(make_logistic().named_steps["clf"])
    gb_est = clone(make_gbm().named_steps["clf"])
    models = {
        "Logistic, no resampling": make_logistic(None).fit(X_tr, y_tr),
        "Logistic, class_weight=balanced": make_logistic("balanced").fit(X_tr, y_tr),
        "Logistic, SMOTE (own implementation)": fit_with_smote(lr_est, True, X_tr, y_tr),
        "Gradient boosting, no resampling": make_gbm(None).fit(X_tr, y_tr),
        "Gradient boosting, class_weight=balanced": make_gbm("balanced").fit(X_tr, y_tr),
        "Gradient boosting, SMOTE (own implementation)": fit_with_smote(gb_est, False, X_tr, y_tr),
    }
    for name, pipe in _imblearn_variants().items():
        models[name] = pipe.fit(X_tr, y_tr)

    rows = []
    for name, model in models.items():
        ev = evaluate(y_va, model.predict_proba(X_va)[:, 1])
        rows.append({"variant": name, **{k: round(v, 4) for k, v in ev.items() if k != "base_rate"}})
        print(rows[-1])
    path = C.REPORTS_DIR / "resampling_check.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
