"""Train, compare, select and evaluate models. Run:  python -m creditrisk.train"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from . import config as C  # noqa: E402
from .data import load_raw, split_data, xy  # noqa: E402
from .explain import reference_values  # noqa: E402
from .metrics import band_table, calibration_table, confusion_at, cost_threshold, evaluate  # noqa: E402
from .models import MODEL_FACTORIES  # noqa: E402


def _roc_plot(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (validation set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _calibration_plot(cal: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    top = float(max(cal["mean_predicted"].max(), cal["observed"].max())) * 1.1
    ax.plot([0, top], [0, top], "k--", alpha=0.4, label="Perfect calibration")
    ax.plot(cal["mean_predicted"], cal["observed"], "o-", label="Model (score deciles)")
    ax.set(xlabel="Mean predicted default probability", ylabel="Observed default rate",
           title="Calibration (test set)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _score_hist(y, p, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 0], bins=bins, alpha=0.6, density=True, label="Repaid")
    ax.hist(p[y == 1], bins=bins, alpha=0.6, density=True, label="Defaulted")
    ax.set(xlabel="Predicted probability of default", ylabel="Density", title="Score distribution (test set)")
    ax.set_yscale("log"); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _importance_plot(imp: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp.sort_values().plot.barh(ax=ax)
    ax.set(xlabel="Drop in ROC-AUC when shuffled", title="Permutation importance (validation sample)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _model_card(m: dict) -> str:
    t, c = m["test"], m["confusion_test"]
    rows = "\n".join(
        f"| {n} | {r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f} | {r['validation']['roc_auc']:.4f} | "
        f"{r['validation']['pr_auc']:.4f} | {r['validation']['ks']:.3f} | {r['validation']['brier']:.4f} |"
        for n, r in m["candidates"].items())
    bands = "\n".join(f"| {b['band']} | {b['loans']:,} | {b['observed_default_rate']:.2%} |" for b in m["bands_test"])
    return f"""# Model card: credit-risk scorecard

**Task:** predict serious delinquency (90+ days past due) within two years.
**Data:** Give Me Some Credit training file, 150,000 borrowers, {m['base_rate']:.2%} default rate. Competition data from 2011 whose population is not documented; results are not evidence about any current lender.

## Method
- Split 60/20/20 (train/validation/test), stratified, seed {C.RANDOM_STATE}. The test set is used once, after all choices were made.
- Features built inside the pipeline (medians and caps learned on training data only). Special past-due codes 96/98 are treated as flags, not counts; missing income and dependents get explicit missing-value flags.
- Candidates: logistic regression (scorecard) and gradient boosting. Champion rule: keep the simpler model unless the challenger beats it by more than {C.CHALLENGER_MARGIN} validation ROC-AUC.

## Model comparison (train CV, validation)
| Model | 5-fold CV ROC-AUC | Val ROC-AUC | Val PR-AUC | Val KS | Val Brier |
|---|---|---|---|---|---|
{rows}

**Champion:** {m['champion']}.

## Final test-set result (single evaluation)
ROC-AUC **{t['roc_auc']:.4f}**, PR-AUC {t['pr_auc']:.4f}, KS {t['ks']:.3f}, Brier {t['brier']:.4f} (a model with no skill has PR-AUC about {t['base_rate']:.3f}).

## Decision threshold
Chosen on the validation set by minimising cost, assuming approving a defaulter costs {C.COST_FN:g}x as much as declining a good borrower. This ratio is an **assumption**; change `COST_FN` in `creditrisk/config.py` for a different business. Threshold: **{m['threshold']:.2f}**. On the test set this declines {c['decline_rate']:.1%} of applicants and catches {c['recall']:.1%} of defaulters at {c['precision']:.1%} precision.

## Risk bands (test set)
| Band | Loans | Observed default rate |
|---|---|---|
{bands}

## Limitations
- Old competition data of unknown origin; no out-of-time validation is possible.
- Age is a model input. Many jurisdictions restrict using age in lending decisions; this is a modelling exercise, not a compliant credit policy.
- No fairness analysis has been run.
- Rejected applicants are absent from the data (selection bias).
- Probabilities were checked for calibration on test data but not recalibrated.
"""


def main(data_path: Path, models_dir: Path, reports_dir: Path, folds: int = 5) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = reports_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(data_path)
    train, val, test = split_data(df)
    X_tr, y_tr = xy(train); X_va, y_va = xy(val); X_te, y_te = xy(test)
    print(f"rows train/val/test: {len(train):,}/{len(val):,}/{len(test):,}; default rate {df[C.TARGET].mean():.4f}")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=C.RANDOM_STATE)
    candidates, fitted, val_scores, curves = {}, {}, {}, {}
    for name, factory in MODEL_FACTORIES.items():
        cv_scores = cross_val_score(factory(), X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
        model = factory().fit(X_tr, y_tr)
        p_va = model.predict_proba(X_va)[:, 1]
        ev = evaluate(y_va, p_va)
        candidates[name] = {"cv_auc_mean": float(cv_scores.mean()), "cv_auc_std": float(cv_scores.std()), "validation": ev}
        fitted[name], val_scores[name] = model, p_va
        fpr, tpr, _ = roc_curve(y_va, p_va); curves[name] = (fpr, tpr, ev["roc_auc"])
        print(f"{name}: CV {cv_scores.mean():.4f}+/-{cv_scores.std():.4f}  val AUC {ev['roc_auc']:.4f}  PR {ev['pr_auc']:.4f}  KS {ev['ks']:.3f}")

    names = list(MODEL_FACTORIES)
    base, chall = names[0], names[1]
    gain = candidates[chall]["validation"]["roc_auc"] - candidates[base]["validation"]["roc_auc"]
    champion = chall if gain > C.CHALLENGER_MARGIN else base
    print(f"challenger gain {gain:+.4f} -> champion: {champion}")

    model = fitted[champion]
    threshold, val_cost = cost_threshold(y_va, val_scores[champion])

    p_te = model.predict_proba(X_te)[:, 1]
    test_metrics = evaluate(y_te, p_te)
    conf = confusion_at(y_te, p_te, threshold)
    cal = calibration_table(y_te, p_te)
    bands = band_table(y_te, p_te)
    print(f"TEST  AUC {test_metrics['roc_auc']:.4f}  PR {test_metrics['pr_auc']:.4f}  KS {test_metrics['ks']:.3f}  Brier {test_metrics['brier']:.4f}")

    sample = X_va.sample(n=min(20000, len(X_va)), random_state=C.RANDOM_STATE)
    pi = permutation_importance(model, sample, y_va.loc[sample.index], scoring="roc_auc",
                                n_repeats=5, random_state=C.RANDOM_STATE, n_jobs=1)
    importance = pd.Series(pi.importances_mean, index=C.RAW_FEATURES).sort_values(ascending=False)

    _roc_plot(curves, fig_dir / "roc_validation.png")
    _calibration_plot(cal, fig_dir / "calibration_test.png")
    _score_hist(y_te.to_numpy(), p_te, fig_dir / "score_distribution_test.png")
    _importance_plot(importance, fig_dir / "permutation_importance.png")
    cal.to_csv(reports_dir / "calibration_test.csv", index=False)
    importance.rename("auc_drop").to_csv(reports_dir / "permutation_importance.csv", index_label="feature")

    metrics = {
        "champion": champion, "threshold": threshold, "validation_cost_at_threshold": val_cost,
        "base_rate": float(df[C.TARGET].mean()), "candidates": candidates,
        "validation": {n: r["validation"] for n, r in candidates.items()},
        "test": test_metrics, "confusion_test": conf,
        "bands_test": bands.to_dict("records"),
        "assumptions": {"cost_fn": C.COST_FN, "cost_fp": C.COST_FP, "challenger_margin": C.CHALLENGER_MARGIN},
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
        "n_rows": {"train": len(train), "validation": len(val), "test": len(test)},
    }
    joblib.dump(model, models_dir / "model.joblib", compress=3)
    (models_dir / "threshold.json").write_text(json.dumps({"threshold": threshold, "cost_fn": C.COST_FN}, indent=2))
    (models_dir / "reference.json").write_text(json.dumps(reference_values(train), indent=2))
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports_dir / "model_card.md").write_text(_model_card(metrics), encoding="utf-8")
    print("saved artefacts to", models_dir, "and", reports_dir)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=C.DATA_PATH)
    ap.add_argument("--models", type=Path, default=C.MODELS_DIR)
    ap.add_argument("--reports", type=Path, default=C.REPORTS_DIR)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    main(a.data, a.models, a.reports, a.folds)
