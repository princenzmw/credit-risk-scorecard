"""Model factories. Each returns a full pipeline: raw columns in, probability out."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import CreditFeatureBuilder


def make_logistic(class_weight=None) -> Pipeline:
    return Pipeline([
        ("features", CreditFeatureBuilder()),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight=class_weight,
                                   random_state=C.RANDOM_STATE)),
    ])


def make_gbm(class_weight=None) -> Pipeline:
    return Pipeline([
        ("features", CreditFeatureBuilder()),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, class_weight=class_weight, random_state=C.RANDOM_STATE)),
    ])


MODEL_FACTORIES = {"Logistic regression (scorecard)": make_logistic,
                   "Gradient boosting (challenger)": make_gbm}
