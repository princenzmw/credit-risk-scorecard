"""Leak-free feature builder.

Every statistic (medians, caps) is learned in fit() from training data only,
so the builder can sit inside an sklearn Pipeline and be cross-validated safely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C

FEATURE_NAMES = [
    "age",
    "income_log",
    "income_missing",
    "dependents",
    "dependents_missing",
    "utilization",
    "utilization_over_1",
    "debt_ratio_log",
    "open_credit_lines",
    "real_estate_loans",
    "late_30_59",
    "late_60_89",
    "late_90",
    "late_total",
    "any_late",
    "late_code_flag",
]

_CAP_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
    "MonthlyIncome",
]


class CreditFeatureBuilder(BaseEstimator, TransformerMixin):
    """Clean raw bureau-style columns and create model features."""

    def __init__(self, cap_quantile: float = C.CAP_QUANTILE):
        self.cap_quantile = cap_quantile

    @staticmethod
    def _clean(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in C.PAST_DUE_COLS:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X["late_code_flag"] = X[C.PAST_DUE_COLS].isin(C.PAST_DUE_SENTINELS).any(axis=1).astype(float)
        for col in C.PAST_DUE_COLS:
            X.loc[X[col].isin(C.PAST_DUE_SENTINELS), col] = np.nan
        X["age"] = pd.to_numeric(X["age"], errors="coerce")
        X.loc[(X["age"] < 18) | (X["age"] > 110), "age"] = np.nan
        return X

    def fit(self, X: pd.DataFrame, y=None):
        X = self._clean(X)
        self.age_median_ = float(X["age"].median())
        self.caps_ = {c: float(X[c].quantile(self.cap_quantile)) for c in _CAP_COLS}
        self.income_median_ = float(X["MonthlyIncome"].median())
        self.dependents_median_ = float(X["NumberOfDependents"].median())
        self.feature_names_out_ = list(FEATURE_NAMES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._clean(X)
        out = pd.DataFrame(index=X.index)
        out["age"] = X["age"].fillna(self.age_median_)

        income_missing = X["MonthlyIncome"].isna()
        income = X["MonthlyIncome"].fillna(self.income_median_).clip(0, self.caps_["MonthlyIncome"])
        out["income_log"] = np.log1p(income)
        out["income_missing"] = income_missing.astype(float)

        dep_missing = X["NumberOfDependents"].isna()
        dep = X["NumberOfDependents"].fillna(self.dependents_median_)
        out["dependents"] = dep.clip(0, self.caps_["NumberOfDependents"])
        out["dependents_missing"] = dep_missing.astype(float)

        util = X["RevolvingUtilizationOfUnsecuredLines"].fillna(0.0)
        out["utilization"] = util.clip(0, self.caps_["RevolvingUtilizationOfUnsecuredLines"])
        out["utilization_over_1"] = (util > 1).astype(float)

        # When income is missing DebtRatio holds a huge absolute figure, so the
        # model also receives income_missing to tell the two regimes apart.
        out["debt_ratio_log"] = np.log1p(X["DebtRatio"].fillna(0.0).clip(0, self.caps_["DebtRatio"]))
        out["open_credit_lines"] = X["NumberOfOpenCreditLinesAndLoans"].clip(0, self.caps_["NumberOfOpenCreditLinesAndLoans"])
        out["real_estate_loans"] = X["NumberRealEstateLoansOrLines"].clip(0, self.caps_["NumberRealEstateLoansOrLines"])

        late = X[C.PAST_DUE_COLS].fillna(0.0).clip(0, C.PAST_DUE_CAP)
        out["late_30_59"] = late["NumberOfTime30-59DaysPastDueNotWorse"]
        out["late_60_89"] = late["NumberOfTime60-89DaysPastDueNotWorse"]
        out["late_90"] = late["NumberOfTimes90DaysLate"]
        out["late_total"] = late.sum(axis=1)
        out["any_late"] = (out["late_total"] > 0).astype(float)
        out["late_code_flag"] = X["late_code_flag"]
        return out[FEATURE_NAMES].astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)
