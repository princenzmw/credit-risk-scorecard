"""Synthetic data helpers shared by the tests (no real data needed)."""
import numpy as np
import pandas as pd

from creditrisk import config as C


def make_synthetic(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": rng.beta(1, 4, n),
        "age": rng.integers(21, 80, n),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.poisson(0.3, n),
        "DebtRatio": rng.gamma(2, 0.2, n),
        "MonthlyIncome": rng.lognormal(8.7, 0.5, n),
        "NumberOfOpenCreditLinesAndLoans": rng.poisson(8, n),
        "NumberOfTimes90DaysLate": rng.poisson(0.15, n),
        "NumberRealEstateLoansOrLines": rng.poisson(1, n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.1, n),
        "NumberOfDependents": rng.poisson(1, n).astype(float),
    })
    logit = (-4 + 3 * df["RevolvingUtilizationOfUnsecuredLines"] + 1.2 * df["NumberOfTimes90DaysLate"]
             + 0.5 * df["NumberOfTime30-59DaysPastDueNotWorse"] - 0.02 * (df["age"] - 45))
    df[C.TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.loc[rng.random(n) < 0.2, "MonthlyIncome"] = np.nan
    df.loc[rng.random(n) < 0.03, "NumberOfDependents"] = np.nan
    return df
