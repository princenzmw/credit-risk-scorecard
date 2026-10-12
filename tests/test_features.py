import numpy as np

from creditrisk import config as C
from creditrisk.features import FEATURE_NAMES, CreditFeatureBuilder
from tests.conftest_data import make_synthetic


def _fit(n=3000):
    df = make_synthetic(n)
    return df, CreditFeatureBuilder().fit(df[C.RAW_FEATURES])


def test_output_has_expected_columns_and_no_nans():
    df, fb = _fit()
    out = fb.transform(df[C.RAW_FEATURES])
    assert list(out.columns) == FEATURE_NAMES
    assert not out.isna().any().any()


def test_sentinel_codes_become_flags_not_counts():
    df, fb = _fit()
    row = df[C.RAW_FEATURES].head(1).copy()
    row["NumberOfTimes90DaysLate"] = 98
    out = fb.transform(row)
    assert out["late_code_flag"].iloc[0] == 1.0
    assert out["late_90"].iloc[0] == 0.0


def test_missing_income_flagged_and_imputed_with_training_median():
    df, fb = _fit()
    row = df[C.RAW_FEATURES].head(1).copy()
    row["MonthlyIncome"] = np.nan
    out = fb.transform(row)
    assert out["income_missing"].iloc[0] == 1.0
    assert np.isclose(out["income_log"].iloc[0], np.log1p(fb.income_median_))


def test_statistics_come_from_fit_data_only():
    df, _ = _fit()
    train = df[C.RAW_FEATURES].iloc[:2000]
    fb = CreditFeatureBuilder().fit(train)
    assert np.isclose(fb.income_median_, train["MonthlyIncome"].median())


def test_extreme_values_are_capped_and_invalid_age_imputed():
    df, fb = _fit()
    row = df[C.RAW_FEATURES].head(1).copy()
    row["RevolvingUtilizationOfUnsecuredLines"] = 50000
    row["age"] = 0
    out = fb.transform(row)
    assert out["utilization"].iloc[0] <= fb.caps_["RevolvingUtilizationOfUnsecuredLines"]
    assert out["age"].iloc[0] == fb.age_median_
