import numpy as np

from creditrisk import config as C
from creditrisk.explain import occlusion_contributions, reference_values
from creditrisk.models import make_gbm, make_logistic
from creditrisk.predict import score_dataframe
from tests.conftest_data import make_synthetic


def _trained(factory):
    df = make_synthetic(4000)
    return df, factory().fit(df[C.RAW_FEATURES], df[C.TARGET])


def test_probabilities_valid_and_better_than_chance():
    from sklearn.metrics import roc_auc_score
    for factory in (make_logistic, make_gbm):
        df, model = _trained(factory)
        p = model.predict_proba(df[C.RAW_FEATURES])[:, 1]
        assert ((p >= 0) & (p <= 1)).all()
        assert roc_auc_score(df[C.TARGET], p) > 0.65


def test_more_90_day_lates_means_higher_risk():
    df, model = _trained(make_logistic)
    row = df[C.RAW_FEATURES].head(1).copy()
    low = row.copy(); low["NumberOfTimes90DaysLate"] = 0
    high = row.copy(); high["NumberOfTimes90DaysLate"] = 5
    assert model.predict_proba(high)[0, 1] > model.predict_proba(low)[0, 1]


def test_score_dataframe_output_columns():
    df, model = _trained(make_logistic)
    out = score_dataframe(model, df.head(20), threshold=0.2)
    assert set(out.columns) == {"pd", "band", "decision"}
    assert set(out["decision"]).issubset({"Approve", "Decline / review"})


def test_occlusion_zero_when_row_equals_reference():
    df, model = _trained(make_logistic)
    ref = reference_values(df)
    row = df[C.RAW_FEATURES].head(1).copy()
    for k, v in ref.items():
        row[k] = v
    contrib = occlusion_contributions(model, row, ref)
    assert np.allclose(contrib.values, 0.0, atol=1e-12)
    assert set(contrib.index) == set(C.RAW_FEATURES)
