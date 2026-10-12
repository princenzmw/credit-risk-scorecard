import pandas as pd

from creditrisk import config as C
from creditrisk.data import split_data, validate_schema
from tests.conftest_data import make_synthetic


def test_validate_schema_raises_on_missing_column():
    df = make_synthetic(50).drop(columns=["age"])
    try:
        validate_schema(df)
    except ValueError as exc:
        assert "age" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_split_sizes_and_stratification():
    df = make_synthetic(5000)
    tr, va, te = split_data(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert abs(len(te) / len(df) - C.TEST_SIZE) < 0.01
    assert abs(len(va) / len(df) - C.VAL_SIZE) < 0.01
    rates = [x[C.TARGET].mean() for x in (tr, va, te)]
    assert max(rates) - min(rates) < 0.01


def test_split_is_deterministic():
    df = make_synthetic(2000)
    a = split_data(df)[2]
    b = split_data(df)[2]
    pd.testing.assert_frame_equal(a, b)
