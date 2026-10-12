import numpy as np

from creditrisk.resampling import smote_oversample


def _toy(n_major=300, n_minor=30, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (n_major, 3)), rng.normal(4, 0.5, (n_minor, 3))])
    y = np.r_[np.zeros(n_major), np.ones(n_minor)].astype(int)
    return X, y


def test_smote_balances_classes_and_keeps_original_rows():
    X, y = _toy()
    Xb, yb = smote_oversample(X, y)
    assert (yb == 1).sum() == (yb == 0).sum()
    assert np.array_equal(Xb[: len(X)], X)
    assert np.array_equal(yb[: len(y)], y)


def test_synthetic_points_stay_inside_minority_region():
    X, y = _toy()
    Xb, _ = smote_oversample(X, y)
    synth, minority = Xb[len(X):], X[y == 1]
    assert (synth >= minority.min(axis=0) - 1e-9).all()
    assert (synth <= minority.max(axis=0) + 1e-9).all()


def test_already_balanced_data_is_unchanged():
    X, y = _toy(n_major=50, n_minor=50)
    Xb, yb = smote_oversample(X, y)
    assert Xb.shape == X.shape and np.array_equal(yb, y)
