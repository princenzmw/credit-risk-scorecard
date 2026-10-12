import numpy as np

from creditrisk.metrics import assign_band, calibration_table, cost_threshold, ks_statistic


def test_ks_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert ks_statistic(y, np.r_[np.zeros(50), np.ones(50)]) == 1.0
    rng = np.random.default_rng(0)
    assert ks_statistic(rng.integers(0, 2, 5000), rng.random(5000)) < 0.1


def test_cost_threshold_moves_with_cost_ratio():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    p = np.clip(0.35 * y + rng.normal(0.3, 0.15, 4000), 0, 1)
    low_t, _ = cost_threshold(y, p, cost_fn=1, cost_fp=1)
    high_t, _ = cost_threshold(y, p, cost_fn=20, cost_fp=1)
    assert high_t <= low_t  # costlier misses -> flag more -> lower threshold


def test_bands_are_monotonic_and_cover_range():
    labels = assign_band([0.0, 0.02, 0.05, 0.15, 0.9, 1.0]).astype(str).tolist()
    assert labels == ["Low", "Low", "Moderate", "High", "Very high", "Very high"]


def test_calibration_table_shape():
    rng = np.random.default_rng(2)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    cal = calibration_table(y, p, bins=10)
    assert len(cal) == 10 and cal["n"].sum() == 1000
