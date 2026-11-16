"""Тести детекторів викидів і їх порівняння (ЛР1, R9)."""

import numpy as np
import pandas as pd
import pytest

from mllabs.labs.lab01.outliers import (
    IqrDetector,
    IsolationForestDetector,
    LofDetector,
    OutlierComparison,
    ZScoreDetector,
    project_to_plane,
)


def test_iqr_bounds_follow_tukey_rule_on_known_quartiles():
    # Квартилі 0..100 з кроком 1: Q1 = 25, Q3 = 75, IQR = 50.
    features = pd.DataFrame({"x": np.arange(101, dtype=float)})

    bounds = IqrDetector(factor=1.5).bounds(features)

    assert bounds.loc["x", "lower"] == pytest.approx(25 - 75)
    assert bounds.loc["x", "upper"] == pytest.approx(75 + 75)


def test_iqr_marks_values_beyond_fences_but_not_missing():
    features = pd.DataFrame({"x": [*range(1, 21), 150.0, -100.0, np.nan]})

    flags = IqrDetector(factor=1.5).detect_by_feature(features)["x"]

    assert flags.tolist() == [False] * 20 + [True, True, False]


def test_univariate_row_is_outlier_if_any_feature_is():
    features = pd.DataFrame({"a": [1.0, 2, 3, 4, 100], "b": [1.0, 2, 3, 4, 5]})

    rows = IqrDetector(1.5).detect(features)

    assert rows.tolist() == [False, False, False, False, True]


def test_zscore_detects_far_value_and_ignores_typical_ones():
    rng = np.random.default_rng(0)
    values = rng.normal(size=1000)
    values[0] = 10.0
    features = pd.DataFrame({"x": values})

    flags = ZScoreDetector(threshold=3.0).detect(features)

    assert flags.iloc[0]
    assert flags.mean() < 0.01


@pytest.mark.parametrize(
    "detector",
    [IsolationForestDetector(0.05, 50, random_state=0), LofDetector(0.05, n_neighbors=10)],
)
def test_multivariate_detectors_flag_requested_share_and_handle_missing(detector):
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "c"])
    features.loc[0, "a"] = np.nan
    features.loc[1] = [8.0, -8.0, 8.0]  # очевидна аномалія

    flags = detector.detect(features)

    assert flags.mean() == pytest.approx(0.05, abs=0.01)
    assert flags.iloc[1]


def test_comparison_reports_jaccard_and_positive_rate():
    features = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 100]})
    is_positive = pd.Series([False] * 9 + [True])
    comparison = OutlierComparison(
        univariate=[IqrDetector(1.5), ZScoreDetector(2.0)], multivariate=[]
    )

    report = comparison.run(features, is_positive)

    assert report.overlap.loc["IQR", "Z-score"] == pytest.approx(1.0)
    summary = report.by_method.set_index("Метод")
    assert summary.loc["IQR", "Викидів"] == 1
    assert summary.loc["IQR", "Клас 1 серед викидів, %"] == pytest.approx(100.0)
    metrics = report.to_metrics()
    assert metrics["iqr_share"] == pytest.approx(0.1)
    assert metrics["all_methods_share"] == pytest.approx(0.1)
    assert metrics["positive_rate_in_outliers"]["zscore"] == pytest.approx(1.0)


def test_projection_returns_requested_number_of_points():
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(100, 4)))

    projection, explained = project_to_plane(features, sample_size=30, random_state=0)

    assert projection.shape == (30, 2)
    assert 0 < explained[1] <= explained[0] < 1
