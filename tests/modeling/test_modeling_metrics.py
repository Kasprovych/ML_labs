"""Тести метрик, вибору порога, кривих і бутстрепу спільного пакета modeling."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

from mllabs.modeling import CurveBuilder, F1ThresholdSelector, MetricsCalculator, PairedBootstrap


def test_metrics_match_hand_calculation_on_known_example():
    target = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.7, 0.2, 0.9, 0.3, 0.8])

    metrics = MetricsCalculator(positive_label=1).compute(target, proba, threshold=0.5)

    # При порозі 0,5 позначено рядки 2, 3, 5, 7: TP = 3 (3, 5, 7), FP = 1 (2), FN = 1 (4).
    assert metrics.precision == pytest.approx(3 / 4)
    assert metrics.recall == pytest.approx(3 / 4)
    assert metrics.f1 == pytest.approx(3 / 4)
    assert metrics.accuracy == pytest.approx(6 / 8)
    assert metrics.brier == pytest.approx(np.mean((proba - target) ** 2))
    assert metrics.mean_proba == pytest.approx(proba.mean())


def test_metrics_respect_positive_label_other_than_one():
    target = np.array([2, 2, 5, 5])
    proba = np.array([0.9, 0.8, 0.1, 0.2])  # ймовірність класу «2»

    metrics = MetricsCalculator(positive_label=2).compute(target, proba)

    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)


def test_precision_is_zero_when_nothing_is_flagged():
    metrics = MetricsCalculator().compute(np.array([0, 1, 0, 1]), np.array([0.1, 0.2, 0.3, 0.4]))

    assert metrics.precision == 0.0
    assert metrics.f1 == 0.0


def test_majority_accuracy_is_share_of_negative_class():
    assert MetricsCalculator().majority_accuracy(np.array([0, 0, 0, 1])) == pytest.approx(0.75)


def test_selected_threshold_maximizes_f1():
    rng = np.random.default_rng(0)
    target = (rng.random(500) < 0.15).astype(int)
    proba = np.clip(0.15 + 0.3 * target + rng.normal(0, 0.15, size=500), 0, 1)

    threshold = F1ThresholdSelector().select(target, proba)

    best = f1_score(target, proba >= threshold)
    for candidate in np.unique(proba):
        assert f1_score(target, proba >= candidate) <= best + 1e-12


def test_threshold_curve_is_thinned_but_keeps_ends():
    rng = np.random.default_rng(1)
    target = (rng.random(2000) < 0.2).astype(int)
    proba = rng.random(2000)

    curve = CurveBuilder(n_points=50).by_threshold(target, proba)

    assert len(curve) <= 50
    assert curve["threshold"].iloc[0] == pytest.approx(proba.min())
    assert {"threshold", "precision", "recall"} == set(curve.columns)


def test_bootstrap_of_identical_predictions_gives_zero_interval():
    rng = np.random.default_rng(2)
    target = (rng.random(300) < 0.3).astype(int)
    proba = rng.random(300)

    interval = PairedBootstrap(200, 0.95, random_state=0).interval(
        target, proba, proba, average_precision_score
    )

    assert (interval.delta, interval.low, interval.high) == (0.0, 0.0, 0.0)
    assert not interval.excludes_zero


def test_bootstrap_detects_clearly_better_model():
    rng = np.random.default_rng(3)
    target = (rng.random(1000) < 0.3).astype(int)
    noise = rng.random(1000)
    good = 0.7 * target + 0.3 * noise

    interval = PairedBootstrap(200, 0.95, random_state=0).interval(
        target, noise, good, average_precision_score
    )

    assert interval.low > 0
    assert interval.excludes_zero
