"""Тести власної логіки ЛР4: вибір конфігурації, фронт Парето, позначки рішень, приклади карт."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mllabs.labs.lab04.evaluation import outcome_labels, pick_map_examples
from mllabs.labs.lab04.profiling_study import pareto_mask
from mllabs.labs.lab04.studies import select_best


@dataclass(frozen=True)
class _Run:
    val_pixel_auroc: float


def test_select_best_prefers_earlier_variant_within_tolerance():
    runs = [_Run(0.900), _Run(0.9015), _Run(0.880)]

    assert select_best(runs, tolerance=0.002) is runs[0]
    assert select_best(runs, tolerance=0.001) is runs[1]


def test_pareto_mask_keeps_only_non_dominated_points():
    cost = np.array([1.0, 2.0, 3.0, 2.5])
    quality = np.array([0.80, 0.90, 0.95, 0.85])

    assert pareto_mask(cost, quality).tolist() == [True, True, True, False]


def test_outcome_labels_cover_confusion_matrix():
    labels = np.array([1, 1, 0, 0])
    decisions = np.array([True, False, True, False])

    assert outcome_labels(labels, decisions).tolist() == ["TP", "FN", "FP", "TN"]


def test_pick_map_examples_prefers_missed_anomaly_and_false_alarm():
    predictions = pd.DataFrame(
        {
            "outcome": ["TP", "FN", "FN", "FP", "FP", "TN"],
            "score": [5.0, 1.0, 1.5, 3.0, 4.0, 0.5],
            "defect_area_share": [0.01, 0.002, 0.03, 0.0, 0.0, 0.0],
        }
    )

    assert pick_map_examples(predictions) == [2, 4]


def test_pick_map_examples_falls_back_to_true_positives():
    predictions = pd.DataFrame(
        {
            "outcome": ["TP", "TP", "TP", "TN"],
            "score": [3.0, 1.0, 2.0, 0.1],
            "defect_area_share": [0.01, 0.02, 0.03, 0.0],
        }
    )

    assert pick_map_examples(predictions) == [2, 0]
