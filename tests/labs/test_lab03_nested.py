"""Тести вкладеної крос-валідації: зовнішній val не потрапляє у внутрішній пошук."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from mllabs.labs.lab03.nested import NestedCrossValidation
from mllabs.labs.lab03.search import EvaluationRecord, SearchOutcome
from mllabs.modeling import CrossValidator, MetricsCalculator


class _SpySearch:
    """Шпигун-стратегія: запам'ятовує індекси рядків, на яких її запущено."""

    key = "spy"
    label = "Spy"

    def __init__(self) -> None:
        self.seen_indexes: list[set] = []

    def run(self, pipeline, features, target, splits):
        self.seen_indexes.append(set(features.index))
        assert all(max(train.max(), valid.max()) < len(features) for train, valid in splits)
        record = EvaluationRecord(0, {"learning_rate": 0.1}, 0.9, 0.0, (0.9,), 0.0)
        return SearchOutcome(
            "spy",
            "Spy",
            1,
            len(splits),
            0.0,
            0.0,
            {"learning_rate": 0.1},
            0.9,
            0.0,
            (record,),
            1,
            1,
        )


def test_outer_validation_rows_never_reach_the_inner_search():
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    target = pd.Series((features["a"] + rng.normal(size=200) > 1).astype(int))
    pipeline = Pipeline([("model", HistGradientBoostingClassifier(max_iter=5, random_state=0))])
    nested = NestedCrossValidation(
        CrossValidator(4, 0), CrossValidator(3, 0), MetricsCalculator(), "model", refit_threads=1
    )
    spy = _SpySearch()

    result = nested.run(spy, pipeline, features, target)

    for seen, (_, valid_index) in zip(
        spy.seen_indexes, nested.outer_splits(features, target), strict=True
    ):
        assert seen.isdisjoint(set(features.index[valid_index]))
        assert len(seen) == len(features) - len(valid_index)
    assert len(result.folds) == 4
    assert all(fold.inner_best == 0.9 for fold in result.folds)
    optimism, _ = result.optimism()
    assert optimism == np.mean([0.9 - fold.outer_pr_auc for fold in result.folds])
