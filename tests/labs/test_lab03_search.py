"""Тести стратегій пошуку на мінізадачі (без довгого навчання)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from mllabs.labs.lab03.search import (
    GridSearchStrategy,
    OptunaSearchStrategy,
    RandomSearchStrategy,
)
from mllabs.labs.lab03.search_space import ParameterRange, SearchSpace

_SPACE = SearchSpace(
    ranges=(
        ParameterRange("learning_rate", "крок", 0.05, 0.3, (0.05, 0.3), log=True, integer=False),
        ParameterRange("max_leaf_nodes", "листки", 4, 16, (4, 16), log=True, integer=True),
    )
)
_OPTIONS = {
    "space": _SPACE,
    "scoring": "average_precision",
    "prefix": "model",
    "n_jobs": 1,
    "threads_per_worker": 1,
}


@pytest.fixture(scope="module")
def task():
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    target = pd.Series((features["a"] + rng.normal(size=300) > 1).astype(int))
    pipeline = Pipeline([("model", HistGradientBoostingClassifier(max_iter=5, random_state=0))])
    splits = list(StratifiedKFold(3, shuffle=True, random_state=0).split(features, target))
    return pipeline, features, target, splits


def test_grid_evaluates_every_point_and_reports_best(task):
    outcome = GridSearchStrategy(**_OPTIONS).run(*task)

    assert outcome.n_candidates == 4 and outcome.n_fits == 12
    assert outcome.best_score == max(record.mean_score for record in outcome.history)
    assert outcome.best_so_far()[-1] == outcome.best_score
    assert set(outcome.best_params) == {"learning_rate", "max_leaf_nodes"}


def test_random_search_with_more_iterations_extends_the_same_sequence(task):
    short = RandomSearchStrategy(3, 7, **_OPTIONS).run(*task)
    long = RandomSearchStrategy(6, 7, **_OPTIONS).run(*task)

    assert [r.params for r in long.history[:3]] == [r.params for r in short.history]
    assert long.key == "random_6"


def _optuna(seed: int) -> OptunaSearchStrategy:
    return OptunaSearchStrategy(
        space=_SPACE,
        scoring="average_precision",
        prefix="model",
        n_trials=6,
        startup_trials=2,
        multivariate=True,
        random_state=seed,
        fold_n_jobs=1,
        threads_per_worker=1,
    )


def test_optuna_with_same_seed_reproduces_history(task):
    first = _optuna(3).run(*task)
    second = _optuna(3).run(*task)

    assert [r.params for r in first.history] == [r.params for r in second.history]
    assert [r.mean_score for r in first.history] == [r.mean_score for r in second.history]
    assert first.n_fits == 18


def test_optuna_and_grid_score_the_same_point_identically(task):
    pipeline, features, target, splits = task
    grid = GridSearchStrategy(**_OPTIONS).run(pipeline, features, target, splits)
    optuna_run = _optuna(0).run(pipeline, features, target, splits)
    record = optuna_run.history[0]
    fixed = SearchSpace(
        ranges=tuple(
            ParameterRange(
                item.name,
                item.description,
                item.low,
                item.high,
                (item.low, record.params[item.name], item.high),
                item.log,
                item.integer,
            )
            for item in _SPACE.ranges
        )
    )

    regrid = GridSearchStrategy(**(_OPTIONS | {"space": fixed})).run(
        pipeline, features, target, splits
    )
    same = next(r for r in regrid.history if r.params == record.params)
    assert same.fold_scores == pytest.approx(record.fold_scores)
    assert grid.n_candidates == 4
