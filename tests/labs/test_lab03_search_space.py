"""Тести простору гіперпараметрів: однакові межі для Grid, Random і Optuna."""

import numpy as np
import optuna
import pytest
from sklearn.model_selection import ParameterSampler

from mllabs.labs.lab03.search_space import (
    IntLogUniform,
    InvalidSearchSpaceError,
    ParameterRange,
    hgb_search_space,
)


def test_full_grid_has_81_points_within_bounds():
    space = hgb_search_space()
    grid = space.param_grid("model")

    assert space.grid_size == 81
    for item in space.ranges:
        values = grid[f"model__{item.name}"]
        assert min(values) == item.low and max(values) == item.high


def test_random_distributions_stay_within_bounds_and_keep_integer_types():
    space = hgb_search_space()
    samples = list(ParameterSampler(space.distributions("model"), n_iter=300, random_state=0))

    for item in space.ranges:
        values = [sample[f"model__{item.name}"] for sample in samples]
        assert all(item.low <= value <= item.high for value in values)
        if item.integer:
            assert all(isinstance(value, int) for value in values)


def test_optuna_suggestions_stay_within_bounds():
    space = hgb_search_space()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for _ in range(50):
        trial = study.ask()
        params = space.suggest(trial)
        study.tell(trial, 0.0)
        for item in space.ranges:
            assert item.low <= params[item.name] <= item.high
            assert isinstance(params[item.name], int) == item.integer


def test_int_log_uniform_covers_both_ends_and_prefers_small_values():
    values = IntLogUniform(15, 63).rvs(size=5000, random_state=0)

    assert values.min() == 15 and values.max() == 63
    assert np.median(values) < (15 + 63) / 2  # лог-шкала: малі значення частіші


def test_range_rejects_grid_without_bounds():
    with pytest.raises(InvalidSearchSpaceError):
        ParameterRange("x", "опис", 1.0, 10.0, (2.0, 10.0), log=True, integer=False)


def test_strip_and_prefixed_are_inverse():
    space = hgb_search_space()
    params = {
        "learning_rate": 0.05,
        "max_leaf_nodes": 20,
        "min_samples_leaf": 50,
        "max_features": 0.5,
    }

    assert space.strip(space.prefixed(params, "model"), "model") == params
