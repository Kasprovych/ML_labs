"""Тести крос-валідатора та фабрики моделей."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mllabs.modeling import CrossValidator, MetricsCalculator, ModelFactory, ModelKind


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "c"])
    target = pd.Series((features["a"] + rng.normal(0, 1, 400) > 1.2).astype(int))
    return features, target


class _CountingInspector:
    def __call__(self, pipeline, features, target):
        return {"rows": len(features), "positives": int(target.sum())}


def test_oof_covers_every_row_exactly_once(toy_data):
    features, target = toy_data

    result = CrossValidator(5, random_state=0).evaluate(LogisticRegression(), features, target)

    valid = np.concatenate([fold.valid_index for fold in result.folds])
    assert sorted(valid) == list(range(len(features)))
    assert not np.isnan(result.oof_proba).any()


def test_folds_are_identical_between_calls(toy_data):
    features, target = toy_data
    validator = CrossValidator(5, random_state=0)

    first = validator.evaluate(LogisticRegression(), features, target)
    second = validator.evaluate(LogisticRegression(C=0.01), features, target)

    for left, right in zip(first.folds, second.folds, strict=True):
        np.testing.assert_array_equal(left.valid_index, right.valid_index)


def test_train_size_keeps_class_share_and_full_validation(toy_data):
    features, target = toy_data

    result = CrossValidator(4, random_state=0).evaluate(
        LogisticRegression(), features, target, train_size=100
    )

    share = target.mean()
    for fold in result.folds:
        assert len(fold.train_index) == 100
        assert len(fold.valid_index) == 100  # валідація повна (400 / 4)
        assert target.iloc[fold.train_index].mean() == pytest.approx(share, abs=0.02)
        assert not set(fold.train_index) & set(fold.valid_index)


def test_parallel_and_sequential_runs_agree(toy_data):
    features, target = toy_data

    sequential = CrossValidator(3, random_state=0, n_jobs=1).evaluate(
        LogisticRegression(), features, target
    )
    parallel = CrossValidator(3, random_state=0, n_jobs=2).evaluate(
        LogisticRegression(), features, target
    )

    np.testing.assert_allclose(sequential.oof_proba, parallel.oof_proba)


def test_inspector_sees_only_training_rows_of_fold(toy_data):
    features, target = toy_data

    result = CrossValidator(4, random_state=0).evaluate(
        LogisticRegression(), features, target, inspector=_CountingInspector()
    )

    assert result.extras("rows") == [300] * 4
    assert sum(result.extras("positives")) == 3 * target.sum()


def test_fold_metrics_and_train_pr_auc_have_one_value_per_fold(toy_data):
    features, target = toy_data
    calculator = MetricsCalculator()

    result = CrossValidator(5, random_state=0).evaluate(LogisticRegression(), features, target)

    assert len(result.fold_metrics(calculator)) == 5
    assert result.train_pr_auc(calculator).shape == (5,)
    assert result.n_features == 3


@pytest.mark.parametrize("kind", list(ModelKind))
def test_model_factory_creates_unfitted_reproducible_models(kind):
    factory = ModelFactory(random_state=7)

    model = factory.create(kind)

    assert model.get_params()["random_state"] == 7
    assert not hasattr(model, "classes_")
    assert factory.describe(kind)
    assert "class_weight" not in factory.describe(kind)


def test_boosting_has_no_internal_early_stopping():
    model = ModelFactory().create(ModelKind.HIST_GRADIENT_BOOSTING)

    assert model.get_params()["early_stopping"] is False
