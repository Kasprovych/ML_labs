"""Тести прогнозу на train-рядках в ознаках часу fit (TargetEncoder, семплер, пошук)."""

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from mllabs.modeling import (
    FIT_INPUT_STEP,
    RESAMPLED_INPUT_STEP,
    CrossValidator,
    FitInputMismatchError,
    fit_input_proba,
    model_input_width,
    with_fit_input_recorder,
)


@pytest.fixture
def categorical_data():
    rng = np.random.default_rng(0)
    category = rng.integers(0, 60, size=600)  # багато рідкісних категорій
    target = pd.Series((rng.random(600) < 0.2 + 0.4 * (category % 3 == 0)).astype(int))
    features = pd.DataFrame({"category": category.astype(str)})
    return features, target


def _encoder_pipeline():
    cross_fitting = StratifiedKFold(5, shuffle=True, random_state=0)
    return Pipeline(
        [
            ("encoder", TargetEncoder(target_type="binary", cv=cross_fitting)),
            ("model", LogisticRegression()),
        ]
    )


def test_train_proba_is_computed_on_the_matrix_seen_during_fit(categorical_data):
    features, target = categorical_data

    result = CrossValidator(3, random_state=0).evaluate(_encoder_pipeline(), features, target)

    fold = result.folds[0]
    x_train, y_train = features.iloc[fold.train_index], target.iloc[fold.train_index]
    reference = clone(_encoder_pipeline()).fit(x_train, y_train)
    fit_matrix = clone(reference[:-1]).fit_transform(x_train, y_train)  # крос-фітинг, як у fit
    expected = reference[-1].predict_proba(fit_matrix)[:, 1]
    naive = reference.predict_proba(x_train)[:, 1]  # transform: власна мітка рядка «протікає»
    np.testing.assert_allclose(fold.train_proba, expected)
    assert not np.allclose(fold.train_proba, naive)


def test_recorder_goes_before_sampler_and_keeps_only_real_rows(categorical_data):
    features, target = categorical_data
    pipeline = ImbalancedPipeline(
        [
            (
                "encoder",
                TargetEncoder(
                    target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=0)
                ),
            ),
            ("sampler", SMOTE(random_state=0)),
            ("model", LogisticRegression()),
        ]
    )

    prepared = with_fit_input_recorder(pipeline)
    prepared.fit(features, target)

    assert [name for name, _ in prepared.steps] == [
        "encoder",
        FIT_INPUT_STEP,
        "sampler",
        RESAMPLED_INPUT_STEP,
        "model",
    ]
    assert len(fit_input_proba(prepared, features)) == len(features)


def test_search_wrapper_gets_recorder_and_reports_model_width(categorical_data):
    features, target = categorical_data
    search = GridSearchCV(_encoder_pipeline(), {"model__C": [0.1, 1.0]}, cv=3)

    prepared = with_fit_input_recorder(search)
    prepared.fit(features, target)

    assert FIT_INPUT_STEP in prepared.best_estimator_.named_steps
    assert model_input_width(prepared) == 1  # ширина входу моделі, а не сирих даних
    assert len(fit_input_proba(prepared, features)) == len(features)


def test_plain_model_is_left_unchanged_and_predicts_on_raw_rows():
    features = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 0.0]})
    target = pd.Series([0, 0, 1, 1])
    model = LogisticRegression()

    prepared = with_fit_input_recorder(model)
    prepared.fit(features, target)

    assert prepared is model
    np.testing.assert_allclose(
        fit_input_proba(prepared, features), model.predict_proba(features)[:, 1]
    )


@pytest.fixture
def numeric_data():
    rng = np.random.default_rng(1)
    features = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "c"])
    target = pd.Series((features["a"] + rng.normal(scale=0.5, size=400) > 1.0).astype(int))
    return features, target


def _scaled_after_sampler(sampler):
    return ImbalancedPipeline(
        [
            ("pre", StandardScaler()),
            ("sampler", sampler),
            ("post", StandardScaler()),  # навчається вже на вибірці з синтетикою
            ("model", LogisticRegression()),
        ]
    )


def test_steps_between_sampler_and_model_are_applied_to_recorded_matrix(numeric_data):
    features, target = numeric_data
    prepared = with_fit_input_recorder(_scaled_after_sampler(SMOTE(random_state=0)))
    prepared.fit(features, target)

    pre, post, model = (prepared.named_steps[name] for name in ("pre", "post", "model"))
    expected = model.predict_proba(post.transform(pre.transform(features)))[:, 1]
    np.testing.assert_allclose(fit_input_proba(prepared, features), expected)


def test_recorders_do_not_change_what_the_model_learns(numeric_data):
    features, target = numeric_data
    plain = _scaled_after_sampler(SMOTE(random_state=0)).fit(features, target)
    prepared = with_fit_input_recorder(_scaled_after_sampler(SMOTE(random_state=0)))
    prepared.fit(features, target)

    np.testing.assert_allclose(plain.predict_proba(features), prepared.predict_proba(features))


def test_undersampler_that_drops_real_rows_is_reported(numeric_data):
    features, target = numeric_data
    prepared = with_fit_input_recorder(_scaled_after_sampler(RandomUnderSampler(random_state=0)))
    prepared.fit(features, target)

    with pytest.raises(FitInputMismatchError):
        fit_input_proba(prepared, features)


def test_wrapping_returns_a_copy_and_keeps_pipeline_settings(categorical_data):
    pipeline = Pipeline(_encoder_pipeline().steps, verbose=True)
    search = GridSearchCV(pipeline, {"model__C": [0.1, 1.0]}, cv=3)

    prepared_pipeline = with_fit_input_recorder(pipeline)
    prepared_search = with_fit_input_recorder(search)

    assert prepared_pipeline.verbose is True
    assert FIT_INPUT_STEP not in pipeline.named_steps
    assert FIT_INPUT_STEP not in search.estimator.named_steps
    assert FIT_INPUT_STEP in prepared_search.estimator.named_steps


def test_feature_names_pass_through_the_recorder(numeric_data):
    features, target = numeric_data
    prepared = with_fit_input_recorder(
        Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])
    )
    prepared.fit(features, target)

    assert list(prepared[:-1].get_feature_names_out()) == ["a", "b", "c"]


def test_search_without_refit_raises_a_clear_error(categorical_data):
    features, target = categorical_data
    search = GridSearchCV(_encoder_pipeline(), {"model__C": [0.1, 1.0]}, cv=3, refit=False)
    prepared = with_fit_input_recorder(search)
    prepared.fit(features, target)

    with pytest.raises(FitInputMismatchError, match="refit"):
        fit_input_proba(prepared, features)
