"""Тести дослідження витоку: некоректні варіанти справді «підглядають» мітки."""

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from mllabs.labs.lab03.leakage import (
    LeakageExperiment,
    LeakageVariant,
    LeakyMatrixBuilder,
    SmoteBeforeSplitEvaluator,
)
from mllabs.modeling import CrossValidator, MetricsCalculator


@pytest.fixture(scope="module")
def identifier_data():
    """Категорія-ідентифікатор унікальна для кожного рядка; ціль — чистий шум."""
    rng = np.random.default_rng(0)
    n_rows = 600
    features = pd.DataFrame({"row_id": [f"id{index}" for index in range(n_rows)]})
    target = pd.Series((rng.random(n_rows) < 0.2).astype(int))
    preprocessor = Pipeline(
        [
            (
                "columns",
                ColumnTransformer(
                    [
                        (
                            "te",
                            TargetEncoder(
                                target_type="binary",
                                cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            ),
                            ["row_id"],
                        )
                    ]
                ),
            )
        ]
    )
    return features, target, preprocessor


def _experiment(preprocessor):
    return LeakageExperiment(
        preprocessor=preprocessor,
        model=LogisticRegression(),
        sampler=SMOTE(sampling_strategy=0.5, random_state=0),
        cv=CrossValidator(3, 0),
        calculator=MetricsCalculator(),
        random_state=0,
        n_jobs=1,
        threads_per_worker=1,
    )


def test_fit_all_leaks_own_label_while_correct_and_unsupervised_do_not(identifier_data):
    features, target, preprocessor = identifier_data
    results = _experiment(preprocessor).run(
        features,
        target,
        (LeakageVariant.CORRECT, LeakageVariant.LEAK_UNSUPERVISED, LeakageVariant.LEAK_FIT_ALL),
    )

    share = target.mean()
    assert np.mean(results[LeakageVariant.LEAK_FIT_ALL].valid_pr_auc) > 0.95
    assert np.mean(results[LeakageVariant.CORRECT].valid_pr_auc) < share + 0.1
    assert np.mean(results[LeakageVariant.LEAK_UNSUPERVISED].valid_pr_auc) < share + 0.1


def test_fit_all_matrix_encodes_each_row_with_its_own_label(identifier_data):
    features, target, preprocessor = identifier_data

    matrix = LeakyMatrixBuilder.fit_all_then_transform(preprocessor, features, target)
    honest = LeakyMatrixBuilder.fit_transform_all(preprocessor, features, target)

    assert np.corrcoef(matrix[:, 0], target)[0, 1] > 0.9
    # Крос-фітинг: невідома в інших фолдах категорія отримує середнє фолда — без власної мітки.
    assert np.ptp(honest[:, 0]) < np.ptp(matrix[:, 0]) / 10


def test_smote_before_split_keeps_real_rows_in_their_folds():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(300, 3))
    target = pd.Series((rng.random(300) < 0.15).astype(int))
    cv = CrossValidator(3, 0)
    evaluator = SmoteBeforeSplitEvaluator(
        SMOTE(sampling_strategy=0.5, random_state=0),
        LogisticRegression(),
        cv,
        MetricsCalculator(),
        random_state=0,
        n_jobs=1,
        threads_per_worker=1,
    )

    resampled, labels, fold_of_row, n_real = evaluator.resample_with_folds(matrix, target)

    negatives = int((target == 0).sum())
    assert n_real == 300
    assert len(labels) - n_real == int(negatives * 0.5) - int(target.sum())
    np.testing.assert_array_equal(resampled[:n_real], matrix)
    for fold, (_, valid_index) in enumerate(cv.splits(pd.DataFrame(matrix), target)):
        assert (fold_of_row[valid_index] == fold).all()
    assert set(np.unique(fold_of_row[n_real:])) <= {0, 1, 2}
