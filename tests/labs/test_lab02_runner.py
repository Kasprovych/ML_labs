"""Тести запуску конфігурацій ЛР2: синтетика, валідаційні рядки, поріг без test."""

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mllabs.data import ProcessedDatasetRepository
from mllabs.labs.lab02.comparison import GridResult
from mllabs.labs.lab02.runner import (
    ConfigurationKey,
    ConfigurationRunner,
    SyntheticFitProbe,
    SyntheticRecorder,
)
from mllabs.labs.lab02.scenarios import ScenarioCatalog, ScenarioId
from mllabs.labs.lab02.test_evaluation import HoldoutEvaluator
from mllabs.modeling import (
    CrossValidator,
    F1ThresholdSelector,
    MetricsCalculator,
    ModelFactory,
    ModelKind,
    ModelSettings,
)
from mllabs.preprocessing import PreprocessorFactory


@pytest.fixture
def imbalanced():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(300, 4))
    labels = (features[:, 0] + rng.normal(0, 0.5, 300) > 1.3).astype(int)
    return features, labels


@pytest.fixture(scope="module")
def lab1_data(lab1_project):
    repository = ProcessedDatasetRepository(lab1_project.data_processed_dir, "kick")
    return repository.load_schema(), repository.load()


@pytest.fixture(scope="module")
def runner_parts(lab1_data):
    schema, split = lab1_data
    catalog = ScenarioCatalog(PreprocessorFactory(schema, 0), 0.5, 5, 0.05, 0)
    models = ModelFactory(ModelSettings(forest_estimators=10, boosting_max_iter=10), 0)
    runner = ConfigurationRunner(
        catalog,
        models,
        CrossValidator(3, 0),
        MetricsCalculator(),
        F1ThresholdSelector(),
        split.x_train,
        split.y_train,
    )
    return catalog, models, runner, split


def test_recorder_keeps_real_rows_first_and_records_synthetic_tail(imbalanced):
    features, labels = imbalanced
    recorder = SyntheticRecorder(SMOTE(sampling_strategy=0.5, random_state=0))

    resampled, resampled_labels = recorder.fit_resample(features, labels)

    expected, _ = SMOTE(sampling_strategy=0.5, random_state=0).fit_resample(features, labels)
    np.testing.assert_array_equal(resampled, expected)
    np.testing.assert_array_equal(resampled[: len(features)], features)
    np.testing.assert_array_equal(recorder.synthetic_features_, expected[len(features) :])
    assert (recorder.synthetic_target_ == 1).all()
    assert len(recorder.synthetic_features_) == (resampled_labels == 1).sum() - labels.sum()


def test_recorder_does_not_change_what_the_model_learns(imbalanced):
    features, labels = imbalanced
    plain = ImbalancedPipeline(
        [
            ("scale", StandardScaler()),
            ("sampler", SMOTE(random_state=0)),
            ("model", LogisticRegression()),
        ]
    )
    recorded = ImbalancedPipeline(
        [
            ("scale", StandardScaler()),
            ("sampler", SyntheticRecorder(SMOTE(random_state=0))),
            ("model", LogisticRegression()),
        ]
    )

    plain.fit(features, labels)
    recorded.fit(features, labels)

    np.testing.assert_allclose(plain.predict_proba(features), recorded.predict_proba(features))
    # Інспектор дає ймовірність для кожного синтетичного рядка, на якому вчилася модель.
    probe = SyntheticFitProbe()(recorded, pd.DataFrame(features), pd.Series(labels))
    assert len(probe["synthetic_proba"]) == len(recorded.named_steps["sampler"].synthetic_features_)


def test_sampler_does_not_touch_validation_rows(runner_parts):
    _, _, runner, split = runner_parts

    result = runner.run(ConfigurationKey(ModelKind.LOGREG, ScenarioId.C_SMOTE, 0.5))

    # OOF має рівно стільки значень, скільки реальних рядків train, — синтетичних серед них немає.
    assert len(result.oof_proba) == len(split.x_train)
    assert not np.isnan(result.oof_proba).any()
    positives = int(result.train_positives)
    negatives = result.train_rows - positives
    # SMOTE доводить клас 1 до половини класу 0 лише в навчальній частині фолда.
    assert len(result.synthetic_proba) == 3 * (negatives // 2 - positives)
    assert result.augmented_train_pr_auc is not None


def test_configuration_without_sampler_has_no_synthetic_rows(runner_parts):
    _, _, runner, _ = runner_parts

    result = runner.run(ConfigurationKey(ModelKind.HIST_GRADIENT_BOOSTING, ScenarioId.B))

    assert result.synthetic_proba is None
    assert result.augmented_gap is None
    assert 0 < result.threshold < 1
    assert len(result.folds_default) == 3


def test_threshold_is_chosen_without_test_data(runner_parts):
    catalog, models, runner, split = runner_parts
    result = runner.run(ConfigurationKey(ModelKind.LOGREG, ScenarioId.B))
    grid = GridResult({(ModelKind.LOGREG, ScenarioId.B): result})
    evaluator = HoldoutEvaluator(catalog, models, MetricsCalculator(), n_jobs=1)
    specs = [catalog.spec(ScenarioId.B)]
    train = (split.x_train, split.y_train)
    shuffled = pd.Series(
        np.random.default_rng(0).permutation(split.y_test.to_numpy()), index=split.y_test.index
    )

    original = evaluator.evaluate(
        grid, [ModelKind.LOGREG], specs, train, (split.x_test, split.y_test)
    )
    with_other_labels = evaluator.evaluate(
        grid, [ModelKind.LOGREG], specs, train, (split.x_test, shuffled)
    )

    used = [
        holdout.get(ModelKind.LOGREG, ScenarioId.B).at_threshold.threshold
        for holdout in (original, with_other_labels)
    ]
    assert used == [result.threshold, result.threshold]


class _SpyValidator(CrossValidator):
    """Крос-валідатор, що запам'ятовує, які рядки йому передали."""

    def __init__(self):
        super().__init__(3, 0)
        self.seen = []

    def evaluate(self, pipeline, features, target, **kwargs):
        self.seen.append(features)
        return super().evaluate(pipeline, features, target, **kwargs)


def test_runner_never_receives_test_rows(lab1_data):
    schema, split = lab1_data
    spy = _SpyValidator()
    runner = ConfigurationRunner(
        ScenarioCatalog(PreprocessorFactory(schema, 0), 0.5, 5, 0.05, 0),
        ModelFactory(random_state=0),
        spy,
        MetricsCalculator(),
        F1ThresholdSelector(),
        split.x_train,
        split.y_train,
    )

    for scenario in (ScenarioId.A, ScenarioId.B, ScenarioId.C_SMOTE):
        runner.run(ConfigurationKey(ModelKind.LOGREG, scenario, 0.5))

    assert len(spy.seen) == 3
    # Індекси train і test після читання CSV обидва починаються з 0, тож перевіряємо
    # саму таблицю: крос-валідація отримує саме об'єкт x_train і нічого більше.
    assert all(features is split.x_train for features in spy.seen)
