"""Фінальна оцінка на відкладеній тестовій вибірці.

Тест використовується ОДИН раз і лише для заздалегідь заданого переліку
конфігурацій (3 моделі × сценарії А, Б, В-SMOTE, В-ADASYN). Кожна
конфігурація навчається на всьому train; поріг класифікації береться готовим
з OOF-прогнозів крос-валідації. За тестовими числами нічого не обирається:
ні модель, ні поріг, ні сценарій.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, confusion_matrix

from mllabs.labs.lab02.comparison import GridResult
from mllabs.labs.lab02.scenarios import ScenarioCatalog, ScenarioId, ScenarioSpec
from mllabs.modeling import (
    BootstrapInterval,
    ClassificationMetrics,
    CurveBuilder,
    MetricsCalculator,
    ModelFactory,
    ModelKind,
    PairedBootstrap,
    positive_proba,
)

#: Колонки довгих таблиць кривих.
COLUMN_MODEL = "model"
COLUMN_SCENARIO = "scenario"

#: Порівняння сценаріїв на test (базовий, порівнюваний) — як у CV.
TEST_PAIRS: tuple[tuple[ScenarioId, ScenarioId], ...] = (
    (ScenarioId.A, ScenarioId.B),
    (ScenarioId.B, ScenarioId.C_SMOTE),
    (ScenarioId.B, ScenarioId.C_ADASYN),
)


@dataclass(frozen=True)
class HoldoutPrediction:
    """Прогноз однієї конфігурації на test.

    Attributes:
        model: Модель.
        scenario: Сценарій.
        proba: Ймовірності класу 1 для рядків test.
        at_default: Метрики при порозі 0.5.
        at_threshold: Метрики при порозі з OOF-прогнозів train.
        fit_seconds: Час навчання на всьому train, с.
        convergence_warnings: Скільки ``ConvergenceWarning`` видав ``fit``.
    """

    model: ModelKind
    scenario: ScenarioId
    proba: np.ndarray
    at_default: ClassificationMetrics
    at_threshold: ClassificationMetrics
    fit_seconds: float
    convergence_warnings: int


@dataclass(frozen=True)
class HoldoutResult:
    """Усі прогнози на test.

    Attributes:
        predictions: Прогноз за ключем ``(модель, сценарій)``.
        target: Мітки test.
    """

    predictions: Mapping[tuple[ModelKind, ScenarioId], HoldoutPrediction]
    target: np.ndarray

    def get(self, model: ModelKind, scenario: ScenarioId) -> HoldoutPrediction:
        """Прогноз конфігурації."""
        return self.predictions[(model, scenario)]


class HoldoutEvaluator:
    """Навчає конфігурації на всьому train і оцінює їх на test.

    Args:
        catalog: Сценарії.
        models: Фабрика моделей.
        calculator: Калькулятор метрик.
        n_jobs: Скільки конфігурацій навчати паралельно.
    """

    def __init__(
        self,
        catalog: ScenarioCatalog,
        models: ModelFactory,
        calculator: MetricsCalculator,
        n_jobs: int,
    ) -> None:
        self._catalog = catalog
        self._models = models
        self._calculator = calculator
        self._n_jobs = n_jobs

    def evaluate(
        self,
        grid: GridResult,
        model_kinds: Sequence[ModelKind],
        specs: Sequence[ScenarioSpec],
        train: tuple[pd.DataFrame, pd.Series],
        test: tuple[pd.DataFrame, pd.Series],
    ) -> HoldoutResult:
        """Навчає кожну конфігурацію на train і рахує метрики на test.

        Args:
            grid: Результати CV — звідти береться поріг кожної конфігурації.
            model_kinds: Моделі.
            specs: Сценарії (лише ті, що позначені ``on_test``).
            train: Ознаки й мітки train.
            test: Ознаки й мітки test.
        """
        x_train, y_train = train
        x_test, y_test = test
        configurations = [(model, spec) for model in model_kinds for spec in specs if spec.on_test]
        outputs = Parallel(n_jobs=self._n_jobs)(
            delayed(_fit_and_predict)(
                self._catalog.pipeline(spec, self._models.create(model)),
                x_train,
                y_train,
                x_test,
                self._calculator.positive_label,
            )
            for model, spec in configurations
        )
        predictions = {}
        for (model, spec), (proba, fit_seconds, n_warnings) in zip(
            configurations, outputs, strict=True
        ):
            threshold = grid.get(model, spec.scenario_id).threshold
            predictions[(model, spec.scenario_id)] = HoldoutPrediction(
                model=model,
                scenario=spec.scenario_id,
                proba=proba,
                at_default=self._calculator.compute(y_test, proba),
                at_threshold=self._calculator.compute(y_test, proba, threshold),
                fit_seconds=fit_seconds,
                convergence_warnings=n_warnings,
            )
        return HoldoutResult(predictions, y_test.to_numpy())

    def bootstrap(
        self,
        result: HoldoutResult,
        model_kinds: Sequence[ModelKind],
        bootstrap: PairedBootstrap,
    ) -> dict[tuple[ModelKind, ScenarioId], BootstrapInterval]:
        """95 % ДІ різниць PR-AUC на test для пар сценаріїв ``TEST_PAIRS``.

        Returns:
            Інтервал за ключем ``(модель, порівнюваний сценарій)``.
        """
        truth = self._calculator.is_positive(result.target)
        pairs = [(model, base, other) for model in model_kinds for base, other in TEST_PAIRS]
        intervals = Parallel(n_jobs=self._n_jobs)(
            delayed(bootstrap.interval)(
                truth,
                result.get(model, base).proba,
                result.get(model, other).proba,
                average_precision_score,
            )
            for model, base, other in pairs
        )
        return {
            (model, other): interval
            for (model, _, other), interval in zip(pairs, intervals, strict=True)
        }

    # ------------------------------------------------------------------ таблиці

    def results_table(self, result: HoldoutResult) -> pd.DataFrame:
        """Метрики test при порозі з OOF-прогнозів train (головна таблиця test)."""
        rows = [
            {
                "Конфігурація": self._configuration_label(prediction),
                "Accuracy": prediction.at_threshold.accuracy,
                "Precision": prediction.at_threshold.precision,
                "Recall": prediction.at_threshold.recall,
                "F1": prediction.at_threshold.f1,
                "ROC-AUC": prediction.at_threshold.roc_auc,
                "PR-AUC": prediction.at_threshold.pr_auc,
            }
            for prediction in result.predictions.values()
        ]
        return pd.DataFrame(rows)

    def threshold_effect_table(self, result: HoldoutResult) -> pd.DataFrame:
        """Порогові метрики test при 0,5 і при порозі з OOF-прогнозів."""
        rows = [
            {
                "Конфігурація": self._configuration_label(prediction),
                "Поріг": prediction.at_threshold.threshold,
                "Precision (0,5)": prediction.at_default.precision,
                "Recall (0,5)": prediction.at_default.recall,
                "F1 (0,5)": prediction.at_default.f1,
                "Recall (поріг)": prediction.at_threshold.recall,
                "F1 (поріг)": prediction.at_threshold.f1,
            }
            for prediction in result.predictions.values()
        ]
        return pd.DataFrame(rows)

    def metrics_matrix(self, result: HoldoutResult) -> pd.DataFrame:
        """Матриця «конфігурація × метрика» для теплової карти (при OOF-порозі)."""
        return self.results_table(result).set_index("Конфігурація")

    def curves(
        self, result: HoldoutResult, builder: CurveBuilder
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Довгі таблиці кривих PR і ROC на test для всіх конфігурацій."""
        pr_frames, roc_frames = [], []
        for (model, scenario), prediction in result.predictions.items():
            labels = {COLUMN_MODEL: model.value, COLUMN_SCENARIO: scenario.value}
            pr_frames.append(
                builder.precision_recall(result.target, prediction.proba).assign(**labels)
            )
            roc_frames.append(builder.roc(result.target, prediction.proba).assign(**labels))
        return pd.concat(pr_frames, ignore_index=True), pd.concat(roc_frames, ignore_index=True)

    def confusion_matrices(
        self, result: HoldoutResult, model: ModelKind, scenarios: Sequence[ScenarioId]
    ) -> dict[str, np.ndarray]:
        """Матриці помилок моделі на test при OOF-порозі: підпис сценарію → матриця 2×2."""
        truth = self._calculator.is_positive(result.target)
        matrices = {}
        for scenario in scenarios:
            prediction = result.get(model, scenario)
            predicted = (prediction.proba >= prediction.at_threshold.threshold).astype(int)
            matrices[self._catalog.label(scenario)] = confusion_matrix(
                truth, predicted, labels=[0, 1]
            )
        return matrices

    def to_metrics(self, result: HoldoutResult) -> dict[str, Any]:
        """Розділ ``test`` для ``metrics.json``."""
        metrics: dict[str, Any] = {}
        for (model, scenario), prediction in result.predictions.items():
            tuned, default = prediction.at_threshold, prediction.at_default
            metrics.setdefault(model.value, {})[scenario.value] = {
                "pr_auc": tuned.pr_auc,
                "roc_auc": tuned.roc_auc,
                "accuracy": tuned.accuracy,
                "precision": tuned.precision,
                "recall": tuned.recall,
                "f1": tuned.f1,
                "threshold": tuned.threshold,
                "accuracy_at_05": default.accuracy,
                "precision_at_05": default.precision,
                "recall_at_05": default.recall,
                "f1_at_05": default.f1,
                "brier": tuned.brier,
                "mean_proba": tuned.mean_proba,
                "fit_seconds_full_train": prediction.fit_seconds,
                "convergence_warnings": prediction.convergence_warnings,
            }
        metrics["majority_baseline_accuracy"] = self._calculator.majority_accuracy(result.target)
        metrics["n_rows"] = len(result.target)
        return metrics

    def flat_metrics(self, result: HoldoutResult) -> dict[tuple[ModelKind, ScenarioId], dict]:
        """Ті самі числа за ключем ``(модель, сценарій)`` — для інших зведень."""
        nested = self.to_metrics(result)
        return {key: nested[key[0].value][key[1].value] for key in result.predictions}

    def _configuration_label(self, prediction: HoldoutPrediction) -> str:
        return (
            f"{ModelFactory.short_name(prediction.model)} · "
            f"{self._catalog.label(prediction.scenario)}"
        )


def _fit_and_predict(
    pipeline: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    positive_label: int,
) -> tuple[np.ndarray, float, int]:
    """Навчає пайплайн на всьому train; повертає ймовірності test, час і кількість попереджень."""
    # Попередження про незбіжність записуються як факт (див. CrossValidator), а не друкуються.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
    n_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    return positive_proba(pipeline, x_test, positive_label), fit_seconds, n_warnings
