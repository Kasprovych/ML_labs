"""Фінальна оцінка на test заздалегідь визначеного переліку моделей (D8).

Test уже використовувався в ЛР2 для інших конфігурацій. Повторне використання
доречне лише за умови, що за ним НІЧОГО НЕ ОБИРАЄТЬСЯ: перелік моделей
(baseline + найкращі параметри кожного методу з самостійного пошуку)
визначено до запуску, кожна модель оцінюється один раз, а різниця з baseline
подається з бутстреп-інтервалом.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from mllabs.labs.lab03.search_space import SearchSpace
from mllabs.modeling import BootstrapInterval, MetricsCalculator, PairedBootstrap, positive_proba

#: Ключ baseline у результатах.
BASELINE_KEY = "baseline"


@dataclass(frozen=True)
class HoldoutResult:
    """Прогнози всіх моделей на test.

    Attributes:
        truth: Мітки test (0/1, 1 — позитивний клас).
        probas: Ключ моделі → ймовірності класу 1 на test.
        pr_auc: Ключ моделі → PR-AUC на test.
        roc_auc: Ключ моделі → ROC-AUC на test.
    """

    truth: np.ndarray
    probas: Mapping[str, np.ndarray]
    pr_auc: Mapping[str, float]
    roc_auc: Mapping[str, float]


class FinalModelEvaluator:
    """Навчає моделі з заданими параметрами на dev і оцінює на test.

    Args:
        calculator: Калькулятор метрик.
        bootstrap: Парний бутстреп для різниць із baseline.
        prefix: Назва кроку моделі в пайплайні.
        threads: Ліміт потоків OpenMP у головному процесі.
    """

    def __init__(
        self,
        calculator: MetricsCalculator,
        bootstrap: PairedBootstrap,
        prefix: str,
        threads: int,
    ) -> None:
        self._calculator = calculator
        self._bootstrap = bootstrap
        self._prefix = prefix
        self._threads = threads

    def evaluate(
        self,
        pipeline: BaseEstimator,
        candidates: Mapping[str, Mapping[str, int | float]],
        train: tuple[pd.DataFrame, pd.Series],
        test: tuple[pd.DataFrame, pd.Series],
    ) -> HoldoutResult:
        """Навчає кожного кандидата на ``train`` і прогнозує ``test``.

        Args:
            pipeline: Нефітований пайплайн з параметрами baseline.
            candidates: Ключ моделі → параметри моделі (порожні — baseline).
            train: Ознаки й мітки dev-вибірки.
            test: Ознаки й мітки test.
        """
        (x_train, y_train), (x_test, y_test) = train, test
        probas = {}
        with threadpool_limits(self._threads):
            for key, params in candidates.items():
                model = clone(pipeline).set_params(**SearchSpace.prefixed(params, self._prefix))
                model.fit(x_train, y_train)
                probas[key] = positive_proba(model, x_test, self._calculator.positive_label)
        metrics = {key: self._calculator.compute(y_test, proba) for key, proba in probas.items()}
        return HoldoutResult(
            truth=self._calculator.is_positive(y_test),
            probas=probas,
            pr_auc={key: value.pr_auc for key, value in metrics.items()},
            roc_auc={key: value.roc_auc for key, value in metrics.items()},
        )

    def intervals(self, result: HoldoutResult) -> dict[str, BootstrapInterval]:
        """Парний бутстреп «модель − baseline» (PR-AUC) для кожної моделі, крім baseline."""
        reference = result.probas[BASELINE_KEY]
        return {
            key: self._bootstrap.interval(result.truth, reference, proba, average_precision_score)
            for key, proba in result.probas.items()
            if key != BASELINE_KEY
        }
