"""Вкладена крос-валідація (Nested CV): незміщена оцінка якості всієї процедури підбору.

Чому «Best Score» пошуку оптимістичний: пошук бачить N кандидатів, кожна
оцінка на 3 фолдах трохи «шумить», і максимум із N шумних чисел систематично
вищий за справжню якість переможця (як найкращий із 81 кидків кубика). Що
більше кандидатів, то більший цей «оптимізм».

Вкладена CV відокремлює ВИБІР від ОЦІНКИ:

* **зовнішня петля** (K = 5): дані ділиться на 5 частин; кожна по черзі —
  «зовнішня валідація», якої пошук не бачить узагалі;
* **внутрішня петля** (K = 3): на решті 4/5 запускається ВЕСЬ метод пошуку
  зі своєю 3-кратною CV; найкращі параметри → навчання на всіх 4/5 →
  оцінка на відкладеній 1/5.

Середнє по 5 зовнішніх фолдах оцінює не «одну модель», а процедуру «метод
пошуку + навчання» — саме вона буде застосована в реальній задачі.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from threadpoolctl import threadpool_limits

from mllabs.labs.lab03.search import HyperparameterSearch
from mllabs.labs.lab03.search_space import SearchSpace
from mllabs.modeling import CrossValidator, MetricsCalculator, mean_and_std, positive_proba


@dataclass(frozen=True)
class NestedFoldResult:
    """Один зовнішній фолд.

    Attributes:
        fold: Номер зовнішнього фолда (з 0).
        inner_best: Best Score внутрішнього пошуку (середній PR-AUC на 3 фолдах).
        outer_pr_auc: PR-AUC переможця на зовнішній валідації.
        outer_roc_auc: ROC-AUC переможця на зовнішній валідації.
        best_params: Параметри, обрані внутрішнім пошуком.
        search_wall_seconds: Час внутрішнього пошуку, с.
        refit_seconds: Час навчання переможця на зовнішньому train, с.
    """

    fold: int
    inner_best: float
    outer_pr_auc: float
    outer_roc_auc: float
    best_params: dict[str, int | float]
    search_wall_seconds: float
    refit_seconds: float

    @property
    def optimism(self) -> float:
        """Наскільки внутрішня оцінка вища за зовнішню на цьому фолді."""
        return self.inner_best - self.outer_pr_auc


@dataclass(frozen=True)
class NestedResult:
    """Вкладена CV одного методу пошуку.

    Attributes:
        key: Ключ методу.
        label: Назва методу.
        folds: Результати зовнішніх фолдів.
        wall_seconds: Загальний час вкладеної CV, с.
    """

    key: str
    label: str
    folds: tuple[NestedFoldResult, ...]
    wall_seconds: float

    @property
    def fold_scores(self) -> np.ndarray:
        """Зовнішні PR-AUC по фолдах."""
        return np.array([fold.outer_pr_auc for fold in self.folds])

    @property
    def mean_std(self) -> tuple[float, float]:
        """Unbiased Score: середній зовнішній PR-AUC і його σ."""
        return mean_and_std(self.fold_scores)

    @property
    def roc_auc(self) -> float:
        """Середній зовнішній ROC-AUC."""
        return float(np.mean([fold.outer_roc_auc for fold in self.folds]))

    @property
    def inner_mean(self) -> float:
        """Середній Best Score внутрішніх пошуків."""
        return float(np.mean([fold.inner_best for fold in self.folds]))

    @property
    def fold_optimism(self) -> np.ndarray:
        """Сирий оптимізм «inner best − outer» по зовнішніх фолдах."""
        return np.array([fold.optimism for fold in self.folds])

    def optimism(self) -> tuple[float, float]:
        """СИРИЙ оптимізм: середнє і σ різниць «inner best − outer» по фолдах.

        Змішує два ефекти: оптимізм відбору (максимум із N шумних оцінок) і
        зсув, що є і без відбору (внутрішні моделі вчаться на меншій частині
        даних, фолди різні за складністю). Розділяє їх :meth:`selection_optimism`.
        """
        return mean_and_std(self.fold_optimism)

    def selection_optimism(self, baseline_shift: Sequence[float]) -> tuple[float, float]:
        """Оптимізм ВІДБОРУ: сирий оптимізм мінус зсув baseline на тих самих фолдах.

        По фолду k: ``(inner_k − outer_k)`` методу ``− (inner_k − outer_k)``
        baseline. Baseline нічого не обирає, тож його «inner − outer» — це
        зсув без відбору; різниця лишає лише ефект вибору найкращого з N.

        Args:
            baseline_shift: «Inner − outer» baseline по тих самих зовнішніх фолдах.
        """
        return mean_and_std(self.fold_optimism - np.asarray(baseline_shift))


class NestedCrossValidation:
    """Зовнішня петля вкладеної CV для будь-якого :class:`HyperparameterSearch`.

    Зовнішні й внутрішні розбиття будуються тими самими ``CrossValidator``, що й
    у baseline та самостійному пошуку, — тож усі методи й baseline оцінюються
    на ОДНАКОВИХ зовнішніх фолдах, і різниці між ними можна брати попарно.

    Args:
        outer_cv: Розбиття зовнішньої петлі (K = 5).
        inner_cv: Розбиття внутрішньої петлі (K = 3), застосовуються до зовнішнього train.
        calculator: Калькулятор метрик.
        prefix: Назва кроку моделі в пайплайні.
        refit_threads: Ліміт потоків OpenMP для навчання переможця в головному процесі.
    """

    def __init__(
        self,
        outer_cv: CrossValidator,
        inner_cv: CrossValidator,
        calculator: MetricsCalculator,
        prefix: str,
        refit_threads: int,
    ) -> None:
        self._outer_cv = outer_cv
        self._inner_cv = inner_cv
        self._calculator = calculator
        self._prefix = prefix
        self._refit_threads = refit_threads

    def outer_splits(
        self, features: pd.DataFrame, target: pd.Series
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Зовнішні розбиття (однакові для всіх методів і baseline)."""
        return self._outer_cv.splits(features, target)

    def run(
        self,
        strategy: HyperparameterSearch,
        pipeline: BaseEstimator,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> NestedResult:
        """Виконує вкладену CV одного методу пошуку.

        Args:
            strategy: Метод пошуку (той самий об'єкт, що й у самостійному пошуку).
            pipeline: Нефітований пайплайн «препроцесор → модель».
            features: Ознаки dev-вибірки.
            target: Мітки dev-вибірки.
        """
        started = time.perf_counter()
        folds = [
            self._run_fold(strategy, pipeline, features, target, fold, train_index, valid_index)
            for fold, (train_index, valid_index) in enumerate(self.outer_splits(features, target))
        ]
        return NestedResult(
            key=strategy.key,
            label=strategy.label,
            folds=tuple(folds),
            wall_seconds=time.perf_counter() - started,
        )

    def _run_fold(
        self,
        strategy: HyperparameterSearch,
        pipeline: BaseEstimator,
        features: pd.DataFrame,
        target: pd.Series,
        fold: int,
        train_index: np.ndarray,
        valid_index: np.ndarray,
    ) -> NestedFoldResult:
        """Внутрішній пошук на зовнішньому train → refit → оцінка на зовнішньому val."""
        x_train, y_train = features.iloc[train_index], target.iloc[train_index]
        x_valid, y_valid = features.iloc[valid_index], target.iloc[valid_index]
        # Внутрішні розбиття — позиції ВСЕРЕДИНІ x_train: зовнішній val пошук не бачить.
        inner_splits = self._inner_cv.splits(x_train, y_train)
        outcome = strategy.run(pipeline, x_train, y_train, inner_splits)

        winner = clone(pipeline).set_params(
            **SearchSpace.prefixed(outcome.best_params, self._prefix)
        )
        # threadpool_limits обмежує потоки OpenMP у ГОЛОВНОМУ процесі (refit іде тут).
        with threadpool_limits(self._refit_threads):
            refit_started = time.perf_counter()
            winner.fit(x_train, y_train)
            refit_seconds = time.perf_counter() - refit_started
            proba = positive_proba(winner, x_valid, self._calculator.positive_label)
        metrics = self._calculator.compute(y_valid, proba)
        return NestedFoldResult(
            fold=fold,
            inner_best=outcome.best_score,
            outer_pr_auc=metrics.pr_auc,
            outer_roc_auc=metrics.roc_auc,
            best_params=dict(outcome.best_params),
            search_wall_seconds=outcome.wall_seconds,
            refit_seconds=refit_seconds,
        )
