"""Три методи оптимізації гіперпараметрів з однаковим інтерфейсом (патерн «Стратегія»).

* **Grid Search** (``GridSearchCV``) — перебір УСІХ точок сітки. Простий і
  паралелиться ідеально, але кількість точок росте експоненційно з кількістю
  параметрів (3 значення × 4 параметри = 81, × 5 = 243).
* **Random Search** (``RandomizedSearchCV``) — N випадкових точок із
  розподілів. Кожен параметр отримує N різних значень (а не 3, як у сітці),
  тому за той самий бюджет він частіше натрапляє на добрі значення важливих
  параметрів (Bergstra & Bengio, 2012).
* **Bayesian Optimization** (Optuna, семплер TPE) — наступна точка обирається
  за результатами попередніх: TPE будує дві оцінки щільності — «де були добрі
  результати» і «де погані» — і пропонує точку з найбільшим їх відношенням.
  Тому спроби йдуть ПОСЛІДОВНО (кожна чекає на попередні).

Усі стратегії отримують ті самі внутрішні розбиття (``splits``) і ту саму
метрику, повертають однакову структуру :class:`SearchOutcome` і нічого не
навчають «остаточно» (``refit=False``): час — це лише оцінювання кандидатів.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, override

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs, parallel_config
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._search import BaseSearchCV

from mllabs.labs.lab03.config import Lab3Config
from mllabs.labs.lab03.search_space import SearchSpace
from mllabs.modeling import mean_and_std

#: Розбиття «позиції навчальних рядків / позиції валідаційних рядків».
Splits = Sequence[tuple[np.ndarray, np.ndarray]]


class SearchKind(StrEnum):
    """Сімейство методу пошуку."""

    GRID = "grid"
    RANDOM = "random"
    OPTUNA = "optuna"


@dataclass(frozen=True)
class EvaluationRecord:
    """Одна оцінена точка простору (кандидат).

    Attributes:
        index: Номер оцінки в порядку виконання (з 0).
        params: Значення гіперпараметрів (без префікса кроку).
        mean_score: Середній PR-AUC по внутрішніх фолдах.
        std_score: Вибіркове σ PR-AUC по фолдах.
        fold_scores: PR-AUC кожного фолда.
        compute_seconds: Сумарний час навчання й оцінювання всіх фолдів, с.
    """

    index: int
    params: Mapping[str, int | float]
    mean_score: float
    std_score: float
    fold_scores: tuple[float, ...]
    compute_seconds: float


@dataclass(frozen=True)
class SearchOutcome:
    """Результат одного запуску методу пошуку.

    Attributes:
        key: Ключ методу (``grid``, ``random_20``, ...).
        label: Назва для таблиць і графіків.
        n_candidates: Скільки точок оцінено.
        n_fits: Скільки разів навчалася модель (кандидати × фолди).
        wall_seconds: Реальний («настінний») час усього пошуку, с.
        compute_seconds: Сума часу всіх навчань і оцінювань (без паралелізму), с.
        best_params: Найкращі параметри за середнім PR-AUC.
        best_score: Найкращий середній PR-AUC (Best Score, Inner CV).
        best_score_std: σ PR-AUC найкращого кандидата по фолдах.
        history: Усі оцінки в порядку виконання.
        workers: Скільки навчань ішло одночасно (процесів joblib).
        threads_per_fit: Потоків OpenMP в одному навчанні. Через це
            ``compute_seconds`` різних методів НЕ порівнянні: навчання в 3
            потоки (Optuna) коротше за навчання в 1 потік (Grid/Random), хоча
            займає втричі більше ядер. Методи порівнюються за ``wall_seconds``.
    """

    key: str
    label: str
    n_candidates: int
    n_fits: int
    wall_seconds: float
    compute_seconds: float
    best_params: Mapping[str, int | float]
    best_score: float
    best_score_std: float
    history: tuple[EvaluationRecord, ...]
    workers: int
    threads_per_fit: int

    def best_so_far(self) -> np.ndarray:
        """Найкращий PR-AUC серед перших k оцінок, k = 1..N (крива збіжності)."""
        # maximum.accumulate — «біжучий максимум»: [0.3, 0.5, 0.4] → [0.3, 0.5, 0.5].
        return np.maximum.accumulate([record.mean_score for record in self.history])


class HyperparameterSearch(Protocol):
    """Інтерфейс методу пошуку (як ``interface`` у Java, але структурний)."""

    @property
    def key(self) -> str:
        """Ключ методу для ``metrics.json``."""
        ...

    @property
    def label(self) -> str:
        """Назва методу для звіту."""
        ...

    def run(
        self, pipeline: BaseEstimator, features: pd.DataFrame, target: pd.Series, splits: Splits
    ) -> SearchOutcome:
        """Оцінює кандидатів на заданих розбиттях і повертає найкращого."""
        ...


def _build_outcome(
    key: str,
    label: str,
    history: list[EvaluationRecord],
    n_splits: int,
    wall_seconds: float,
    workers: int,
    threads_per_fit: int,
) -> SearchOutcome:
    """Збирає :class:`SearchOutcome` з історії оцінок (спільне для всіх методів)."""
    # Перший із найкращих — так само обирає sklearn (rank_test_score == 1).
    best = max(history, key=lambda record: record.mean_score)
    return SearchOutcome(
        key=key,
        label=label,
        n_candidates=len(history),
        n_fits=len(history) * n_splits,
        wall_seconds=wall_seconds,
        compute_seconds=float(sum(record.compute_seconds for record in history)),
        best_params=dict(best.params),
        best_score=best.mean_score,
        best_score_std=best.std_score,
        history=tuple(history),
        workers=workers,
        threads_per_fit=threads_per_fit,
    )


class SklearnSearchStrategy(ABC):
    """Спільна частина Grid і Random Search (обидва — нащадки ``BaseSearchCV``).

    Паралелізм — по парах «кандидат × фолд» (``n_jobs`` процесів), а потоки
    OpenMP у кожному процесі обмежено ``threads_per_worker``: паралельним є
    лише один рівень, інакше 10 процесів × 10 потоків бустингу змагалися б
    за 10 ядер.

    Args:
        space: Простір гіперпараметрів.
        scoring: Метрика sklearn (``"average_precision"``).
        prefix: Назва кроку моделі в пайплайні (префікс параметрів).
        n_jobs: Кількість процесів (−1 — усі ядра).
        threads_per_worker: Ліміт потоків OpenMP/BLAS у кожному процесі.
    """

    def __init__(
        self,
        space: SearchSpace,
        scoring: str,
        prefix: str,
        n_jobs: int,
        threads_per_worker: int,
    ) -> None:
        self._space = space
        self._scoring = scoring
        self._prefix = prefix
        self._n_jobs = n_jobs
        self._threads_per_worker = threads_per_worker

    @property
    @abstractmethod
    def key(self) -> str:
        """Ключ методу."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Назва методу для звіту."""

    @abstractmethod
    def _build(self, pipeline: BaseEstimator, splits: Splits) -> BaseSearchCV:
        """Створює нефітований об'єкт пошуку sklearn."""

    def run(
        self, pipeline: BaseEstimator, features: pd.DataFrame, target: pd.Series, splits: Splits
    ) -> SearchOutcome:
        """Запускає пошук і перетворює ``cv_results_`` на :class:`SearchOutcome`."""
        search = self._build(pipeline, splits)
        # parallel_config задає ліміт потоків OpenMP у процесах-працівниках loky.
        with parallel_config(backend="loky", inner_max_num_threads=self._threads_per_worker):
            started = time.perf_counter()
            search.fit(features, target)
            wall_seconds = time.perf_counter() - started
        return _build_outcome(
            self.key,
            self.label,
            self._history(search.cv_results_, len(splits)),
            len(splits),
            wall_seconds,
            # effective_n_jobs перетворює −1 на фактичну кількість процесів (ядер).
            effective_n_jobs(self._n_jobs),
            self._threads_per_worker,
        )

    def _history(self, results: Mapping[str, Any], n_splits: int) -> list[EvaluationRecord]:
        """Записи оцінок із ``cv_results_`` (словник масивів, по елементу на кандидата)."""
        history = []
        for index, params in enumerate(results["params"]):
            folds = tuple(
                float(results[f"split{fold}_test_score"][index]) for fold in range(n_splits)
            )
            mean, spread = mean_and_std(folds)
            # mean_fit_time — середнє по фолдах; × n_splits = сумарний час кандидата.
            per_fold = results["mean_fit_time"][index] + results["mean_score_time"][index]
            history.append(
                EvaluationRecord(
                    index=index,
                    params=self._space.strip(params, self._prefix),
                    mean_score=mean,
                    std_score=spread,
                    fold_scores=folds,
                    compute_seconds=float(per_fold * n_splits),
                )
            )
        return history


class GridSearchStrategy(SklearnSearchStrategy):
    """Повний перебір сітки ``SearchSpace.param_grid`` (``GridSearchCV``)."""

    @property
    @override
    def key(self) -> str:
        return SearchKind.GRID.value

    @property
    @override
    def label(self) -> str:
        return "Grid Search"

    @override
    def _build(self, pipeline: BaseEstimator, splits: Splits) -> BaseSearchCV:
        return GridSearchCV(
            pipeline,
            self._space.param_grid(self._prefix),
            scoring=self._scoring,
            cv=list(splits),
            refit=False,
            n_jobs=self._n_jobs,
            error_score="raise",
        )


class RandomSearchStrategy(SklearnSearchStrategy):
    """N випадкових точок із розподілів ``SearchSpace.distributions`` (``RandomizedSearchCV``).

    З тим самим цілим ``random_state`` sklearn генерує ту саму послідовність
    кандидатів, тож перші 20 точок пошуку з N = 50 збігаються з пошуком N = 20:
    різниця між ними — рівно «ще 30 ітерацій», а не везіння іншого зерна.

    Args:
        n_iter: Кількість кандидатів.
        random_state: Зерно генератора кандидатів.
        space: Простір гіперпараметрів.
        scoring: Метрика sklearn.
        prefix: Назва кроку моделі в пайплайні.
        n_jobs: Кількість процесів.
        threads_per_worker: Ліміт потоків OpenMP у кожному процесі.
    """

    def __init__(
        self,
        n_iter: int,
        random_state: int,
        *,
        space: SearchSpace,
        scoring: str,
        prefix: str,
        n_jobs: int,
        threads_per_worker: int,
    ) -> None:
        # «*» у сигнатурі: усі параметри після нього передаються лише за іменем.
        super().__init__(space, scoring, prefix, n_jobs, threads_per_worker)
        self._n_iter = n_iter
        self._random_state = random_state

    @property
    @override
    def key(self) -> str:
        return f"{SearchKind.RANDOM.value}_{self._n_iter}"

    @property
    @override
    def label(self) -> str:
        return f"Random Search (N={self._n_iter})"

    @override
    def _build(self, pipeline: BaseEstimator, splits: Splits) -> BaseSearchCV:
        return RandomizedSearchCV(
            pipeline,
            self._space.distributions(self._prefix),
            n_iter=self._n_iter,
            scoring=self._scoring,
            cv=list(splits),
            refit=False,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
            error_score="raise",
        )


class OptunaSearchStrategy:
    """Байєсівська оптимізація: Optuna з семплером TPE.

    Спроби виконуються строго послідовно — лише так TPE з фіксованим ``seed``
    відтворюваний і справді «вчиться» на попередніх спробах (з паралельними
    спробами половина з них стартувала б, не знаючи результатів інших). Зате
    3 фолди кожної спроби рахуються паралельно (``fold_n_jobs`` процесів по
    ``threads_per_worker`` потоків OpenMP).

    Прунер (раннє відкидання поганих спроб) не використовується: кожна спроба
    проходить ті самі фолди, інакше «кількість оцінок» не була б порівнянна з
    Grid і Random.

    Args:
        space: Простір гіперпараметрів.
        scoring: Метрика sklearn (та сама, що в Grid/Random).
        prefix: Назва кроку моделі в пайплайні.
        n_trials: Кількість спроб.
        startup_trials: Скільки перших спроб TPE робить випадково.
        multivariate: Чи моделювати параметри спільно.
        random_state: Зерно семплера TPE.
        fold_n_jobs: Скільки фолдів спроби рахувати паралельно.
        threads_per_worker: Ліміт потоків OpenMP у кожному процесі.
    """

    def __init__(
        self,
        space: SearchSpace,
        scoring: str,
        prefix: str,
        n_trials: int,
        startup_trials: int,
        multivariate: bool,
        random_state: int,
        fold_n_jobs: int,
        threads_per_worker: int,
    ) -> None:
        self._space = space
        self._scoring = scoring
        self._prefix = prefix
        self._n_trials = n_trials
        self._startup_trials = startup_trials
        self._multivariate = multivariate
        self._random_state = random_state
        self._fold_n_jobs = fold_n_jobs
        self._threads_per_worker = threads_per_worker

    @property
    def key(self) -> str:
        """Ключ методу."""
        return f"{SearchKind.OPTUNA.value}_{self._n_trials}"

    @property
    def label(self) -> str:
        """Назва методу для звіту."""
        return f"Bayesian / Optuna TPE (N={self._n_trials})"

    @property
    def startup_trials(self) -> int:
        """Кількість стартових випадкових спроб (для графіка збіжності)."""
        return self._startup_trials

    def run(
        self, pipeline: BaseEstimator, features: pd.DataFrame, target: pd.Series, splits: Splits
    ) -> SearchOutcome:
        """Виконує ``n_trials`` спроб і повертає історію в порядку виконання."""
        # Optuna за замовчуванням пише рядок у журнал на кожну спробу — лишаємо лише попередження.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        with warnings.catch_warnings():
            # multivariate=True позначено в Optuna як «експериментальне» — це відоме
            # попередження про API, а не про наші дані; свідомо його приглушуємо.
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            sampler = optuna.samplers.TPESampler(
                seed=self._random_state,
                multivariate=self._multivariate,
                n_startup_trials=self._startup_trials,
            )
        study = optuna.create_study(direction="maximize", sampler=sampler)

        with (
            parallel_config(backend="loky", inner_max_num_threads=self._threads_per_worker),
            # Один пул процесів на всі спроби: не запускати процеси заново 20 разів.
            Parallel(n_jobs=self._fold_n_jobs) as parallel,
        ):
            started = time.perf_counter()
            study.optimize(
                lambda trial: self._objective(trial, parallel, pipeline, features, target, splits),
                n_trials=self._n_trials,
            )
            wall_seconds = time.perf_counter() - started

        history = [
            EvaluationRecord(
                index=trial.number,
                params={name: trial.params[name] for name in self._space.names},
                mean_score=float(trial.value),
                std_score=float(trial.user_attrs["std_score"]),
                fold_scores=tuple(trial.user_attrs["fold_scores"]),
                compute_seconds=float(trial.user_attrs["compute_seconds"]),
            )
            for trial in study.trials
        ]
        return _build_outcome(
            self.key,
            self.label,
            history,
            len(splits),
            wall_seconds,
            self._fold_n_jobs,
            self._threads_per_worker,
        )

    def _objective(
        self,
        trial: optuna.Trial,
        parallel: Parallel,
        pipeline: BaseEstimator,
        features: pd.DataFrame,
        target: pd.Series,
        splits: Splits,
    ) -> float:
        """Ціль спроби: середній PR-AUC на тих самих фолдах, що й у Grid/Random."""
        params = self._space.suggest(trial)
        candidate = clone(pipeline).set_params(**SearchSpace.prefixed(params, self._prefix))
        results = parallel(
            delayed(score_fold)(candidate, features, target, train, valid, self._scoring)
            for train, valid in splits
        )
        scores = [score for score, _ in results]
        mean, spread = mean_and_std(scores)
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("std_score", spread)
        trial.set_user_attr("compute_seconds", float(sum(seconds for _, seconds in results)))
        return mean


def score_fold(
    pipeline: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    scoring: str,
) -> tuple[float, float]:
    """Навчає копію пайплайна на фолді й рахує метрику так само, як ``GridSearchCV``.

    Returns:
        Пара «значення метрики, час навчання + оцінювання, с».
    """
    started = time.perf_counter()
    model = clone(pipeline).fit(features.iloc[train_index], target.iloc[train_index])
    score = get_scorer(scoring)(model, features.iloc[valid_index], target.iloc[valid_index])
    return float(score), time.perf_counter() - started


class SearchStrategyFactory:
    """Створює стратегії пошуку з параметрами конфігурації в порядку завдання.

    Args:
        config: Параметри лабораторної.
        prefix: Назва кроку моделі в пайплайні.
        random_state: Зерно Random Search і TPE.
    """

    def __init__(self, config: Lab3Config, prefix: str, random_state: int) -> None:
        self._config = config
        self._prefix = prefix
        self._random_state = random_state

    def create_all(self) -> list[HyperparameterSearch]:
        """Grid → Random (кожне N) → Optuna — порядок таблиці методів."""
        config = self._config
        sklearn_options = {
            "space": config.search_space,
            "scoring": config.scoring,
            "prefix": self._prefix,
            "n_jobs": config.search_n_jobs,
            "threads_per_worker": config.search_threads_per_worker,
        }
        strategies: list[HyperparameterSearch] = [GridSearchStrategy(**sklearn_options)]
        strategies.extend(
            RandomSearchStrategy(n_iter, self._random_state, **sklearn_options)
            for n_iter in config.random_iterations
        )
        strategies.append(
            OptunaSearchStrategy(
                space=config.search_space,
                scoring=config.scoring,
                prefix=self._prefix,
                n_trials=config.optuna_trials,
                startup_trials=config.optuna_startup_trials,
                multivariate=config.optuna_multivariate,
                random_state=self._random_state,
                fold_n_jobs=config.optuna_fold_n_jobs,
                threads_per_worker=config.optuna_threads_per_worker,
            )
        )
        return strategies
