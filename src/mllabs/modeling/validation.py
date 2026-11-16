"""Крос-валідація з OOF-прогнозами та прогнозами на навчальних рядках.

Стандартна ``sklearn.model_selection.cross_validate`` повертає лише числа
метрик. Для ЛР2–3 потрібно більше:

* **OOF-прогнози** (out-of-fold): кожен рядок train отримує ймовірність від
  моделі, яка цього рядка НЕ бачила. На них обирається поріг класифікації;
* **прогнози на власних навчальних рядках** фолда — щоб виміряти розрив
  «train − валідація», тобто перенавчання;
* **однакові фолди** для всіх моделей і сценаріїв — умова парних порівнянь;
* **зменшення навчальної частини** фолда (криві навчання) при повній валідації;
* **«інспектор» фолда** — довільна функція, що після навчання дивиться на
  навчену модель (наприклад, на синтетичні рядки SMOTE).

Тому цикл по фолдах написано вручну; фолди можна виконувати паралельно.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, train_test_split

from mllabs.modeling.fit_input import fit_input_proba, model_input_width, with_fit_input_recorder
from mllabs.modeling.metrics import DEFAULT_THRESHOLD, MetricsCalculator


class FoldInspector(Protocol):
    """Інтерфейс «інспектора»: що ще виміряти на навченому пайплайні фолда.

    ``Protocol`` — структурний інтерфейс: підходить будь-який об'єкт із таким
    методом ``__call__`` (тобто який можна викликати як функцію), явно
    «implements» писати не треба. Під час паралельного виконання інспектор
    передається в інший процес, тому має бути класом рівня модуля.
    """

    def __call__(
        self, pipeline: BaseEstimator, features: pd.DataFrame, target: pd.Series
    ) -> Mapping[str, Any]:
        """Повертає довільні результати для фолда.

        Args:
            pipeline: Пайплайн, навчений на навчальній частині фолда.
            features: Навчальна частина фолда (ті самі рядки, що бачив ``fit``).
            target: Мітки навчальної частини.
        """
        ...


@dataclass(frozen=True)
class FoldPrediction:
    """Результат одного фолда.

    Attributes:
        fold: Номер фолда (з 0).
        train_index: Позиції рядків, на яких навчалася модель.
        valid_index: Позиції валідаційних рядків.
        valid_proba: Ймовірності позитивного класу для валідаційних рядків.
        train_proba: Ймовірності для власних навчальних рядків (реальних, без
            синтетичних), пораховані на ТІЙ САМІЙ матриці ознак, яку модель
            отримала під час ``fit`` (див. ``mllabs.modeling.fit_input``), а не
            через ``predict_proba(x_train)``: ``TargetEncoder`` кодує навчальні
            рядки під час ``fit`` інакше (з крос-фітингом), ніж під час ``transform``.
        fit_seconds: Час навчання всього пайплайна, с.
        predict_seconds: Час прогнозу валідаційної частини, с.
        n_features: Кількість ознак на вході моделі.
        convergence_warnings: Скільки ``ConvergenceWarning`` видав ``fit``.
        extras: Результат інспектора фолда (порожній, якщо його немає).
    """

    fold: int
    train_index: np.ndarray
    valid_index: np.ndarray
    valid_proba: np.ndarray
    train_proba: np.ndarray
    fit_seconds: float
    predict_seconds: float
    n_features: int
    convergence_warnings: int
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossValidationResult:
    """Усі фолди однієї конфігурації «препроцесинг → [семплер] → модель».

    Attributes:
        folds: Результати фолдів у порядку номерів.
        target: Мітки всіх рядків, переданих у крос-валідацію.
    """

    folds: tuple[FoldPrediction, ...]
    target: np.ndarray

    @property
    def oof_proba(self) -> np.ndarray:
        """OOF-ймовірності: кожен рядок — від моделі, що його не бачила."""
        oof = np.full(len(self.target), np.nan)
        for fold in self.folds:
            oof[fold.valid_index] = fold.valid_proba
        return oof

    def fold_metrics(
        self, calculator: MetricsCalculator, threshold: float = DEFAULT_THRESHOLD
    ) -> pd.DataFrame:
        """Метрики валідаційної частини кожного фолда (рядок — фолд)."""
        rows = [
            calculator.compute(self.target[fold.valid_index], fold.valid_proba, threshold).to_dict()
            for fold in self.folds
        ]
        return pd.DataFrame(rows, index=pd.Index([fold.fold for fold in self.folds], name="fold"))

    def train_pr_auc(self, calculator: MetricsCalculator) -> np.ndarray:
        """PR-AUC моделі на її ВЛАСНИХ навчальних рядках, по фолдах."""
        return np.array(
            [
                calculator.pr_auc(self.target[fold.train_index], fold.train_proba)
                for fold in self.folds
            ]
        )

    def valid_pr_auc(self, calculator: MetricsCalculator) -> np.ndarray:
        """PR-AUC на валідаційних частинах, по фолдах."""
        return np.array(
            [
                calculator.pr_auc(self.target[fold.valid_index], fold.valid_proba)
                for fold in self.folds
            ]
        )

    @property
    def fit_seconds(self) -> float:
        """Середній час навчання одного фолда, с."""
        return float(np.mean([fold.fit_seconds for fold in self.folds]))

    @property
    def predict_seconds(self) -> float:
        """Середній час прогнозу валідаційної частини одного фолда, с."""
        return float(np.mean([fold.predict_seconds for fold in self.folds]))

    @property
    def n_features(self) -> int:
        """Кількість ознак на вході моделі (у першому фолді)."""
        return self.folds[0].n_features

    @property
    def train_rows(self) -> int:
        """Кількість реальних навчальних рядків у фолді (у першому фолді)."""
        return len(self.folds[0].train_index)

    @property
    def convergence_warnings(self) -> int:
        """Сумарна кількість ``ConvergenceWarning`` по всіх фолдах."""
        return sum(fold.convergence_warnings for fold in self.folds)

    def extras(self, key: str) -> list[Any]:
        """Значення ``key`` з результатів інспектора по фолдах."""
        return [fold.extras[key] for fold in self.folds]


class CrossValidator:
    """Стратифікована k-кратна крос-валідація з однаковими фолдами.

    Для кожного фолда пайплайн КЛОНУЄТЬСЯ (чиста ненавчена копія) і навчається
    лише на навчальній частині: імп'ютери, енкодери й семплер не бачать
    валідаційних рядків. Семплер (SMOTE тощо) у ``imblearn.pipeline.Pipeline``
    працює лише під час ``fit``, тож валідація відбувається на реальних рядках.

    Перепідписка потоків (oversubscription). Паралельність тут має три рівні:
    фолди цього класу (``n_jobs`` процесів), ``n_jobs`` всередині оцінювача
    (ліс, ``GridSearchCV``) і потоки OpenMP/BLAS моделі (бустинг, лінійна
    алгебра). Якщо всі три просять «усі ядра», потоків стає в десятки разів
    більше за ядра, і навчання сповільнюється. Що робить joblib:

    * у кожному процесі-працівнику потоки OpenMP/BLAS обмежуються до
      ``ядра // n_jobs`` — або до ``threads_per_worker``, якщо його задано;
    * вкладений ``Parallel`` усередині працівника (``n_jobs`` лісу чи пошуку)
      виконується в ПОТОКАХ, і це обмеження на нього не діє.

    Правило для ЛР3: паралельним має бути лише один рівень — або фолди тут
    (``n_jobs > 1``, а у пошуку й моделі ``n_jobs=1``), або пошук/модель
    (тут ``n_jobs=1``).

    Args:
        n_splits: Кількість фолдів.
        random_state: Зерно перемішування фолдів і підвибірок.
        positive_label: Мітка позитивного класу (чия ймовірність береться).
        n_jobs: Скільки фолдів виконувати паралельно (1 — послідовно).
        threads_per_worker: Ліміт потоків OpenMP/BLAS у кожному процесі-працівнику;
            ``None`` — стандарт joblib (``ядра // n_jobs``).
    """

    def __init__(
        self,
        n_splits: int,
        random_state: int,
        positive_label: int = 1,
        n_jobs: int = 1,
        threads_per_worker: int | None = None,
    ) -> None:
        self._n_splits = n_splits
        self._random_state = random_state
        self._positive_label = positive_label
        self._n_jobs = n_jobs
        self._threads_per_worker = threads_per_worker

    @property
    def n_splits(self) -> int:
        """Кількість фолдів."""
        return self._n_splits

    def splits(
        self, features: pd.DataFrame, target: pd.Series
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Позиції рядків «навчання / валідація» для кожного фолда.

        Залежить лише від міток, кількості фолдів і зерна, тому однакове в усіх
        викликах на тих самих даних — умова парного порівняння конфігурацій.
        """
        folds = StratifiedKFold(
            n_splits=self._n_splits, shuffle=True, random_state=self._random_state
        )
        return list(folds.split(features, target))

    def evaluate(
        self,
        pipeline: BaseEstimator,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        train_size: int | None = None,
        inspector: FoldInspector | None = None,
        n_jobs: int | None = None,
    ) -> CrossValidationResult:
        """Навчає й оцінює пайплайн на кожному фолді.

        Args:
            pipeline: Нефітований пайплайн, модель або обгортка пошуку
                (клонується для кожного фолда; у копію вставляється рекордер
                матриці ``fit`` — див. ``mllabs.modeling.fit_input``).
            features: Ознаки навчальної вибірки (НЕ тестової).
            target: Мітки навчальної вибірки.
            train_size: Якщо задано — навчальна частина кожного фолда
                стратифіковано зменшується до стількох рядків (валідаційна —
                повна). Для кривих навчання.
            inspector: Необов'язкова функція, що виконується після ``fit`` фолда.
            n_jobs: Паралельність саме цього виклику; ``None`` — як у конструкторі.

        Returns:
            Результати всіх фолдів.
        """
        labels = target.to_numpy()
        tasks = []
        for fold, (train_index, valid_index) in enumerate(self.splits(features, target)):
            used_index = (
                train_index
                if train_size is None
                else self._subsample(train_index, labels, train_size)
            )
            # delayed(f)(args) не викликає f одразу, а запам'ятовує «завдання» —
            # Parallel виконає завдання у кількох процесах і поверне результати по порядку.
            tasks.append(
                delayed(_run_fold)(
                    with_fit_input_recorder(clone(pipeline)),
                    features,
                    target,
                    fold,
                    used_index,
                    valid_index,
                    self._positive_label,
                    inspector,
                )
            )
        effective_jobs = n_jobs or self._n_jobs
        if effective_jobs == 1:
            folds = Parallel(n_jobs=1)(tasks)
        else:
            # parallel_config задає ліміт потоків OpenMP/BLAS у процесах-працівниках loky.
            with parallel_config(backend="loky", inner_max_num_threads=self._threads_per_worker):
                folds = Parallel(n_jobs=effective_jobs)(tasks)
        return CrossValidationResult(folds=tuple(folds), target=labels)

    def _subsample(self, train_index: np.ndarray, labels: np.ndarray, size: int) -> np.ndarray:
        """Стратифікована підвибірка позицій навчальної частини фолда."""
        if size >= len(train_index):
            return train_index
        picked, _ = train_test_split(
            train_index,
            train_size=size,
            stratify=labels[train_index],
            random_state=self._random_state,
        )
        return np.sort(picked)


def _run_fold(
    pipeline: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    fold: int,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    positive_label: int,
    inspector: FoldInspector | None,
) -> FoldPrediction:
    """Навчає пайплайн на одному фолді й збирає прогнози (виконується у процесі-працівнику)."""
    x_train, y_train = features.iloc[train_index], target.iloc[train_index]
    x_valid = features.iloc[valid_index]

    # catch_warnings(record=True) перехоплює попередження в список замість друку:
    # ConvergenceWarning — факт для аналізу (чутливість до масштабу), а не шум у журналі.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
    n_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in caught)

    started = time.perf_counter()
    valid_proba = positive_proba(pipeline, x_valid, positive_label)
    predict_seconds = time.perf_counter() - started
    # Не predict_proba(x_train): потрібні ознаки, які модель справді бачила під час fit.
    train_proba = fit_input_proba(pipeline, x_train, positive_label)

    extras = dict(inspector(pipeline, x_train, y_train)) if inspector is not None else {}
    return FoldPrediction(
        fold=fold,
        train_index=train_index,
        valid_index=valid_index,
        valid_proba=valid_proba,
        train_proba=train_proba,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        n_features=model_input_width(pipeline),
        convergence_warnings=n_warnings,
        extras=extras,
    )


def positive_proba(
    model: BaseEstimator, features: pd.DataFrame | np.ndarray, positive_label: int = 1
) -> np.ndarray:
    """Ймовірність позитивного класу з ``predict_proba`` навченої моделі.

    ``predict_proba`` повертає матрицю «рядок × клас»; потрібна колонка
    позитивного класу, порядок класів — у ``classes_``.
    """
    column = list(model.classes_).index(positive_label)
    return model.predict_proba(features)[:, column]


def mean_and_std(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Середнє і вибіркове σ (``ddof=1``: ділення на n − 1, чесніше на кількох фолдах)."""
    array = np.asarray(values, dtype=float)
    spread = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return float(array.mean()), spread
