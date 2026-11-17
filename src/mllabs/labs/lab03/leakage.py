"""Дослідження витоку даних (Data Leakage).

УВАГА: ЦЕЙ МОДУЛЬ НАВМИСНО МІСТИТЬ НЕКОРЕКТНИЙ КОД — ДЕМОНСТРАЦІЮ ПОМИЛКИ.
Класи :class:`LeakyMatrixBuilder`, :class:`SmoteBeforeSplitEvaluator` і
некоректна гілка :class:`LeakageHoldoutCheck` показують, ЯК НЕ ТРЕБА робити.
Це єдине місце в проєкті з таким кодом; більше ніде воно не використовується.

Витік даних — ситуація, коли під час навчання модель (або препроцесинг)
отримує інформацію, якої в реальному застосуванні не буде, — найчастіше про
валідаційні чи тестові рядки. Оцінка якості тоді завищена «штучно».

Класичний приклад — препроцесор, навчений ДО розбиття на фолди на всіх
даних: медіани, параметри масштабування, а головне — ``TargetEncoder``
(середня частка класу 1 у категорії) «бачать» валідаційні рядки разом з їхніми
мітками. Якщо категорія рідкісна (кілька рядків), закодоване значення фактично
містить відповідь.

Варіанти (усі — та сама модель, ті самі 5 фолдів):

* ``correct`` — препроцесор у ``Pipeline``, ``fit`` лише на навчальній частині фолда;
* ``leak_unsupervised`` — ``препроцесор.fit_transform(усі дані)`` до розбиття;
  ``TargetEncoder`` тут кодує з крос-фітингом (без власної мітки рядка), але
  статистики всіх інших кроків порахована і по валідаційних рядках;
* ``leak_fit_all`` — ``препроцесор.fit(усі дані)``, потім ``transform(усі дані)``:
  кожен рядок закодовано статистикою, у якій є ЙОГО ВЛАСНА мітка (головний
  «некоректний» пайплайн);
* ``correct_smote`` — SMOTE усередині ``imblearn.Pipeline`` (лише на train фолда);
* ``leak_smote`` — SMOTE на всіх даних до розбиття: синтетичні точки,
  інтерпольовані з валідаційних позитивів, потрапляють у train.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from mllabs.modeling import (
    BootstrapInterval,
    CrossValidationResult,
    CrossValidator,
    MetricsCalculator,
    PairedBootstrap,
    positive_proba,
)
from mllabs.preprocessing import build_model_pipeline

#: Функція «препроцесор, ознаки, мітки → матриця» (методи :class:`LeakyMatrixBuilder`).
MatrixBuilder = Callable[[BaseEstimator, pd.DataFrame, pd.Series], np.ndarray]


class LeakageVariant(StrEnum):
    """Варіант пайплайна в дослідженні витоку."""

    CORRECT = "correct"
    LEAK_UNSUPERVISED = "leak_unsupervised"
    LEAK_FIT_ALL = "leak_fit_all"
    CORRECT_SMOTE = "correct_smote"
    LEAK_SMOTE = "leak_smote"


#: Назви варіантів для таблиць і графіків.
VARIANT_LABELS = {
    LeakageVariant.CORRECT: "Коректний",
    LeakageVariant.LEAK_UNSUPERVISED: "Препроцесинг fit_transform(усі)",
    LeakageVariant.LEAK_FIT_ALL: "Некоректний: fit(усі)",
    LeakageVariant.CORRECT_SMOTE: "Коректний + SMOTE",
    LeakageVariant.LEAK_SMOTE: "SMOTE до розбиття",
}

#: Що саме «протікає» у валідацію.
VARIANT_LEAKS = {
    LeakageVariant.CORRECT: "нічого",
    LeakageVariant.LEAK_UNSUPERVISED: "статистики імп'ютерів, скейлерів, агрегатів по val",
    LeakageVariant.LEAK_FIT_ALL: "те саме + TargetEncoder з власною міткою рядка",
    LeakageVariant.CORRECT_SMOTE: "нічого",
    # Порівнюється з «Коректний + SMOTE», але препроцесор тут навчено на всіх
    # даних (fit_transform, як у LEAK_UNSUPERVISED), тож Δ змішує два джерела;
    # внесок препроцесингу на повному train ≈ 0,002 (див. LEAK_UNSUPERVISED).
    LeakageVariant.LEAK_SMOTE: "синтетика з val-позитивів у train + fit_transform(усі)",
}

#: З яким коректним варіантом порівнюється кожен некоректний.
REFERENCE_VARIANT = {
    LeakageVariant.LEAK_UNSUPERVISED: LeakageVariant.CORRECT,
    LeakageVariant.LEAK_FIT_ALL: LeakageVariant.CORRECT,
    LeakageVariant.LEAK_SMOTE: LeakageVariant.CORRECT_SMOTE,
}


@dataclass(frozen=True)
class LeakageResult:
    """Оцінки одного варіанта по фолдах.

    Attributes:
        variant: Варіант пайплайна.
        valid_pr_auc: PR-AUC на (реальних) валідаційних рядках фолдів.
        valid_roc_auc: ROC-AUC на (реальних) валідаційних рядках фолдів.
        train_pr_auc: PR-AUC на власних навчальних рядках (реальних) фолдів.
        naive_pr_auc: Лише для ``leak_smote``: PR-AUC, якщо валідувати і на
            синтетичних рядках (як зробив би новачок). Частка класу 1 там інша,
            тож з іншими варіантами ця PR-AUC непорівнянна.
        naive_roc_auc: Те саме для ROC-AUC (від частки класів не залежить).
    """

    variant: LeakageVariant
    valid_pr_auc: tuple[float, ...]
    valid_roc_auc: tuple[float, ...]
    train_pr_auc: tuple[float, ...]
    naive_pr_auc: tuple[float, ...] = field(default=())
    naive_roc_auc: tuple[float, ...] = field(default=())


class LeakyMatrixBuilder:
    """ДЕМОНСТРАЦІЯ ПОМИЛКИ: препроцесинг на всіх даних до розбиття на фолди.

    Обидва методи повертають матрицю ознак, у якій кожен рядок (включно з
    майбутніми валідаційними) уже закодовано статистиками, порахованими з
    участю валідаційних рядків. Будь-яка подальша крос-валідація на такій
    матриці оцінює модель на даних, «знайомих» препроцесору.
    """

    @staticmethod
    def fit_transform_all(
        preprocessor: BaseEstimator, features: pd.DataFrame, target: pd.Series
    ) -> np.ndarray:
        """НЕКОРЕКТНО: ``fit_transform`` на всіх даних (варіант ``leak_unsupervised``).

        ``TargetEncoder`` у ``fit_transform`` кодує з крос-фітингом, тож власна
        мітка рядка в його код не потрапляє, але мітки інших валідаційних
        рядків — потрапляють; решта кроків (медіани, Yeo-Johnson, агрегати)
        порахована по всіх рядках.
        """
        return clone(preprocessor).fit_transform(features, target)

    @staticmethod
    def fit_all_then_transform(
        preprocessor: BaseEstimator, features: pd.DataFrame, target: pd.Series
    ) -> np.ndarray:
        """НЕКОРЕКТНО: ``fit`` на всіх даних, потім ``transform`` усіх (``leak_fit_all``).

        Саме ``fit(...)`` + окремий ``transform(...)``, а не ``fit_transform``:
        ``transform`` у ``TargetEncoder`` кодує рядок статистикою ВСІХ рядків
        ``fit``, включно з ним самим, — у рідкісній категорії це майже його мітка.
        """
        fitted = clone(preprocessor).fit(features, target)
        return fitted.transform(features)


class SmoteBeforeSplitEvaluator:
    """ДЕМОНСТРАЦІЯ ПОМИЛКИ: SMOTE на всіх даних ДО розбиття на фолди.

    Реальні рядки зберігають свої фолди (ті самі, що в інших варіантах), а
    синтетичні розподіляються по фолдах випадково — так їх розподілила б
    звичайна крос-валідація на «розширеному» наборі. Синтетична точка — це
    інтерполяція двох сусідніх позитивів; якщо один із них валідаційний, у
    train потрапляє його «копія», і модель бачить відповідь.

    Оцінка рахується двічі: (а) лише на реальних валідаційних рядках — чесне
    порівняння з ``correct_smote``; (б) на валідаційних разом із синтетичними
    — так «оцінив» би новачок (частка класу 1 там інша).

    Args:
        sampler: Нефітований семплер (SMOTE).
        model: Нефітована модель.
        cv: Крос-валідатор (джерело тих самих фолдів).
        calculator: Калькулятор метрик.
        random_state: Зерно розподілу синтетичних рядків по фолдах.
        n_jobs: Кількість паралельних фолдів.
        threads_per_worker: Ліміт потоків OpenMP у процесах.
    """

    def __init__(
        self,
        sampler: BaseEstimator,
        model: BaseEstimator,
        cv: CrossValidator,
        calculator: MetricsCalculator,
        random_state: int,
        n_jobs: int,
        threads_per_worker: int,
    ) -> None:
        self._sampler = sampler
        self._model = model
        self._cv = cv
        self._calculator = calculator
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._threads_per_worker = threads_per_worker

    def resample_with_folds(
        self, matrix: np.ndarray, target: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """НЕКОРЕКТНО: аугментує всі дані й призначає кожному рядку фолд.

        Returns:
            Ознаки після SMOTE, мітки, номер фолда кожного рядка, кількість
            реальних рядків (реальні йдуть першими в початковому порядку).
        """
        resampled, labels = clone(self._sampler).fit_resample(matrix, target.to_numpy())
        n_real = len(target)
        if not np.allclose(resampled[:n_real], matrix, equal_nan=True):
            raise RuntimeError("Семплер мав залишити реальні рядки першими в початковому порядку")
        fold_of_row = np.empty(len(labels), dtype=int)
        for fold, (_, valid_index) in enumerate(self._cv.splits(pd.DataFrame(matrix), target)):
            fold_of_row[valid_index] = fold
        generator = np.random.default_rng(self._random_state)
        fold_of_row[n_real:] = generator.integers(0, self._cv.n_splits, size=len(labels) - n_real)
        return resampled, np.asarray(labels), fold_of_row, n_real

    def evaluate(self, matrix: np.ndarray, target: pd.Series) -> LeakageResult:
        """Крос-валідація на аугментованій до розбиття матриці."""
        resampled, labels, fold_of_row, n_real = self.resample_with_folds(matrix, target)
        is_real = np.arange(len(labels)) < n_real
        tasks = [
            delayed(_fit_smote_fold)(
                clone(self._model),
                resampled,
                labels,
                fold_of_row != fold,
                (fold_of_row == fold) & is_real,
                fold_of_row == fold,
                is_real,
                self._calculator,
            )
            for fold in range(self._cv.n_splits)
        ]
        with parallel_config(backend="loky", inner_max_num_threads=self._threads_per_worker):
            folds = Parallel(n_jobs=self._n_jobs)(tasks)
        # zip(*rows) «транспонує» список кортежів: [(a1, b1), (a2, b2)] → (a1, a2), (b1, b2).
        valid_pr, valid_roc, train_pr, naive_pr, naive_roc = zip(*folds, strict=True)
        return LeakageResult(
            variant=LeakageVariant.LEAK_SMOTE,
            valid_pr_auc=valid_pr,
            valid_roc_auc=valid_roc,
            train_pr_auc=train_pr,
            naive_pr_auc=naive_pr,
            naive_roc_auc=naive_roc,
        )


def _fit_smote_fold(
    model: BaseEstimator,
    resampled: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    real_valid_mask: np.ndarray,
    naive_valid_mask: np.ndarray,
    is_real: np.ndarray,
    calculator: MetricsCalculator,
) -> tuple[float, float, float, float, float]:
    """Один фолд варіанта «SMOTE до розбиття» (виконується у процесі-працівнику)."""
    model.fit(resampled[train_mask], labels[train_mask])

    def scores(mask: np.ndarray) -> tuple[float, float]:
        metrics = calculator.compute(labels[mask], positive_proba(model, resampled[mask]))
        return metrics.pr_auc, metrics.roc_auc

    valid_pr, valid_roc = scores(real_valid_mask)
    naive_pr, naive_roc = scores(naive_valid_mask)
    train_real = train_mask & is_real
    train_pr = calculator.pr_auc(labels[train_real], positive_proba(model, resampled[train_real]))
    return valid_pr, valid_roc, train_pr, naive_pr, naive_roc


class LeakageExperiment:
    """Оцінює варіанти витоку на тих самих фолдах.

    Матричні варіанти перевикористовують ``CrossValidator.evaluate`` з самою
    моделлю на готовій матриці: фолди залежать лише від міток і зерна, тож
    збігаються з фолдами коректного пайплайна — різниці можна брати попарно.

    Args:
        preprocessor: Нефітований препроцесор сценарію Б (ЛР1).
        model: Нефітована модель (HGB з параметрами ЛР2).
        sampler: Нефітований SMOTE для варіантів зі SMOTE.
        cv: Крос-валідатор (5 фолдів).
        calculator: Калькулятор метрик.
        random_state: Зерно розподілу синтетики по фолдах.
        n_jobs: Паралельні фолди.
        threads_per_worker: Ліміт потоків OpenMP у процесах.
    """

    def __init__(
        self,
        preprocessor: BaseEstimator,
        model: BaseEstimator,
        sampler: BaseEstimator,
        cv: CrossValidator,
        calculator: MetricsCalculator,
        random_state: int,
        n_jobs: int,
        threads_per_worker: int,
    ) -> None:
        self._preprocessor = preprocessor
        self._model = model
        self._sampler = sampler
        self._cv = cv
        self._calculator = calculator
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._threads_per_worker = threads_per_worker

    def run(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        variants: Sequence[LeakageVariant] = tuple(LeakageVariant),
    ) -> dict[LeakageVariant, LeakageResult]:
        """Оцінює перелічені варіанти на ``features``/``target``."""
        return {variant: self._evaluate(variant, features, target) for variant in variants}

    def run_by_size(
        self, samples: Mapping[int, tuple[pd.DataFrame, pd.Series]]
    ) -> dict[int, dict[LeakageVariant, LeakageResult]]:
        """Головна пара «коректний / некоректний» на вибірках різного обсягу."""
        pair = (LeakageVariant.CORRECT, LeakageVariant.LEAK_FIT_ALL)
        return {
            size: self.run(features, target, pair) for size, (features, target) in samples.items()
        }

    def _evaluate(
        self, variant: LeakageVariant, features: pd.DataFrame, target: pd.Series
    ) -> LeakageResult:
        """Один варіант."""
        match variant:
            case LeakageVariant.CORRECT:
                pipeline = build_model_pipeline(self._preprocessor, self._model)
                return self._from_cv(variant, self._cross_validate(pipeline, features, target))
            case LeakageVariant.CORRECT_SMOTE:
                pipeline = build_model_pipeline(self._preprocessor, self._model, self._sampler)
                return self._from_cv(variant, self._cross_validate(pipeline, features, target))
            case LeakageVariant.LEAK_UNSUPERVISED:
                matrix = self._leaky_matrix(LeakyMatrixBuilder.fit_transform_all, features, target)
                return self._from_cv(variant, self._cross_validate(self._model, matrix, target))
            case LeakageVariant.LEAK_FIT_ALL:
                matrix = self._leaky_matrix(
                    LeakyMatrixBuilder.fit_all_then_transform, features, target
                )
                return self._from_cv(variant, self._cross_validate(self._model, matrix, target))
            case LeakageVariant.LEAK_SMOTE:
                matrix = self._leaky_matrix(LeakyMatrixBuilder.fit_transform_all, features, target)
                evaluator = SmoteBeforeSplitEvaluator(
                    self._sampler,
                    self._model,
                    self._cv,
                    self._calculator,
                    self._random_state,
                    self._n_jobs,
                    self._threads_per_worker,
                )
                return evaluator.evaluate(matrix.to_numpy(), target)

    def _leaky_matrix(
        self, builder: MatrixBuilder, features: pd.DataFrame, target: pd.Series
    ) -> pd.DataFrame:
        """Будує «протеклу» матрицю у головному процесі з обмеженням потоків."""
        # Препроцесор навчається в головному процесі, поки процеси фолдів простоюють,
        # тож йому віддаємо стільки ж потоків, скільки всі фолди разом.
        with threadpool_limits(self._threads_per_worker * self._n_jobs):
            matrix = builder(self._preprocessor, features, target)
        # CrossValidator вибирає рядки через iloc, тож потрібна таблиця pandas.
        return pd.DataFrame(np.asarray(matrix), index=features.index)

    def _cross_validate(
        self, estimator: BaseEstimator, features: pd.DataFrame, target: pd.Series
    ) -> CrossValidationResult:
        """Звичайна крос-валідація ``CrossValidator``."""
        return self._cv.evaluate(estimator, features, target, n_jobs=self._n_jobs)

    def _from_cv(self, variant: LeakageVariant, result: CrossValidationResult) -> LeakageResult:
        """Перетворює результат ``CrossValidator`` на :class:`LeakageResult`."""
        return LeakageResult(
            variant=variant,
            valid_pr_auc=tuple(result.valid_pr_auc(self._calculator).tolist()),
            valid_roc_auc=tuple(result.fold_metrics(self._calculator)["roc_auc"].tolist()),
            train_pr_auc=tuple(result.train_pr_auc(self._calculator).tolist()),
        )


@dataclass(frozen=True)
class HoldoutLeakageResult:
    """Перевірка «реальністю»: коректний і некоректний пайплайни на test.

    Attributes:
        correct_pr_auc: PR-AUC коректного пайплайна на test.
        leaky_pr_auc: PR-AUC некоректного пайплайна на test.
        correct_roc_auc: ROC-AUC коректного на test.
        leaky_roc_auc: ROC-AUC некоректного на test.
        interval: Парний бутстреп «некоректний − коректний» (PR-AUC).
    """

    correct_pr_auc: float
    leaky_pr_auc: float
    correct_roc_auc: float
    leaky_roc_auc: float
    interval: BootstrapInterval


class LeakageHoldoutCheck:
    """Обидва пайплайни навчаються на всьому train і оцінюються на test.

    Некоректна гілка — ДЕМОНСТРАЦІЯ ПОМИЛКИ: препроцесор ``fit(train)``, потім
    ``transform(train)`` (кожен навчальний рядок закодовано з його власною
    міткою) → модель. Test кодується статистикою train — як було б «у
    продакшні». Модель, що звикла довіряти «підглянутому» кодуванню, на нових
    даних цієї підказки не отримує.

    Args:
        preprocessor: Нефітований препроцесор.
        model: Нефітована модель.
        calculator: Калькулятор метрик.
        bootstrap: Парний бутстреп.
        threads: Ліміт потоків OpenMP у головному процесі.
    """

    def __init__(
        self,
        preprocessor: BaseEstimator,
        model: BaseEstimator,
        calculator: MetricsCalculator,
        bootstrap: PairedBootstrap,
        threads: int,
    ) -> None:
        self._preprocessor = preprocessor
        self._model = model
        self._calculator = calculator
        self._bootstrap = bootstrap
        self._threads = threads

    def evaluate(
        self,
        train: tuple[pd.DataFrame, pd.Series],
        test: tuple[pd.DataFrame, pd.Series],
    ) -> HoldoutLeakageResult:
        """Навчає обидва пайплайни на train і порівнює на test."""
        (x_train, y_train), (x_test, y_test) = train, test
        label = self._calculator.positive_label
        with threadpool_limits(self._threads):
            correct = build_model_pipeline(self._preprocessor, self._model).fit(x_train, y_train)
            correct_proba = positive_proba(correct, x_test, label)

            leaky_preprocessor = clone(self._preprocessor).fit(x_train, y_train)
            leaky_model = clone(self._model).fit(leaky_preprocessor.transform(x_train), y_train)
            leaky_proba = positive_proba(leaky_model, leaky_preprocessor.transform(x_test), label)

        correct_metrics = self._calculator.compute(y_test, correct_proba)
        leaky_metrics = self._calculator.compute(y_test, leaky_proba)
        return HoldoutLeakageResult(
            correct_pr_auc=correct_metrics.pr_auc,
            leaky_pr_auc=leaky_metrics.pr_auc,
            correct_roc_auc=correct_metrics.roc_auc,
            leaky_roc_auc=leaky_metrics.roc_auc,
            interval=self._bootstrap.interval(
                self._calculator.is_positive(y_test),
                correct_proba,
                leaky_proba,
                average_precision_score,
            ),
        )
