"""Запуск однієї конфігурації «модель × сценарій» у крос-валідації та її підсумок.

Конфігурація — це модель, сценарій даних і, можливо, змінена частота
аугментації чи зменшений розмір навчальної частини. Для кожної:

1. крос-валідація на train (однакові фолди для всіх конфігурацій);
2. поріг класифікації — за максимумом F1 на OOF-прогнозах (лише train);
3. метрики кожного фолда при порозі 0.5 і при обраному порозі;
4. PR-AUC на власних навчальних рядках (для розриву train − CV);
5. для сценаріїв з аугментацією — прогнози на синтетичних рядках.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from mllabs.labs.lab02.scenarios import ScenarioCatalog, ScenarioId
from mllabs.modeling import (
    CrossValidationResult,
    CrossValidator,
    F1ThresholdSelector,
    MetricsCalculator,
    ModelFactory,
    ModelKind,
    mean_and_std,
    positive_proba,
)
from mllabs.preprocessing import SAMPLER_STEP

#: Ключ результату інспектора: ймовірності класу 1 на синтетичних рядках.
SYNTHETIC_PROBA = "synthetic_proba"


class SyntheticRecorder(BaseEstimator):
    """Обгортка семплера, що запам'ятовує створені ним синтетичні рядки.

    Працює як звичайний семплер imblearn (метод ``fit_resample``), тож стоїть
    у пайплайні на місці SMOTE/ADASYN і НЕ змінює навчання: модель бачить
    рівно ті самі рядки. Після ``fit`` пайплайна в атрибуті
    ``synthetic_features_`` лежать синтетичні рядки — саме ті, на яких
    навчалася модель (а не відтворені повторним запуском).

    SMOTE і ADASYN повертають спершу всі реальні рядки в початковому порядку,
    а синтетичні дописують у кінець, тому синтетичні — це «хвіст» результату.

    Args:
        sampler: Нефітований семплер imblearn.
    """

    def __init__(self, sampler: BaseEstimator) -> None:
        self.sampler = sampler

    def fit_resample(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        **params: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Викликає семплер і зберігає синтетичні рядки.

        Імена ``X``/``y`` вимушені: так метод називається в API imblearn.
        """
        self.sampler_ = clone(self.sampler)
        resampled_features, resampled_target = self.sampler_.fit_resample(X, y, **params)
        n_real = len(y)
        self.synthetic_features_ = np.asarray(resampled_features)[n_real:]
        self.synthetic_target_ = np.asarray(resampled_target)[n_real:]
        return resampled_features, resampled_target


class SyntheticFitProbe:
    """Інспектор фолда: ймовірності, які навчена модель дає синтетичним рядкам.

    Якщо модель «запам'ятала» синтетику, на синтетичних рядках вона буде
    майже впевнена (ймовірність ≈ 1), а на реальних позитивних валідації —
    ні. Це ознака перенавчання на вигаданих точках.

    Args:
        positive_label: Мітка позитивного класу.
    """

    def __init__(self, positive_label: int = 1) -> None:
        self._positive_label = positive_label

    def __call__(
        self, pipeline: BaseEstimator, features: pd.DataFrame, target: pd.Series
    ) -> dict[str, Any]:
        """Повертає ``{SYNTHETIC_PROBA: масив}`` для навченого пайплайна фолда."""
        recorder = pipeline.named_steps[SAMPLER_STEP]
        model = pipeline[-1]
        return {
            SYNTHETIC_PROBA: positive_proba(
                model, recorder.synthetic_features_, self._positive_label
            )
        }


@dataclass(frozen=True)
class ConfigurationKey:
    """Що саме оцінювалося.

    Attributes:
        model: Модель.
        scenario: Сценарій даних.
        sampling_strategy: Частота аугментації (0 — без аугментації).
        train_size: Розмір навчальної частини фолда (``None`` — повна).
    """

    model: ModelKind
    scenario: ScenarioId
    sampling_strategy: float = 0.0
    train_size: int | None = None


@dataclass(frozen=True)
class ConfigurationResult:
    """Підсумок крос-валідації однієї конфігурації.

    Attributes:
        key: Що оцінювалося.
        oof_proba: OOF-ймовірності класу 1 для всіх рядків train.
        target: Мітки train (у тому ж порядку).
        threshold: Поріг з максимальним F1 на OOF-прогнозах.
        folds_default: Метрики фолдів при порозі 0.5 (рядок — фолд).
        folds_tuned: Метрики фолдів при обраному порозі.
        train_pr_auc: PR-AUC на власних навчальних рядках, по фолдах.
        augmented_train_pr_auc: PR-AUC на навчальних рядках РАЗОМ із
            синтетичними, по фолдах (``None`` без аугментації).
        synthetic_proba: Ймовірності класу 1 на синтетичних рядках усіх
            фолдів (``None`` без аугментації).
        train_positive_proba: Ймовірності на реальних позитивних навчальних
            рядках усіх фолдів.
        fit_seconds: Середній час навчання фолда, с.
        predict_seconds: Середній час прогнозу валідаційної частини, с.
        n_features: Кількість ознак на вході моделі.
        train_rows: Реальних навчальних рядків у фолді.
        train_positives: Реальних позитивних навчальних рядків у фолді.
        convergence_warnings: Скільки ``ConvergenceWarning`` було в усіх фолдах.
    """

    key: ConfigurationKey
    oof_proba: np.ndarray
    target: np.ndarray
    threshold: float
    folds_default: pd.DataFrame
    folds_tuned: pd.DataFrame
    train_pr_auc: np.ndarray
    augmented_train_pr_auc: np.ndarray | None
    synthetic_proba: np.ndarray | None
    train_positive_proba: np.ndarray
    fit_seconds: float
    predict_seconds: float
    n_features: int
    train_rows: int
    train_positives: int
    convergence_warnings: int

    @classmethod
    def from_cv(
        cls,
        key: ConfigurationKey,
        cv: CrossValidationResult,
        calculator: MetricsCalculator,
        selector: F1ThresholdSelector,
    ) -> Self:
        """Рахує підсумок із сирих результатів крос-валідації."""
        oof = cv.oof_proba
        threshold = selector.select(cv.target, oof)
        is_positive = calculator.is_positive(cv.target)
        augmented, synthetic = None, None
        if cv.folds[0].extras:
            synthetic_by_fold = cv.extras(SYNTHETIC_PROBA)
            augmented = np.array(
                [
                    calculator.pr_auc(
                        np.concatenate([is_positive[fold.train_index], np.ones(len(proba))]),
                        np.concatenate([fold.train_proba, proba]),
                    )
                    for fold, proba in zip(cv.folds, synthetic_by_fold, strict=True)
                ]
            )
            synthetic = np.concatenate(synthetic_by_fold)
        train_positive = np.concatenate(
            [fold.train_proba[is_positive[fold.train_index] == 1] for fold in cv.folds]
        )
        return cls(
            key=key,
            oof_proba=oof,
            target=cv.target,
            threshold=threshold,
            folds_default=cv.fold_metrics(calculator),
            folds_tuned=cv.fold_metrics(calculator, threshold),
            train_pr_auc=cv.train_pr_auc(calculator),
            augmented_train_pr_auc=augmented,
            synthetic_proba=synthetic,
            train_positive_proba=train_positive,
            fit_seconds=cv.fit_seconds,
            predict_seconds=cv.predict_seconds,
            n_features=cv.n_features,
            train_rows=cv.train_rows,
            train_positives=int(is_positive[cv.folds[0].train_index].sum()),
            convergence_warnings=cv.convergence_warnings,
        )

    def cv_mean(self, metric: str, *, tuned: bool = False) -> float:
        """Середнє метрики по фолдах (при 0.5 або при обраному порозі)."""
        frame = self.folds_tuned if tuned else self.folds_default
        return float(frame[metric].mean())

    def cv_std(self, metric: str, *, tuned: bool = False) -> float:
        """Вибіркове σ метрики по фолдах."""
        frame = self.folds_tuned if tuned else self.folds_default
        return mean_and_std(frame[metric])[1]

    @property
    def fold_pr_auc(self) -> np.ndarray:
        """PR-AUC валідації по фолдах (для парних різниць)."""
        return self.folds_default["pr_auc"].to_numpy()

    @property
    def pr_auc(self) -> float:
        """Середній PR-AUC валідації."""
        return self.cv_mean("pr_auc")

    @property
    def train_gap(self) -> float:
        """Розрив «PR-AUC на реальних навчальних рядках − PR-AUC валідації»."""
        return float(self.train_pr_auc.mean() - self.pr_auc)

    @property
    def augmented_gap(self) -> float | None:
        """Розрив «PR-AUC на навчальних + синтетичних рядках − PR-AUC валідації»."""
        if self.augmented_train_pr_auc is None:
            return None
        return float(self.augmented_train_pr_auc.mean() - self.pr_auc)

    def valid_positive_proba(self, calculator: MetricsCalculator) -> np.ndarray:
        """OOF-ймовірності реальних позитивних рядків (кожен — від моделі, що його не бачила)."""
        return self.oof_proba[calculator.is_positive(self.target) == 1]


class ConfigurationRunner:
    """Виконує крос-валідацію конфігурацій на навчальній вибірці.

    Args:
        catalog: Сценарії даних.
        models: Фабрика моделей.
        validator: Крос-валідатор (однакові фолди для всіх конфігурацій).
        calculator: Калькулятор метрик.
        selector: Вибір порога на OOF-прогнозах.
        features: Ознаки train.
        target: Мітки train.
    """

    def __init__(
        self,
        catalog: ScenarioCatalog,
        models: ModelFactory,
        validator: CrossValidator,
        calculator: MetricsCalculator,
        selector: F1ThresholdSelector,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> None:
        self._catalog = catalog
        self._models = models
        self._validator = validator
        self._calculator = calculator
        self._selector = selector
        self._features = features
        self._target = target

    def run(self, key: ConfigurationKey) -> ConfigurationResult:
        """Оцінює конфігурацію в крос-валідації."""
        spec = self._catalog.spec(key.scenario)
        augmented = spec.is_augmented
        pipeline = self._catalog.pipeline(
            spec,
            self._models.create(key.model),
            sampling_strategy=key.sampling_strategy if augmented else None,
            sampler_wrapper=SyntheticRecorder if augmented else None,
        )
        cv = self._validator.evaluate(
            pipeline,
            self._features,
            self._target,
            train_size=key.train_size,
            inspector=SyntheticFitProbe() if augmented else None,
        )
        return ConfigurationResult.from_cv(key, cv, self._calculator, self._selector)
