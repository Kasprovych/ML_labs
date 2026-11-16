"""Метрики бінарної класифікації з дисбалансом класів.

Метрики двох типів:

* **ранжувальні** (PR-AUC, ROC-AUC) — оцінюють, наскільки добре модель
  впорядковує рядки за ймовірністю; від порога не залежать;
* **порогові** (Accuracy, Precision, Recall, F1) — оцінюють рішення «так/ні»
  після порівняння ймовірності з порогом.

Brier score — середній квадрат різниці між ймовірністю і справжньою міткою
(0/1): оцінює, наскільки ймовірності «чесні» (відкалібровані). Чим менше, тим краще.

Позитивний клас — рідкісний (у ``kick`` — «погана» покупка); саме для нього
рахуються Precision, Recall, F1 і PR-AUC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: Стандартний поріг: «позитивний», якщо ймовірність ≥ 0.5.
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class ClassificationMetrics:
    """Метрики одного набору прогнозів.

    Attributes:
        pr_auc: Площа під кривою precision–recall (``average_precision``) —
            головна метрика при дисбалансі; випадковий класифікатор має PR-AUC,
            що дорівнює частці позитивного класу.
        roc_auc: Площа під ROC-кривою; випадковий класифікатор — 0.5.
        brier: Brier score (менше — краще).
        accuracy: Частка правильних відповідей (при дисбалансі оманлива).
        precision: Частка справді позитивних серед позначених моделлю.
        recall: Частка знайдених позитивних серед усіх позитивних.
        f1: Гармонійне середнє precision і recall.
        threshold: Поріг, за яким пораховано порогові метрики.
        mean_proba: Середня передбачена ймовірність позитивного класу.
    """

    pr_auc: float
    roc_auc: float
    brier: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float
    mean_proba: float

    def to_dict(self) -> dict[str, float]:
        """Словник «назва метрики → значення»."""
        return asdict(self)


class MetricsCalculator:
    """Рахує :class:`ClassificationMetrics` за мітками і ймовірностями.

    Args:
        positive_label: Мітка позитивного (рідкісного) класу.
    """

    def __init__(self, positive_label: int = 1) -> None:
        self._positive_label = positive_label

    @property
    def positive_label(self) -> int:
        """Мітка позитивного класу."""
        return self._positive_label

    def is_positive(self, target: ArrayLike) -> np.ndarray:
        """Перетворює мітки на 0/1: 1 — позитивний клас."""
        return (np.asarray(target) == self._positive_label).astype(int)

    def compute(
        self, target: ArrayLike, proba: ArrayLike, threshold: float = DEFAULT_THRESHOLD
    ) -> ClassificationMetrics:
        """Рахує всі метрики.

        Args:
            target: Справжні мітки класів.
            proba: Передбачені ймовірності ПОЗИТИВНОГО класу.
            threshold: Поріг для порогових метрик.
        """
        truth = self.is_positive(target)
        scores = np.asarray(proba, dtype=float)
        predicted = (scores >= threshold).astype(int)
        return ClassificationMetrics(
            pr_auc=float(average_precision_score(truth, scores)),
            roc_auc=float(roc_auc_score(truth, scores)),
            brier=float(brier_score_loss(truth, scores)),
            accuracy=float(accuracy_score(truth, predicted)),
            # zero_division=0: якщо модель не позначила жодного рядка, precision
            # невизначена (0/0) — вважаємо її нулем, а не падаємо з попередженням.
            precision=float(precision_score(truth, predicted, zero_division=0)),
            recall=float(recall_score(truth, predicted, zero_division=0)),
            f1=float(f1_score(truth, predicted, zero_division=0)),
            threshold=float(threshold),
            mean_proba=float(scores.mean()),
        )

    def pr_auc(self, target: ArrayLike, proba: ArrayLike) -> float:
        """Лише PR-AUC — найчастіша окрема метрика (швидше за :meth:`compute`)."""
        return float(average_precision_score(self.is_positive(target), np.asarray(proba)))

    def majority_accuracy(self, target: ArrayLike) -> float:
        """Accuracy «моделі», що завжди відповідає «негативний».

        Опорна лінія: при 12 % позитивних вона ≈ 0.88, тож Accuracy 0.89
        — зовсім не успіх.
        """
        return float(1.0 - self.is_positive(target).mean())
