"""Вибір порога класифікації.

Модель повертає ймовірність, а рішення «позитивний / ні» залежить від порога.
Стандартні 0.5 при 12 % позитивних — погана точка: модель позначає майже
нікого. Поріг — це теж «параметр», тому його обирають на даних, які не
використовуються для фінальної оцінки: на OOF-прогнозах крос-валідації
(out-of-fold — прогноз кожного рядка train моделлю, яка цього рядка не бачила).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import precision_recall_curve


class F1ThresholdSelector:
    """Обирає поріг, за якого F1 позитивного класу максимальний.

    Args:
        positive_label: Мітка позитивного класу.
    """

    def __init__(self, positive_label: int = 1) -> None:
        self._positive_label = positive_label

    def select(self, target: ArrayLike, proba: ArrayLike) -> float:
        """Повертає поріг з максимальним F1.

        Args:
            target: Справжні мітки (лише навчальні/OOF дані, НЕ тестові).
            proba: Ймовірності позитивного класу для тих самих рядків.
        """
        precision, recall, thresholds = precision_recall_curve(
            np.asarray(target), np.asarray(proba, dtype=float), pos_label=self._positive_label
        )
        # Остання точка кривої (precision=1, recall=0) не має порога —
        # зрізом [:-1] відкидаємо її, щоб масиви мали однакову довжину.
        precision, recall = precision[:-1], recall[:-1]
        denominator = precision + recall
        # np.divide з where: там, де знаменник 0, F1 лишається нулем (без ділення на 0).
        f1 = np.divide(
            2 * precision * recall,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        return float(thresholds[int(np.argmax(f1))])
