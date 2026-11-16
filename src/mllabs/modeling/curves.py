"""Криві якості класифікатора у вигляді таблиць (для графіків і CSV).

Криві PR і ROC мають стільки точок, скільки різних значень ймовірності —
десятки тисяч. Для графіка достатньо кількох сотень, тому криві проріджуються
рівномірно за індексом (крайні точки зберігаються).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import precision_recall_curve, roc_curve


class CurveBuilder:
    """Будує проріджені криві PR, ROC і «метрика від порога».

    Args:
        n_points: Максимальна кількість точок кривої.
        positive_label: Мітка позитивного класу.
    """

    def __init__(self, n_points: int, positive_label: int = 1) -> None:
        self._n_points = n_points
        self._positive_label = positive_label

    def precision_recall(self, target: ArrayLike, proba: ArrayLike) -> pd.DataFrame:
        """Крива precision–recall: колонки ``recall``, ``precision``."""
        precision, recall, _ = precision_recall_curve(
            np.asarray(target), np.asarray(proba, dtype=float), pos_label=self._positive_label
        )
        return self._thin(pd.DataFrame({"recall": recall, "precision": precision}))

    def roc(self, target: ArrayLike, proba: ArrayLike) -> pd.DataFrame:
        """ROC-крива: колонки ``fpr`` (частка хибних тривог), ``tpr`` (= recall)."""
        fpr, tpr, _ = roc_curve(
            np.asarray(target), np.asarray(proba, dtype=float), pos_label=self._positive_label
        )
        return self._thin(pd.DataFrame({"fpr": fpr, "tpr": tpr}))

    def by_threshold(self, target: ArrayLike, proba: ArrayLike) -> pd.DataFrame:
        """Precision і recall як функції порога.

        Returns:
            Колонки ``threshold``, ``precision``, ``recall``.
        """
        precision, recall, thresholds = precision_recall_curve(
            np.asarray(target), np.asarray(proba, dtype=float), pos_label=self._positive_label
        )
        # Останню точку (без порога) відкидаємо зрізом [:-1].
        frame = pd.DataFrame(
            {"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]}
        )
        return self._thin(frame)

    def _thin(self, curve: pd.DataFrame) -> pd.DataFrame:
        """Залишає не більше ``n_points`` рівномірно розташованих рядків."""
        if len(curve) <= self._n_points:
            return curve.reset_index(drop=True)
        positions = np.unique(np.linspace(0, len(curve) - 1, self._n_points).round().astype(int))
        return curve.iloc[positions].reset_index(drop=True)
