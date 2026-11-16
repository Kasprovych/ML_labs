"""Парний бутстреп: довірчий інтервал різниці метрики двох моделей на тих самих рядках.

Одне число на тесті (наприклад, «PR-AUC 0.48 проти 0.47») не каже, чи різниця
реальна, чи це випадковість конкретної тестової вибірки. Бутстреп імітує
«інші тестові вибірки»: багато разів вибирає з тесту рядки з поверненням,
рахує метрику обох моделей на ТИХ САМИХ вибраних рядках і бере різницю.
Перцентилі цих різниць дають довірчий інтервал (ДІ). Якщо 95 % ДІ містить 0 —
різниця статистично не підтверджена.

«Парний» означає, що обидві моделі оцінюються на однакових підвибірках:
«важкі» рядки однаково впливають на обидві, і ця спільна мінливість у
різниці скорочується (так само, як парні різниці по фолдах у ЛР1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

#: Метрика: (мітки 0/1, ймовірності) -> число.
MetricFunction = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class BootstrapInterval:
    """Різниця метрики «B − A» та її довірчий інтервал.

    Attributes:
        delta: Різниця на повній вибірці (без перевибірки).
        low: Нижня межа ДІ.
        high: Верхня межа ДІ.
        confidence_level: Рівень довіри (0.95 — 95 %).
        n_repeats: Кількість бутстреп-повторів.
    """

    delta: float
    low: float
    high: float
    confidence_level: float
    n_repeats: int

    @property
    def excludes_zero(self) -> bool:
        """Чи лежить увесь ДІ по один бік від нуля (різниця «значуща»)."""
        return self.low > 0 or self.high < 0


class PairedBootstrap:
    """Парний бутстреп різниці метрики двох наборів прогнозів.

    Args:
        n_repeats: Кількість повторів; 1000 дає стабільні межі 95 % ДІ.
        confidence_level: Рівень довіри.
        random_state: Зерно генератора — однакові індекси для всіх пар моделей.
    """

    def __init__(self, n_repeats: int, confidence_level: float, random_state: int) -> None:
        self._n_repeats = n_repeats
        self._confidence_level = confidence_level
        self._random_state = random_state

    def interval(
        self,
        truth: ArrayLike,
        proba_a: ArrayLike,
        proba_b: ArrayLike,
        metric: MetricFunction,
    ) -> BootstrapInterval:
        """Рахує ``metric(B) − metric(A)`` і її ДІ.

        Args:
            truth: Мітки 0/1 (1 — позитивний клас).
            proba_a: Ймовірності моделі A (базової).
            proba_b: Ймовірності моделі B (порівнюваної) для тих самих рядків.
            metric: Функція метрики, напр. ``sklearn.metrics.average_precision_score``.
        """
        labels = np.asarray(truth)
        first, second = np.asarray(proba_a, dtype=float), np.asarray(proba_b, dtype=float)
        # Новий генератор із тим самим зерном у кожному виклику: усі пари моделей
        # оцінюються на ОДНАКОВИХ бутстреп-вибірках.
        generator = np.random.default_rng(self._random_state)
        n_rows = len(labels)
        differences = np.empty(self._n_repeats)
        for repeat in range(self._n_repeats):
            rows = generator.integers(0, n_rows, size=n_rows)  # вибір із поверненням
            differences[repeat] = metric(labels[rows], second[rows]) - metric(
                labels[rows], first[rows]
            )
        tail = (1 - self._confidence_level) / 2 * 100
        low, high = np.percentile(differences, [tail, 100 - tail])
        return BootstrapInterval(
            delta=float(metric(labels, second) - metric(labels, first)),
            low=float(low),
            high=float(high),
            confidence_level=self._confidence_level,
            n_repeats=self._n_repeats,
        )
