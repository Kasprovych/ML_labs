"""Єдине правило вибору між варіантами препроцесингу.

Усі рішення сценарію ``FULL`` (ЛР1) ухвалюються за одним правилом, щоб вони
були узгоджені між собою й відтворювані:

    обидва варіанти оцінюються на ОДНИХ І ТИХ САМИХ фолдах крос-валідації;
    на кожному фолді k береться різниця d_k = PR-AUC складнішого − PR-AUC
    простішого; Δ = mean(d), s = std(d, ddof=1). Складніший варіант (більше
    ознак, вищий VIF, повільніший метод) обирається, лише якщо Δ > s;
    інакше (Δ ≤ s) обирається простіший.

Чому парні різниці, а не σ одного варіанта: фолди «важкі» чи «легкі» для обох
варіантів однаково, і в різниці ця спільна мінливість скорочується. Тоді s —
розкид саме РІЗНИЦІ, а не якості окремого варіанта, і не залежить від того,
чиє σ взяти. ``ddof=1`` — вибіркове стандартне відхилення (ділення на n − 1):
на кількох фолдах звичайне ``ddof=0`` помітно його занижує.

Свідомі винятки. Відступ від правила допускається лише як явно оголошений
у коді виняток з обґрунтуванням (``_DECLARED_EXCEPTIONS`` у підсумку ЛР1); у
таблиці рішень він позначається «свідомий виняток», а будь-яка неоголошена
розбіжність — «УВАГА».
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: Формулювання правила для таблиць і звіту (одне на весь проєкт).
SELECTION_RULE = (
    "на тих самих фолдах d_k = PR-AUC складнішого − PR-AUC простішого; "
    "Δ = mean(d), s = std(d, ddof=1); складніший варіант обирається, лише якщо "
    "Δ > s, інакше (Δ ≤ s) — простіший (менше ознак, нижчий VIF, швидший метод)"
)


class FoldMismatchError(ValueError):
    """Варіанти оцінено на різній кількості фолдів — попарне порівняння неможливе."""


@dataclass(frozen=True)
class PairedDifference:
    """Парне порівняння двох варіантів по фолдах.

    Attributes:
        delta: Середня різниця PR-AUC «складніший − простіший».
        spread: Вибіркове стандартне відхилення різниць (``ddof=1``).
        n_folds: Кількість фолдів (пар).
    """

    delta: float
    spread: float
    n_folds: int

    @property
    def complex_wins(self) -> bool:
        """Чи обрати складніший варіант: лише якщо Δ > s."""
        return self.delta > self.spread


def paired_difference(
    simpler_folds: Sequence[float], complex_folds: Sequence[float]
) -> PairedDifference:
    """Рахує Δ і s за PR-AUC двох варіантів на тих самих фолдах.

    Args:
        simpler_folds: PR-AUC простішого варіанта по фолдах.
        complex_folds: PR-AUC складнішого варіанта на тих самих фолдах, у тому
            самому порядку.

    Raises:
        FoldMismatchError: Якщо кількість фолдів різна або менша за 2.
    """
    if len(simpler_folds) != len(complex_folds) or len(simpler_folds) < 2:
        raise FoldMismatchError(
            f"Потрібні PR-AUC на тих самих ≥ 2 фолдах; отримано {len(simpler_folds)} "
            f"і {len(complex_folds)}"
        )
    differences = np.asarray(complex_folds, dtype=float) - np.asarray(simpler_folds, dtype=float)
    return PairedDifference(
        delta=float(differences.mean()),
        spread=float(differences.std(ddof=1)),
        n_folds=len(differences),
    )
