"""Стан даних у кожному сценарії: що отримує модель після перетворень.

Для звіту про «стан даних та виконані перетворення» (пункт 6 завдання):
скільки ознак на вході моделі, скільки рядків і яка частка класу 1 у
навчальних даних після аугментації. Препроцесор і семплер навчаються лише
на train; тестова вибірка не аугментується ніколи.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from mllabs.labs.lab02.scenarios import ScenarioCatalog, ScenarioSpec


@dataclass(frozen=True)
class ScenarioState:
    """Стан навчальних даних одного сценарію.

    Attributes:
        spec: Сценарій.
        n_features: Кількість ознак на вході моделі.
        train_rows: Рядків у навчальних даних моделі (з синтетичними).
        synthetic_rows: Скільки з них синтетичних.
        minority_share: Частка класу 1 у навчальних даних моделі.
        convergence_warnings: Скільки ``ConvergenceWarning`` видав препроцесор
            (MICE може не зійтися за ``mice_max_iter`` ітерацій).
    """

    spec: ScenarioSpec
    n_features: int
    train_rows: int
    synthetic_rows: int
    minority_share: float
    convergence_warnings: int


class DataStateReporter:
    """Рахує стан даних сценаріїв, навчаючи препроцесор і семплер на всьому train.

    Args:
        catalog: Сценарії.
        positive_label: Мітка класу 1.
    """

    def __init__(self, catalog: ScenarioCatalog, positive_label: int = 1) -> None:
        self._catalog = catalog
        self._positive_label = positive_label

    def describe(
        self, specs: Sequence[ScenarioSpec], features: pd.DataFrame, target: pd.Series
    ) -> list[ScenarioState]:
        """Стан навчальних даних кожного сценарію.

        Args:
            specs: Сценарії.
            features: Ознаки train (НЕ test).
            target: Мітки train.
        """
        states = []
        for spec in specs:
            # Попередження записуються як факт у metrics.json, а не друкуються в журнал.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                matrix = clone(self._catalog.preprocessor(spec)).fit_transform(features, target)
            labels = target.to_numpy()
            sampler = self._catalog.sampler(spec)
            if sampler is not None:
                matrix, labels = sampler.fit_resample(matrix, labels)
            states.append(
                ScenarioState(
                    spec=spec,
                    n_features=int(matrix.shape[1]),
                    train_rows=int(matrix.shape[0]),
                    synthetic_rows=int(matrix.shape[0] - len(features)),
                    minority_share=float(np.mean(np.asarray(labels) == self._positive_label)),
                    convergence_warnings=sum(
                        issubclass(item.category, ConvergenceWarning) for item in caught
                    ),
                )
            )
        return states

    def table(self, states: Sequence[ScenarioState]) -> pd.DataFrame:
        """Таблиця стану даних для звіту."""
        rows = [
            {
                "Сценарій": state.spec.label,
                "Перетворення": self._catalog.transformations(state.spec),
                "Ознак": state.n_features,
                "Рядків навчання": state.train_rows,
                "Синтетичних": state.synthetic_rows,
                "Частка класу 1, %": state.minority_share * 100,
                "Test аугментується": "ні",
            }
            for state in states
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def to_metrics(states: Sequence[ScenarioState]) -> dict[str, Any]:
        """Розділ ``scenarios`` для ``metrics.json``."""
        return {
            state.spec.scenario_id.value: {
                "label": state.spec.label,
                "n_features": state.n_features,
                "train_rows_after_sampling": state.train_rows,
                "synthetic_rows": state.synthetic_rows,
                "minority_share_after_sampling": state.minority_share,
                "on_test": state.spec.on_test,
                "preprocessor_convergence_warnings": state.convergence_warnings,
            }
            for state in states
        }
