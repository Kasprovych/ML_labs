"""Кодування категоріальних ознак: план за кардинальністю та порівняння методів."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pandas as pd

from mllabs.data.schema import DatasetSchema
from mllabs.labs.lab01.timing import FOLD_TIME_COLUMN, approximate_seconds
from mllabs.preprocessing import (
    CvScores,
    HighCardinalityEncoding,
    PreprocessingOptions,
    PreprocessorFactory,
    ProxyModelEvaluator,
)

ENCODING_LABELS = {
    HighCardinalityEncoding.TARGET: "Target",
    HighCardinalityEncoding.FREQUENCY: "Frequency",
    HighCardinalityEncoding.ONEHOT: "One-Hot",
    HighCardinalityEncoding.ORDINAL: "Ordinal",
}
_BINARY_CARDINALITY = 2
#: Кодування, які завдання прямо називає для високопотужних ознак (One-Hot та
#: Ordinal порівнюються з ними як базові варіанти).
_REQUIRED_HIGH_CARDINALITY = (HighCardinalityEncoding.TARGET, HighCardinalityEncoding.FREQUENCY)


class CardinalityMismatchError(ValueError):
    """Поділ ознак у схемі не відповідає порогу кардинальності конфігурації."""


class EncodingBenchmark:
    """Аналізує кардинальність і порівнює кодування високопотужних ознак.

    Кардинальність — кількість різних категорій ознаки. Від неї залежить
    вибір кодування: що більше категорій, то менше рядків припадає на кожну і
    то гірше працює «колонка на категорію».

    Args:
        schema: Схема очищеного датасету.
        high_cardinality_threshold: Від скількох категорій ознака вважається
            високопотужною.
        rare_category_size: Менше стількох рядків — категорія рідкісна.
    """

    def __init__(
        self, schema: DatasetSchema, high_cardinality_threshold: int, rare_category_size: int
    ) -> None:
        self._schema = schema
        self._threshold = high_cardinality_threshold
        self._rare_size = rare_category_size

    def cardinality(self, features: pd.DataFrame) -> pd.DataFrame:
        """Таблиця кардинальності категоріальних ознак навчальної вибірки.

        Returns:
            «Ознака», «Категорій», кількість рідкісних категорій, частка
            найчастішої категорії та частка пропусків.
        """
        rows = []
        for column in self._schema.categorical:
            counts = features[column].value_counts()
            rows.append(
                {
                    "Ознака": self._schema.label(column),
                    "Категорій": len(counts),
                    f"Рідкісних (< {self._rare_size} рядків)": int(
                        (counts < self._rare_size).sum()
                    ),
                    "Найчастіша, %": counts.iloc[0] / len(features) * 100,
                    "Пропуски, %": features[column].isna().mean() * 100,
                }
            )
        return pd.DataFrame(rows).sort_values("Категорій", ascending=False).reset_index(drop=True)

    def check_schema(self, features: pd.DataFrame) -> None:
        """Перевіряє, що поділ ознак у схемі збігається з порогом кардинальності.

        Пайплайн (``PreprocessorFactory``) бере перелік високопотужних ознак зі
        схеми (``categorical_high``), а план кодування й таблиці — з порогу
        ``high_cardinality_threshold``. Щоб це не були два незалежні джерела,
        перевіряється: кожна ознака ``categorical_high`` має на train не менше
        порогу категорій, а кожна інша категоріальна — менше.

        Raises:
            CardinalityMismatchError: Якщо хоч одна ознака порушує поділ.
        """
        schema = self._schema
        counts = {column: features[column].nunique() for column in schema.categorical}
        too_small = [c for c in schema.categorical_high if counts[c] < self._threshold]
        too_large = [
            c for c in (*schema.categorical_low, *schema.binary) if counts[c] >= self._threshold
        ]
        if too_small or too_large:
            raise CardinalityMismatchError(
                f"Поріг {self._threshold} категорій не збігається зі схемою: "
                f"у categorical_high менше порогу — {too_small}; "
                f"поза categorical_high не менше порогу — {too_large}"
            )

    def plan(self, features: pd.DataFrame) -> pd.DataFrame:
        """План кодування: метод для кожної ознаки залежно від кардинальності.

        Returns:
            «Ознака», «Категорій», «Метод», «Причина».
        """
        rows = []
        for column in self._schema.categorical:
            n_categories = features[column].nunique()
            method, reason = self._choose(n_categories)
            rows.append(
                {
                    "Ознака": self._schema.label(column),
                    "Категорій": n_categories,
                    "Метод": method,
                    "Причина": reason,
                }
            )
        return pd.DataFrame(rows).sort_values("Категорій", ascending=False).reset_index(drop=True)

    def _choose(self, n_categories: int) -> tuple[str, str]:
        if n_categories >= self._threshold:
            return (
                "Target Encoding",
                f"≥ {self._threshold} категорій: One-Hot дав би багато розріджених колонок",
            )
        if n_categories <= _BINARY_CARDINALITY:
            return "Ordinal (0/1)", "два значення: достатньо однієї колонки"
        return "One-Hot", "мало категорій без природного порядку"

    def compare(
        self,
        factory: PreprocessorFactory,
        evaluator: ProxyModelEvaluator,
        base_options: PreprocessingOptions,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> dict[HighCardinalityEncoding, CvScores]:
        """Оцінює кожне кодування високопотужних ознак проксі-моделлю.

        Змінюється тільки кодування високопотужних ознак; усе інше — ``base_options``.
        """
        return {
            kind: evaluator.evaluate(
                factory.build_from_options(replace(base_options, high_cardinality_encoding=kind)),
                features,
                target,
            )
            for kind in HighCardinalityEncoding
        }

    @staticmethod
    def comparison_table(scores: Mapping[HighCardinalityEncoding, CvScores]) -> pd.DataFrame:
        """Таблиця метрик проксі-моделі для кожного кодування."""
        return pd.DataFrame(
            [
                {
                    "Кодування": ENCODING_LABELS[kind],
                    "Ознак": result.n_features,
                    "PR-AUC": result.pr_auc,
                    "σ PR-AUC": result.pr_auc_std,
                    "ROC-AUC": result.roc_auc,
                    "F1": result.f1,
                    FOLD_TIME_COLUMN: approximate_seconds(result.fit_seconds),
                }
                for kind, result in scores.items()
            ]
        )

    @staticmethod
    def to_metrics(
        scores: Mapping[HighCardinalityEncoding, CvScores], plan: pd.DataFrame
    ) -> dict[str, object]:
        """Словник для ``metrics.json``.

        Args:
            scores: Результат :meth:`compare`.
            plan: Результат :meth:`plan` — звідти береться найбільша кількість
                категорій серед ознак, що кодуються One-Hot.
        """
        pr_auc = {kind.value: result.pr_auc for kind, result in scores.items()}
        # Серед кодувань, яких вимагає завдання для високої кардинальності.
        required = {kind.value: pr_auc[kind.value] for kind in _REQUIRED_HIGH_CARDINALITY}
        return {
            "pr_auc": pr_auc,
            "pr_auc_std": {kind.value: result.pr_auc_std for kind, result in scores.items()},
            "pr_auc_folds": {
                kind.value: list(result.fold_pr_auc) for kind, result in scores.items()
            },
            "n_output_features": {kind.value: result.n_features for kind, result in scores.items()},
            "best_method": max(pr_auc, key=pr_auc.get),  # ключ із найбільшим значенням
            "best_required_method": max(required, key=required.get),
            "max_onehot_cardinality": int(plan.loc[plan["Метод"] == "One-Hot", "Категорій"].max()),
        }
