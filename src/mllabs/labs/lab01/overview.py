"""Опис датасету: розмір, типи ознак, кардинальність, пропуски, баланс класів."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from mllabs.data.schema import DatasetSchema


@dataclass(frozen=True)
class DatasetProfile:
    """Ключові числа про очищений датасет.

    Attributes:
        n_rows: Кількість рядків.
        n_features: Кількість ознак після очищення (без цілі).
        n_numeric: Кількість числових ознак.
        n_categorical: Кількість категоріальних ознак.
        n_high_cardinality: Кількість категоріальних ознак із кардинальністю
            не нижче порога.
        max_cardinality: Найбільша кількість категорій в одній ознаці.
        minority_share: Частка позитивного (рідкісного) класу.
        imbalance_ratio: Скільки рядків мажоритарного класу припадає на один
            рядок міноритарного.
        missing_cells_share: Частка порожніх клітинок серед ознак.
        rows_with_missing_share: Частка рядків, що мають хоча б один пропуск.
    """

    n_rows: int
    n_features: int
    n_numeric: int
    n_categorical: int
    n_high_cardinality: int
    max_cardinality: int
    minority_share: float
    imbalance_ratio: float
    missing_cells_share: float
    rows_with_missing_share: float

    def to_metrics(self) -> dict[str, float]:
        """Словник для ``metrics.json``."""
        return asdict(self)


class DatasetProfiler:
    """Рахує описові характеристики очищеного датасету.

    Args:
        schema: Схема очищеного датасету.
        high_cardinality_threshold: Від скількох категорій ознака вважається
            високопотужною.
    """

    def __init__(self, schema: DatasetSchema, high_cardinality_threshold: int) -> None:
        self._schema = schema
        self._threshold = high_cardinality_threshold

    def profile(self, data: pd.DataFrame) -> DatasetProfile:
        """Збирає ключові числа про датасет.

        Args:
            data: Очищена таблиця з ознаками та ціллю.
        """
        schema = self._schema
        features = data[list(schema.features)]
        cardinality = features[list(schema.categorical)].nunique()
        is_positive = data[schema.target] == schema.positive_label
        n_positive = int(is_positive.sum())
        return DatasetProfile(
            n_rows=len(data),
            n_features=features.shape[1],
            n_numeric=len(schema.numeric),
            n_categorical=len(schema.categorical),
            n_high_cardinality=int((cardinality >= self._threshold).sum()),
            max_cardinality=int(cardinality.max()),
            minority_share=n_positive / len(data),
            imbalance_ratio=(len(data) - n_positive) / n_positive,
            # isna() дає таблицю True/False; mean() двічі — частка True у всій таблиці.
            missing_cells_share=float(features.isna().mean().mean()),
            # any(axis=1): чи є хоч один пропуск у рядку (axis=1 — «уздовж колонок»).
            rows_with_missing_share=float(features.isna().any(axis=1).mean()),
        )

    def overview_table(
        self, profile: DatasetProfile, raw_shape: tuple[int, int], hidden_missing_cells: int
    ) -> pd.DataFrame:
        """Таблиця «Показник — Значення» для звіту.

        Args:
            profile: Результат :meth:`profile`.
            raw_shape: Розмір сирої таблиці (рядки, колонки разом із ціллю).
            hidden_missing_cells: Скільки значень очищення перетворило на пропуски.
        """
        date_note = " + дата покупки" if self._schema.date_column else ""
        rows = [
            ("Датасет", f"Carvana «Don't Get Kicked!» ({self._schema.name})"),
            ("Рядків", f"{profile.n_rows}"),
            ("Ознак у сирих даних", f"{raw_shape[1] - 1}"),
            ("Ознак після очищення", f"{profile.n_features}"),
            ("з них числових", f"{profile.n_numeric}{date_note}"),
            ("з них категоріальних", f"{profile.n_categorical}"),
            (
                f"Категоріальних із ≥ {self._threshold} категорій",
                f"{profile.n_high_cardinality} (максимум {profile.max_cardinality})",
            ),
            ("Цільова змінна", f"{self._schema.target} (1 — «погана» покупка)"),
            ("Частка класу 1, %", f"{profile.minority_share * 100:.2f}"),
            ("Дисбаланс (клас 0 : клас 1)", f"{profile.imbalance_ratio:.1f} : 1"),
            ("Прихованих пропусків, клітинок", f"{hidden_missing_cells}"),
            ("Порожніх клітинок після очищення, %", f"{profile.missing_cells_share * 100:.2f}"),
            ("Рядків із пропусками, %", f"{profile.rows_with_missing_share * 100:.2f}"),
        ]
        return pd.DataFrame(rows, columns=["Показник", "Значення"])

    def feature_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Таблиця ознак: група, кількість унікальних значень, частка пропусків."""
        schema = self._schema
        groups: dict[str, str] = {}
        # dict.fromkeys(ключі, значення) — словник, де всім ключам відповідає одне значення;
        # оператор |= доливає його в наявний словник (як Map.putAll).
        groups |= dict.fromkeys(schema.numeric, "числова")
        groups |= dict.fromkeys(schema.categorical_low, "номінальна")
        groups |= dict.fromkeys(schema.categorical_high, "високої кардинальності")
        groups |= dict.fromkeys(schema.binary, "бінарна")
        if schema.date_column:
            groups[schema.date_column] = "дата"
        columns = list(schema.features)
        return pd.DataFrame(
            {
                "Ознака": columns,
                "Скорочення": [schema.label(column) for column in columns],
                "Група": [groups[column] for column in columns],
                "Унікальних": data[columns].nunique().to_numpy(),
                "Пропуски, %": (data[columns].isna().mean() * 100).to_numpy(),
            }
        )

    def class_balance(self, target: pd.Series) -> pd.DataFrame:
        """Кількість і частка рядків кожного класу цільової змінної."""
        counts = target.value_counts().sort_index()
        return pd.DataFrame(
            {
                "Клас": counts.index.to_numpy(),
                "Кількість": counts.to_numpy(),
                "Частка, %": (counts / counts.sum() * 100).to_numpy(),
            }
        )
