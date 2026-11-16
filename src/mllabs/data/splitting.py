"""Розбиття датасету на навчальну та тестову вибірки."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    """Результат розбиття: ознаки й ціль окремо для train і test.

    Attributes:
        x_train: Ознаки навчальної вибірки — на них «вчиться» весь препроцесинг.
        x_test: Ознаки тестової вибірки — відкладені дані, які не можна
            використовувати для жодних рішень до фінальної оцінки моделі.
        y_train: Цільова змінна навчальної вибірки.
        y_test: Цільова змінна тестової вибірки.
    """

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def to_metrics(self, positive_label: int) -> dict[str, object]:
        """Розміри частин і частка позитивного класу в кожній (для ``metrics.json``).

        Args:
            positive_label: Мітка позитивного класу (``DatasetSchema.positive_label``).
        """
        return {
            "train_rows": len(self.x_train),
            "test_rows": len(self.x_test),
            "train_minority_share": float((self.y_train == positive_label).mean()),
            "test_minority_share": float((self.y_test == positive_label).mean()),
        }


class StratifiedSplitter:
    """Ділить таблицю на train/test зі збереженням часток класів.

    Стратифікація гарантує, що рідкісний клас (тут ~12 %) потрапить в обидві
    частини в тій самій пропорції; без неї тест міг би випадково отримати
    помітно іншу частку «поганих» авто, і метрики були б зміщені.

    Args:
        test_size: Частка рядків, що відкладається в тест (0..1).
        random_state: Зерно генератора — однакове зерно дає однакове розбиття.
    """

    def __init__(self, test_size: float, random_state: int) -> None:
        self._test_size = test_size
        self._random_state = random_state

    def split(self, data: pd.DataFrame, target_column: str) -> DatasetSplit:
        """Розбиває таблицю на навчальну й тестову частини.

        Args:
            data: Очищена таблиця з ознаками та цільовою колонкою.
            target_column: Назва цільової колонки.

        Returns:
            Розбиття з перенумерованими (0..n-1) індексами рядків.
        """
        features = data.drop(columns=[target_column])
        target = data[target_column]
        # Функція повертає кортеж із чотирьох елементів; запис «a, b, c, d = ...»
        # розпаковує його в чотири змінні за один раз.
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self._test_size,
            stratify=target,
            random_state=self._random_state,
        )
        return DatasetSplit(
            x_train=x_train.reset_index(drop=True),
            x_test=x_test.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
        )
