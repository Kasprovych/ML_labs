"""Власні кодувальники категоріальних ознак."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Замінює категорію її часткою в навчальній вибірці (Frequency Encoding).

    Ідея: для ознаки з сотнями категорій (модель авто) One-Hot дав би сотні
    колонок. Натомість кожна категорія стає одним числом — «наскільки вона
    поширена»: ``Model = "IMPALA"`` → ``0.027``, якщо 2,7 % авто в train — Impala.
    На відміну від Target Encoding, цільова змінна не використовується, тому
    кодування не може «підглянути» відповідь.

    Частки запам'ятовуються у ``fit`` (лише train). Категорія, якої в train не
    було (і пропуск), кодується нулем — «таких авто ми не бачили».

    Успадкування від ``BaseEstimator`` і ``TransformerMixin`` — це спосіб
    реалізувати «інтерфейс трансформера» sklearn: з'являються ``fit_transform``,
    ``get_params`` і сумісність із ``Pipeline``.

    Attributes:
        frequencies_: Список словників «категорія → частка», по одному на колонку.
            Суфікс ``_`` — угода sklearn: атрибут з'являється лише після ``fit``.
    """

    def fit(self, features: pd.DataFrame | np.ndarray, target: object = None) -> Self:
        """Запам'ятовує частки категорій кожної колонки.

        Args:
            features: Категоріальні колонки навчальної вибірки.
            target: Не використовується; параметр потрібен інтерфейсу sklearn.
        """
        table = pd.DataFrame(features)
        self.feature_names_in_ = np.asarray([str(column) for column in table.columns])
        self.frequencies_ = [
            table[column].value_counts(normalize=True).to_dict() for column in table.columns
        ]
        return self

    def transform(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Замінює кожну категорію запам'ятованою часткою.

        Returns:
            Числова матриця тієї самої форми, що й вхід.
        """
        table = pd.DataFrame(features)
        encoded = [
            table[column].map(frequencies).fillna(0.0).astype(float)
            # zip іде по двох послідовностях одночасно: (колонка, її словник часток).
            for column, frequencies in zip(table.columns, self.frequencies_, strict=True)
        ]
        return np.column_stack(encoded)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Назви вихідних колонок збігаються з вхідними (потрібно для ``Pipeline``)."""
        return self.feature_names_in_
