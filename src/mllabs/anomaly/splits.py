"""Розбиття VisA на train / val_normal / val (розмічений) / test.

Офіційне розбиття «1 клас» має лише train (норма) і test (норма + аномалії); валідації
немає. Тому:

* офіційний train → ``train`` (85 %) і ``val_normal`` (15 %) — обидві лише норма; на
  ``train`` моделі навчаються, на ``val_normal`` — раннє зупинення (втрати без міток);
* офіційний test → ``val`` і ``test`` навпіл, стратифіковано за міткою норма/аномалія:
  на ``val`` обираються конфігурації та пороги, ``test`` — лише фінальна оцінка.

Моделі ніколи не бачать аномалій під час навчання (unsupervised); мітки ``val`` потрібні
лише для вибору гіперпараметрів і порогів — це явно «напівнаглядова» селекція.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mllabs.anomaly.visa import VisaSample


class SplitPart(StrEnum):
    """Частини розбиття (``StrEnum`` — значення є рядками й пишуться в CSV як є)."""

    TRAIN = "train"
    VAL_NORMAL = "val_normal"
    VAL = "val"
    TEST = "test"


#: Колонки ``splits.csv`` у фіксованому порядку.
SPLIT_COLUMNS = (
    "category",
    "official_split",
    "part",
    "label",
    "image",
    "mask",
    "defect_types",
)


class SplitIntegrityError(ValueError):
    """Розбиття порушує правила (аномалія у train, перетин частин тощо)."""


class AnomalySplitter:
    """Детерміноване (за seed) розбиття зразків однієї категорії.

    Args:
        train_fraction: Частка офіційного train, що йде на навчання (решта — ``val_normal``).
        val_fraction_of_test: Частка офіційного test, що йде у розмічений ``val``.
        random_state: Зерно генератора.
    """

    def __init__(
        self, train_fraction: float, val_fraction_of_test: float, random_state: int
    ) -> None:
        self._train_fraction = train_fraction
        self._val_fraction = val_fraction_of_test
        self._random_state = random_state

    def split(self, samples: Sequence[VisaSample]) -> pd.DataFrame:
        """Повертає таблицю зразків із колонкою ``part`` (див. :class:`SplitPart`)."""
        frame = pd.DataFrame([asdict(sample) for sample in samples])
        official_train = frame[frame["official_split"] == "train"]
        official_test = frame[frame["official_split"] == "test"]
        train, val_normal = train_test_split(
            official_train, train_size=self._train_fraction, random_state=self._random_state
        )
        # stratify — однакова частка аномалій у val і test.
        val, test = train_test_split(
            official_test,
            train_size=self._val_fraction,
            random_state=self._random_state,
            stratify=official_test["label"],
        )
        parts = {
            SplitPart.TRAIN: train,
            SplitPart.VAL_NORMAL: val_normal,
            SplitPart.VAL: val,
            SplitPart.TEST: test,
        }
        # assign повертає НОВУ таблицю з доданою колонкою (без inplace).
        result = pd.concat([part.assign(part=str(name)) for name, part in parts.items()])
        result = result.sort_values(["part", "image"], kind="stable").reset_index(drop=True)
        validate_split(result)
        return result.loc[:, list(SPLIT_COLUMNS)]


def validate_split(frame: pd.DataFrame) -> None:
    """Перевіряє інваріанти розбиття.

    Raises:
        SplitIntegrityError: Якщо в train / val_normal є аномалії або зображення повторюються.
    """
    unsupervised = frame["part"].isin([SplitPart.TRAIN, SplitPart.VAL_NORMAL])
    if frame.loc[unsupervised, "label"].any():
        raise SplitIntegrityError("У train або val_normal потрапили аномальні зображення")
    if frame["image"].duplicated().any():
        raise SplitIntegrityError("Те саме зображення потрапило в кілька частин розбиття")


class SplitRepository:
    """Зберігає та читає ``data/processed/visa/splits.csv`` (його перевикористовує ЛР5).

    Args:
        path: Шлях до CSV.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Шлях до CSV."""
        return self._path

    def save(self, frame: pd.DataFrame) -> Path:
        """Записує розбиття (усі категорії в одному файлі)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        frame.loc[:, list(SPLIT_COLUMNS)].to_csv(self._path, index=False)
        return self._path

    def load(self) -> pd.DataFrame:
        """Читає розбиття; порожні маски/типи — порожні рядки, а не NaN."""
        frame = pd.read_csv(self._path, dtype={"mask": str, "defect_types": str})
        return frame.fillna({"mask": "", "defect_types": ""})
