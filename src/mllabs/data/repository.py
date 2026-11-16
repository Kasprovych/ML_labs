"""Сховище підготовленого датасету в ``data/processed/<назва>/``.

ЛР1 записує сюди очищені, але НЕ трансформовані train/test (без імп'ютації,
кодування й масштабування), а ЛР2–3 читають їх і самі будують пайплайни.
Якби на диск ішла вже трансформована матриця, чесна крос-валідація в
наступних лабораторних була б неможливою: трансформери вже «бачили» всі дані.

Поруч зі схемою ЛР1 записує ``preprocessing.json`` — фактичні налаштування
препроцесингу, обґрунтовані в ЛР1. Сховище не тлумачить його вміст (пакет
``mllabs.data`` не залежить від ``mllabs.preprocessing``), лише зберігає й читає.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from mllabs.data.schema import DatasetSchema
from mllabs.data.splitting import DatasetSplit

_TRAIN_FILE = "train.csv"
_TEST_FILE = "test.csv"
_SCHEMA_FILE = "schema.json"
_PREPROCESSING_FILE = "preprocessing.json"


class ProcessedDatasetNotFoundError(FileNotFoundError):
    """Підготовленого датасету ще немає на диску."""


class ProcessedDatasetRepository:
    """Записує та читає розбитий датасет разом зі схемою.

    CSV не зберігає типів колонок: поштовий індекс ``33619`` після читання став
    би числом, а дата — рядком. Тому поруч лежить ``schema.json``, і читання
    відновлює типи за схемою.

    Args:
        processed_dir: Каталог ``data/processed`` проєкту.
        dataset_name: Ім'я датасету — назва підкаталогу.
    """

    def __init__(self, processed_dir: Path, dataset_name: str) -> None:
        self._directory = processed_dir / dataset_name

    @property
    def directory(self) -> Path:
        """Каталог, у якому лежать файли датасету."""
        return self._directory

    def save(
        self,
        split: DatasetSplit,
        schema: DatasetSchema,
        preprocessing: Mapping[str, Any] | None = None,
    ) -> None:
        """Записує ``train.csv``, ``test.csv`` (з цільовою колонкою) і ``schema.json``.

        Args:
            split: Розбиття, отримане в ЛР1.
            schema: Схема очищеного датасету.
            preprocessing: Фактичні налаштування препроцесингу
                (``PreprocessorFactory.manifest()``); якщо задано — записуються
                у ``preprocessing.json``.
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        for file_name, features, target in (
            (_TRAIN_FILE, split.x_train, split.y_train),
            (_TEST_FILE, split.x_test, split.y_test),
        ):
            table = features.assign(**{schema.target: target.to_numpy()})
            table.to_csv(self._directory / file_name, index=False)
        self._write_json(_SCHEMA_FILE, schema.to_dict())
        if preprocessing is not None:
            self._write_json(_PREPROCESSING_FILE, preprocessing)

    def load_preprocessing(self) -> dict[str, Any]:
        """Читає налаштування препроцесингу, записані ЛР1.

        Raises:
            ProcessedDatasetNotFoundError: Якщо ЛР1 ще не запускали.
        """
        settings_file = self._directory / _PREPROCESSING_FILE
        if not settings_file.is_file():
            raise ProcessedDatasetNotFoundError(
                f"Не знайдено {settings_file}. Спершу виконайте: uv run mllabs run 1"
            )
        return json.loads(settings_file.read_text(encoding="utf-8"))

    def _write_json(self, file_name: str, payload: Mapping[str, Any]) -> None:
        (self._directory / file_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_schema(self) -> DatasetSchema:
        """Читає схему датасету.

        Raises:
            ProcessedDatasetNotFoundError: Якщо ЛР1 ще не запускали.
        """
        schema_file = self._directory / _SCHEMA_FILE
        if not schema_file.is_file():
            raise ProcessedDatasetNotFoundError(
                f"Не знайдено {schema_file}. Спершу виконайте: uv run mllabs run 1"
            )
        return DatasetSchema.from_dict(json.loads(schema_file.read_text(encoding="utf-8")))

    def load(self, train_sample_size: int | None = None, random_state: int = 0) -> DatasetSplit:
        """Читає розбиття з диска, відновлюючи типи колонок за схемою.

        Args:
            train_sample_size: Якщо задано — повертається стратифікована
                підвибірка train такого розміру (для повільних моделей на кшталт
                SVM). Тест завжди повертається повністю.
            random_state: Зерно для підвибірки.

        Raises:
            ProcessedDatasetNotFoundError: Якщо ЛР1 ще не запускали.
        """
        schema = self.load_schema()
        train = self._read(_TRAIN_FILE, schema)
        test = self._read(_TEST_FILE, schema)
        if train_sample_size is not None and train_sample_size < len(train):
            train, _ = train_test_split(
                train,
                train_size=train_sample_size,
                stratify=train[schema.target],
                random_state=random_state,
            )
            train = train.reset_index(drop=True)
        return DatasetSplit(
            x_train=train.drop(columns=[schema.target]),
            x_test=test.drop(columns=[schema.target]),
            y_train=train[schema.target],
            y_test=test[schema.target],
        )

    def _read(self, file_name: str, schema: DatasetSchema) -> pd.DataFrame:
        """Читає один CSV: категорії — як текст, числа — як float, дата — як datetime."""
        csv_file = self._directory / file_name
        if not csv_file.is_file():
            raise ProcessedDatasetNotFoundError(
                f"Не знайдено {csv_file}. Спершу виконайте: uv run mllabs run 1"
            )
        dtypes: dict[str, type] = {column: str for column in schema.categorical}
        dtypes.update({column: float for column in schema.numeric})
        dtypes[schema.target] = int
        table = pd.read_csv(
            csv_file,
            dtype=dtypes,
            parse_dates=[schema.date_column] if schema.date_column else None,
        )
        # Той самий тип object, що й після RawDataCleaner: пайплайни ЛР1–3 бачать
        # однакові дані незалежно від того, чи прийшли вони з пам'яті, чи з диска.
        categorical = list(schema.categorical)
        table[categorical] = table[categorical].astype(object)
        return table
