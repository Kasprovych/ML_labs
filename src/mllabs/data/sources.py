"""Джерела сирих даних.

Інтерфейс :class:`DatasetSource` відокремлює «звідки беруться дані» від
усього іншого коду: у тестах замість мережі підставляється фальшиве джерело
з маленькою таблицею (як mock-реалізація інтерфейсу в Java).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import pandas as pd

_logger = logging.getLogger(__name__)


class DatasetDownloadError(RuntimeError):
    """Не вдалося завантажити датасет із зовнішнього джерела."""


class DatasetSource(Protocol):
    """Будь-що, що вміє повернути сиру таблицю датасету.

    ``Protocol`` — структурний інтерфейс: клас не мусить явно писати
    ``implements``, достатньо мати метод ``load`` з такою сигнатурою.
    """

    def load(self) -> pd.DataFrame:
        """Повертає сиру таблицю (ознаки разом із цільовою колонкою)."""
        ...


class OpenMlDatasetSource:
    """Завантажує датасет із репозиторію OpenML і кешує його на диску.

    Перший виклик качає файл ARFF із мережі, наступні — читають кеш із
    ``cache_dir``, тому повторні запуски лабораторної не залежать від мережі.

    Args:
        data_id: Ідентифікатор датасету на openml.org.
        cache_dir: Каталог кешу (усередині ``data/raw/``).
    """

    def __init__(self, data_id: int, cache_dir: Path) -> None:
        self._data_id = data_id
        self._cache_dir = cache_dir

    def load(self) -> pd.DataFrame:
        """Повертає сиру таблицю датасету.

        Raises:
            DatasetDownloadError: Якщо датасету немає в кеші, а мережа недоступна.
        """
        # Імпорт усередині методу: sklearn.datasets потрібен лише тут.
        from sklearn.datasets import fetch_openml

        try:
            # parser="liac-arff": для деяких датасетів (зокрема id 41162) швидший
            # парсер "pandas" падає з TypeError на номінальних колонках із числами.
            bunch = fetch_openml(
                data_id=self._data_id,
                parser="liac-arff",
                as_frame=True,
                data_home=str(self._cache_dir),
            )
        except OSError as error:  # URLError і помилки сокетів — нащадки OSError
            raise DatasetDownloadError(
                f"Не вдалося завантажити датасет OpenML id={self._data_id}: {error}. "
                f"Перевірте мережу або покладіть кеш у {self._cache_dir}."
            ) from error
        _logger.info("Датасет OpenML id=%d: %d x %d", self._data_id, *bunch.frame.shape)
        return bunch.frame
