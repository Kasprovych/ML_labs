"""Спільний пакет даних: схема, завантаження, очищення, розбиття, сховище.

Цим пакетом користуються лабораторні №1–3: ЛР1 готує дані й записує їх у
``data/processed/``, ЛР2–3 читають їх через :class:`ProcessedDatasetRepository`.
"""

from mllabs.data.cleaning import CleaningResult, RawDataCleaner
from mllabs.data.repository import ProcessedDatasetNotFoundError, ProcessedDatasetRepository
from mllabs.data.schema import KICK_SCHEMA, DatasetSchema
from mllabs.data.sources import DatasetDownloadError, DatasetSource, OpenMlDatasetSource
from mllabs.data.splitting import DatasetSplit, StratifiedSplitter

__all__ = [
    "KICK_SCHEMA",
    "CleaningResult",
    "DatasetDownloadError",
    "DatasetSchema",
    "DatasetSource",
    "DatasetSplit",
    "OpenMlDatasetSource",
    "ProcessedDatasetNotFoundError",
    "ProcessedDatasetRepository",
    "RawDataCleaner",
    "StratifiedSplitter",
]
