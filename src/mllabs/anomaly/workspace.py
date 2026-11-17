"""Робочий простір даних VisA (ЛР4 і ЛР5): розбиття, зменшені зображення, ознаки.

Клас :class:`VisaWorkspace` — фасад над класами цього пакета: код лабораторної просить
«зображення категорії» чи «ознаки train з агрегацією p», не думаючи про кеші, порядок
рядків і розпакування. Параметри задає :class:`WorkspaceSettings` (без залежності від
конфігурації конкретної лабораторної).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from mllabs.anomaly.backbones import BACKBONE_SPECS, BackboneKind, FeatureExtractorFactory
from mllabs.anomaly.features import FeatureCache, PatchAggregator
from mllabs.anomaly.images import ImageBatch, ImageStore
from mllabs.anomaly.splits import AnomalySplitter, SplitPart, SplitRepository
from mllabs.anomaly.visa import VisaDatasetSource
from mllabs.core import ProjectPaths

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkspaceSettings:
    """Параметри даних, від яких залежать розбиття, кеші зображень і ознак.

    Attributes:
        categories: Категорії, що розпаковуються повністю й розбиваються.
        background_categories: Категорії для виміру тла (з них — лише вибірка норм).
        background_sample: Скільки нормальних зображень брати з таких категорій.
        train_fraction: Частка офіційного train на навчання (решта — val_normal).
        val_fraction_of_test: Частка офіційного test у розмічений val.
        image_size: Сторона зображення (тайла) для backbone.
        eval_size: Сторона масок і карт для метрик.
        feature_dim: Розмірність ознак ``D``.
        extraction_batch_size: Пакет проходу backbone.
    """

    categories: tuple[str, ...]
    background_categories: tuple[str, ...] = ()
    background_sample: int = 50
    train_fraction: float = 0.85
    val_fraction_of_test: float = 0.5
    image_size: int = 256
    eval_size: int = 256
    feature_dim: int = 512
    extraction_batch_size: int = 16


@dataclass(frozen=True)
class FeatureKey:
    """Що саме за ознаки: backbone і тайли.

    Attributes:
        backbone: Тип backbone.
        tile_grid: ``g``.
    """

    backbone: BackboneKind
    tile_grid: int

    @property
    def group(self) -> str:
        """Підкаталог кешу."""
        return f"{self.backbone}_g{self.tile_grid}"


class VisaWorkspace:
    """Дані VisA для лабораторної.

    Args:
        project: Шляхи проєкту.
        config: Параметри даних.
        source: Джерело VisA (у тестах — підмінене).
        weights: Ваги backbone (``"imagenet"``; у тестах — ``None``).
    """

    def __init__(
        self,
        project: ProjectPaths,
        config: WorkspaceSettings,
        source: VisaDatasetSource,
        weights: str | None = "imagenet",
    ) -> None:
        self._config = config
        self._source = source
        processed = project.data_processed_dir / "visa"
        self._images = ImageStore(source.root, processed / "images")
        self._features = FeatureCache(processed / "features")
        self._splits_repository = SplitRepository(processed / "splits.csv")
        self._weights = weights
        self._splits: pd.DataFrame | None = None
        self._extractors: dict[FeatureKey, keras.Model] = {}
        self._batches: dict[tuple[str, int], ImageBatch] = {}

    @property
    def source(self) -> VisaDatasetSource:
        """Джерело VisA."""
        return self._source

    @property
    def splits_path(self) -> Path:
        """Де лежить ``splits.csv`` (для ЛР5)."""
        return self._splits_repository.path

    def prepare(self, seed: int) -> pd.DataFrame:
        """Розпаковує дані, будує й зберігає розбиття всіх категорій."""
        config = self._config
        sample_categories = [c for c in config.background_categories if c not in config.categories]
        self._source.extract(config.categories, sample_categories, config.background_sample)
        splitter = AnomalySplitter(config.train_fraction, config.val_fraction_of_test, seed)
        frames = [splitter.split(self._source.read_samples(c)) for c in config.categories]
        self._splits = pd.concat(frames, ignore_index=True)
        self._splits_repository.save(self._splits)
        return self._splits

    def load_splits(self) -> pd.DataFrame:
        """Читає вже збережене розбиття (ЛР5): без розпакування архіву й перерозбиття.

        Raises:
            FileNotFoundError: Якщо ``splits.csv`` ще немає (спершу ``mllabs run 4``).
        """
        if not self._splits_repository.path.is_file():
            raise FileNotFoundError(
                f"Немає {self._splits_repository.path}: спершу виконайте `uv run mllabs run 4`"
            )
        self._splits = self._splits_repository.load()
        return self._splits

    @property
    def splits(self) -> pd.DataFrame:
        """Розбиття (після :meth:`prepare`)."""
        if self._splits is None:
            raise RuntimeError("Спершу викличте prepare()")
        return self._splits

    def category_rows(self, category: str) -> pd.DataFrame:
        """Рядки категорії в порядку, у якому лежать її зображення й ознаки в кешах."""
        return self.splits[self.splits["category"] == category].reset_index(drop=True)

    def part_mask(self, category: str, part: SplitPart) -> np.ndarray:
        """Булева маска рядків категорії, що належать частині ``part``."""
        return (self.category_rows(category)["part"] == str(part)).to_numpy()

    def images(self, category: str, size: int | None = None) -> ImageBatch:
        """Зменшені зображення й маски всієї категорії."""
        key = (category, size or self._config.image_size)
        if key not in self._batches:  # кеш у пам'яті: зображення — memmap, маски — компактні
            rows = self.category_rows(category)
            self._batches[key] = self._images.load(
                category, rows["image"].tolist(), rows["mask"].tolist(), key[1]
            )
        return self._batches[key]

    def background_images(self, category: str, count: int) -> np.ndarray:
        """Перші ``count`` нормальних зображень категорії (для виміру дисперсії тла)."""
        paths = self._source.normal_images(category)[:count]
        relative = [str(path.relative_to(self._source.root)) for path in paths]
        batch = self._images.load(
            f"background_{category}", relative, [""] * len(relative), self._config.image_size
        )
        return np.asarray(batch.images)

    def masks(self, category: str, part: SplitPart) -> np.ndarray:
        """Маски частини на сітці оцінювання ``eval_size``."""
        return np.asarray(
            self.images(category, self._config.eval_size).masks[self.part_mask(category, part)]
        )

    def labels(self, category: str, part: SplitPart) -> np.ndarray:
        """Мітки зображень частини."""
        rows = self.category_rows(category)
        return rows.loc[self.part_mask(category, part), "label"].to_numpy()

    def raw_features(self, category: str, key: FeatureKey) -> np.ndarray:
        """Ознаки всієї категорії до агрегації (``memmap``); рахуються при першому запиті."""
        config = self._config
        spec = BACKBONE_SPECS[key.backbone]
        side = config.image_size * key.tile_grid
        batch = self.images(category, side)
        manifest = {
            "backbone": str(key.backbone),
            "layers": list(spec.layers),
            "feature_dim": config.feature_dim,
            "channel_pooling": "gather-weighted-mean",
            "image_size": config.image_size,
            "tile_grid": key.tile_grid,
            "weights": self._weights,
            "images": self.category_rows(category)["image"].tolist(),
            "tf_version": tf.__version__,
            "keras_version": keras.__version__,
        }
        return self._features.get_or_compute(
            key.group,
            category,
            manifest,
            self._extractor(key),
            batch.images,
            config.extraction_batch_size,
        )

    def features(
        self, category: str, key: FeatureKey, patch_size: int, part: SplitPart
    ) -> np.ndarray:
        """Агреговані (``p×p``) ознаки однієї частини розбиття, ``float16`` у пам'яті."""
        raw = self.raw_features(category, key)
        rows = np.flatnonzero(self.part_mask(category, part))
        return PatchAggregator(patch_size).apply(raw, rows)

    def _extractor(self, key: FeatureKey) -> keras.Model:
        """Екстрактор (один на ключ за запуск)."""
        if key not in self._extractors:
            factory = FeatureExtractorFactory(
                self._config.image_size, self._config.feature_dim, self._weights
            )
            self._extractors[key] = factory.build(key.backbone, key.tile_grid)
        return self._extractors[key]

    def extractor_factory(self) -> FeatureExtractorFactory:
        """Фабрика екстракторів із параметрами роботи."""
        return FeatureExtractorFactory(
            self._config.image_size, self._config.feature_dim, self._weights
        )
