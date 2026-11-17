"""Кеш ознак backbone на диску і агрегація патчів ``p×p``.

Прохід backbone — найдорожча операція (≈ 20 мс на зображення для ResNet50 на CPU),
а ознаки одного зображення не залежать від головки детектора. Тому ознаки рахуються один
раз і зберігаються ``float16`` у ``.npy`` (через ``open_memmap`` — масив на диску, з яким
працюють як зі звичайним NumPy-масивом, не завантажуючи весь у пам'ять). Поруч лежить
JSON-маніфест: якщо backbone, шари чи склад зображень змінилися — кеш перераховується.

Ознаки зберігаються ДО агрегації ``p×p``: агрегація дешева й робиться на льоту.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

_logger = logging.getLogger(__name__)


class FeatureCache:
    """Сховище ознак: ``<root>/<group>/<name>.npy`` + ``.json``.

    Args:
        root: Каталог кешу, напр. ``data/processed/visa/features``.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def get_or_compute(
        self,
        group: str,
        name: str,
        manifest: Mapping[str, Any],
        extractor: keras.Model,
        images: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """Повертає ознаки (``memmap`` тільки для читання), за потреби — рахує їх.

        Args:
            group: Підкаталог, напр. ``"resnet50_g1"``.
            name: Ім'я набору, напр. категорія.
            manifest: Усе, від чого залежать ознаки (backbone, шари, D, хеш файлів…).
            extractor: Модель «зображення → ознаки».
            images: ``(N, H, W, 3)`` ``uint8``.
            batch_size: Розмір пакета для проходу backbone.
        """
        array_path = self._root / group / f"{name}.npy"
        manifest_path = array_path.with_suffix(".json")
        if array_path.is_file() and manifest_path.is_file():
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            if stored == dict(manifest):
                return np.load(array_path, mmap_mode="r")
        array_path.parent.mkdir(parents=True, exist_ok=True)
        output_shape = (len(images), *extractor.output_shape[1:])
        _logger.info("Ознаки %s/%s: %s", group, name, output_shape)
        target = np.lib.format.open_memmap(
            array_path, mode="w+", dtype=np.float16, shape=output_shape
        )
        extract_features(extractor, images, batch_size, target)
        target.flush()
        del target  # закриває файл до запису маніфесту
        manifest_path.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
        return np.load(array_path, mmap_mode="r")


def extract_features(
    extractor: keras.Model, images: np.ndarray, batch_size: int, target: np.ndarray
) -> None:
    """Проганяє зображення через екстрактор пакетами й пише результат у ``target``."""
    run = tf.function(lambda batch: extractor(batch, training=False))
    for start in range(0, len(images), batch_size):
        batch = np.asarray(images[start : start + batch_size], dtype=np.float32)
        target[start : start + len(batch)] = run(tf.constant(batch)).numpy().astype(np.float16)


class PatchAggregator:
    """Локальна агрегація ознак ``p×p`` (визначення «розміру патча» з PatchCore).

    Кожен вектор сітки замінюється середнім по вікну ``p×p`` навколо нього (крок 1,
    доповнення «same»: на краях усереднюються лише наявні сусіди). Сітка не змінюється,
    але кожен вектор «бачить» ширшу ділянку зображення — приблизно ``8·p`` пікселів понад
    рецептивне поле backbone.

    Args:
        patch_size: ``p``; 1 — без агрегації.
    """

    def __init__(self, patch_size: int) -> None:
        self._patch_size = patch_size

    @property
    def patch_size(self) -> int:
        """Розмір вікна агрегації ``p``."""
        return self._patch_size

    def layer(self) -> keras.layers.Layer:
        """Той самий оператор як шар Keras (для наскрізної моделі ЛР5)."""
        return keras.layers.AveragePooling2D(
            self._patch_size,
            strides=1,
            padding="same",
            name=f"patch_aggregation_p{self._patch_size}",
        )

    def apply(
        self, features: np.ndarray, rows: np.ndarray | None = None, batch_size: int = 32
    ) -> np.ndarray:
        """Агрегує ознаки пакетами; повертає новий ``float16``-масив у пам'яті.

        Args:
            features: ``(N, h, w, D)`` — зокрема ``memmap`` з кешу.
            rows: Які рядки взяти (``None`` — усі); читаються з диска пакетами, тож у пам'яті
                не з'являється зайва повна копія.
            batch_size: Зображень за раз.
        """
        indices = np.arange(len(features)) if rows is None else np.asarray(rows)
        result = np.empty((len(indices), *features.shape[1:]), dtype=np.float16)
        pool = self.layer()
        for start in range(0, len(indices), batch_size):
            chunk = np.asarray(features[indices[start : start + batch_size]], dtype=np.float32)
            if self._patch_size > 1:
                chunk = pool(tf.constant(chunk)).numpy()
            result[start : start + len(chunk)] = chunk.astype(np.float16)
        return result
