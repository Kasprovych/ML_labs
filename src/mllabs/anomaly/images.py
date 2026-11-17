"""Зменшені копії зображень і масок VisA з кешем на диску.

Оригінали VisA — ~1500×1000 JPG; декодування одного займає ~10 мс, тому зменшені до
квадрата ``size×size`` копії зберігаються як ``uint8`` у ``.npy`` (читаються миттєво).
Зображення зменшується ЦІЛКОМ, без кропу (дефекти біля країв не губляться), тому
пропорції спотворюються — однаково для train і test. Маска зменшується методом
найближчого сусіда й бінаризується (``> 0``: у VisA значення маски — індекс дефекту).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_logger = logging.getLogger(__name__)

#: Потоків для декодування JPEG (Pillow відпускає GIL під час декодування).
_DECODE_WORKERS = 8


@dataclass(frozen=True)
class ImageBatch:
    """Зображення й маски однієї вибірки.

    Attributes:
        images: ``(N, S, S, 3)`` ``uint8``.
        masks: ``(N, S, S)`` ``bool`` — ``True`` на пікселях дефекту.
    """

    images: np.ndarray
    masks: np.ndarray


def load_resized_image(path: Path, size: int) -> np.ndarray:
    """Читає RGB-зображення й зменшує до ``size×size`` (білінійно, з антиаліасингом Pillow)."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR))


def load_resized_mask(path: Path | None, size: int) -> np.ndarray:
    """Читає маску й зменшує до ``size×size`` найближчим сусідом; без маски — нулі."""
    if path is None:
        return np.zeros((size, size), dtype=bool)
    with Image.open(path) as mask:
        return np.asarray(mask.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0


class ImageStore:
    """Кеш зменшених зображень і масок однієї категорії.

    Args:
        raw_root: Каталог VisA (відносно нього задано шляхи з ``splits.csv``).
        cache_dir: Каталог кешу, напр. ``data/processed/visa/images``.
    """

    def __init__(self, raw_root: Path, cache_dir: Path) -> None:
        self._raw_root = raw_root
        self._cache_dir = cache_dir

    def load(self, name: str, images: Sequence[str], masks: Sequence[str], size: int) -> ImageBatch:
        """Повертає зменшені зображення та маски; при першому виклику — рахує й кешує.

        Args:
            name: Ім'я набору в кеші (напр. категорія).
            images: Відносні шляхи зображень.
            masks: Відносні шляхи масок (``""`` — маски немає).
            size: Сторона квадрата в пікселях.
        """
        key = f"{name}_{size}"
        image_file = self._cache_dir / f"{key}_images.npy"
        mask_file = self._cache_dir / f"{key}_masks.npy"
        manifest_file = self._cache_dir / f"{key}.json"
        fingerprint = _fingerprint(images, masks)
        if manifest_file.is_file() and image_file.is_file() and mask_file.is_file():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") == fingerprint:
                return ImageBatch(np.load(image_file, mmap_mode="r"), np.load(mask_file))
        _logger.info("Декодування %d зображень (%s, %d px)", len(images), name, size)
        batch = self._decode(images, masks, size)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(image_file, batch.images)
        np.save(mask_file, batch.masks)
        manifest_file.write_text(
            json.dumps({"fingerprint": fingerprint, "count": len(images), "size": size}),
            encoding="utf-8",
        )
        return ImageBatch(np.load(image_file, mmap_mode="r"), batch.masks)

    def _decode(self, images: Sequence[str], masks: Sequence[str], size: int) -> ImageBatch:
        """Паралельне декодування (потоки, бо Pillow відпускає GIL)."""
        with ThreadPoolExecutor(_DECODE_WORKERS) as pool:
            decoded = list(
                pool.map(lambda rel: load_resized_image(self._raw_root / rel, size), images)
            )
            decoded_masks = list(
                pool.map(
                    lambda rel: load_resized_mask(self._raw_root / rel if rel else None, size),
                    masks,
                )
            )
        return ImageBatch(np.stack(decoded), np.stack(decoded_masks))


def _fingerprint(images: Sequence[str], masks: Sequence[str]) -> str:
    """Хеш переліку файлів — кеш недійсний, якщо змінився склад або порядок зображень."""
    return hashlib.sha256("\n".join([*images, "|", *masks]).encode("utf-8")).hexdigest()


def coarse_brightness_std(images: np.ndarray, cells: int) -> np.ndarray:
    """σ яскравості на грубій сітці ``cells×cells`` для кожного зображення.

    Зображення переводиться у відтінки сірого й усереднюється блоками — лишаються лише
    повільні зміни яскравості (освітлення, неоднорідне тло, розташування об'єктів).
    Їх стандартне відхилення — проста міра «неоднорідності тла» одного зображення.

    Args:
        images: ``(N, S, S, 3)`` ``uint8``; ``S`` ділиться на ``cells``.
        cells: Кількість блоків уздовж сторони.

    Returns:
        ``(N,)`` у одиницях яскравості 0–255.
    """
    gray = np.asarray(images, dtype=np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    n_images, side = gray.shape[0], gray.shape[1]
    block = side // cells
    # reshape (N, cells, block, cells, block) і середнє по осях блоку = середнє кожної клітинки.
    coarse = gray.reshape(n_images, cells, block, cells, block).mean(axis=(2, 4))
    return coarse.reshape(n_images, -1).std(axis=1)


def pixelwise_std(images: np.ndarray) -> float:
    """Середнє по пікселях σ яскравості між зображеннями (міра мінливості сцени між кадрами)."""
    gray = np.asarray(images, dtype=np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    return float(gray.std(axis=0).mean())
