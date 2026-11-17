"""Найближчі нормальні зображення train — «еталон норми» для IG і для вилучення патчів.

Для детектора аномалій нейтрального зображення немає: і чорне, і розмите, і середнє — самі
аномальні. Найчистіша альтернатива — схожа НОРМА з train: той самий фон, освітлення й
(якщо пощастить) положення об'єктів. Схожість — евклідова відстань мініатюр 32×32 у
відтінках сірого (дешево й нечутливо до дрібних деталей); серед ``n`` найближчих обирається
та, яку сам детектор вважає найнормальнішою (мінімальна оцінка).
"""

from __future__ import annotations

import numpy as np

#: Сторона мініатюри для пошуку сусідів.
THUMBNAIL_SIDE = 32
_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], np.float32)


def thumbnails(images: np.ndarray, side: int = THUMBNAIL_SIDE, batch: int = 64) -> np.ndarray:
    """``(N, S, S, 3)`` ``uint8`` → ``(N, side²)`` сірі мініатюри (середнє блоками)."""
    count, size = len(images), images.shape[1]
    block = size // side
    result = np.empty((count, side * side), np.float32)
    for start in range(0, count, batch):
        gray = np.asarray(images[start : start + batch], np.float32) @ _GRAY_WEIGHTS
        reduced = gray.reshape(len(gray), side, block, side, block).mean(axis=(2, 4))
        result[start : start + len(gray)] = reduced.reshape(len(gray), -1)
    return result


class NormalReferenceFinder:
    """Пошук схожих нормальних зображень train.

    Args:
        train_images: ``(N, S, S, 3)`` нормальні зображення train.
        train_scores: ``(N,)`` їхні оцінки аномальності (детектор ЛР4).
        candidates: Скільки найближчих розглядати.
        batch_size: Зображень у пакеті підрахунку середнього (обмежує пам'ять).
    """

    def __init__(
        self,
        train_images: np.ndarray,
        train_scores: np.ndarray,
        candidates: int,
        batch_size: int = 64,
    ) -> None:
        self._images = train_images
        self._scores = np.asarray(train_scores)
        self._thumbnails = thumbnails(train_images)
        self._candidates = candidates
        self._batch = batch_size

    def nearest(self, image: np.ndarray, count: int) -> np.ndarray:
        """Номери ``count`` найближчих train-зображень за зростанням відстані."""
        query = thumbnails(image[None])[0]
        distances = np.square(self._thumbnails - query).sum(axis=1)
        return np.argsort(distances, kind="stable")[:count]

    def reference_index(self, image: np.ndarray) -> int:
        """Серед ``candidates`` найближчих — номер найнормальнішого (мінімальна оцінка)."""
        nearest = self.nearest(image, self._candidates)
        return int(nearest[np.argmin(self._scores[nearest])])

    def reference(self, image: np.ndarray) -> np.ndarray:
        """Еталонне нормальне зображення ``(S, S, 3)`` ``uint8``."""
        return np.asarray(self._images[self.reference_index(image)])

    def references(self, image: np.ndarray, count: int) -> np.ndarray:
        """``count`` найближчих норм ``(count, S, S, 3)`` (для Expected Gradients)."""
        return np.asarray(self._images[np.sort(self.nearest(image, count))])

    def mean_image(self) -> np.ndarray:
        """Середнє зображення train (база IG «середнє»), ``float32``."""
        total = np.zeros(self._images.shape[1:], np.float64)
        for start in range(0, len(self._images), self._batch):
            chunk = self._images[start : start + self._batch]
            total += np.asarray(chunk, np.float64).sum(axis=0)
        return (total / len(self._images)).astype(np.float32)
