"""Пакетні прогони детектора ЛР4: із кешу ознак (швидко) або наскрізно із зображень.

* :class:`CachedScorer` — ознаки backbone вже пораховані ЛР4 (кеш ``float16``), тому для
  незмінених зображень достатньо прогнати лише головку FAE і пост-обробку. Саме так ЛР4
  рахувала оцінки test — результат збігається з ``test_predictions.csv``.
* :class:`DetectorRunner` — повна модель «зображення → карта»; потрібна, коли зображення
  змінене (perturbation, дрейф, рандомізація ваг).
"""

from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np
import tensorflow as tf

from mllabs.anomaly.postprocess import AnomalyMapPostprocessor


@dataclass(frozen=True)
class PartOutputs:
    """Результат детектора для набору зображень.

    Attributes:
        patch_scores: ``(N, h, w)`` помилка відновлення кожного патча.
        maps: ``(N, E, E)`` карти аномалій (або ``None``, якщо не зберігалися).
        scores: ``(N,)`` оцінки зображень (максимум карти).
    """

    patch_scores: np.ndarray
    maps: np.ndarray | None
    scores: np.ndarray


class CachedScorer:
    """Головка FAE + пост-обробка на кешованих ознаках.

    Args:
        head: Головка «ознаки ``(B, h, w, D)`` → помилка ``(B, h, w)``».
        eval_size: Сторона карти аномалій.
        blur_sigma: σ гаусового згладжування (як у детекторі ЛР4).
        batch_size: Зображень у пакеті.
    """

    def __init__(
        self, head: keras.Model, eval_size: int, blur_sigma: float, batch_size: int
    ) -> None:
        self._run = tf.function(lambda batch: head(batch, training=False), reduce_retracing=True)
        self._postprocessor = AnomalyMapPostprocessor(eval_size, blur_sigma, batch_size)
        self._batch = batch_size

    def patch_scores(self, features: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Оцінки патчів для рядків ``rows`` кешу ознак."""
        parts = []
        for start in range(0, len(rows), self._batch):
            chunk = np.asarray(features[rows[start : start + self._batch]], np.float32)
            parts.append(self._run(tf.constant(chunk)).numpy())
        return np.concatenate(parts) if parts else np.empty((0, *features.shape[1:3]), np.float32)

    def maps(self, patch_scores: np.ndarray) -> np.ndarray:
        """Карти аномалій з оцінок патчів (збільшення + згладжування)."""
        return self._postprocessor.maps(patch_scores)

    def score(self, features: np.ndarray, rows: np.ndarray, keep_maps: bool = True) -> PartOutputs:
        """Повний результат для рядків ``rows``.

        Args:
            features: Кеш ознак категорії.
            rows: Номери рядків.
            keep_maps: Чи зберігати карти (для train не потрібно — 200 МБ пам'яті).
        """
        patch = self.patch_scores(features, rows)
        maps = self.maps(patch)
        scores = maps.reshape(len(maps), -1).max(axis=1)
        return PartOutputs(patch, maps if keep_maps else None, scores)


class DetectorRunner:
    """Наскрізний прогін детектора на зображеннях ``uint8`` пакетами.

    Args:
        detector: Наскрізна модель з виходами ``anomaly_map``, ``image_score``,
            ``patch_scores``.
        batch_size: Зображень у пакеті.
    """

    def __init__(self, detector: keras.Model, batch_size: int) -> None:
        self._batch = batch_size

        @tf.function
        def run(images: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            outputs = detector(images, training=False)
            return outputs["patch_scores"], outputs["anomaly_map"], outputs["image_score"]

        self._run = run

    def run(self, images: np.ndarray, keep_maps: bool = True) -> PartOutputs:
        """Оцінки для ``(N, S, S, 3)``; останній пакет доповнюється, щоб не перебудовувати граф."""
        patch_parts, map_parts, score_parts = [], [], []
        for start in range(0, len(images), self._batch):
            chunk = np.asarray(images[start : start + self._batch], np.float32)
            size = len(chunk)
            if size < self._batch:  # доповнення до сталого розміру пакета (той самий граф)
                padding = np.repeat(chunk[-1:], self._batch - size, axis=0)
                chunk = np.concatenate([chunk, padding])
            patch, maps, scores = self._run(tf.constant(chunk))
            patch_parts.append(patch.numpy()[:size])
            score_parts.append(scores.numpy()[:size])
            if keep_maps:
                map_parts.append(maps.numpy()[:size])
        return PartOutputs(
            np.concatenate(patch_parts),
            np.concatenate(map_parts) if keep_maps else None,
            np.concatenate(score_parts),
        )
