"""Реімплементація PatchCore (Roth et al., 2022) на спільних ознаках патчів.

PatchCore зберігає «банк пам'яті» векторів нормальних патчів train. Оцінка патча — відстань
до найближчого сусіда в банку: якщо такого вигляду патча серед норми не траплялося, відстань
велика. Усіх векторів train сотні тисяч, тому банк проріджують жадібним k-center coreset:
послідовно додають вектор, НАЙДАЛЬШИЙ від уже обраних, — так невелика підмножина (1 %)
рівномірно покриває весь простір нормальних патчів. Для швидкості відстані в coreset
рахуються після випадкової проєкції Джонсона–Лінденштраусса (512 → 128 вимірів, відстані
зберігаються приблизно). Перезважування оцінки за сусідами зі статті не використовується.
Це відтворення алгоритму за статтею, а не бібліотечна реалізація Anomalib.
"""

from __future__ import annotations

from typing import Any, override

import keras
import numpy as np
import tensorflow as tf


class GreedyCoresetSampler:
    """Жадібний k-center coreset.

    Args:
        ratio: Частка векторів, що потрапляє в банк.
        projection_dim: Розмірність випадкової проєкції.
        random_state: Зерно проєкції та стартової точки.
    """

    def __init__(self, ratio: float, projection_dim: int, random_state: int) -> None:
        self._ratio = ratio
        self._projection_dim = projection_dim
        self._random_state = random_state

    def select(self, vectors: np.ndarray) -> np.ndarray:
        """Повертає індекси обраних векторів ``(M,)``, ``M = ceil(ratio · N)``."""
        n_vectors, depth = vectors.shape
        count = max(1, int(np.ceil(self._ratio * n_vectors)))
        generator = np.random.default_rng(self._random_state)
        projection = generator.normal(size=(depth, self._projection_dim)) / np.sqrt(
            self._projection_dim
        )
        projected = tf.constant(np.asarray(vectors, np.float32) @ projection.astype(np.float32))
        start = int(generator.integers(n_vectors))
        return _greedy_k_center(projected, tf.constant(count), tf.constant(start)).numpy()


@tf.function
def _greedy_k_center(points: tf.Tensor, count: tf.Tensor, start: tf.Tensor) -> tf.Tensor:
    """Цикл coreset у графі TensorFlow (без накладних витрат Python на кожній ітерації).

    ``min_distance[i]`` — квадрат відстані від точки ``i`` до найближчої вже обраної.
    Відстань до нової точки ``c``: ``‖x‖² − 2·x·c + ‖c‖²`` — одне множення матриці на вектор.
    """
    norms = tf.reduce_sum(tf.square(points), axis=1)

    def squared_distance(index: tf.Tensor) -> tf.Tensor:
        center = points[index]
        return norms - 2.0 * tf.linalg.matvec(points, center) + norms[index]

    selected = tf.TensorArray(tf.int32, size=count)
    selected = selected.write(0, tf.cast(start, tf.int32))
    min_distance = squared_distance(start)
    for step in tf.range(1, count):  # у tf.function цикл по tf.range стає графовим циклом
        index = tf.cast(tf.argmax(min_distance), tf.int32)
        selected = selected.write(step, index)
        min_distance = tf.minimum(min_distance, squared_distance(index))
    return selected.stack()


@keras.saving.register_keras_serializable(package="mllabs")
class NearestNeighborScoring(keras.layers.Layer):
    """Шар ``(B, h, w, D)`` → ``(B, h, w)``: відстань до найближчого вектора банку.

    Args:
        bank_size: Кількість векторів у банку ``M``.
        feature_dim: Розмірність ``D``.
        **kwargs: Стандартні параметри шару Keras.
    """

    def __init__(self, bank_size: int, feature_dim: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._bank_size = bank_size
        self._feature_dim = feature_dim
        self.bank = self.add_weight(
            shape=(bank_size, feature_dim), initializer="zeros", trainable=False, name="memory_bank"
        )

    @override
    def call(self, features: tf.Tensor) -> tf.Tensor:
        height, width = features.shape[1], features.shape[2]
        vectors = tf.reshape(features, (-1, self._feature_dim))
        squared = (
            tf.reduce_sum(tf.square(vectors), axis=1, keepdims=True)
            - 2.0 * tf.matmul(vectors, self.bank, transpose_b=True)
            + tf.reduce_sum(tf.square(self.bank), axis=1)[tf.newaxis, :]
        )
        nearest = tf.reduce_min(squared, axis=1)
        return tf.reshape(tf.sqrt(tf.maximum(nearest, 0.0)), (-1, height, width))

    @override
    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        return tuple(input_shape[:3])

    @override
    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "bank_size": self._bank_size,
            "feature_dim": self._feature_dim,
        }


class PatchCore:
    """PatchCore: банк пам'яті нормальних патчів + найближчий сусід.

    Args:
        sampler: Вибір coreset.
        max_train_vectors: Верхня межа векторів train перед coreset (випадкова підвибірка
            понад неї) — обмежує пам'ять і час для великих сіток (тайли 2×2).
        random_state: Зерно підвибірки.
    """

    def __init__(
        self, sampler: GreedyCoresetSampler, max_train_vectors: int, random_state: int
    ) -> None:
        self._sampler = sampler
        self._max_train_vectors = max_train_vectors
        self._random_state = random_state
        self._layer: NearestNeighborScoring | None = None
        self._n_train_vectors = 0

    def fit(self, train_features: np.ndarray) -> PatchCore:
        """Будує банк із нормальних карт train ``(N, h, w, D)``."""
        depth = train_features.shape[-1]
        flat = train_features.reshape(-1, depth)
        if len(flat) > self._max_train_vectors:
            generator = np.random.default_rng(self._random_state)
            rows = np.sort(generator.choice(len(flat), self._max_train_vectors, replace=False))
            flat = flat[rows]
        vectors = np.asarray(flat, np.float32)
        indices = self._sampler.select(vectors)
        layer = NearestNeighborScoring(len(indices), depth, name="patchcore_scoring")
        layer.bank.assign(vectors[indices])
        self._layer = layer
        self._n_train_vectors = len(vectors)
        return self

    @property
    def bank_size(self) -> int:
        """Розмір банку ``M``."""
        return int(self.scoring_layer().bank.shape[0])

    @property
    def n_train_vectors(self) -> int:
        """Скільки векторів train було до прорідження."""
        return self._n_train_vectors

    def scoring_layer(self) -> keras.layers.Layer:
        """Шар оцінювання (після :meth:`fit`)."""
        if self._layer is None:
            raise RuntimeError("PatchCore ще не навчено: спершу викличте fit()")
        return self._layer

    def memory_bytes(self) -> int:
        """Банк пам'яті, ``float32``."""
        return int(np.prod(self.scoring_layer().bank.shape) * 4)

    def head_flops(self, n_positions: int) -> int:
        """FLOPs на зображення: на позицію ``2·M·D`` (скалярні добутки з усім банком)."""
        bank = self.scoring_layer().bank.shape
        return int(2 * bank[0] * bank[1] * n_positions)
