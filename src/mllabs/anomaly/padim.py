"""Реімплементація PaDiM (Defard et al., 2020) на спільних ознаках патчів.

PaDiM моделює кожну позицію сітки окремим багатовимірним нормальним розподілом:
для позиції ``(i, j)`` по всіх нормальних зображеннях train рахуються середнє ``μ`` і
коваріація ``Σ`` векторів ознак. Оцінка патча — відстань Махаланобіса
``sqrt((x − μ)ᵀ Σ⁻¹ (x − μ))``: «на скільки стандартних відхилень» патч далекий від
звичайного для цього місця вигляду. Щоб коваріації були невеликими, як у статті береться
випадкова підмножина ``d`` каналів. Це НЕ бібліотечна реалізація Anomalib, а відтворення
алгоритму за статтею на тих самих ознаках, що й власна модель.
"""

from __future__ import annotations

from typing import Any, override

import keras
import numpy as np
import tensorflow as tf

#: Скільки позицій сітки обробляти за раз під час оцінки коваріацій.
_POSITION_CHUNK = 128


@keras.saving.register_keras_serializable(package="mllabs")
class MahalanobisScoring(keras.layers.Layer):
    """Шар ``(B, h, w, D)`` → ``(B, h, w)``: відстань Махаланобіса в кожній позиції.

    Args:
        channels: Індекси обраних каналів ``(d,)``.
        grid: Сторони сітки ``(h, w)``.
        **kwargs: Стандартні параметри шару Keras.
    """

    def __init__(self, channels: list[int], grid: tuple[int, int], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._channels = [int(index) for index in channels]
        self._grid = (int(grid[0]), int(grid[1]))
        positions, depth = self._grid[0] * self._grid[1], len(self._channels)
        self.mean = self.add_weight(
            shape=(positions, depth), initializer="zeros", trainable=False, name="mean"
        )
        self.inverse_covariance = self.add_weight(
            shape=(positions, depth, depth),
            initializer="zeros",
            trainable=False,
            name="inverse_covariance",
        )

    @override
    def call(self, features: tf.Tensor) -> tf.Tensor:
        height, width = self._grid
        selected = tf.gather(features, self._channels, axis=-1)
        diff = tf.reshape(selected, (-1, height * width, len(self._channels))) - self.mean
        # einsum: для кожного зображення b і позиції p — квадратична форма diffᵀ · Σ⁻¹ · diff.
        projected = tf.einsum("bpi,pij->bpj", diff, self.inverse_covariance)
        squared = tf.reduce_sum(projected * diff, axis=-1)
        return tf.reshape(tf.sqrt(tf.maximum(squared, 0.0)), (-1, height, width))

    @override
    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        return (input_shape[0], *self._grid)

    @override
    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "channels": self._channels, "grid": self._grid}


class Padim:
    """PaDiM: гаусова модель кожної позиції сітки.

    Args:
        n_channels: Скільки випадкових каналів ``d`` залишити.
        epsilon: Регуляризація коваріації ``Σ + ε·I`` (щоб вона була оборотною).
        random_state: Зерно вибору каналів.
    """

    def __init__(self, n_channels: int, epsilon: float, random_state: int) -> None:
        self._n_channels = n_channels
        self._epsilon = epsilon
        self._random_state = random_state
        self._layer: MahalanobisScoring | None = None
        self._depth = n_channels  # фактичне d стане відомим після fit (min(d, D))

    def fit(self, train_features: np.ndarray) -> Padim:
        """Оцінює ``μ`` і ``Σ⁻¹`` кожної позиції за нормальними картами train ``(N, h, w, D)``."""
        n_images, height, width, depth = train_features.shape
        generator = np.random.default_rng(self._random_state)
        channels = np.sort(generator.choice(depth, min(self._n_channels, depth), replace=False))
        flat = train_features.reshape(n_images, height * width, depth)
        positions = height * width
        mean = np.empty((positions, len(channels)), np.float32)
        inverse = np.empty((positions, len(channels), len(channels)), np.float32)
        # Позиції обробляються шматками, щоб не тримати в пам'яті всі вектори у float64.
        for start in range(0, positions, _POSITION_CHUNK):
            chunk = np.asarray(flat[:, start : start + _POSITION_CHUNK][..., channels], np.float64)
            chunk_mean = chunk.mean(axis=0)
            centered = chunk - chunk_mean
            covariance = np.einsum("npi,npj->pij", centered, centered) / max(n_images - 1, 1)
            covariance += self._epsilon * np.eye(len(channels))
            mean[start : start + len(chunk_mean)] = chunk_mean
            inverse[start : start + len(chunk_mean)] = np.linalg.inv(covariance)
        self._depth = len(channels)
        layer = MahalanobisScoring(channels.tolist(), (height, width), name="padim_scoring")
        layer.mean.assign(mean)
        layer.inverse_covariance.assign(inverse)
        self._layer = layer
        return self

    def scoring_layer(self) -> keras.layers.Layer:
        """Шар оцінювання (після :meth:`fit`)."""
        if self._layer is None:
            raise RuntimeError("PaDiM ще не навчено: спершу викличте fit()")
        return self._layer

    def memory_bytes(self) -> int:
        """Середні та обернені коваріації, ``float32``."""
        layer = self.scoring_layer()
        return int((np.prod(layer.mean.shape) + np.prod(layer.inverse_covariance.shape)) * 4)

    def head_flops(self, n_positions: int) -> int:
        """FLOPs на зображення: ``2d² + 2d`` на позицію, ``d = min(n_channels, D)``."""
        depth = self._depth
        return int((2 * depth * depth + 2 * depth) * n_positions)
