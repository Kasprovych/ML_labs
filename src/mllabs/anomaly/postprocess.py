"""Пост-обробка карт аномалій: збільшення до роздільності оцінювання й гаусове згладжування.

Головка детектора дає оцінку для кожного патча сітки 32×32. Щоб порівнювати з піксельною
маскою, карту збільшують до 256×256 (білінійно) і згладжують гаусовим ядром (σ = 4 пікс.),
як у PaDiM/PatchCore: це прибирає «сходинки» сітки й поодинокі шумові піки. Оцінка
зображення — максимум карти. Усе реалізовано операціями TensorFlow (а не SciPy), щоб
градієнт проходив крізь пост-обробку — це потрібно для Integrated Gradients у ЛР5.
"""

from __future__ import annotations

from typing import Any, override

import keras
import numpy as np
import tensorflow as tf

#: Ядро обрізається на відстані 4σ (далі ваги < 0,04 % від центральної).
_TRUNCATE_SIGMAS = 4.0


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Нормоване одновимірне гаусове ядро довжини ``2·round(4σ)+1``."""
    radius = int(_TRUNCATE_SIGMAS * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


@keras.saving.register_keras_serializable(package="mllabs")
class AnomalyMapPostprocessing(keras.layers.Layer):
    """Шар ``(B, h, w)`` → ``(B, S, S)``: білінійне збільшення + гаусове згладжування.

    Гаусове ядро сепарабельне: 2-D згортка = 1-D по рядках, потім 1-D по стовпцях
    (``2·k`` множень на піксель замість ``k²``). Краї доповнюються дзеркально (``REFLECT``),
    щоб нулі доповнення не «затемнювали» карту біля меж зображення.

    Args:
        output_size: Сторона вихідної карти ``S``.
        sigma: σ гаусового ядра в пікселях вихідної карти (0 — без згладжування).
        **kwargs: Стандартні параметри шару Keras.
    """

    def __init__(self, output_size: int, sigma: float, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._output_size = output_size
        self._sigma = sigma

    @override
    def call(self, patch_maps: tf.Tensor) -> tf.Tensor:
        # expand_dims/squeeze замість зрізів [..., None] — зрізи (StridedSlice) заважають
        # конвертації моделі в TFLite для edge-пристроїв (ЛР5).
        maps = tf.image.resize(
            tf.expand_dims(patch_maps, -1), (self._output_size, self._output_size)
        )
        if self._sigma > 0:
            kernel = gaussian_kernel_1d(self._sigma)
            radius = len(kernel) // 2
            maps = tf.pad(
                maps, [[0, 0], [radius, radius], [radius, radius], [0, 0]], mode="REFLECT"
            )
            column = tf.constant(kernel.reshape(-1, 1, 1, 1), dtype=maps.dtype)
            row = tf.constant(kernel.reshape(1, -1, 1, 1), dtype=maps.dtype)
            maps = tf.nn.conv2d(maps, column, strides=1, padding="VALID")
            maps = tf.nn.conv2d(maps, row, strides=1, padding="VALID")
        return tf.squeeze(maps, -1)

    @override
    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        return (input_shape[0], self._output_size, self._output_size)

    @override
    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "output_size": self._output_size, "sigma": self._sigma}


class AnomalyMapPostprocessor:
    """Зручна обгортка для NumPy: пакетна пост-обробка вже обчислених карт патчів.

    Args:
        output_size: Сторона вихідної карти.
        sigma: σ гаусового згладжування.
        batch_size: Скільки карт обробляти за раз.
    """

    def __init__(self, output_size: int, sigma: float, batch_size: int = 64) -> None:
        self._layer = AnomalyMapPostprocessing(output_size, sigma)
        self._output_size = output_size
        self._batch_size = batch_size
        self._run = tf.function(self._layer)

    def maps(self, patch_maps: np.ndarray) -> np.ndarray:
        """``(N, h, w)`` → ``(N, S, S)`` ``float32``."""
        result = np.empty((len(patch_maps), self._output_size, self._output_size), np.float32)
        for start in range(0, len(patch_maps), self._batch_size):
            batch = tf.constant(
                np.asarray(patch_maps[start : start + self._batch_size], np.float32)
            )
            result[start : start + len(batch)] = self._run(batch).numpy()
        return result


def image_scores(anomaly_maps: np.ndarray) -> np.ndarray:
    """Оцінка зображення — максимум його карти аномалій."""
    return anomaly_maps.reshape(len(anomaly_maps), -1).max(axis=1)
