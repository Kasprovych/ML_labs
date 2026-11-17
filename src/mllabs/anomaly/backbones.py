"""Екстрактори ознак патчів на основі переднавчених backbone з ``keras.applications``.

Ідея (як у PaDiM / PatchCore): згорткова мережа, навчена на ImageNet, уже вміє описувати
локальні фрагменти зображення. Беремо два ПРОМІЖНІ шари (крок 8 і крок 16 пікселів):
перший — дрібні текстури, другий — ширший контекст. Глибший шар збільшуємо вдвічі
(білінійно), склеюємо з першим по каналах і зводимо кількість каналів до ``D`` фіксованим
усередненням сусідніх каналів (без навчання). Результат — сітка ``(S/8)×(S/8)`` векторів:
кожен вектор описує «патч» зображення з кроком 8 пікселів.

Усе — шари Keras, тож модель «зображення 0–255 → ознаки» диференційовна (потрібно ЛР5).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, override

import keras
import numpy as np
import tensorflow as tf

#: Пакет, під яким власні шари реєструються для збереження/завантаження ``.keras``.
_PACKAGE = "mllabs"


class BackboneKind(StrEnum):
    """Підтримувані backbone."""

    RESNET50 = "resnet50"
    MOBILENET_V2 = "mobilenet_v2"
    EFFICIENTNET_B0 = "efficientnet_b0"


class PreprocessMode(StrEnum):
    """Як backbone очікує нормалізований вхід (так його навчали на ImageNet)."""

    CAFFE = "caffe"  # RGB→BGR, мінус середні ImageNet (ResNet50)
    TF = "tf"  # масштаб до [-1, 1] (MobileNetV2)
    NONE = "none"  # нормалізація вбудована в саму мережу (EfficientNet)


@dataclass(frozen=True)
class BackboneSpec:
    """Опис backbone: шари-джерела ознак і препроцесинг.

    Attributes:
        kind: Тип backbone.
        layers: Імена шарів із кроком 8 і 16 пікселів.
        channels: Кількість каналів цих шарів.
        preprocess: Режим препроцесингу.
        label: Підпис для таблиць і графіків.
    """

    kind: BackboneKind
    layers: tuple[str, str]
    channels: tuple[int, int]
    preprocess: PreprocessMode
    label: str


BACKBONE_SPECS: dict[BackboneKind, BackboneSpec] = {
    BackboneKind.RESNET50: BackboneSpec(
        BackboneKind.RESNET50,
        ("conv3_block4_out", "conv4_block6_out"),
        (512, 1024),
        PreprocessMode.CAFFE,
        "ResNet50",
    ),
    BackboneKind.MOBILENET_V2: BackboneSpec(
        BackboneKind.MOBILENET_V2,
        ("block_6_expand_relu", "block_13_expand_relu"),
        (192, 576),
        PreprocessMode.TF,
        "MobileNetV2",
    ),
    BackboneKind.EFFICIENTNET_B0: BackboneSpec(
        BackboneKind.EFFICIENTNET_B0,
        ("block4a_expand_activation", "block6a_expand_activation"),
        (240, 672),
        PreprocessMode.NONE,
        "EfficientNetB0",
    ),
}

_APPLICATIONS: dict[BackboneKind, Callable[..., keras.Model]] = {
    BackboneKind.RESNET50: keras.applications.ResNet50,
    BackboneKind.MOBILENET_V2: keras.applications.MobileNetV2,
    BackboneKind.EFFICIENTNET_B0: keras.applications.EfficientNetB0,
}

#: Середні значення каналів ImageNet у порядку BGR (препроцесинг «caffe»).
_CAFFE_MEAN_BGR = (103.939, 116.779, 123.68)


@keras.saving.register_keras_serializable(package=_PACKAGE)
class BackbonePreprocessing(keras.layers.Layer):
    """Переводить зображення 0–255 у масштаб, на якому навчався backbone.

    Args:
        mode: Значення :class:`PreprocessMode`.
        **kwargs: Стандартні параметри шару Keras (``name`` тощо).
    """

    def __init__(self, mode: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._mode = PreprocessMode(mode)

    @override
    def call(self, images: tf.Tensor) -> tf.Tensor:
        if self._mode is PreprocessMode.CAFFE:
            bgr = tf.reverse(images, axis=[-1])  # розвертає порядок каналів RGB→BGR
            return bgr - tf.constant(_CAFFE_MEAN_BGR, dtype=images.dtype)
        if self._mode is PreprocessMode.TF:
            return images / 127.5 - 1.0
        return images

    @override
    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "mode": str(self._mode)}


def channel_pooling_matrix(in_channels: int, out_channels: int) -> np.ndarray:
    """Матриця ``(in, out)`` усереднення сусідніх каналів (як ``adaptive_avg_pool1d``).

    Вихідний канал ``i`` — середнє вхідних каналів з відрізка
    ``[floor(i·in/out), ceil((i+1)·in/out))``.
    """
    matrix = np.zeros((in_channels, out_channels), dtype=np.float32)
    for index in range(out_channels):
        start = (index * in_channels) // out_channels
        end = -((-(index + 1) * in_channels) // out_channels)  # ціле ділення зі стелею
        matrix[start:end, index] = 1.0 / (end - start)
    return matrix


@keras.saving.register_keras_serializable(package=_PACKAGE)
class ChannelPooling(keras.layers.Layer):
    """Фіксоване (не навчається) зведення каналів до ``out_channels`` усередненням.

    Кожен вихідний канал — середнє 1–3 сусідніх вхідних (див. :func:`channel_pooling_matrix`).
    Замість множення на розріджену матрицю ``(C, D)`` (≈ 0,8–1,6 GFLOPs зайвої роботи на
    зображення) беремо лише потрібні канали (``tf.gather``) і зважено підсумовуємо:
    ``D × w`` множень на позицію, де ``w`` — ширина вікна. Результат той самий.

    Args:
        out_channels: Кількість вихідних каналів ``D``.
        **kwargs: Стандартні параметри шару Keras.
    """

    def __init__(self, out_channels: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._out_channels = out_channels
        self._indices: tf.Tensor | None = None
        self._weights: tf.Tensor | None = None

    @override
    def build(self, input_shape: tuple[int | None, ...]) -> None:
        matrix = channel_pooling_matrix(int(input_shape[-1]), self._out_channels)
        width = int((matrix > 0).sum(axis=0).max())
        indices = np.zeros((self._out_channels, width), dtype=np.int32)
        weights = np.zeros((self._out_channels, width), dtype=np.float32)
        for column in range(self._out_channels):
            rows = np.flatnonzero(matrix[:, column])
            indices[column, : len(rows)] = rows
            weights[column, : len(rows)] = matrix[rows, column]
        self._indices = tf.constant(indices.reshape(-1))
        self._weights = tf.constant(weights)

    @override
    def call(self, features: tf.Tensor) -> tf.Tensor:
        gathered = tf.gather(features, self._indices, axis=-1)  # (..., D·w)
        shape = tf.concat([tf.shape(features)[:-1], tf.shape(self._weights)], axis=0)
        windows = tf.reshape(gathered, shape)  # (..., D, w)
        return tf.reduce_sum(windows * tf.cast(self._weights, features.dtype), axis=-1)

    @override
    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        return (*input_shape[:-1], self._out_channels)

    @override
    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "out_channels": self._out_channels}


@keras.saving.register_keras_serializable(package=_PACKAGE)
class TileSplit(keras.layers.Layer):
    """Ділить зображення ``(B, g·S, g·S, C)`` на ``g×g`` тайлів ``(B·g², S, S, C)``.

    Args:
        grid: Кількість тайлів уздовж сторони ``g`` (1 — без поділу).
        **kwargs: Стандартні параметри шару Keras.
    """

    def __init__(self, grid: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._grid = grid

    @override
    def call(self, images: tf.Tensor) -> tf.Tensor:
        if self._grid == 1:
            return images
        g = self._grid
        side, channels = images.shape[1] // g, images.shape[-1]
        tiles = tf.reshape(images, (-1, g, side, g, side, channels))
        tiles = tf.transpose(tiles, (0, 1, 3, 2, 4, 5))  # (B, рядок, стовпець, S, S, C)
        return tf.reshape(tiles, (-1, side, side, channels))

    @override
    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "grid": self._grid}


@keras.saving.register_keras_serializable(package=_PACKAGE)
class TileMerge(keras.layers.Layer):
    """Склеює карти тайлів ``(B·g², h, w, D)`` назад у ``(B, g·h, g·w, D)``.

    Args:
        grid: Кількість тайлів уздовж сторони ``g``.
        **kwargs: Стандартні параметри шару Keras.
    """

    def __init__(self, grid: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._grid = grid

    @override
    def call(self, maps: tf.Tensor) -> tf.Tensor:
        if self._grid == 1:
            return maps
        g = self._grid
        height, width, depth = maps.shape[1], maps.shape[2], maps.shape[3]
        merged = tf.reshape(maps, (-1, g, g, height, width, depth))
        merged = tf.transpose(merged, (0, 1, 3, 2, 4, 5))  # (B, рядок, h, стовпець, w, D)
        return tf.reshape(merged, (-1, g * height, g * width, depth))

    @override
    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "grid": self._grid}


@dataclass(frozen=True)
class FeatureTensors:
    """Символьні тензори екстрактора всередині функціональної моделі.

    Attributes:
        features: Ознаки патчів ``(B, g·S/8, g·S/8, D)``.
        shallow: Активації шару з кроком 8 ``(B·g², S/8, S/8, C₁)`` (по тайлах).
        deep: Активації шару з кроком 16 ``(B·g², S/16, S/16, C₂)``.
    """

    features: keras.KerasTensor
    shallow: keras.KerasTensor
    deep: keras.KerasTensor


class FeatureExtractorFactory:
    """Будує Keras-модель «зображення 0–255 → сітка ознак патчів».

    Args:
        image_size: Сторона зображення (або тайла), яке бачить backbone, ``S``.
        feature_dim: Розмірність вектора ознак ``D`` після зведення каналів.
        weights: ``"imagenet"`` або ``None`` (випадкові ваги — для тестів і замірів швидкодії).
    """

    def __init__(self, image_size: int, feature_dim: int, weights: str | None = "imagenet") -> None:
        self._image_size = image_size
        self._feature_dim = feature_dim
        self._weights = weights

    @property
    def feature_dim(self) -> int:
        """Розмірність ``D`` вектора ознак."""
        return self._feature_dim

    def grid_size(self, tile_grid: int = 1) -> int:
        """Сторона сітки патчів: ``g · S / 8``."""
        return tile_grid * self._image_size // 8

    def build(self, kind: BackboneKind, tile_grid: int = 1) -> keras.Model:
        """Модель з входом ``(g·S, g·S, 3)`` і виходом ``(g·S/8, g·S/8, D)``.

        Args:
            kind: Тип backbone.
            tile_grid: Поділ на ``g×g`` тайлів; кожен тайл обробляється окремо.
        """
        size = tile_grid * self._image_size
        inputs = keras.Input((size, size, 3), name="image")
        return keras.Model(
            inputs, self.wire(inputs, kind, tile_grid).features, name="feature_extractor"
        )

    def wire(
        self, images: keras.KerasTensor, kind: BackboneKind, tile_grid: int = 1
    ) -> FeatureTensors:
        """Під'єднує шари екстрактора до символьного входу ``images`` (функціональний API).

        На відміну від :meth:`build`, шари не загортаються в окрему модель, тож проміжні
        активації backbone лишаються «видимими» в зовнішній моделі — це потрібно Grad-CAM (ЛР5).
        """
        spec = BACKBONE_SPECS[kind]
        tiles = TileSplit(tile_grid, name="tile_split")(images)
        prepared = BackbonePreprocessing(str(spec.preprocess), name="preprocess")(tiles)
        shallow, deep = self.build_backbone(kind)(prepared)
        deep_up = keras.layers.UpSampling2D(2, interpolation="bilinear", name="upsample")(deep)
        stacked = keras.layers.Concatenate(name="concat")([shallow, deep_up])
        pooled = ChannelPooling(self._feature_dim, name="channel_pooling")(stacked)
        merged = TileMerge(tile_grid, name="tile_merge")(pooled)
        return FeatureTensors(merged, shallow, deep)

    def build_backbone(self, kind: BackboneKind) -> keras.Model:
        """Обрізаний backbone: вхід ``(S, S, 3)`` → два проміжні шари (ім'я моделі ``backbone``)."""
        spec = BACKBONE_SPECS[kind]
        full = _APPLICATIONS[kind](
            include_top=False,
            weights=self._weights,
            input_shape=(self._image_size, self._image_size, 3),
        )
        outputs = [full.get_layer(name).output for name in spec.layers]
        return keras.Model(full.input, outputs, name="backbone")
