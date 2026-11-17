"""Спільний інтерфейс головок детектора і наскрізна модель «зображення → карта аномалій».

Будь-який детектор у цій роботі має однакову будову::

    зображення 0–255 → backbone (ознаки патчів) → агрегація p×p → ГОЛОВКА → карта патчів
                     → збільшення до 256 + гаусове згладжування → карта аномалій, max → оцінка

Різняться лише головки: Feature Autoencoder (власна мережа), PaDiM (гаусова модель
кожної позиції) і PatchCore (найближчий сусід у банку пам'яті). Тому вони реалізують один
інтерфейс :class:`PatchScorer`, а :func:`build_detector` збирає з будь-якої з них
диференційовну ``keras.Model`` — її бере ЛР5 для Integrated Gradients і Grad-CAM.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import keras
import numpy as np
import tensorflow as tf

from mllabs.anomaly.backbones import BackboneKind, FeatureExtractorFactory
from mllabs.anomaly.features import PatchAggregator
from mllabs.anomaly.postprocess import AnomalyMapPostprocessing


class PatchScorer(Protocol):
    """Головка детектора: ознаки патчів ``(B, h, w, D)`` → оцінки аномальності ``(B, h, w)``.

    ``Protocol`` — структурний інтерфейс (як interface у Java, але без ``implements``):
    класу достатньо мати ці методи.
    """

    def scoring_layer(self) -> keras.layers.Layer:
        """Шар Keras, що рахує оцінки патчів (працює в наскрізній моделі)."""
        ...

    def memory_bytes(self) -> int:
        """Скільки байтів займають параметри головки (ваги, банк, коваріації)."""
        ...


def score_patches(scorer: PatchScorer, features: np.ndarray, batch_size: int) -> np.ndarray:
    """Оцінки патчів для ``(N, h, w, D)`` пакетами; повертає ``(N, h, w)`` ``float32``."""
    layer = scorer.scoring_layer()
    run = tf.function(lambda batch: layer(batch, training=False))
    result = np.empty(features.shape[:3], dtype=np.float32)
    for start in range(0, len(features), batch_size):
        batch = tf.constant(np.asarray(features[start : start + batch_size], dtype=np.float32))
        result[start : start + len(batch)] = run(batch).numpy()
    return result


@dataclass(frozen=True)
class DetectorConfig:
    """Усе, що потрібно, щоб відтворити детектор (зберігається в ``detector.json``).

    Attributes:
        category: Категорія VisA.
        head: ``"fae"``, ``"padim"`` або ``"patchcore"``.
        backbone: Значення :class:`BackboneKind`.
        layers: Шари backbone — джерела ознак (для Grad-CAM у ЛР5).
        feature_dim: Розмірність вектора ознак ``D``.
        patch_size: Агрегація ``p×p``.
        tile_grid: Поділ на ``g×g`` тайлів (вхід моделі — ``g·image_size``).
        image_size: Сторона зображення/тайла для backbone.
        eval_size: Сторона карти аномалій.
        blur_sigma: σ гаусового згладжування карти.
        threshold_image: Поріг оцінки зображення (обрано на розміченому val за max F1).
        threshold_pixel: Поріг пікселя карти (обрано на розміченому val за max F1).
        seed: Зерно експерименту.
        tf_version: Версія TensorFlow.
        keras_version: Версія Keras.
    """

    category: str
    head: str
    backbone: str
    layers: tuple[str, ...]
    feature_dim: int
    patch_size: int
    tile_grid: int
    image_size: int
    eval_size: int
    blur_sigma: float
    threshold_image: float
    threshold_pixel: float
    seed: int
    tf_version: str
    keras_version: str

    @property
    def input_size(self) -> int:
        """Сторона вхідного зображення наскрізної моделі."""
        return self.tile_grid * self.image_size

    def save(self, path: Path) -> Path:
        """Записує конфіг у JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> DetectorConfig:
        """Читає конфіг із JSON (``**`` розпаковує словник в іменовані аргументи)."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{**data, "layers": tuple(data["layers"])})


def build_detector(
    config: DetectorConfig,
    scoring_layer: keras.layers.Layer,
    weights: str | None = "imagenet",
    expose_backbone: bool = False,
) -> keras.Model:
    """Наскрізна диференційовна модель «зображення → карта аномалій».

    Args:
        config: Параметри детектора.
        scoring_layer: Головка (``PatchScorer.scoring_layer()`` або завантажена FAE).
        weights: Ваги backbone (``"imagenet"``; ``None`` — випадкові, для тестів).
        expose_backbone: Додати виходи ``backbone_shallow`` і ``backbone_deep`` — активації
            шарів ``config.layers``, з яких будуються ознаки (для Grad-CAM у ЛР5). При
            тайлах (``tile_grid > 1``) їхній перший вимір — ``B·g²`` (по тайлу на рядок).

    Returns:
        Модель із входом ``(input_size, input_size, 3)`` у діапазоні 0–255 і словником
        виходів: ``anomaly_map`` ``(B, S, S)``, ``image_score`` ``(B,)``,
        ``patch_scores`` ``(B, h, w)`` (оцінки патчів до збільшення = помилка відновлення
        FAE), за потреби — ``backbone_shallow``/``backbone_deep``. Шари доступні за іменами:
        ``backbone`` (обрізана мережа ImageNet), ``fae_head`` (головка FAE), ``postprocess``.
    """
    factory = FeatureExtractorFactory(config.image_size, config.feature_dim, weights)
    images = keras.Input((config.input_size, config.input_size, 3), name="image")
    tensors = factory.wire(images, BackboneKind(config.backbone), config.tile_grid)
    aggregated = PatchAggregator(config.patch_size).layer()(tensors.features)
    patch_scores = scoring_layer(aggregated)
    anomaly_map = AnomalyMapPostprocessing(config.eval_size, config.blur_sigma, name="postprocess")(
        patch_scores
    )
    image_score = keras.ops.max(anomaly_map, axis=(1, 2))
    outputs = {"anomaly_map": anomaly_map, "image_score": image_score, "patch_scores": patch_scores}
    if expose_backbone:
        outputs |= {"backbone_shallow": tensors.shallow, "backbone_deep": tensors.deep}
    return keras.Model(images, outputs, name=f"detector_{config.head}")


#: Імена файлів збереженого детектора.
HEAD_FILE = "fae_head.keras"
CONFIG_FILE = "detector.json"


def save_bundle(directory: Path, head: keras.Model, config: DetectorConfig) -> Path:
    """Зберігає головку FAE і конфіг детектора в ``directory`` (для ЛР5)."""
    directory.mkdir(parents=True, exist_ok=True)
    # Нова модель на тих самих шарах — без compile() і стану оптимізатора Adam.
    inference_head = keras.Model(head.input, head.output, name=head.name)
    inference_head.save(directory / HEAD_FILE)
    config.save(directory / CONFIG_FILE)
    return directory


def load_bundle(
    directory: Path, weights: str | None = "imagenet", expose_backbone: bool = False
) -> tuple[DetectorConfig, keras.Model]:
    """Відновлює детектор із ``directory``: повертає конфіг і наскрізну модель.

    Імпорт модулів із власними шарами (``fae``) реєструє їх для ``load_model``.

    Args:
        directory: Каталог із ``fae_head.keras`` і ``detector.json``.
        weights: Ваги backbone (``"imagenet"`` — як під час навчання).
        expose_backbone: Див. :func:`build_detector` (виходи для Grad-CAM).
    """
    from mllabs.anomaly import fae  # noqa: F401 — реєстрація PatchReconstructionError

    config = DetectorConfig.load(directory / CONFIG_FILE)
    head = keras.models.load_model(directory / HEAD_FILE)
    return config, build_detector(config, head, weights, expose_backbone)
