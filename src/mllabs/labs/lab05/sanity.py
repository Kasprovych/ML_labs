"""Перевірки адекватності пояснень (R3 б, в): рандомізація моделі та стійкість до шуму.

* **Sanity check** (Adebayo et al., 2018): якщо замінити навчені ваги випадковими, а
  пояснення майже не зміниться — метод пояснює не модель, а саме зображення (працює як
  детектор країв). Рівні: «випадкова головка FAE» (backbone навчений на ImageNet) і
  «випадкова вся модель» (backbone теж випадковий). Мала схожість з поясненням навченої
  моделі = тест пройдено.
* **Стійкість**: малий шум на вході (σ = 2 одиниці яскравості, непомітний оку) не має
  змінювати пояснення. Велика схожість = тест пройдено.

Схожість карт — ранговий Spearman ρ і структурна схожість SSIM (``tf.image.ssim``) на
рангових картах. Для кожного рівня рандомізації окремо міряється Pixel AUROC самого
рандомізованого детектора: висновок «пояснення не залежить від моделі» має сенс лише там,
де рандомізований детектор справді втратив локалізацію.
"""

from __future__ import annotations

from collections.abc import Iterator

import keras
import numpy as np
import tensorflow as tf
from scipy.stats import rankdata

from mllabs.anomaly.detector import DetectorConfig, build_detector
from mllabs.labs.lab05.localization import downsample
from mllabs.labs.lab05.weights_analysis import spearman


def _all_layers(model: keras.Model) -> Iterator[keras.layers.Layer]:
    """Усі шари моделі, включно з вкладеними підмоделями (обхід у глибину)."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            yield from _all_layers(layer)  # yield from — «віддати всі елементи генератора»
        else:
            yield layer


def _he_normal(shape: tuple[int, ...], fan_in: int, generator: np.random.Generator) -> np.ndarray:
    """Ініціалізація He: N(0, 2/fan_in) — стандартна для згорток з ReLU/Swish."""
    return generator.normal(0.0, np.sqrt(2.0 / max(fan_in, 1)), shape).astype(np.float32)


def reinitialize(model: keras.Model, generator: np.random.Generator) -> None:
    """Детерміновано (за ``generator``) замінює навчені ваги моделі випадковими.

    Ядра згорток і ``Dense`` — He-нормальні, зсуви — нулі, BatchNormalization — тотожна
    (γ = 1, β = 0, середнє 0, дисперсія 1). Шари без ядер (препроцесинг) не змінюються.
    Власний генератор NumPy замість ``keras.utils.set_random_seed`` не чіпає глобальний
    стан випадковості посеред прогону.
    """
    for layer in _all_layers(model):
        if isinstance(layer, keras.layers.BatchNormalization):
            gamma, beta, mean, variance = layer.get_weights()  # порядок ваг шару BN у Keras
            layer.set_weights(
                [
                    np.ones_like(gamma),
                    np.zeros_like(beta),
                    np.zeros_like(mean),
                    np.ones_like(variance),
                ]
            )
        elif getattr(layer, "kernel", None) is not None:
            kernel = np.asarray(layer.kernel)
            if isinstance(layer, keras.layers.DepthwiseConv2D):
                fan_in = kernel.shape[0] * kernel.shape[1]
            else:
                fan_in = int(np.prod(kernel.shape[:-1]))
            weights = [_he_normal(kernel.shape, fan_in, generator)]
            weights += [np.zeros_like(weight) for weight in layer.get_weights()[1:]]
            layer.set_weights(weights)


def randomized_head(head: keras.Model, seed: int, reset_normalization: bool = False) -> keras.Model:
    """Копія головки FAE з випадковими вагами ``Dense``.

    Args:
        head: Навчена головка.
        seed: Зерно випадкових ваг.
        reset_normalization: ``False`` — нормалізація μ/σ з train зберігається (тоді «випадкова»
            головка лишається працездатним детектором: випадкова проекція нормалізованих
            ознак ≈ діагональна гаусова відстань до норми); ``True`` — μ = 0, σ = 1.
    """

    def clone_layer(layer: keras.layers.Layer) -> keras.layers.Layer:
        if reset_normalization and isinstance(layer, keras.layers.Normalization):
            dim = int(np.asarray(layer.mean).size)
            return keras.layers.Normalization(
                mean=np.zeros(dim, np.float32), variance=np.ones(dim, np.float32), name=layer.name
            )
        return layer.__class__.from_config(layer.get_config())

    clone = keras.models.clone_model(head, clone_function=clone_layer)
    generator = np.random.default_rng(seed)
    for layer in _all_layers(clone):
        if isinstance(layer, keras.layers.Dense):
            kernel = np.asarray(layer.kernel)
            fan_in, fan_out = kernel.shape
            limit = np.sqrt(6.0 / (fan_in + fan_out))  # Glorot uniform, як у Keras за замовч.
            bias = np.zeros(fan_out, np.float32)
            layer.set_weights(
                [generator.uniform(-limit, limit, kernel.shape).astype(np.float32), bias]
            )
    return clone


class ModelRandomization:
    """Детектори з поступовою рандомізацією ваг.

    Args:
        config: Конфіг детектора ЛР4.
        head: Навчена головка FAE.
        weights: Ваги backbone навченої моделі (``"imagenet"``).
        seed: Зерно випадкових ваг.
    """

    #: Рівні рандомізації: (ключ, підпис).
    LEVELS = (
        ("head", "випадкова головка FAE (μ/σ з train)"),
        ("head_no_norm", "випадкова головка FAE без нормалізації"),
        ("full", "випадкова вся модель"),
    )

    def __init__(
        self, config: DetectorConfig, head: keras.Model, weights: str | None, seed: int
    ) -> None:
        self._config = config
        self._head = head
        self._weights = weights
        self._seed = seed

    def detector(self, level: str) -> keras.Model:
        """Детектор заданого рівня рандомізації (див. :attr:`LEVELS`)."""
        if level == "head":
            head = randomized_head(self._head, self._seed)
            return build_detector(self._config, head, self._weights, expose_backbone=True)
        if level == "head_no_norm":
            head = randomized_head(self._head, self._seed, reset_normalization=True)
            return build_detector(self._config, head, self._weights, expose_backbone=True)
        if level == "full":
            head = randomized_head(self._head, self._seed, reset_normalization=True)
            detector = build_detector(self._config, head, None, expose_backbone=True)
            reinitialize(detector.get_layer("backbone"), np.random.default_rng(self._seed + 1))
            return detector
        raise ValueError(f"Невідомий рівень рандомізації: {level}")


def _unit_ranks(values: np.ndarray) -> np.ndarray:
    """Ранги значень (нічиї — середній ранг), зведені до [0, 1]; форма зберігається."""
    ranks = rankdata(values.reshape(-1)).reshape(values.shape)
    return (ranks - 1) / max(ranks.size - 1, 1)


def map_similarity(first: np.ndarray, second: np.ndarray, grid: int) -> tuple[float, float]:
    """Spearman ρ і SSIM рангових карт на сітці ``grid×grid``.

    SSIM рахується на РАНГАХ, а не на сирих значеннях: розріджені карти (HiResCAM) майже
    всюди нульові, і на сирих значеннях SSIM завищувала б спільне «чорне тло».
    """
    small_first, small_second = downsample(first, grid), downsample(second, grid)
    rho = spearman(small_first, small_second)
    ssim = tf.image.ssim(
        tf.constant(_unit_ranks(small_first)[..., None], tf.float32),
        tf.constant(_unit_ranks(small_second)[..., None], tf.float32),
        max_val=1.0,
        filter_size=min(11, grid),
    )
    return float(np.nan_to_num(rho)), float(ssim)


def add_noise(images: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Гаусів шум σ на зображеннях ``uint8`` (з округленням і обрізанням до 0–255)."""
    generator = np.random.default_rng(seed)
    noisy = np.asarray(images, np.float32) + generator.normal(0, sigma, images.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
