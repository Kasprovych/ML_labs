"""Пояснення рішень детектора (R2): Integrated Gradients, Grad-CAM, HiResCAM.

Ціль усіх методів — ``image_score`` (максимум карти аномалій): саме ця величина вирішує
«аномалія / норма». Тому пояснення відповідають на питання «які пікселі зробили найаномаль-
нішу точку аномальною», а не «що на всій карті».

* **Integrated Gradients** (Sundararajan et al., 2017): ``IG_i = (x_i − b_i)·∫₀¹ ∂F/∂x_i
  (b + α(x − b)) dα``. Аксіома повноти: ``Σ IG = F(x) − F(b)`` — перевіряється для кожного
  зображення; якщо похибка інтегрування велика, кількість кроків подвоюється.
* **Grad-CAM** (Selvaraju et al., 2017): ваги каналу — середній градієнт по площині,
  карта = ReLU(Σ_c α_c·A_c). **HiResCAM** (Draelos & Carin, 2020): ReLU(Σ_c ∂F/∂A_c ⊙ A_c) —
  без усереднення градієнта, тому вірніше локалізує, коли ціль — локальний максимум.

Градієнти рахує ``tf.GradientTape`` — «магнітофон», що записує операції прямого проходу,
щоб потім пройти їх у зворотному порядку (автоматичне диференціювання).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import keras
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter, zoom

from mllabs.labs.lab05.config import BaselineKind
from mllabs.labs.lab05.references import NormalReferenceFinder

#: Функція «пакет зображень float32 → оцінки (B,)».
ScoreFunction = Callable[[tf.Tensor], tf.Tensor]


class Explainer(Protocol):
    """Будь-який метод пояснення: зображення ``(N, S, S, 3)`` → карти ``(N, E, E)``."""

    def explain(self, images: np.ndarray) -> np.ndarray:
        """Карти важливості (більше — важливіше для рішення «аномалія»)."""
        ...


@dataclass(frozen=True)
class IgSettings:
    """Параметри Integrated Gradients.

    Attributes:
        steps: Початкова кількість кроків m (правило трапецій, m + 1 точка).
        max_steps: Межа подвоєння m.
        alpha_batch: Точок шляху в одному пакеті градієнта.
        tolerance: Допустима похибка повноти ε (у одиницях ``scale``).
        scale: Шкала похибки — поріг зображення ЛР4 категорії. Однакова для всіх баз, тож
            критерій не залежить від того, наскільки база «аномальна» (|F(x) − F(b)|).
        blur_sigma: σ згладжування карти |IG|.
        output_size: Сторона карти (як у карти аномалій).
    """

    steps: int
    max_steps: int
    alpha_batch: int
    tolerance: float
    scale: float
    blur_sigma: float
    output_size: int


@dataclass(frozen=True)
class IgResult:
    """Integrated Gradients одного зображення.

    Attributes:
        map: ``(E, E)`` згладжена сума |IG| по каналах (для метрик і perturbation).
        signed: ``(S, S)`` знакова сума IG по каналах (для панелей).
        steps: Кількість кроків, на якій зупинилася ескалація.
        completeness_error: ``|ΣIG − (F(x) − F(b))| / threshold_image`` (похибка повноти
            в частках порога рішення ЛР4).
        score: F(x).
        baseline_score: F(b) (для Expected Gradients — середнє по базах).
    """

    map: np.ndarray
    signed: np.ndarray
    steps: int
    completeness_error: float
    score: float
    baseline_score: float

    @property
    def delta(self) -> float:
        """F(x) − F(b): скільки «аномальності» розподіляє IG."""
        return self.score - self.baseline_score


def trapezoid_weights(steps: int) -> np.ndarray:
    """Ваги правила трапецій на ``steps + 1`` рівномірних точках відрізка [0, 1]."""
    weights = np.full(steps + 1, 1.0 / steps)
    weights[[0, -1]] *= 0.5
    return weights


class IntegratedGradients:
    """Integrated Gradients із перевіркою аксіоми повноти та ескалацією кроків.

    Args:
        score_function: ``F``: пакет зображень → оцінки.
        settings: Параметри.
    """

    def __init__(self, score_function: ScoreFunction, settings: IgSettings) -> None:
        self._settings = settings

        @tf.function(reduce_retracing=True)
        def gradients(points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            with tf.GradientTape() as tape:
                tape.watch(points)  # вхід — не змінна, тож стрічку просимо стежити явно
                scores = score_function(points)
            return scores, tape.gradient(scores, points)

        self._gradients = gradients

    @property
    def settings(self) -> IgSettings:
        """Параметри."""
        return self._settings

    def explain(
        self, image: np.ndarray, baselines: np.ndarray, max_steps: int | None = None
    ) -> IgResult:
        """IG для ``image`` ``(S, S, 3)`` відносно бази(баз) ``(K, S, S, 3)``.

        При ``K > 1`` — Expected Gradients: середнє IG по базах, кроки діляться між ними.

        Args:
            image: Зображення.
            baselines: Бази.
            max_steps: Межа ескалації замість ``settings.max_steps`` (``None`` — з налаштувань).
        """
        settings = self._settings
        limit = max_steps or settings.max_steps
        image = np.asarray(image, np.float32)
        baselines = np.asarray(baselines, np.float32)
        steps = settings.steps
        while True:
            attribution, score, baseline_score = self._integrate(image, baselines, steps)
            delta = score - baseline_score
            error = abs(float(attribution.sum()) - delta) / settings.scale
            if error <= settings.tolerance or steps >= limit:
                break
            steps *= 2
        magnitude = np.abs(attribution).sum(axis=-1)
        return IgResult(
            map=self._smooth(magnitude),
            signed=attribution.sum(axis=-1),
            steps=steps,
            completeness_error=error,
            score=score,
            baseline_score=baseline_score,
        )

    def _integrate(
        self, image: np.ndarray, baselines: np.ndarray, steps: int
    ) -> tuple[np.ndarray, float, float]:
        """Сума IG по базах і оцінки F(x), середня F(b) за ``steps`` кроків."""
        per_baseline = max(steps // len(baselines), 1)
        alphas = np.linspace(0.0, 1.0, per_baseline + 1, dtype=np.float32)
        weights = trapezoid_weights(per_baseline)
        attribution = np.zeros(image.shape, np.float64)
        scores_x, scores_b = [], []
        for baseline in baselines:
            difference = image - baseline
            scores, average = self._weighted_gradient(baseline, difference, alphas, weights)
            attribution += difference * average
            scores_b.append(scores[0])
            scores_x.append(scores[-1])
        attribution /= len(baselines)
        return attribution.astype(np.float32), float(np.mean(scores_x)), float(np.mean(scores_b))

    def _weighted_gradient(
        self, baseline: np.ndarray, difference: np.ndarray, alphas: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """``Σ_i w_i·∇F(b + α_i(x − b))`` пакетами сталого розміру й оцінки в усіх точках.

        Сума накопичується по пакетах, тож у пам'яті не лежать градієнти всіх m + 1 точок.
        Останній пакет доповнюється повторами — граф ``tf.function`` не перебудовується.
        """
        size = self._settings.alpha_batch
        total = np.zeros(difference.shape, np.float64)
        scores = []
        for start in range(0, len(alphas), size):
            chunk_alphas = alphas[start : start + size]
            count = len(chunk_alphas)
            padded = np.concatenate([chunk_alphas, np.repeat(chunk_alphas[-1:], size - count)])
            # Broadcasting: (n, 1, 1, 1) * (S, S, 3) → (n, S, S, 3) точок шляху.
            points = baseline + padded[:, None, None, None] * difference
            batch_scores, gradients = self._gradients(tf.constant(points))
            scores.append(batch_scores.numpy()[:count])
            # tensordot по осі пакета: Σ_i w_i·g_i без явного циклу.
            total += np.tensordot(weights[start : start + count], gradients.numpy()[:count], axes=1)
        return np.concatenate(scores), total

    def _smooth(self, magnitude: np.ndarray) -> np.ndarray:
        """Гаусове згладжування (σ як у карти моделі) і приведення до розміру карти."""
        size = self._settings.output_size
        if magnitude.shape[0] != size:
            magnitude = zoom(magnitude, size / magnitude.shape[0], order=1)
        if self._settings.blur_sigma > 0:
            magnitude = gaussian_filter(magnitude, self._settings.blur_sigma, mode="reflect")
        return magnitude.astype(np.float32)


class BaselineFactory:
    """Будує бази IG різних типів для конкретного зображення.

    Args:
        references: Пошук нормальних зображень train.
        mean_image: Середнє зображення train.
        blur_sigma: σ розмиття для бази ``BLUR``.
        expected_k: Кількість баз Expected Gradients.
    """

    def __init__(
        self,
        references: NormalReferenceFinder,
        mean_image: np.ndarray,
        blur_sigma: float,
        expected_k: int,
    ) -> None:
        self._references = references
        self._mean = mean_image
        self._blur = blur_sigma
        self._expected_k = expected_k

    def make(self, kind: BaselineKind, image: np.ndarray) -> np.ndarray:
        """Бази ``(K, S, S, 3)`` ``float32`` (``K = 1``, крім Expected Gradients)."""
        image = np.asarray(image)
        match kind:  # match — аналог switch у Java
            case BaselineKind.BLACK:
                bases = np.zeros((1, *image.shape))
            case BaselineKind.MEAN:
                bases = self._mean[None]
            case BaselineKind.BLUR:
                blurred = gaussian_filter(image.astype(np.float32), (self._blur, self._blur, 0))
                bases = blurred[None]
            case BaselineKind.NEAREST_NORMAL:
                bases = self._references.reference(image)[None]
            case BaselineKind.EXPECTED:
                bases = self._references.references(image, self._expected_k)
        return np.asarray(bases, np.float32)


class IgExplainer:
    """IG як :class:`Explainer` з фіксованим типом бази.

    Args:
        ig: Обчислювач IG.
        baselines: Фабрика баз.
        kind: Тип бази.
    """

    def __init__(
        self, ig: IntegratedGradients, baselines: BaselineFactory, kind: BaselineKind
    ) -> None:
        self._ig = ig
        self._baselines = baselines
        self._kind = kind

    def explain_one(self, image: np.ndarray) -> IgResult:
        """Повний результат IG для одного зображення."""
        return self._ig.explain(image, self._baselines.make(self._kind, image))

    def explain(self, images: np.ndarray) -> np.ndarray:
        """Карти |IG| для пакета."""
        return np.stack([self.explain_one(image).map for image in images])


class CamGraphError(RuntimeError):
    """Градієнт цілі за активаціями backbone не існує (граф розірвано)."""


#: Варіанти CAM і шари backbone.
CAM_VARIANTS = ("gradcam", "hirescam")
CAM_LAYERS = ("shallow", "deep", "fused")


def cam_key(variant: str, layer: str) -> str:
    """Ключ карти, напр. ``"hirescam_fused"``."""
    return f"{variant}_{layer}"


class GradCam:
    """Grad-CAM і HiResCAM за двома виходами backbone (крок 8 і крок 16).

    Args:
        detector: Модель із виходами ``image_score``, ``backbone_shallow``, ``backbone_deep``.
        output_size: Сторона карти.
        batch_size: Зображень у пакеті.
    """

    def __init__(self, detector: keras.Model, output_size: int, batch_size: int) -> None:
        self._size = output_size
        self._batch = batch_size

        # autograph=False: цикли тут — звичайний Python під час трасування, а виняток
        # CamGraphError має дійти до викликача як є (autograph загорнув би його).
        @tf.function(reduce_retracing=True, autograph=False)
        def run(images: tf.Tensor) -> dict[str, tf.Tensor]:
            with tf.GradientTape() as tape:
                outputs = detector(images, training=False)
            activations = [outputs["backbone_shallow"], outputs["backbone_deep"]]
            gradients = tape.gradient(outputs["image_score"], activations)
            # None (а не нулі) означає розрив графа: ціль не залежить від активацій шару —
            # тоді CAM беззмістовний, і краще впасти з поясненням, ніж малювати нулі.
            if any(gradient is None for gradient in gradients):
                raise CamGraphError(
                    "image_score не залежить від backbone_shallow/backbone_deep — "
                    "перевірте, що детектор зібрано з expose_backbone=True"
                )
            maps = {}
            for layer, values, grads in zip(
                ("shallow", "deep"), activations, gradients, strict=True
            ):
                weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # α_c Grad-CAM
                maps[cam_key("gradcam", layer)] = tf.nn.relu(tf.reduce_sum(weights * values, -1))
                maps[cam_key("hirescam", layer)] = tf.nn.relu(tf.reduce_sum(grads * values, -1))
            return {
                key: tf.image.resize(value[..., None], (output_size, output_size))[..., 0]
                for key, value in maps.items()
            }

        self._run = run

    def maps(self, images: np.ndarray) -> dict[str, np.ndarray]:
        """Усі карти: ключі ``<варіант>_<шар>``, шари ``shallow``/``deep``/``fused``."""
        parts: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(images), self._batch):
            chunk = np.asarray(images[start : start + self._batch], np.float32)
            for key, value in self._run(tf.constant(chunk)).items():
                parts.setdefault(key, []).append(value.numpy())
        result = {key: np.concatenate(values) for key, values in parts.items()}
        for variant in CAM_VARIANTS:
            shallow = _normalize_max(result[cam_key(variant, "shallow")])
            deep = _normalize_max(result[cam_key(variant, "deep")])
            result[cam_key(variant, "fused")] = (shallow + deep) / 2
        return result


def _normalize_max(maps: np.ndarray) -> np.ndarray:
    """Ділить кожну карту на її максимум (нульова карта лишається нульовою)."""
    peaks = maps.reshape(len(maps), -1).max(axis=1)
    return maps / np.where(peaks > 0, peaks, 1.0)[:, None, None]


class CamExplainer:
    """Один варіант CAM як :class:`Explainer`.

    Args:
        cam: Обчислювач CAM.
        key: Ключ карти, напр. ``"hirescam_fused"``.
    """

    def __init__(self, cam: GradCam, key: str) -> None:
        self._cam = cam
        self._key = key

    def explain(self, images: np.ndarray) -> np.ndarray:
        """Карти обраного варіанта."""
        return self._cam.maps(images)[self._key]


class RandomExplainer:
    """Нижня межа: рівномірний шум замість пояснення.

    Args:
        output_size: Сторона карти.
        seed: Зерно.
    """

    def __init__(self, output_size: int, seed: int) -> None:
        self._size = output_size
        self._generator = np.random.default_rng(seed)

    def explain(self, images: np.ndarray) -> np.ndarray:
        """Випадкові карти."""
        return self._generator.random((len(images), self._size, self._size)).astype(np.float32)
