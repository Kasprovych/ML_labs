"""Контрольований дрейф розподілу (D8): як детектор поводиться, коли «світ» змінився.

Детектор навчено на зображеннях однієї камери при одному освітленні. На виробництві лампа
старіє, камера зсувається, з'являється шум сенсора чи сильніше стиснення JPEG. Тут ці зміни
моделюються детермінованими трансформаціями test-зображень і для кожного рівня міряється:

* якість (Image/Pixel AUROC, Pixel AP), FPR і TPR при порозі ЛР4;
* зсув оцінок НОРМАЛЬНИХ кадрів і KS-відстань їх розподілу від чистого;
* **монітор дрейфу без міток**: чи відрізняє медіана помилки патчів кадру «дрейф-норми» від
  чистих норм (AUROC) — тобто чи може edge-пристрій сам помітити, що умови змінилися.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Protocol, override

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from mllabs.anomaly.metrics import HistogramAuc
from mllabs.labs.lab05.config import DriftLevel
from mllabs.labs.lab05.inference import PartOutputs


class ImageTransform(Protocol):
    """Трансформація пакета: зображення ``uint8`` і маски → нові зображення й маски."""

    def apply(
        self, images: np.ndarray, masks: np.ndarray, generator: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Нові ``(images, masks)``; маски змінюються лише геометричними трансформаціями."""
        ...


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


class _Photometric:
    """База фотометричних трансформацій: маска не змінюється (шаблонний метод)."""

    def apply(
        self, images: np.ndarray, masks: np.ndarray, generator: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Застосовує :meth:`_pixels` до зображень, маски повертає як є."""
        return self._pixels(np.asarray(images), generator), masks

    def _pixels(self, images: np.ndarray, generator: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class BrightnessGain(_Photometric):
    """Множення яскравості на ``gain`` (інша експозиція / лампа)."""

    def __init__(self, gain: float) -> None:
        self._gain = gain

    @override
    def _pixels(self, images: np.ndarray, generator: np.random.Generator) -> np.ndarray:
        return _to_uint8(images.astype(np.float32) * self._gain)


class IlluminationGradient(_Photometric):
    """Нерівномірне освітлення: множник лінійно від ``1 − a/2`` (ліворуч) до ``1 + a/2``."""

    def __init__(self, amplitude: float) -> None:
        self._amplitude = amplitude

    @override
    def _pixels(self, images: np.ndarray, generator: np.random.Generator) -> np.ndarray:
        width = images.shape[2]
        ramp = 1 + self._amplitude * (np.linspace(0, 1, width, dtype=np.float32) - 0.5)
        return _to_uint8(images.astype(np.float32) * ramp[None, None, :, None])


class GaussianNoise(_Photometric):
    """Адитивний гаусів шум σ (шум сенсора при слабкому світлі)."""

    def __init__(self, sigma: float) -> None:
        self._sigma = sigma

    @override
    def _pixels(self, images: np.ndarray, generator: np.random.Generator) -> np.ndarray:
        if self._sigma == 0:
            return images.copy()
        noise = generator.normal(0, self._sigma, images.shape)
        return _to_uint8(images.astype(np.float32) + noise)


class GaussianBlur(_Photometric):
    """Розмиття гаусовим ядром σ пікс. (розфокусування, рух)."""

    def __init__(self, sigma: float) -> None:
        self._sigma = sigma

    @override
    def _pixels(self, images: np.ndarray, generator: np.random.Generator) -> np.ndarray:
        if self._sigma == 0:
            return images.copy()
        blurred = gaussian_filter(images.astype(np.float32), (0, self._sigma, self._sigma, 0))
        return _to_uint8(blurred)


class JpegCompression(_Photometric):
    """Повторне стиснення JPEG з якістю ``quality`` (Pillow)."""

    def __init__(self, quality: int) -> None:
        self._quality = quality

    @override
    def _pixels(self, images: np.ndarray, generator: np.random.Generator) -> np.ndarray:
        result = np.empty_like(images)
        for index, image in enumerate(images):
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="JPEG", quality=self._quality)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                result[index] = np.asarray(decoded.convert("RGB"))
        return result


class Translation:
    """Зсув кадру на ``shift`` пікселів праворуч і вниз з дзеркальним доповненням.

    Геометрична трансформація: маска зсувається так само, як зображення.
    """

    def __init__(self, shift: int) -> None:
        self._shift = shift

    def apply(
        self, images: np.ndarray, masks: np.ndarray, generator: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Зсунуті зображення й маски."""
        return self._shifted(np.asarray(images)), self._shifted(np.asarray(masks))

    def _shifted(self, values: np.ndarray) -> np.ndarray:
        shift = self._shift
        if shift == 0:
            return values.copy()
        pad = [(0, 0), (shift, 0), (shift, 0)] + [(0, 0)] * (values.ndim - 3)
        padded = np.pad(values, pad, mode="reflect")
        return padded[:, : values.shape[1], : values.shape[2]]


#: Фабрики трансформацій за ключем родини.
TRANSFORMS: dict[str, Callable[[float], ImageTransform]] = {
    "brightness_down": BrightnessGain,
    "brightness_up": BrightnessGain,
    "illumination": IlluminationGradient,
    "noise": GaussianNoise,
    "blur": GaussianBlur,
    "jpeg": lambda value: JpegCompression(int(value)),
    "shift": lambda value: Translation(int(value)),
}

#: Підписи родин для таблиць і графіків.
FAMILY_LABELS = {
    "clean": "без змін",
    "brightness_down": "затемнення (×)",
    "brightness_up": "підсилення яскравості (×)",
    "illumination": "градієнт освітлення (амплітуда)",
    "noise": "гаусів шум (σ)",
    "blur": "розмиття (σ, пікс.)",
    "jpeg": "JPEG (якість)",
    "shift": "зсув кадру (пікс.)",
}


def make_transform(level: DriftLevel) -> ImageTransform:
    """Трансформація для рівня дрейфу."""
    return TRANSFORMS[level.family](level.value)


@dataclass(frozen=True)
class DriftMetrics:
    """Якість детектора на одному рівні дрейфу однієї категорії.

    Attributes:
        pixel_auroc: Pixel AUROC.
        pixel_ap: Pixel AP.
        image_auroc: Image AUROC.
        fpr: Частка норм з оцінкою ≥ порога ЛР4.
        tpr: Частка аномалій з оцінкою ≥ порога ЛР4.
        score_shift: Зсув медіани оцінок норм відносно чистих.
        ks: KS-статистика оцінок норм проти чистих.
        monitor_auroc: AUROC медіани помилки патчів «дрейф-норми vs чисті норми».
        pointing: Частка аномалій, у яких argmax карти в (розширеній) масці.
    """

    pixel_auroc: float
    pixel_ap: float
    image_auroc: float
    fpr: float
    tpr: float
    score_shift: float
    ks: float
    monitor_auroc: float
    pointing: float


class DriftEvaluator:
    """Метрики :class:`DriftMetrics` відносно чистого прогону.

    Args:
        histogram: Швидкі піксельні метрики.
        threshold: Поріг зображення ЛР4.
        pointing_px: Розширення маски для pointing game, пікс.
    """

    def __init__(self, histogram: HistogramAuc, threshold: float, pointing_px: int) -> None:
        self._histogram = histogram
        self._threshold = threshold
        self._pointing_px = pointing_px

    def evaluate(
        self,
        outputs: PartOutputs,
        masks: np.ndarray,
        labels: np.ndarray,
        clean: PartOutputs,
    ) -> DriftMetrics:
        """Метрики прогону ``outputs`` (маски — уже трансформовані)."""
        pixel_auroc, pixel_ap = self._histogram.metrics(outputs.maps, masks)
        decisions = outputs.scores >= self._threshold
        normal = labels == 0
        drift_median = np.median(outputs.patch_scores.reshape(len(labels), -1), axis=1)
        clean_median = np.median(clean.patch_scores.reshape(len(labels), -1), axis=1)
        monitor_truth = np.r_[np.zeros(normal.sum()), np.ones(normal.sum())]
        monitor_scores = np.r_[clean_median[normal], drift_median[normal]]
        anomalies = np.flatnonzero(labels == 1)
        dilated = binary_dilation(masks[anomalies], iterations=self._pointing_px, axes=(1, 2))
        peaks = outputs.maps[anomalies].reshape(len(anomalies), -1).argmax(axis=1)
        pointing = dilated.reshape(len(anomalies), -1)[np.arange(len(anomalies)), peaks]
        return DriftMetrics(
            pixel_auroc=pixel_auroc,
            pixel_ap=pixel_ap,
            image_auroc=float(roc_auc_score(labels, outputs.scores)),
            fpr=float(decisions[normal].mean()),
            tpr=float(decisions[~normal].mean()),
            score_shift=float(np.median(outputs.scores[normal]) - np.median(clean.scores[normal])),
            ks=float(ks_2samp(outputs.scores[normal], clean.scores[normal]).statistic),
            monitor_auroc=float(roc_auc_score(monitor_truth, monitor_scores)),
            pointing=float(pointing.mean()),
        )


def first_failing_level(
    levels: list[tuple[float, float, float]],
    clean: tuple[float, float],
    auroc_tolerance: float,
    fpr_tolerance: float,
) -> float | None:
    """Перший рівень (за зростанням сили), де якість виходить за апріорні допуски.

    Args:
        levels: ``(значення рівня, Pixel AUROC, FPR)`` у порядку зростання сили.
        clean: ``(Pixel AUROC, FPR)`` на чистих даних.
        auroc_tolerance: Допустиме падіння Pixel AUROC.
        fpr_tolerance: Допустиме зростання FPR.

    Returns:
        Значення параметра першого «поганого» рівня або ``None`` (межу не досягнуто).
    """
    for value, auroc, fpr in levels:
        if clean[0] - auroc > auroc_tolerance or fpr - clean[1] > fpr_tolerance:
            return value
    return None


def pixel_auroc_interval(
    histogram: HistogramAuc,
    maps: np.ndarray,
    masks: np.ndarray,
    repeats: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Бутстреп-ДІ Pixel AUROC по зображеннях (гістограми зображень + зважені суми)."""
    positive, negative = histogram.per_image_counts(maps, masks, histogram.edges(maps, masks))
    generator = np.random.default_rng(seed)
    count = len(maps)
    # Кожен повтор — скільки разів кожне зображення потрапило у вибірку з поверненням.
    weights = np.stack(
        [np.bincount(generator.integers(0, count, count), minlength=count) for _ in range(repeats)]
    ).astype(np.float64)
    samples = HistogramAuc.auroc(weights @ positive, weights @ negative)
    tail = (1 - confidence_level) / 2 * 100
    low, high = np.nanpercentile(samples, [tail, 100 - tail])
    return float(low), float(high)


@dataclass(frozen=True)
class DriftExample:
    """Одне зображення під усіма родинами (найсильніший рівень) і його карти.

    Attributes:
        category: Категорія.
        images: Родина → зображення (``clean`` — без змін).
        maps: Родина → карта аномалій.
    """

    category: str
    images: dict[str, np.ndarray]
    maps: dict[str, np.ndarray]


class DriftStudy:
    """Прогін test під усіма рівнями дрейфу.

    Args:
        levels: Рівні дрейфу.
        evaluator_factory: Функція «поріг → :class:`DriftEvaluator`».
        repeats: Повторів бутстрепу для ДІ.
        confidence_level: Рівень довіри.
        seed: Зерно (шум і бутстреп).
    """

    def __init__(
        self,
        levels: tuple[DriftLevel, ...],
        evaluator_factory: Callable[[float], DriftEvaluator],
        repeats: int,
        confidence_level: float,
        seed: int,
    ) -> None:
        self._levels = levels
        self._evaluator_factory = evaluator_factory
        self._repeats = repeats
        self._confidence = confidence_level
        self._seed = seed

    def run(
        self,
        category: str,
        run: Callable[[np.ndarray], PartOutputs],
        images: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        histogram: HistogramAuc,
        example_position: int,
    ) -> tuple[list[dict], DriftExample]:
        """Рядки метрик (чистий прогін + усі рівні) і приклад для рисунка."""
        evaluator = self._evaluator_factory(threshold)
        clean = run(images)
        strongest = {level.family: level for level in self._levels}  # останній = найсильніший
        rows = [
            self._row(category, "clean", 0.0, 0, evaluator.evaluate(clean, masks, labels, clean))
        ]
        rows[0] |= self._interval(histogram, clean.maps, masks)
        example_images = {"clean": images[example_position]}
        example_maps = {"clean": clean.maps[example_position]}
        for number, level in enumerate(self._levels):
            generator = np.random.default_rng((self._seed, number))
            changed, changed_masks = make_transform(level).apply(images, masks, generator)
            outputs = run(changed)
            metrics = evaluator.evaluate(outputs, changed_masks, labels, clean)
            row = self._row(category, level.family, level.value, level.severity, metrics)
            if strongest[level.family] is level:
                row |= self._interval(histogram, outputs.maps, changed_masks)
                example_images[level.family] = changed[example_position]
                example_maps[level.family] = outputs.maps[example_position]
            rows.append(row)
        return rows, DriftExample(category, example_images, example_maps)

    def _interval(self, histogram: HistogramAuc, maps: np.ndarray, masks: np.ndarray) -> dict:
        low, high = pixel_auroc_interval(
            histogram, maps, masks, self._repeats, self._confidence, self._seed
        )
        return {"pixel_auroc_low": low, "pixel_auroc_high": high}

    @staticmethod
    def _row(
        category: str, family: str, value: float, severity: int, metrics: DriftMetrics
    ) -> dict:
        return {
            "category": category,
            "family": family,
            "value": value,
            "severity": severity,
            **asdict(metrics),
        }
