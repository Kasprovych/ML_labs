"""Метрики детекції та сегментації аномалій і бутстреп по зображеннях.

* **Pixel AUROC** — площа під ROC-кривою по ВСІХ пікселях усіх зображень (головна метрика
  завдання). Аномальних пікселів < 1 %, тому AUROC оптимістичний: модель, що «підсвічує»
  цілий об'єкт навколо дрібного дефекту, усе одно отримує високий AUROC.
* **Pixel AP** — площа під кривою точність–повнота по пікселях; чутлива до хибних
  спрацьовувань на величезній кількості нормальних пікселів (аналог PR-AUC з ЛР1–3).
* **F1** — за фіксованим порогом, обраним на розміченому val: на рівні зображення
  (аномальне, якщо максимум карти ≥ порога) і на рівні пікселів (сегментація).
* **Image AUROC** — якість ранжування зображень за оцінкою (максимум карти).

Бутстреп ДІ рахується по ЗОБРАЖЕННЯХ (пікселі одного зображення сильно залежні). Щоб не
сортувати мільйони пікселів 1000 разів, кожне зображення стискається до гістограми оцінок
на спільних бінах; бутстреп-повтор — це зважена сума гістограм (одне множення матриць).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

from mllabs.modeling import BootstrapInterval


@dataclass(frozen=True)
class SegmentationMetrics:
    """Метрики одного детектора на одній вибірці.

    Attributes:
        pixel_auroc: Pixel AUROC (точно, ``sklearn``).
        pixel_ap: Pixel AP (точно).
        pixel_f1: Pixel F1 за порогом ``threshold_pixel``.
        image_auroc: Image AUROC.
        image_f1: Image F1 за порогом ``threshold_image``.
        oracle_pixel_f1: Максимально можливий Pixel F1 на ЦІЙ вибірці (верхня межа для
            зіставлення з літературою; для рішень не використовується).
        threshold_pixel: Поріг пікселя.
        threshold_image: Поріг зображення.
    """

    pixel_auroc: float
    pixel_ap: float
    pixel_f1: float
    image_auroc: float
    image_f1: float
    oracle_pixel_f1: float
    threshold_pixel: float
    threshold_image: float


@dataclass(frozen=True)
class MetricInterval:
    """Значення метрики та його бутстреп-ДІ.

    Attributes:
        value: Значення на повній вибірці.
        low: Нижня межа ДІ.
        high: Верхня межа ДІ.
    """

    value: float
    low: float
    high: float


class AnomalyMetricsCalculator:
    """Точні метрики (``sklearn``) для карт аномалій і оцінок зображень."""

    def evaluate(
        self,
        anomaly_maps: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        threshold_image: float,
        threshold_pixel: float,
    ) -> SegmentationMetrics:
        """Усі метрики однієї вибірки.

        Args:
            anomaly_maps: ``(N, S, S)`` карти аномалій.
            masks: ``(N, S, S)`` булеві маски дефектів.
            labels: ``(N,)`` мітки зображень 0/1.
            scores: ``(N,)`` оцінки зображень.
            threshold_image: Поріг зображення (з val).
            threshold_pixel: Поріг пікселя (з val).
        """
        pixel_truth = masks.reshape(-1)
        pixel_scores = anomaly_maps.reshape(-1)
        precision, recall, _ = precision_recall_curve(pixel_truth, pixel_scores)
        return SegmentationMetrics(
            pixel_auroc=float(roc_auc_score(pixel_truth, pixel_scores)),
            # Та сама формула, що в sklearn.average_precision_score: Σ ΔR · P.
            pixel_ap=float(-np.sum(np.diff(recall) * precision[:-1])),
            pixel_f1=float(f1_score(pixel_truth, pixel_scores >= threshold_pixel)),
            image_auroc=float(roc_auc_score(labels, scores)),
            image_f1=float(f1_score(labels, scores >= threshold_image)),
            oracle_pixel_f1=float(_max_f1(precision, recall)),
            threshold_pixel=float(threshold_pixel),
            threshold_image=float(threshold_image),
        )


def _max_f1(precision: np.ndarray, recall: np.ndarray) -> float:
    """Максимум F1 уздовж кривої точність–повнота."""
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    return float(f1.max())


class HistogramAuc:
    """Наближені Pixel AUROC / AP через гістограми оцінок на спільних бінах.

    Межі бінів — квантилі всіх пікселів (половина бінів) і аномальних пікселів (друга
    половина): так роздільність висока і в «хвості», де лежать дефекти. Пікселі в одному
    біні вважаються рівними (як «нічия» у точних метриках), тож похибка — порядку 1e-4.

    Args:
        n_bins: Бажана кількість бінів.
    """

    def __init__(self, n_bins: int) -> None:
        self._n_bins = n_bins

    def edges(self, anomaly_maps: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Внутрішні межі бінів для цієї вибірки."""
        scores = anomaly_maps.reshape(-1)
        positives = scores[masks.reshape(-1)]
        levels = np.linspace(0, 1, self._n_bins // 2 + 1)[1:-1]
        parts = [np.quantile(scores, levels)]
        if positives.size:
            parts.append(np.quantile(positives, levels))
        return np.unique(np.concatenate(parts))

    @staticmethod
    def per_image_counts(
        anomaly_maps: np.ndarray, masks: np.ndarray, edges: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Гістограми аномальних і нормальних пікселів кожного зображення: ``(N, B)`` × 2."""
        n_images, n_bins = len(anomaly_maps), len(edges) + 1
        bins = np.searchsorted(edges, anomaly_maps.reshape(n_images, -1), side="right")
        # Зсув на n_bins·i робить номери бінів різних зображень різними — один bincount на все.
        flat = (bins + np.arange(n_images)[:, None] * n_bins).reshape(-1)
        truth = masks.reshape(-1)
        size = n_images * n_bins
        positive = np.bincount(flat[truth], minlength=size).reshape(n_images, n_bins)
        negative = np.bincount(flat[~truth], minlength=size).reshape(n_images, n_bins)
        return positive.astype(np.float64), negative.astype(np.float64)

    @staticmethod
    def auroc(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
        """AUROC з гістограм (остання вісь — біни за зростанням оцінки)."""
        # Кількість аномальних пікселів у бінах СТРОГО вище поточного.
        above = np.cumsum(positive[..., ::-1], axis=-1)[..., ::-1] - positive
        pairs = negative * (above + 0.5 * positive)  # нічия в біні рахується як ½
        total = positive.sum(axis=-1) * negative.sum(axis=-1)
        # Бутстреп-повтор без аномальних пікселів дає 0/0 → NaN (його пропускає nanpercentile).
        with np.errstate(invalid="ignore", divide="ignore"):
            return pairs.sum(axis=-1) / total

    @staticmethod
    def average_precision(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
        """AP з гістограм: поріг на межі кожного біна, від найвищих оцінок до найнижчих."""
        true_pos = np.cumsum(positive[..., ::-1], axis=-1)
        false_pos = np.cumsum(negative[..., ::-1], axis=-1)
        predicted = true_pos + false_pos
        precision = np.divide(true_pos, predicted, out=np.zeros_like(true_pos), where=predicted > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            recall_step = positive[..., ::-1] / positive.sum(axis=-1, keepdims=True)
        return (recall_step * precision).sum(axis=-1)

    @staticmethod
    def curves(positive: np.ndarray, negative: np.ndarray) -> dict[str, np.ndarray]:
        """Точки ROC- і PR-кривих із сумарних гістограм (для графіків).

        Returns:
            Словник ``fpr``, ``tpr`` (= повнота), ``precision`` — по точці на межу біна,
            від найвищого порога до найнижчого.
        """
        true_pos = np.concatenate([[0.0], np.cumsum(positive[::-1])])
        false_pos = np.concatenate([[0.0], np.cumsum(negative[::-1])])
        predicted = true_pos + false_pos
        precision = np.divide(true_pos, predicted, out=np.ones_like(true_pos), where=predicted > 0)
        return {
            "fpr": false_pos / false_pos[-1],
            "tpr": true_pos / true_pos[-1],
            "precision": precision,
        }

    def metrics(self, anomaly_maps: np.ndarray, masks: np.ndarray) -> tuple[float, float]:
        """Швидкі (наближені) Pixel AUROC і AP — для метрик по епохах."""
        positive, negative = self.per_image_counts(
            anomaly_maps, masks, self.edges(anomaly_maps, masks)
        )
        total_pos, total_neg = positive.sum(axis=0), negative.sum(axis=0)
        return float(self.auroc(total_pos, total_neg)), float(
            self.average_precision(total_pos, total_neg)
        )


@dataclass(frozen=True)
class PixelSummary:
    """Стиснуте представлення карт однієї моделі для бутстрепу.

    Attributes:
        positive: ``(N, B)`` гістограми аномальних пікселів.
        negative: ``(N, B)`` гістограми нормальних пікселів.
        labels: ``(N,)`` мітки зображень.
        scores: ``(N,)`` оцінки зображень.
    """

    positive: np.ndarray
    negative: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class ImageBootstrap:
    """Бутстреп по зображеннях: ДІ метрик однієї моделі та парні ДІ різниць.

    Усі моделі оцінюються на ОДНАКОВИХ перевибірках зображень (генератор із тим самим
    зерном), тому різниці «парні».

    Args:
        n_repeats: Кількість повторів.
        confidence_level: Рівень довіри (0.95).
        random_state: Зерно.
    """

    def __init__(self, n_repeats: int, confidence_level: float, random_state: int) -> None:
        self._n_repeats = n_repeats
        self._confidence = confidence_level
        self._random_state = random_state

    def _weights(self, n_images: int) -> np.ndarray:
        """Матриця ``(R, N)``: скільки разів кожне зображення потрапило в кожен повтор."""
        generator = np.random.default_rng(self._random_state)
        weights = np.zeros((self._n_repeats, n_images))
        for repeat in range(self._n_repeats):
            np.add.at(weights[repeat], generator.integers(0, n_images, n_images), 1)
        return weights

    def samples(self, summary: PixelSummary) -> dict[str, np.ndarray]:
        """Бутстреп-розподіли ``pixel_auroc``, ``pixel_ap``, ``image_auroc`` (по ``R`` значень)."""
        weights = self._weights(len(summary.labels))
        positive, negative = weights @ summary.positive, weights @ summary.negative  # (R, B)
        image_auroc = np.full(self._n_repeats, np.nan)
        for repeat, row in enumerate(weights):
            labels = np.repeat(summary.labels, row.astype(int))
            if labels.min() != labels.max():  # повтор з одним класом не дає AUROC
                image_auroc[repeat] = roc_auc_score(
                    labels, np.repeat(summary.scores, row.astype(int))
                )
        return {
            "pixel_auroc": HistogramAuc.auroc(positive, negative),
            "pixel_ap": HistogramAuc.average_precision(positive, negative),
            "image_auroc": image_auroc,
        }

    def interval(self, values: np.ndarray, point: float) -> MetricInterval:
        """ДІ за перцентилями бутстреп-розподілу."""
        tail = (1 - self._confidence) / 2 * 100
        low, high = np.nanpercentile(values, [tail, 100 - tail])
        return MetricInterval(float(point), float(low), float(high))

    def paired(self, values_a: np.ndarray, values_b: np.ndarray, delta: float) -> BootstrapInterval:
        """ДІ різниці «B − A» за парними бутстреп-розподілами."""
        tail = (1 - self._confidence) / 2 * 100
        low, high = np.nanpercentile(values_b - values_a, [tail, 100 - tail])
        return BootstrapInterval(
            float(delta), float(low), float(high), self._confidence, self._n_repeats
        )
