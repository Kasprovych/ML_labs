"""Кількісна якість пояснень проти маски дефекту і проти карти моделі (D5).

Для кожного аномального зображення й кожної карти:

* **піксельний AUROC / AP** карти проти маски (у межах одного зображення);
* **pointing game** (Zhang et al., 2018): максимум карти лежить у масці, розширеній на
  ``tolerance`` пікселів;
* **частка маси в масці** ``Σ_маска A⁺ / Σ A⁺``, поділена на площу маски (> 1 — карта
  зосереджена на дефекті краще, ніж рівномірна);
* **IoU top-|M|**: найвищі ``|M|`` пікселів карти проти маски ``M``;
* **Spearman ρ** з картою моделі — чи пояснення узгоджене з тим, що модель «бачить».

Середні по зображеннях — з 95 % бутстреп-ДІ (перевибірка зображень).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from mllabs.anomaly.metrics import MetricInterval


@dataclass(frozen=True)
class LocalizationMetrics:
    """Метрики однієї карти на одному зображенні.

    Attributes:
        pixel_auroc: AUROC пікселів карти проти маски.
        pixel_ap: AP пікселів.
        pointing: 1, якщо argmax карти в розширеній масці.
        mass_ratio: (частка маси в масці) / (частка площі маски).
        iou_topk: IoU найвищих |M| пікселів з маскою.
        spearman_model: ρ з картою моделі.
    """

    pixel_auroc: float
    pixel_ap: float
    pointing: float
    mass_ratio: float
    iou_topk: float
    spearman_model: float


def downsample(image_map: np.ndarray, side: int) -> np.ndarray:
    """Середнє блоками до ``side×side`` (для швидкої рангової кореляції)."""
    block = image_map.shape[0] // side
    return image_map.reshape(side, block, side, block).mean(axis=(1, 3))


class LocalizationEvaluator:
    """Рахує :class:`LocalizationMetrics`.

    Args:
        pointing_tolerance_px: Розширення маски для pointing game.
        correlation_grid: Сторона сітки для Spearman ρ.
    """

    def __init__(self, pointing_tolerance_px: int, correlation_grid: int) -> None:
        self._tolerance = pointing_tolerance_px
        self._grid = correlation_grid

    def evaluate(
        self, explanation: np.ndarray, mask: np.ndarray, model_map: np.ndarray
    ) -> LocalizationMetrics:
        """Метрики карти ``explanation`` для зображення з маскою ``mask``."""
        truth = mask.reshape(-1).astype(bool)
        values = explanation.reshape(-1).astype(np.float64)
        dilated = binary_dilation(mask, iterations=self._tolerance) if self._tolerance else mask
        peak = np.unravel_index(int(np.argmax(explanation)), explanation.shape)
        positive = np.clip(values, 0, None)
        area = truth.mean()
        mass = positive[truth].sum() / positive.sum() if positive.sum() > 0 else 0.0
        top = np.argsort(-values, kind="stable")[: int(truth.sum())]
        predicted = np.zeros_like(truth)
        predicted[top] = True
        union = np.logical_or(predicted, truth).sum()
        rho = spearmanr(
            downsample(explanation, self._grid).reshape(-1),
            downsample(model_map, self._grid).reshape(-1),
        ).statistic
        return LocalizationMetrics(
            pixel_auroc=float(roc_auc_score(truth, values)),
            pixel_ap=float(average_precision_score(truth, values)),
            pointing=float(dilated[peak]),
            mass_ratio=float(mass / area),
            iou_topk=float(np.logical_and(predicted, truth).sum() / union),
            spearman_model=float(np.nan_to_num(rho)),
        )

    def table(
        self,
        category: str,
        images: Sequence[str],
        maps: Mapping[str, np.ndarray],
        masks: np.ndarray,
        model_maps: np.ndarray,
    ) -> pd.DataFrame:
        """Метрики всіх методів на всіх зображеннях: рядок = (зображення, метод)."""
        rows = []
        for method, method_maps in maps.items():
            for index, image in enumerate(images):
                metrics = self.evaluate(method_maps[index], masks[index], model_maps[index])
                rows.append(
                    {"category": category, "image": image, "method": method} | asdict(metrics)
                )
        return pd.DataFrame(rows)


def bootstrap_mean(
    values: np.ndarray, repeats: int, confidence_level: float, seed: int
) -> MetricInterval:
    """Середнє і перцентильний бутстреп-ДІ (перевибірка зображень з поверненням)."""
    values = np.asarray(values, np.float64)
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    tail = (1 - confidence_level) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return MetricInterval(float(values.mean()), float(low), float(high))


def area_terciles(areas: np.ndarray, strata: int) -> np.ndarray:
    """Номер страти площі (0 — найдрібніші) за квантилями площ цієї ж вибірки."""
    edges = np.quantile(areas, np.linspace(0, 1, strata + 1)[1:-1])
    return np.searchsorted(edges, areas, side="right")
