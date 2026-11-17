"""Граничні випадки (R4, D7): дескриптори зображень, причини FP/FN, крайові артефакти.

Хибні спрацьовування (FP) і пропуски (FN) беруться з ``test_predictions.csv`` ЛР4 (пороги
обрано на val). Для КОЖНОГО зображення test рахуються прості «фізичні» дескриптори —
яскравість, текстура, пересвіт, сила країв і положення максимуму карти, контраст дефекту —
і групи FP↔TN, FN↔TP порівнюються непараметричним тестом Манна–Вітні з поправкою Голма.
Причини помилок — прозорі правила з порогами з ``config`` (гіпотези, а не доведення).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, laplace, sobel
from scipy.stats import mannwhitneyu
from sklearn.metrics import f1_score, roc_auc_score

from mllabs.anomaly.images import coarse_brightness_std
from mllabs.modeling import F1ThresholdSelector

_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], np.float32)

#: Дескриптори, що порівнюються між групами, і їхні підписи для таблиць.
DESCRIPTOR_LABELS = {
    "brightness_std": "σ яскравості (сітка 16×16)",
    "texture_energy": "текстурна енергія (дисперсія лапласіана)",
    "brightness_mean": "середня яскравість",
    "saturated_share": "частка пересвіту",
    "global_error": "медіанна помилка патчів",
    "locality": "локальність max/медіана",
    "edge_strength": "сила країв біля argmax",
    "window_brightness": "яскравість біля argmax − медіана",
    "window_saturated": "пересвіт біля argmax",
    "border_distance": "відстань argmax до краю, пікс.",
}
#: Додаткові дескриптори аномалій (для FN ↔ TP).
DEFECT_LABELS = {
    "defect_area": "площа дефекту (частка)",
    "defect_contrast": "контраст дефекту",
}

#: Підписи причин.
CAUSE_LABELS = {
    "global": "глобальна текстура / освітлення",
    "glare": "локальний відблиск",
    "object_edge": "край об'єкта / структура",
    "frame_border": "край кадру",
    "frame_corner": "кут кадру",
    "small": "дрібний дефект",
    "low_contrast": "низький контраст",
    "off_map": "дефект не дав максимуму карти",
    "undetermined": "не визначено",
}


def gray(images: np.ndarray) -> np.ndarray:
    """``(N, S, S, 3)`` → ``(N, S, S)`` яскравість (стандарт ITU-R BT.601)."""
    return np.asarray(images, np.float32) @ _GRAY_WEIGHTS


@dataclass(frozen=True)
class DescriptorSettings:
    """Параметри дескрипторів.

    Attributes:
        window: Сторона вікна навколо argmax, пікс.
        saturation_level: Поріг пересвіту.
        shadow_level: Поріг тіні.
        brightness_cells: Сітка σ яскравості.
        ring_px: Ширина кільця навколо маски.
        pointing_px: Розширення маски для «argmax у масці».
        border_px: Ширина смуги краю кадру.
    """

    window: int
    saturation_level: int
    shadow_level: int
    brightness_cells: int
    ring_px: int
    pointing_px: int
    border_px: int


class DescriptorCalculator:
    """Дескриптори кожного зображення.

    Args:
        settings: Параметри.
    """

    def __init__(self, settings: DescriptorSettings) -> None:
        self._settings = settings

    def compute(
        self,
        images: np.ndarray,
        maps: np.ndarray,
        patch_scores: np.ndarray,
        masks: np.ndarray,
    ) -> pd.DataFrame:
        """Таблиця дескрипторів: рядок — зображення.

        Args:
            images: ``(N, S, S, 3)`` ``uint8``.
            maps: ``(N, S, S)`` карти аномалій (сторона = стороні зображення).
            patch_scores: ``(N, h, w)`` оцінки патчів.
            masks: ``(N, S, S)`` маски (порожні для норм).
        """
        settings = self._settings
        rows = []
        stds = coarse_brightness_std(np.asarray(images), settings.brightness_cells)
        for index in range(len(images)):
            luminance = gray(images[index])
            image_map, mask = maps[index], masks[index]
            side = image_map.shape[0]
            peak_row, peak_col = np.unravel_index(int(np.argmax(image_map)), image_map.shape)
            half = settings.window // 2
            window = (
                slice(max(peak_row - half, 0), min(peak_row + half, side)),
                slice(max(peak_col - half, 0), min(peak_col + half, side)),
            )
            edges = np.hypot(sobel(luminance, axis=0), sobel(luminance, axis=1))
            distances = (peak_row, peak_col, side - 1 - peak_row, side - 1 - peak_col)
            near_edges = sum(distance < settings.border_px for distance in distances)
            median_error = float(np.median(patch_scores[index]))
            row = {
                "brightness_std": float(stds[index]),
                "texture_energy": float(laplace(luminance).var()),
                "brightness_mean": float(luminance.mean()),
                "saturated_share": float((luminance >= settings.saturation_level).mean()),
                "shadow_share": float((luminance <= settings.shadow_level).mean()),
                "global_error": median_error,
                "locality": float(image_map.max() / max(median_error, 1e-12)),
                "edge_strength": float(edges[window].mean()),
                "window_brightness": float(luminance[window].mean() - np.median(luminance)),
                "window_saturated": float((luminance[window] >= settings.saturation_level).mean()),
                "border_distance": float(min(distances)),
                "border_edges": int(near_edges),
                "peak_row": int(peak_row),
                "peak_col": int(peak_col),
            }
            row |= self._defect(luminance, image_map, mask, (peak_row, peak_col))
            rows.append(row)
        return pd.DataFrame(rows)

    def _defect(
        self, luminance: np.ndarray, image_map: np.ndarray, mask: np.ndarray, peak: tuple
    ) -> dict[str, float]:
        """Площа, контраст дефекту, чи argmax у масці, AUROC карти в межах зображення."""
        if not mask.any():
            return {
                "defect_area": 0.0,
                "defect_contrast": np.nan,
                "argmax_in_mask": np.nan,
                "map_auroc": np.nan,
            }
        settings = self._settings
        ring = binary_dilation(mask, iterations=settings.ring_px) & ~mask
        contrast = abs(float(luminance[mask].mean()) - float(luminance[ring].mean()))
        tolerance = binary_dilation(mask, iterations=settings.pointing_px)
        return {
            "defect_area": float(mask.mean()),
            "defect_contrast": contrast,
            "argmax_in_mask": float(tolerance[peak]),
            "map_auroc": float(roc_auc_score(mask.reshape(-1), image_map.reshape(-1))),
        }


@dataclass(frozen=True)
class CauseRules:
    """Прозорі правила причин помилок (пороги обчислено до застосування, див. ``fit``).

    Attributes:
        global_error: Медіанна помилка, вище якої FP — «глобальна текстура/освітлення».
        glare_share: Частка пересвіту у вікні для «відблиску».
        edge_strength: Сила країв у вікні для «краю об'єкта».
        border_px: Смуга «краю кадру».
        small_area: Площа дефекту, нижче якої FN — «дрібний».
        low_contrast: Контраст, нижче якого FN — «низький контраст».
        off_map_auroc: AUROC карти зображення для «дефект не дав максимуму».
        fp_order: Порядок правил FP.
        fn_order: Порядок правил FN.
    """

    global_error: float
    glare_share: float
    edge_strength: float
    border_px: int
    small_area: float
    low_contrast: float
    off_map_auroc: float
    fp_order: tuple[str, ...]
    fn_order: tuple[str, ...]

    def fp_cause(self, row: pd.Series) -> str:
        """Перша причина FP, умова якої виконалась."""
        checks = {
            "global": row["global_error"] > self.global_error,
            "glare": row["window_saturated"] > self.glare_share,
            "object_edge": row["edge_strength"] > self.edge_strength,
            "frame_border": row["border_distance"] < self.border_px,
        }
        for cause in self.fp_order:
            if checks[cause]:
                if cause == "frame_border" and row["border_edges"] >= 2:
                    return "frame_corner"
                return cause
        return "undetermined"

    def fn_cause(self, row: pd.Series) -> str:
        """Перша причина FN, умова якої виконалась."""
        checks = {
            "small": row["defect_area"] < self.small_area,
            "low_contrast": row["defect_contrast"] < self.low_contrast,
            "off_map": (row["argmax_in_mask"] == 0) and (row["map_auroc"] >= self.off_map_auroc),
        }
        for cause in self.fn_order:
            if checks[cause]:
                return cause
        return "undetermined"

    def assign(self, cases: pd.DataFrame) -> pd.Series:
        """Причина для кожного рядка FP/FN (порожньо для TP/TN)."""

        # Функція всередині методу — «локальний клас-стратегія» для apply.
        def cause(row: pd.Series) -> str:
            if row["outcome"] == "FP":
                return self.fp_cause(row)
            if row["outcome"] == "FN":
                return self.fn_cause(row)
            return ""

        return cases.apply(cause, axis=1)


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Поправка Голма на множинні порівняння (контроль FWER)."""
    values = np.asarray(p_values, np.float64)
    order = np.argsort(values)
    count = len(values)
    adjusted = np.empty(count)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def compare_groups(
    cases: pd.DataFrame,
    descriptors: Sequence[str],
    first: str,
    second: str,
    min_group_size: int,
) -> pd.DataFrame:
    """Манн–Вітні ``first`` проти ``second`` (за колонкою ``outcome``) для кожного дескриптора.

    Рангово-бісеріальна кореляція ``r = 2U/(n₁n₂) − 1``: +1 — усі значення групи ``first``
    більші, −1 — менші, 0 — групи не відрізняються. При ``n < min_group_size`` тест не
    робиться (лише медіани).
    """
    a = cases[cases["outcome"] == first]
    b = cases[cases["outcome"] == second]
    rows = []
    for descriptor in descriptors:
        values_a, values_b = a[descriptor].dropna(), b[descriptor].dropna()
        row = {
            "descriptor": descriptor,
            "comparison": f"{first} vs {second}",
            "n_first": len(values_a),
            "n_second": len(values_b),
            "median_first": float(values_a.median()) if len(values_a) else np.nan,
            "median_second": float(values_b.median()) if len(values_b) else np.nan,
            "u": np.nan,
            "r": np.nan,
            "p": np.nan,
        }
        if min(len(values_a), len(values_b)) >= min_group_size:
            test = mannwhitneyu(values_a, values_b, alternative="two-sided")
            row |= {
                "u": float(test.statistic),
                "r": float(2 * test.statistic / (len(values_a) * len(values_b)) - 1),
                "p": float(test.pvalue),
            }
        rows.append(row)
    frame = pd.DataFrame(rows)
    tested = frame["p"].notna()
    frame["p_holm"] = np.nan
    if tested.any():
        frame.loc[tested, "p_holm"] = holm(frame.loc[tested, "p"].to_numpy())
    return frame


# ---------------------------------------------------------------- крайові артефакти


def edge_distance_profile(mean_map: np.ndarray, max_distance: int) -> pd.DataFrame:
    """Середнє значення карти залежно від відстані пікселя до найближчого краю кадру."""
    side = mean_map.shape[0]
    rows, cols = np.indices(mean_map.shape)
    distance = np.minimum.reduce([rows, cols, side - 1 - rows, side - 1 - cols])
    selected = distance <= max_distance
    frame = pd.DataFrame({"distance": distance[selected], "value": mean_map[selected]})
    return frame.groupby("distance", as_index=False)["value"].mean()


def band_shares(side: int, border_px: int) -> tuple[float, float]:
    """Частка площі смуги ширини ``border_px`` уздовж країв і частка чотирьох кутів."""
    inner = (side - 2 * border_px) / side
    return 1 - inner**2, 4 * (border_px / side) ** 2


def ring_errors(patch_scores: np.ndarray) -> dict[str, float]:
    """Середня помилка патчів крайового кільця (1-й і 2-й ряд) і центру сітки."""
    grid = patch_scores.shape[1]
    rows, cols = np.indices((grid, grid))
    ring = np.minimum.reduce([rows, cols, grid - 1 - rows, grid - 1 - cols])
    return {
        "ring1": float(patch_scores[:, ring == 0].mean()),
        "ring2": float(patch_scores[:, ring == 1].mean()),
        "center": float(patch_scores[:, ring >= 2].mean()),
    }


def cropped_scores(maps: np.ndarray, band: int) -> np.ndarray:
    """Оцінка зображення без крайової смуги ширини ``band`` (максимум внутрішньої частини)."""
    side = maps.shape[1]
    inner = maps[:, band : side - band, band : side - band] if band else maps
    return inner.reshape(len(maps), -1).max(axis=1)


@dataclass(frozen=True)
class ExclusionResult:
    """Діагностика «оцінка без рамки» для однієї ширини смуги.

    Attributes:
        band: Ширина смуги, пікс.
        threshold: Поріг, перекалібрований на val (max F1).
        image_auroc: Image AUROC на test.
        f1: Image F1 на test.
        fpr: Частка норм test вище порога.
        tpr: Частка аномалій test вище порога.
        masks_touching_band: Скільки масок test перетинають смугу.
    """

    band: int
    threshold: float
    image_auroc: float
    f1: float
    fpr: float
    tpr: float
    masks_touching_band: int


def border_exclusion(
    val_maps: np.ndarray,
    val_labels: np.ndarray,
    test_maps: np.ndarray,
    test_labels: np.ndarray,
    test_masks: np.ndarray,
    band: int,
) -> ExclusionResult:
    """Поріг на val для оцінки без смуги ``band`` і якість на test (модель не змінюється)."""
    val_scores = cropped_scores(val_maps, band)
    test_scores = cropped_scores(test_maps, band)
    threshold = F1ThresholdSelector().select(val_labels, val_scores)
    decisions = test_scores >= threshold
    side = test_masks.shape[1]
    inner = np.zeros((side, side), bool)
    inner[band : side - band, band : side - band] = True
    touching = int((test_masks & ~inner).reshape(len(test_masks), -1).any(axis=1).sum())
    return ExclusionResult(
        band=band,
        threshold=threshold,
        image_auroc=float(roc_auc_score(test_labels, test_scores)),
        f1=float(f1_score(test_labels, decisions)),
        fpr=float(decisions[test_labels == 0].mean()),
        tpr=float(decisions[test_labels == 1].mean()),
        masks_touching_band=touching if band else 0,
    )
