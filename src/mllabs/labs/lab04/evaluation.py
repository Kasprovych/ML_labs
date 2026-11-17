"""Фінальна оцінка на test: пороги з val, метрики, бутстреп-ДІ, прогнози для ЛР5.

Порядок суворий: пороги (max F1 для зображень і для пікселів) обираються на розміченому
val і застосовуються до test БЕЗ змін. Test використовується один раз для фіксованого
переліку моделей (FAE з обраною конфігурацією, PaDiM, PatchCore) і ні на що не впливає.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mllabs.anomaly.images import coarse_brightness_std
from mllabs.anomaly.metrics import (
    AnomalyMetricsCalculator,
    HistogramAuc,
    ImageBootstrap,
    MetricInterval,
    PixelSummary,
    SegmentationMetrics,
)
from mllabs.anomaly.postprocess import image_scores
from mllabs.anomaly.splits import SplitPart
from mllabs.anomaly.workspace import VisaWorkspace
from mllabs.labs.lab04.plots import MapExample
from mllabs.modeling import BootstrapInterval, F1ThresholdSelector

#: Метрики, для яких рахуються бутстреп-ДІ.
CI_METRICS = ("pixel_auroc", "pixel_ap", "image_auroc")


@dataclass(frozen=True)
class MethodMaps:
    """Карти аномалій одного методу на одній категорії.

    Attributes:
        method: ``"FAE"``, ``"PaDiM"`` або ``"PatchCore"``.
        category: Категорія.
        val_maps: ``(N_val, S, S)``.
        test_maps: ``(N_test, S, S)``.
    """

    method: str
    category: str
    val_maps: np.ndarray
    test_maps: np.ndarray


@dataclass(frozen=True)
class GroundTruth:
    """Розмітка val і test однієї категорії.

    Attributes:
        val_masks: Маски val.
        val_labels: Мітки val.
        test_masks: Маски test.
        test_labels: Мітки test.
    """

    val_masks: np.ndarray
    val_labels: np.ndarray
    test_masks: np.ndarray
    test_labels: np.ndarray


@dataclass(frozen=True)
class Evaluation:
    """Результат оцінки методу на test.

    Attributes:
        method: Метод.
        category: Категорія.
        metrics: Точні метрики.
        intervals: Бутстреп-ДІ (``CI_METRICS``).
        samples: Бутстреп-розподіли (для парних різниць).
        test_scores: Оцінки зображень test.
    """

    method: str
    category: str
    metrics: SegmentationMetrics
    intervals: dict[str, MetricInterval]
    samples: dict[str, np.ndarray]
    test_scores: np.ndarray


class TestEvaluator:
    """Пороги з val → метрики test → бутстреп.

    Args:
        bootstrap: Бутстреп по зображеннях (одне зерно для всіх методів → парні ДІ).
        histogram: Гістограми для бутстрепу піксельних метрик.
    """

    __test__ = False  # pytest не вважатиме клас тестом через префікс «Test»

    def __init__(self, bootstrap: ImageBootstrap, histogram: HistogramAuc) -> None:
        self._bootstrap = bootstrap
        self._histogram = histogram
        self._selector = F1ThresholdSelector()
        self._calculator = AnomalyMetricsCalculator()

    def evaluate(self, maps: MethodMaps, truth: GroundTruth) -> Evaluation:
        """Повна оцінка одного методу на одній категорії."""
        val_scores = image_scores(maps.val_maps)
        threshold_image = self._selector.select(truth.val_labels, val_scores)
        threshold_pixel = self._selector.select(
            truth.val_masks.reshape(-1), maps.val_maps.reshape(-1)
        )
        test_scores = image_scores(maps.test_maps)
        metrics = self._calculator.evaluate(
            maps.test_maps,
            truth.test_masks,
            truth.test_labels,
            test_scores,
            threshold_image,
            threshold_pixel,
        )
        edges = self._histogram.edges(maps.test_maps, truth.test_masks)
        positive, negative = self._histogram.per_image_counts(
            maps.test_maps, truth.test_masks, edges
        )
        samples = self._bootstrap.samples(
            PixelSummary(positive, negative, truth.test_labels, test_scores)
        )
        intervals = {
            name: self._bootstrap.interval(samples[name], getattr(metrics, name))
            for name in CI_METRICS
        }
        return Evaluation(maps.method, maps.category, metrics, intervals, samples, test_scores)

    def curves(self, test_maps: np.ndarray, test_masks: np.ndarray) -> dict[str, np.ndarray]:
        """Піксельні ROC/PR-криві на test (гістограмні, для графіка)."""
        edges = self._histogram.edges(test_maps, test_masks)
        positive, negative = self._histogram.per_image_counts(test_maps, test_masks, edges)
        return HistogramAuc.curves(positive.sum(axis=0), negative.sum(axis=0))

    def paired_difference(
        self, model: Evaluation, reference: Evaluation, metric: str
    ) -> BootstrapInterval:
        """ДІ різниці ``model − reference`` (наприклад, FAE − PatchCore)."""
        delta = getattr(model.metrics, metric) - getattr(reference.metrics, metric)
        return self._bootstrap.paired(reference.samples[metric], model.samples[metric], delta)


def outcome_labels(labels: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    """Позначки TP / FP / FN / TN для кожного зображення."""
    return np.select(
        [
            labels.astype(bool) & decisions,
            ~labels.astype(bool) & decisions,
            labels.astype(bool) & ~decisions,
        ],
        ["TP", "FP", "FN"],
        default="TN",
    )


def prediction_rows(
    evaluation: Evaluation,
    test_rows: pd.DataFrame,
    defect_area: np.ndarray,
    brightness_std: np.ndarray,
) -> pd.DataFrame:
    """Прогнози test одного методу (для аналізу FP/FN у ЛР5)."""
    labels = test_rows["label"].to_numpy()
    threshold = evaluation.metrics.threshold_image
    decisions = evaluation.test_scores >= threshold
    return pd.DataFrame(
        {
            "category": evaluation.category,
            "method": evaluation.method,
            "image": test_rows["image"].to_numpy(),
            "label": labels,
            "defect_types": test_rows["defect_types"].to_numpy(),
            "defect_area_share": defect_area,
            "brightness_std": brightness_std,
            "score": evaluation.test_scores,
            "threshold": threshold,
            "decision": decisions.astype(int),
            "outcome": outcome_labels(labels, decisions),
        }
    )


def metrics_row(evaluation: Evaluation) -> dict[str, float | str]:
    """Повний рядок метрик для ``test_metrics_full.csv``."""
    metrics = evaluation.metrics
    row: dict[str, float | str] = {"method": evaluation.method, "category": evaluation.category}
    row |= {
        "pixel_auroc": metrics.pixel_auroc,
        "pixel_ap": metrics.pixel_ap,
        "pixel_f1": metrics.pixel_f1,
        "image_auroc": metrics.image_auroc,
        "image_f1": metrics.image_f1,
        "oracle_pixel_f1": metrics.oracle_pixel_f1,
        "threshold_pixel": metrics.threshold_pixel,
        "threshold_image": metrics.threshold_image,
    }
    for name, interval in evaluation.intervals.items():
        row |= {f"{name}_low": interval.low, f"{name}_high": interval.high}
    return row


def summary_metrics(evaluations: Mapping[tuple[str, str], Evaluation]) -> dict[str, dict]:
    """Розділ ``metrics.test.<метод>.<категорія>`` і середні по категоріях."""
    result: dict[str, dict] = {}
    for (method, category), evaluation in evaluations.items():
        metrics = evaluation.metrics
        entry = {
            "pixel_auroc": metrics.pixel_auroc,
            "pixel_auroc_ci": [
                evaluation.intervals["pixel_auroc"].low,
                evaluation.intervals["pixel_auroc"].high,
            ],
            "pixel_ap": metrics.pixel_ap,
            "pixel_ap_ci": [
                evaluation.intervals["pixel_ap"].low,
                evaluation.intervals["pixel_ap"].high,
            ],
            "pixel_f1": metrics.pixel_f1,
            "image_auroc": metrics.image_auroc,
            "image_auroc_ci": [
                evaluation.intervals["image_auroc"].low,
                evaluation.intervals["image_auroc"].high,
            ],
            "image_f1": metrics.image_f1,
            "threshold_pixel": metrics.threshold_pixel,
            "threshold_image": metrics.threshold_image,
            "oracle_pixel_f1": metrics.oracle_pixel_f1,
        }
        result.setdefault(method, {})[category] = entry
    for entries in result.values():
        names = (
            "pixel_auroc",
            "pixel_ap",
            "pixel_f1",
            "image_auroc",
            "image_f1",
            "oracle_pixel_f1",
        )
        categories = list(entries)
        entries["mean"] = {
            name: float(np.mean([entries[c][name] for c in categories])) for name in names
        }
    return result


def pick_map_examples(predictions: pd.DataFrame) -> list[int]:
    """Два показові зображення test однієї категорії (позиції рядків ``predictions``).

    Перше — аномалія: пропущена (FN) з найбільшою площею дефекту, а якщо пропусків немає —
    знайдена (TP) з медіанною оцінкою. Друге — хибна тривога (FP) з найвищою оцінкою, а якщо
    їх немає — найвпевненіше знайдена аномалія.
    """
    frame = predictions.reset_index(drop=True)
    false_negative = frame[frame["outcome"] == "FN"]
    true_positive = frame[frame["outcome"] == "TP"].sort_values("score")
    false_positive = frame[frame["outcome"] == "FP"]
    if len(false_negative):
        first = int(false_negative["defect_area_share"].idxmax())
    else:
        first = int(true_positive.index[len(true_positive) // 2])
    second = (
        int(false_positive["score"].idxmax())
        if len(false_positive)
        else int(true_positive.index[-1])
    )
    return [first, second] if second != first else [first]


#: Метрики, для яких рахуються парні різниці «FAE − еталон».
PAIRED_METRICS = ("pixel_auroc", "image_auroc", "pixel_ap")


def ground_truth(workspace: VisaWorkspace, category: str) -> GroundTruth:
    """Маски й мітки val і test категорії."""
    return GroundTruth(
        workspace.masks(category, SplitPart.VAL),
        workspace.labels(category, SplitPart.VAL),
        workspace.masks(category, SplitPart.TEST),
        workspace.labels(category, SplitPart.TEST),
    )


@dataclass(frozen=True)
class BenchmarkResult:
    """Оцінка всіх методів на всіх категоріях.

    Attributes:
        evaluations: ``(метод, категорія) → Evaluation``.
        curves: ``категорія → метод → точки ROC/PR``.
        deltas: ``(категорія, «FAE − еталон», метрика) → парний ДІ``.
    """

    evaluations: dict[tuple[str, str], Evaluation]
    curves: dict[str, dict[str, dict[str, np.ndarray]]]
    deltas: dict[tuple[str, str, str], BootstrapInterval]


def evaluate_benchmark(
    evaluator: TestEvaluator,
    maps: Mapping[tuple[str, str], MethodMaps],
    truths: Mapping[str, GroundTruth],
    model: str,
    references: Sequence[str],
) -> BenchmarkResult:
    """Оцінює ``model`` і ``references`` на кожній категорії та рахує парні різниці."""
    evaluations, curves, deltas = {}, {}, {}
    for category, truth in truths.items():
        curves[category] = {}
        for method in (model, *references):
            method_maps = maps[(method, category)]
            evaluations[(method, category)] = evaluator.evaluate(method_maps, truth)
            curves[category][method] = evaluator.curves(method_maps.test_maps, truth.test_masks)
        for reference in references:
            for metric in PAIRED_METRICS:
                deltas[(category, f"{model} − {reference}", metric)] = evaluator.paired_difference(
                    evaluations[(model, category)], evaluations[(reference, category)], metric
                )
    return BenchmarkResult(evaluations, curves, deltas)


def prediction_table(workspace: VisaWorkspace, evaluation: Evaluation, cells: int) -> pd.DataFrame:
    """Прогнози test з площею дефекту й σ яскравості тла (для аналізу FP/FN у ЛР5)."""
    category = evaluation.category
    test = workspace.part_mask(category, SplitPart.TEST)
    batch = workspace.images(category)
    masks = np.asarray(batch.masks[test])
    return prediction_rows(
        evaluation,
        workspace.category_rows(category)[test].reset_index(drop=True),
        masks.reshape(len(masks), -1).mean(axis=1),
        coarse_brightness_std(np.asarray(batch.images[test]), cells),
    )


def _short_types(defect_types: str) -> str:
    """Перший тип дефекту (+ «…», якщо їх кілька) — для підпису рядка сітки карт."""
    first, *rest = defect_types.split(",")
    return first + (" …" if rest else "")


def map_examples(
    workspace: VisaWorkspace,
    predictions: pd.DataFrame,
    result: BenchmarkResult,
    maps: Mapping[tuple[str, str], MethodMaps],
    model: str,
    methods: Sequence[str],
) -> list[MapExample]:
    """Показові карти: по 2 зображення на категорію (FN і FP моделі, якщо вони є).

    Кожна карта ділиться на поріг пікселя свого методу: 1 — межа рішення, тож карти
    різних методів (у різних одиницях) можна порівнювати на одній шкалі.
    """
    examples = []
    for category in dict.fromkeys(predictions["category"]):
        rows = predictions[(predictions["category"] == category) & (predictions["method"] == model)]
        test = workspace.part_mask(category, SplitPart.TEST)
        batch = workspace.images(category)
        images, masks = np.asarray(batch.images[test]), np.asarray(batch.masks[test])
        for index in pick_map_examples(rows):
            row = rows.iloc[index]
            kind = _short_types(row["defect_types"]) if row["label"] else "норма"
            scaled = {
                method: maps[(method, category)].test_maps[index]
                / result.evaluations[(method, category)].metrics.threshold_pixel
                for method in methods
            }
            title = f"{category}, {model}: {row['outcome']}\n{kind}"
            examples.append(MapExample(images[index], masks[index], scaled, title))
    return examples
