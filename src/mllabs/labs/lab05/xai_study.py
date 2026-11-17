"""Сценарії пояснень (R2): вибір бази IG і шару Grad-CAM на val, пояснення на test.

Правило ізоляції: усе, що «обирається» (база IG, шар Grad-CAM), обирається на розміченому
val; test використовується лише для звітних чисел.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mllabs.labs.lab05.attribution import (
    CAM_LAYERS,
    CAM_VARIANTS,
    IgResult,
    IntegratedGradients,
    cam_key,
)
from mllabs.labs.lab05.config import BaselineKind, Lab5Config
from mllabs.labs.lab05.localization import area_terciles
from mllabs.labs.lab05.workbench import CategoryBench

_logger = logging.getLogger(__name__)

#: Підписи методів пояснення.
METHOD_LABELS = {
    "model_map": "карта моделі",
    "ig": "Integrated Gradients",
    "gradcam": "Grad-CAM",
    "hirescam": "HiResCAM",
    "random": "випадкова карта",
    "oracle": "маска-оракул",
}
BASELINE_LABELS = {
    BaselineKind.BLUR: "розмите (σ = 10)",
    BaselineKind.NEAREST_NORMAL: "найближча норма",
    BaselineKind.EXPECTED: "Expected Gradients (4 норми)",
    BaselineKind.MEAN: "середнє train",
    BaselineKind.BLACK: "чорне",
}


def image_auroc(explanation: np.ndarray, mask: np.ndarray) -> float:
    """Піксельний AUROC карти проти маски одного зображення."""
    return float(roc_auc_score(mask.reshape(-1), explanation.reshape(-1)))


@dataclass(frozen=True)
class Selection:
    """Зображення test, для яких рахуються дорогі пояснення (D2).

    Індекси — позиції в масивах test (``bench.test``), не рядки категорії.

    Attributes:
        anomalies: Аномалії, стратифіковані за терцилями площі дефекту.
        false_negatives: Усі FN.
        false_positives: Найвпевненіші FP (за відносним перевищенням порога).
    """

    anomalies: np.ndarray
    false_negatives: np.ndarray
    false_positives: np.ndarray

    @property
    def explained(self) -> np.ndarray:
        """Усі позиції з IG (без повторів, у порядку зростання)."""
        return np.unique(np.r_[self.anomalies, self.false_negatives, self.false_positives])


def select_images(bench: CategoryBench, config: Lab5Config, seed: int) -> Selection:
    """Детермінований вибір зображень test для IG і perturbation."""
    predictions = bench.assets.predictions
    labels = predictions["label"].to_numpy()
    generator = np.random.default_rng(seed)
    anomalies = np.flatnonzero(labels == 1)
    areas = predictions["defect_area_share"].to_numpy()[anomalies]
    strata = area_terciles(areas, config.area_strata)
    per_stratum = config.ig_anomalies_per_category // config.area_strata
    chosen = [
        generator.choice(
            anomalies[strata == stratum],
            min(per_stratum, int((strata == stratum).sum())),
            replace=False,
        )
        for stratum in range(config.area_strata)
    ]
    outcome = predictions["outcome"].to_numpy()
    margin = (predictions["score"] - predictions["threshold"]) / predictions["threshold"]
    false_positives = np.flatnonzero(outcome == "FP")
    confident = false_positives[np.argsort(-margin.to_numpy()[false_positives], kind="stable")]
    return Selection(
        anomalies=np.sort(np.concatenate(chosen)),
        false_negatives=np.flatnonzero(outcome == "FN"),
        false_positives=confident[: config.perturbation_fp_per_category],
    )


@dataclass(frozen=True)
class BaselineExample:
    """Дані для рисунка «бази IG»: одне зображення val і всі бази.

    Attributes:
        category: Категорія.
        image: Зображення.
        mask: Маска.
        baselines: База → перше зображення бази.
        results: База → результат IG.
    """

    category: str
    image: np.ndarray
    mask: np.ndarray
    baselines: dict[str, np.ndarray]
    results: dict[str, IgResult]


@dataclass(frozen=True)
class BaselineSelection:
    """Результат вибору бази IG на val.

    Attributes:
        runs: Рядок на (категорія, база, зображення).
        summary: Зведення по базах і категоріях.
        chosen: Обрана база.
        reason: Пояснення вибору.
        example: Приклад для рисунка.
    """

    runs: pd.DataFrame
    summary: pd.DataFrame
    chosen: BaselineKind
    reason: str
    example: BaselineExample


class XaiStudy:
    """Пояснення детектора: IG, Grad-CAM, HiResCAM.

    Args:
        config: Параметри ЛР5.
        seed: Зерно.
    """

    def __init__(self, config: Lab5Config, seed: int) -> None:
        self._config = config
        self._seed = seed

    # ------------------------------------------------------------ база IG (val)

    def select_baseline(
        self, benches: Sequence[CategoryBench], igs: Mapping[str, IntegratedGradients]
    ) -> BaselineSelection:
        """Усі бази на кількох аномаліях val кожної категорії; вибір за AUROC |IG|."""
        config = self._config
        rows, example = [], None
        for bench in benches:
            generator = np.random.default_rng(self._seed)
            val_rows = bench.val_rows
            anomalies = np.flatnonzero(bench.assets.labels(val_rows) == 1)
            picked = np.sort(
                generator.choice(anomalies, config.val_images_for_selection, replace=False)
            )
            for position in picked:
                row = val_rows[position]
                image = np.asarray(bench.assets.images[row])
                mask = bench.assets.masks[row]
                results, bases = {}, {}
                for kind in config.ig_baselines:
                    baselines = bench.baselines.make(kind, image)
                    result = igs[bench.category].explain(image, baselines)
                    results[str(kind)], bases[str(kind)] = result, baselines[0]
                    rows.append(
                        {
                            "category": bench.category,
                            "baseline": str(kind),
                            "image": bench.assets.rows["image"].iloc[row],
                            "score": result.score,
                            "baseline_score": result.baseline_score,
                            "delta": result.delta,
                            "completeness_error": result.completeness_error,
                            "steps": result.steps,
                            "auroc": image_auroc(result.map, mask),
                        }
                    )
                if example is None:
                    example = BaselineExample(bench.category, image, mask, bases, results)
        runs = pd.DataFrame(rows)
        chosen, reason = self._choose_baseline(runs)
        _logger.info("База IG: %s (%s)", chosen, reason)
        return BaselineSelection(runs, self._baseline_summary(runs), chosen, reason, example)

    def _choose_baseline(self, runs: pd.DataFrame) -> tuple[BaselineKind, str]:
        """Найкращий середній AUROC серед баз, що проходять перевірку повноти."""
        config = self._config
        grouped = runs.groupby("baseline")
        auroc = grouped["auroc"].mean()
        passed = grouped["completeness_error"].apply(
            lambda errors: float((errors <= config.completeness_tolerance).mean())
        )
        eligible = [
            kind
            for kind in config.ig_baselines
            if passed[str(kind)] >= config.min_completeness_pass_share
        ]
        if not eligible:
            eligible = list(config.ig_baselines)
            note = "жодна база не пройшла перевірку повноти — вибір лише за AUROC"
        else:
            note = f"повноту пройшли: {', '.join(str(kind) for kind in eligible)}"
        best = max(auroc[str(kind)] for kind in eligible)
        # Перша (найпростіша) база в межах допуску від найкращої — правило простоти.
        chosen = next(
            kind for kind in eligible if best - auroc[str(kind)] < config.selection_tolerance
        )
        return chosen, f"{note}; AUROC {auroc[str(chosen)]:.3f} (найкращий {best:.3f})"

    def _baseline_summary(self, runs: pd.DataFrame) -> pd.DataFrame:
        tolerance = self._config.completeness_tolerance
        pooled = runs.assign(category="усі")
        frame = pd.concat([runs, pooled], ignore_index=True)
        summary = frame.groupby(["baseline", "category"], sort=False).agg(
            baseline_score=("baseline_score", "median"),
            delta=("delta", "median"),
            pass_share=("completeness_error", lambda errors: float((errors <= tolerance).mean())),
            steps=("steps", "median"),
            auroc=("auroc", "mean"),
        )
        return summary.reset_index()

    # ------------------------------------------------------------ шар Grad-CAM (val)

    def select_cam_layers(
        self, benches: Sequence[CategoryBench]
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """Середній AUROC кожного варіанта/шару CAM на всіх аномаліях val; вибір шару."""
        rows = []
        for bench in benches:
            val_rows = bench.val_rows
            anomalies = val_rows[bench.assets.labels(val_rows) == 1]
            maps = bench.cam.maps(np.asarray(bench.assets.images[anomalies]))
            for variant in CAM_VARIANTS:
                for layer in CAM_LAYERS:
                    values = [
                        image_auroc(explanation, bench.assets.masks[row])
                        for explanation, row in zip(
                            maps[cam_key(variant, layer)], anomalies, strict=True
                        )
                    ]
                    rows.append(
                        {
                            "variant": variant,
                            "layer": layer,
                            "category": bench.category,
                            "auroc": float(np.mean(values)),
                            "n": len(values),
                        }
                    )
        table = pd.DataFrame(rows)
        pooled = table.groupby(["variant", "layer"])["auroc"].mean()
        chosen = {}
        for variant in CAM_VARIANTS:
            best = max(pooled[(variant, layer)] for layer in self._config.gradcam_layers)
            chosen[variant] = next(
                layer
                for layer in self._config.gradcam_layers
                if best - pooled[(variant, layer)] < self._config.selection_tolerance
            )
        return table, chosen

    # ------------------------------------------------------------ test

    def explain_test(
        self,
        bench: CategoryBench,
        ig: IntegratedGradients,
        baseline: BaselineKind,
        selection: Selection,
        cam_layers: Mapping[str, str],
    ) -> TestExplanations:
        """IG на вибраних зображеннях і CAM на всіх зображеннях test категорії."""
        images = bench.assets.images
        test_rows = bench.test_rows
        ig_results = {}
        for position in selection.explained:
            image = np.asarray(images[test_rows[position]])
            ig_results[int(position)] = ig.explain(image, bench.baselines.make(baseline, image))
        cams = bench.cam.maps(np.asarray(images[test_rows]))
        return TestExplanations(
            ig=ig_results,
            gradcam=cams[cam_key("gradcam", cam_layers["gradcam"])],
            hirescam=cams[cam_key("hirescam", cam_layers["hirescam"])],
        )


@dataclass(frozen=True)
class TestExplanations:
    """Пояснення на test однієї категорії (позиції — у масивах test).

    Attributes:
        ig: Позиція → результат IG.
        gradcam: ``(N, E, E)`` Grad-CAM обраного шару для всіх зображень test.
        hirescam: ``(N, E, E)`` HiResCAM обраного шару.
    """

    ig: dict[int, IgResult]
    gradcam: np.ndarray
    hirescam: np.ndarray

    def ig_maps(self, positions: Sequence[int]) -> np.ndarray:
        """Карти |IG| для позицій (усі мають бути пояснені)."""
        return np.stack([self.ig[int(position)].map for position in positions])


def completeness_table(
    category: str, explanations: TestExplanations, names: Sequence[str]
) -> list[dict[str, object]]:
    """Рядки ``ig_completeness.csv``: зображення, m, ε, F(x), F(b)."""
    return [
        {
            "category": category,
            "image": names[position],
            "steps": result.steps,
            "completeness_error": result.completeness_error,
            "score": result.score,
            "baseline_score": result.baseline_score,
        }
        for position, result in explanations.ig.items()
    ]
