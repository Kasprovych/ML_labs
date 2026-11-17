"""Сценарії оцінки якості пояснень (R2 — локалізація, R3 — perturbation, рандомізація, стійкість).

Модулі ``localization``, ``perturbation`` і ``sanity`` містять самі методи; тут — цикли по
категоріях і зображеннях та збирання таблиць.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import keras
import numpy as np
import pandas as pd

from mllabs.anomaly.metrics import HistogramAuc
from mllabs.labs.lab05.attribution import (
    CamExplainer,
    Explainer,
    GradCam,
    IgExplainer,
    IgSettings,
    IntegratedGradients,
    RandomExplainer,
    cam_key,
)
from mllabs.labs.lab05.config import BaselineKind, Lab5Config
from mllabs.labs.lab05.inference import DetectorRunner
from mllabs.labs.lab05.localization import LocalizationEvaluator, area_terciles, bootstrap_mean
from mllabs.labs.lab05.perturbation import (
    NoiseReplacement,
    PerturbationStudy,
    ReferenceReplacement,
    SubstituteSource,
    curve_metrics,
    patch_importance,
)
from mllabs.labs.lab05.sanity import ModelRandomization, add_noise, map_similarity
from mllabs.labs.lab05.weights_analysis import patch_coverage
from mllabs.labs.lab05.workbench import CategoryBench
from mllabs.labs.lab05.xai_study import Selection, TestExplanations

#: Метрики локалізації та їхні підписи.
LOCALIZATION_METRICS = ("pixel_auroc", "pointing", "mass_ratio", "iou_topk", "spearman_model")


class LocalizationStudy:
    """Метрики локалізації всіх методів на аномаліях test.

    Args:
        config: Параметри ЛР5.
        seed: Зерно випадкової карти й бутстрепу.
    """

    def __init__(self, config: Lab5Config, seed: int) -> None:
        self._config = config
        self._seed = seed
        self._evaluator = LocalizationEvaluator(
            config.pointing_tolerance_px, config.correlation_grid
        )

    def per_image(
        self, bench: CategoryBench, explanations: TestExplanations, selection: Selection
    ) -> pd.DataFrame:
        """Рядок на (зображення, метод) для всіх аномалій test; IG — лише на підвибірці."""
        predictions = bench.assets.predictions
        labels = predictions["label"].to_numpy()
        anomalies = np.flatnonzero(labels == 1)
        masks = bench.assets.masks[bench.test_rows]
        model_maps = bench.test.maps
        random = RandomExplainer(bench.assets.config.eval_size, self._seed)
        cheap = {
            "model_map": model_maps[anomalies],
            "gradcam": explanations.gradcam[anomalies],
            "hirescam": explanations.hirescam[anomalies],
            "random": random.explain(anomalies),
        }
        names = predictions["image"].to_numpy()
        table = self._evaluator.table(
            bench.category, names[anomalies], cheap, masks[anomalies], model_maps[anomalies]
        )
        explained = np.array([p for p in selection.explained if labels[p] == 1])
        ig_table = self._evaluator.table(
            bench.category,
            names[explained],
            {"ig": explanations.ig_maps(explained)},
            masks[explained],
            model_maps[explained],
        )
        frame = pd.concat([table, ig_table], ignore_index=True)
        areas = pd.Series(
            predictions["defect_area_share"].to_numpy()[anomalies], index=names[anomalies]
        )
        strata = pd.Series(
            area_terciles(areas.to_numpy(), self._config.area_strata), index=areas.index
        )
        in_subset = set(names[explained])
        # Чи пройшов IG цього зображення перевірку повноти (NaN — IG не рахувався).
        completeness = pd.Series(
            {
                names[position]: explanations.ig[int(position)].completeness_error
                <= self._config.completeness_tolerance
                for position in explained
            },
            dtype=object,
        )
        return frame.assign(
            area=frame["image"].map(areas),
            area_stratum=frame["image"].map(strata),
            ig_subset=frame["image"].isin(in_subset),
            ig_completeness_ok=frame["image"].map(completeness),
        )

    def summary(self, per_image: pd.DataFrame, subset_only: bool) -> pd.DataFrame:
        """Середні й 95 % ДІ по зображеннях: метод × категорія (+ «усі»)."""
        frame = per_image[per_image["ig_subset"]] if subset_only else per_image
        frame = pd.concat([frame, frame.assign(category="усі")], ignore_index=True)
        rows = []
        config = self._config
        for (category, method), group in frame.groupby(["category", "method"], sort=False):
            row = {"category": category, "method": method, "n": len(group)}
            for metric in LOCALIZATION_METRICS:
                interval = bootstrap_mean(
                    group[metric].to_numpy(),
                    config.bootstrap_repeats,
                    config.confidence_level,
                    self._seed,
                )
                row |= {
                    metric: interval.value,
                    f"{metric}_low": interval.low,
                    f"{metric}_high": interval.high,
                }
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def by_area(per_image: pd.DataFrame) -> pd.DataFrame:
        """Середній AUROC і pointing за стратами площі дефекту (на підвибірці IG)."""
        frame = per_image[per_image["ig_subset"]]
        return (
            frame.groupby(["method", "area_stratum"])
            .agg(
                n=("image", "size"),
                pixel_auroc=("pixel_auroc", "mean"),
                pointing=("pointing", "mean"),
            )
            .reset_index()
        )


# ---------------------------------------------------------------- perturbation


class PerturbationRunner:
    """Perturbation analysis на вибраних зображеннях категорії (два режими).

    Args:
        config: Параметри ЛР5.
        seed: Зерно.
    """

    def __init__(self, config: Lab5Config, seed: int) -> None:
        self._config = config
        self._seed = seed

    def modes(self, bench: CategoryBench) -> dict[str, SubstituteSource]:
        """Режими «вилучення» (заміна на норму) і «зашумлення»."""
        return {
            "replace": ReferenceReplacement(bench.references),
            "noise": NoiseReplacement(self._config.noise_sigma, self._seed),
        }

    def run(
        self, bench: CategoryBench, explanations: TestExplanations, selection: Selection
    ) -> pd.DataFrame:
        """Криві для аномалій підвибірки та найвпевненіших FP (рядок на стан кривої)."""
        config = self._config
        grid = bench.assets.grid
        study = PerturbationStudy(
            bench.runner, config.perturbation_steps, grid, config.random_orders, self._seed
        )
        masks = bench.assets.masks[bench.test_rows]
        coverage = patch_coverage(masks, grid).reshape(len(masks), -1)
        names = bench.assets.predictions["image"].to_numpy()
        generator = np.random.default_rng(self._seed)
        frames = []
        groups = (("anomaly", selection.anomalies), ("false_positive", selection.false_positives))
        for group, positions in groups:
            for position in positions:
                image = np.asarray(bench.assets.images[bench.test_rows[position]])
                importances = {
                    "model_map": patch_importance(bench.test.maps[position], grid),
                    "ig": patch_importance(explanations.ig[int(position)].map, grid),
                    "gradcam": patch_importance(explanations.gradcam[position], grid),
                    "hirescam": patch_importance(explanations.hirescam[position], grid),
                }
                zone = coverage[position] > 0
                if zone.any():
                    # Оракул: патчі за покриттям маски; нічиї (нульове покриття) — випадково.
                    jitter = generator.random(grid * grid) * config.oracle_jitter
                    importances["oracle"] = coverage[position] + jitter
                for mode, source in self.modes(bench).items():
                    substitute = source.substitute(image, int(position))
                    curves = study.run(
                        image, substitute, importances, zone if zone.any() else None, int(position)
                    )
                    frames.append(
                        curves.assign(
                            category=bench.category,
                            image=names[position],
                            group=group,
                            mode=mode,
                        )
                    )
        return pd.concat(frames, ignore_index=True)


#: Ключі, що ідентифікують зображення в кривих perturbation.
PERTURBATION_KEYS = ("category", "group", "mode", "image")


def perturbation_metrics(curves: pd.DataFrame) -> pd.DataFrame:
    """AUC/ABPC кожного зображення (див. :func:`curve_metrics`)."""
    return curve_metrics(curves, PERTURBATION_KEYS)


def perturbation_summary(per_image: pd.DataFrame, config: Lab5Config, seed: int) -> pd.DataFrame:
    """Середні по зображеннях з ДІ для ABPC і Δ проти випадкового."""
    pooled = pd.concat([per_image, per_image.assign(category="усі")], ignore_index=True)
    rows = []
    keys = ["category", "group", "mode", "response", "method"]
    for key_values, group in pooled.groupby(keys, sort=False):
        values = group["abpc"].dropna().to_numpy()
        if not len(values):
            continue
        abpc = bootstrap_mean(values, config.bootstrap_repeats, config.confidence_level, seed)
        delta = bootstrap_mean(
            group["delta_vs_random"].dropna().to_numpy(),
            config.bootstrap_repeats,
            config.confidence_level,
            seed,
        )
        rows.append(
            dict(zip(keys, key_values, strict=True))
            | {
                "n": len(values),
                "auc_morf": float(group["auc_morf"].mean()),
                "auc_lerf": float(group["auc_lerf"].mean()),
                "auc_random": float(group["auc_random"].mean()),
                "abpc": abpc.value,
                "abpc_low": abpc.low,
                "abpc_high": abpc.high,
                "delta_vs_random": delta.value,
                "delta_low": delta.low,
                "delta_high": delta.high,
            }
        )
    return pd.DataFrame(rows)


def mean_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Середні криві по зображеннях (для графіка і CSV)."""
    return curves.groupby(
        ["category", "group", "mode", "method", "direction", "k"], as_index=False
    )[["score_rel", "error_rel", "mask_error_rel"]].mean()


# ---------------------------------------------------------------- рандомізація, стійкість


@dataclass(frozen=True)
class SanityExample:
    """Карти одного зображення для рисунка рандомізації.

    Attributes:
        category: Категорія.
        image: Зображення.
        mask: Маска.
        maps: (рівень, метод) → карта.
    """

    category: str
    image: np.ndarray
    mask: np.ndarray
    maps: dict[tuple[str, str], np.ndarray]


class SanityStudy:
    """Рандомізація моделі (Adebayo et al.) і стійкість пояснень до малого шуму.

    Args:
        config: Параметри ЛР5.
        seed: Зерно.
        weights: Ваги backbone навченої моделі.
    """

    def __init__(self, config: Lab5Config, seed: int, weights: str | None) -> None:
        self._config = config
        self._seed = seed
        self._weights = weights
        self._histogram = HistogramAuc(config.histogram_bins)

    def images(self, selection: Selection) -> np.ndarray:
        """Перші аномалії підвибірки (по черзі зі страт площі)."""
        return selection.anomalies[: self._config.sanity_images_per_category]

    def _explainers(
        self,
        bench: CategoryBench,
        detector: keras.Model,
        baseline: BaselineKind,
        layers: Mapping[str, str],
    ) -> dict[str, Explainer]:
        config = self._config
        settings = IgSettings(
            steps=config.sanity_ig_steps,
            max_steps=config.sanity_ig_steps,
            alpha_batch=config.ig_alpha_batch,
            tolerance=config.completeness_tolerance,
            scale=bench.assets.config.threshold_image,
            blur_sigma=config.attribution_blur_sigma,
            output_size=bench.assets.config.eval_size,
        )
        ig = IntegratedGradients(
            lambda batch: detector(batch, training=False)["image_score"], settings
        )
        cam = GradCam(detector, bench.assets.config.eval_size, config.inference_batch)
        return {
            "ig": IgExplainer(ig, bench.baselines, baseline),
            "gradcam": CamExplainer(cam, cam_key("gradcam", layers["gradcam"])),
            "hirescam": CamExplainer(cam, cam_key("hirescam", layers["hirescam"])),
        }

    def run(
        self,
        bench: CategoryBench,
        selection: Selection,
        baseline: BaselineKind,
        layers: Mapping[str, str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, SanityExample]:
        """Таблиці рандомізації та стійкості й приклад для рисунка."""
        config = self._config
        positions = self.images(selection)
        images = np.asarray(bench.assets.images[bench.test_rows[positions]])
        names = bench.assets.predictions["image"].to_numpy()[positions]
        trained = self._explain(
            self._explainers(bench, bench.assets.detector, baseline, layers), images
        )
        randomization = ModelRandomization(
            bench.assets.config, bench.assets.head, self._weights, self._seed
        )
        example_maps = {("trained", method): maps[0] for method, maps in trained.items()}
        test_masks = bench.assets.masks[bench.test_rows]
        trained_auroc, _ = self._histogram.metrics(bench.test.maps, test_masks)
        sanity_rows = []
        for level, _ in ModelRandomization.LEVELS:
            detector = randomization.detector(level)
            # Контроль: чи рандомізований детектор ще локалізує дефекти (Pixel AUROC на test).
            outputs = DetectorRunner(detector, config.inference_batch).run(
                np.asarray(bench.assets.images[bench.test_rows])
            )
            detector_auroc, _ = self._histogram.metrics(outputs.maps, test_masks)
            explainers = self._explainers(bench, detector, baseline, layers)
            randomized = self._explain(explainers, images)
            rows = self._similarity_rows(bench.category, level, names, trained, randomized)
            sanity_rows += [
                row | {"detector_pixel_auroc": detector_auroc, "trained_pixel_auroc": trained_auroc}
                for row in rows
            ]
            example_maps |= {(level, method): maps[0] for method, maps in randomized.items()}
        noisy = add_noise(images, config.stability_noise_sigma, self._seed)
        perturbed = self._explain(
            self._explainers(bench, bench.assets.detector, baseline, layers), noisy
        )
        stability_rows = self._similarity_rows(bench.category, "noise", names, trained, perturbed)
        example = SanityExample(
            bench.category,
            images[0],
            bench.assets.masks[bench.test_rows[positions[0]]],
            example_maps,
        )
        return pd.DataFrame(sanity_rows), pd.DataFrame(stability_rows), example

    @staticmethod
    def _explain(explainers: Mapping[str, Explainer], images: np.ndarray) -> dict[str, np.ndarray]:
        return {method: explainer.explain(images) for method, explainer in explainers.items()}

    def _similarity_rows(
        self,
        category: str,
        level: str,
        names: Sequence[str],
        reference: Mapping[str, np.ndarray],
        other: Mapping[str, np.ndarray],
    ) -> list[dict]:
        rows = []
        for method, maps in reference.items():
            for index, name in enumerate(names):
                rho, ssim = map_similarity(
                    maps[index], other[method][index], self._config.correlation_grid
                )
                rows.append(
                    {
                        "category": category,
                        "level": level,
                        "method": method,
                        "image": name,
                        "spearman": rho,
                        "ssim": ssim,
                    }
                )
        return rows
