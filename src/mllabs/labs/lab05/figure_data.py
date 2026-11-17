"""Підготовка даних для рисунків і зведених рядків (вибір прикладів, нормування карт).

Тут немає обчислень моделей — лише вибір, які зображення показати, і перетворення вже
пораханих результатів у структури, які приймають функції з ``plots``.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from mllabs.labs.lab05.config import Lab5Config
from mllabs.labs.lab05.edge_study import VARIANT_LABELS
from mllabs.labs.lab05.error_cases import CAUSE_LABELS
from mllabs.labs.lab05.errors_study import panel_positions
from mllabs.labs.lab05.perturbation import patch_importance, perturbed, ranking
from mllabs.labs.lab05.plots import LayerMapsRow, XaiRow
from mllabs.labs.lab05.weights_study import CategoryWeights
from mllabs.labs.lab05.workbench import CategoryBench
from mllabs.labs.lab05.xai_study import Selection, TestExplanations


def layer_rows(results: Mapping[str, CategoryWeights]) -> list[LayerMapsRow]:
    """Приклади карт по шарах (по одному на категорію)."""
    return [
        LayerMapsRow(
            category,
            result.example.image,
            result.example.mask,
            result.example.full,
            result.example.shallow,
            result.example.deep,
        )
        for category, result in results.items()
    ]


def latent_rows(results: Mapping[str, CategoryWeights]) -> list[dict]:
    """Рядки таблиці латентного простору / лінійного еталона."""
    rows = []
    for category, result in results.items():
        maps = {row["Карта"]: row for row in result.map_rows}
        rows.append(
            {
                "category": category,
                "participation_ratio": result.participation_ratio,
                "fae_pixel_auroc": maps["FAE (повна)"]["Pixel AUROC"],
                "mahalanobis_pixel_auroc": maps["Махаланобіс у латенті"]["Pixel AUROC"],
                "pca_pixel_auroc": maps["PCA-64 (лінійний АЕ)"]["Pixel AUROC"],
                "fae_pca_spearman": result.pca_fae_spearman,
                "angle_cos_mean": float(result.angle_cosines.mean()),
            }
        )
    return rows


def jacobian_rows(results: Mapping[str, CategoryWeights], threshold: float) -> list[dict]:
    """Рядки таблиці спектрів якобіана."""
    labels = {"normal": "нормальні", "defect": "дефектні"}
    return [
        {
            "category": category,
            "group": labels[group],
            "n": len(summary.projection_error),
            "rank": float(np.median(summary.rank(threshold))),
            "sigma_max": float(np.median(summary.singular_values[:, 0])),
            "projection_error": float(np.median(summary.projection_error)),
        }
        for category, result in results.items()
        for group, summary in result.jacobians.items()
    ]


def weights_metrics(results: Mapping[str, CategoryWeights], threshold: float) -> dict[str, object]:
    """Блок ``metrics.weights``."""
    metrics: dict[str, object] = {"spectra": {}, "latent": {}, "linear": {}, "jacobian": {}}
    for category, result in results.items():
        layers = {row["Група патчів"]: row for row in result.layer_rows}
        maps = {row["Карта"]: row for row in result.map_rows}
        top = {row["k каналів"]: row["Pixel AUROC"] for row in result.topk_rows}
        metrics[category] = {
            "shallow_share_normal": layers["нормальні"]["Частка шару 8"],
            "shallow_share_context": layers["контекст дефекту"]["Частка шару 8"],
            "shallow_share_defect": layers["дефектні"]["Частка шару 8"],
            "defect_patches": layers["дефектні"]["Патчів"],
            "pixel_auroc_full": maps["FAE (повна)"]["Pixel AUROC"],
            "pixel_auroc_shallow_only": maps["лише шар 8"]["Pixel AUROC"],
            "pixel_auroc_deep_only": maps["лише шар 16"]["Pixel AUROC"],
            "top_k_pixel_auroc": top,
            "input_sensitivity_spearman": result.sensitivity_spearman,
        }
        metrics["spectra"][category] = {
            spectrum.layer: {
                "effective_rank": spectrum.effective_rank,
                "sigma_max": float(spectrum.singular_values[0]),
            }
            for spectrum in result.spectra
        }
        metrics["latent"][category] = {
            "participation_ratio": result.participation_ratio,
            "mahalanobis_pixel_auroc": maps["Махаланобіс у латенті"]["Pixel AUROC"],
            "mahalanobis_pixel_ap": maps["Махаланобіс у латенті"]["Pixel AP"],
        }
        metrics["linear"][category] = {
            "pca_pixel_auroc": maps["PCA-64 (лінійний АЕ)"]["Pixel AUROC"],
            "pca_pixel_ap": maps["PCA-64 (лінійний АЕ)"]["Pixel AP"],
            "fae_pixel_ap": maps["FAE (повна)"]["Pixel AP"],
            "fae_pca_spearman": result.pca_fae_spearman,
            "principal_angles_cos_mean": float(result.angle_cosines.mean()),
        }
        metrics["jacobian"][category] = {
            f"{key}_{group}": value
            for group, summary in result.jacobians.items()
            for key, value in (
                ("rank", float(np.median(summary.rank(threshold)))),
                ("projection_error", float(np.median(summary.projection_error))),
            )
        }
    return metrics


def xai_rows(
    bench: CategoryBench, explanations: TestExplanations, selection: Selection, count: int
) -> list[XaiRow]:
    """``count`` аномалій підвибірки з різною площею дефекту (від найменшої до найбільшої)."""
    predictions = bench.assets.predictions
    areas = predictions["defect_area_share"].to_numpy()
    ordered = selection.anomalies[np.argsort(areas[selection.anomalies], kind="stable")]
    picks = ordered[np.unique(np.linspace(0, len(ordered) - 1, count).round().astype(int))]
    return [
        _xai_row(bench, explanations, int(position), _area_title(predictions, position))
        for position in picks
    ]


def _area_title(predictions: pd.DataFrame, position: int) -> str:
    row = predictions.iloc[position]
    return f"площа {100 * row['defect_area_share']:.2f} %\n{row['outcome']}".replace(".", ",")


def _xai_row(
    bench: CategoryBench,
    explanations: TestExplanations,
    position: int,
    title: str,
    caption: str = "",
) -> XaiRow:
    test_row = bench.test_rows[position]
    ig = explanations.ig.get(position)
    return XaiRow(
        title=title,
        image=np.asarray(bench.assets.images[test_row]),
        mask=bench.assets.masks[test_row],
        model_map=bench.test.maps[position] / bench.assets.config.threshold_pixel,
        ig_signed=None if ig is None else ig.signed,
        gradcam=explanations.gradcam[position],
        hirescam=explanations.hirescam[position],
        caption=caption,
    )


def error_rows(
    benches: Sequence[CategoryBench],
    explanations: Mapping[str, TestExplanations],
    cases: pd.DataFrame,
    outcome: str,
    per_category: int,
) -> list[XaiRow]:
    """Найвпевненіші помилки ``outcome`` кожної категорії з підписом причини."""
    rows = []
    for bench in benches:
        frame = cases[cases["category"] == bench.category].reset_index(drop=True)
        for position in panel_positions(cases, bench.category, outcome, per_category):
            row = frame.iloc[position]
            cause = CAUSE_LABELS.get(row["cause"], row["cause"])
            caption = f"оцінка {row['score']:.2f} / поріг {row['threshold']:.2f}".replace(".", ",")
            caption += "\n" + textwrap.fill(f"причина: {cause}", 26)
            title = f"{bench.category}, {outcome}"
            rows.append(_xai_row(bench, explanations[bench.category], position, title, caption))
    return rows


def perturbation_states(
    bench: CategoryBench,
    explanations: TestExplanations,
    selection: Selection,
    steps: Sequence[int],
) -> tuple[list[np.ndarray], list[float]]:
    """Стани аномалії з найбільшим дефектом підвибірки при вилученні top-k патчів за IG.

    Режим — «заміна на норму»; оцінки рахує наскрізний детектор.
    """
    areas = bench.assets.predictions["defect_area_share"].to_numpy()[selection.anomalies]
    position = int(selection.anomalies[int(np.argmax(areas))])
    image = np.asarray(bench.assets.images[bench.test_rows[position]])
    substitute = bench.references.reference(image)
    grid = bench.assets.grid
    order = ranking(patch_importance(explanations.ig[position].map, grid), "morf")
    states = [perturbed(image, substitute, order[:k], grid) for k in steps]
    scores = bench.runner.run(np.stack(states), keep_maps=False).scores
    return states, [float(score) for score in scores]


def tradeoff_points(
    variants: pd.DataFrame, resolution: pd.DataFrame, config: Lab5Config
) -> pd.DataFrame:
    """Точки рисунка «якість ↔ latency»: TFLite-варіанти та роздільності (середнє категорій)."""
    rows = []
    averaged = (
        variants[variants["variant"] != "keras"]
        .groupby("variant", sort=False)
        .mean(numeric_only=True)
    )
    for variant, row in averaged.iterrows():
        rows.append(
            {
                "group": "TFLite, вхід 256",
                "label": VARIANT_LABELS[variant].replace("TFLite ", ""),
                "latency_ms": row["latency_ms_1"],
                "pixel_auroc": row["pixel_auroc"],
            }
        )
    by_size = resolution.groupby("input_size", sort=False).mean(numeric_only=True)
    for size, row in by_size.iterrows():
        if size == max(config.edge_input_sizes):
            continue
        rows.append(
            {
                "group": "TFLite fp32, менший вхід",
                "label": f"{size}×{size}",
                "latency_ms": row["latency_ms_1"],
                "pixel_auroc": row["pixel_auroc"],
            }
        )
    return pd.DataFrame(rows)
