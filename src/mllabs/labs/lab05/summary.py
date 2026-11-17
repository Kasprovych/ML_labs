"""Таблиці для звіту (≤ 7 колонок, українські заголовки) і блоки ``metrics.json``.

Функції лише ПЕРЕТВОРЮЮТЬ уже пораховані результати: перейменовують колонки, зводять
середні, формують словники метрик. Нічого не рахують заново і не пишуть на диск.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict

import numpy as np
import pandas as pd

from mllabs.labs.lab05.config import Lab5Config
from mllabs.labs.lab05.drift import FAMILY_LABELS, first_failing_level
from mllabs.labs.lab05.edge_study import VARIANT_LABELS
from mllabs.labs.lab05.error_cases import CAUSE_LABELS, DEFECT_LABELS, DESCRIPTOR_LABELS
from mllabs.labs.lab05.workbench import CategoryBench
from mllabs.labs.lab05.xai_study import (
    BASELINE_LABELS,
    METHOD_LABELS,
    TestExplanations,
    completeness_table,
)

_RESPONSE_LABELS = {
    "score_rel": "оцінка зображення",
    "error_rel": "середня помилка патчів",
    "mask_error_rel": "помилка в зоні дефекту",
}
_MODE_LABELS = {"replace": "заміна на норму", "noise": "зашумлення"}
#: Знаків після крапки в edge-таблицях звіту: мілісекунди — 1 (точніше вимір не відтворюється
#: між прогонами), мегабайти — 2. Метрики якості лишаються з 4 знаками (формат save_table).
_EDGE_DIGITS = {"Розмір, МБ": 2, "Latency 1 потік, мс": 1, "Latency усі потоки, мс": 1}
_LEVEL_LABELS = {
    "head": "випадкова головка (μ/σ з train)",
    "head_no_norm": "випадкова головка без нормалізації",
    "full": "випадкова вся модель",
    "noise": "шум σ = 2",
}
#: Порожня клітинка таблиці звіту (величина для цього рядка не визначена).
_EMPTY_CELL = "—"


def _interval(value: float, low: float, high: float, digits: int = 3) -> str:
    """«0,951 [0,930; 0,968]» — значення з ДІ одним рядком (десяткова кома)."""
    text = f"{value:.{digits}f} [{low:.{digits}f}; {high:.{digits}f}]"
    return text.replace(".", ",")


def formatted(frame: pd.DataFrame, digits: Mapping[str, int]) -> pd.DataFrame:
    """Копія таблиці, де вказані колонки стали рядками з фіксованою кількістю знаків.

    ``ArtifactStore.save_table`` має один формат дробових чисел на всю таблицю (4 знаки) —
    для часток і метрик якості це доречно, а для кількості кроків («64.0000») чи
    мілісекунд («36.8168») це зайва точність. Колонки з ``digits`` форматуються тут, решта
    лишаються числами. Порожнє значення (NaN) показується як «—».

    Args:
        frame: Таблиця для звіту.
        digits: ``{колонка: знаків після крапки}``; 0 — ціле число. Колонки, яких у
            таблиці немає, пропускаються.
    """

    def text(value: float, places: int) -> str:
        return _EMPTY_CELL if pd.isna(value) else f"{value:.{places}f}"

    # Синтаксис {ключ: значення for ...} — «dict comprehension», короткий запис циклу,
    # що наповнює словник; assign(**словник) додає/замінює колонки й повертає НОВУ таблицю.
    columns = {
        column: frame[column].map(lambda value, places=places: text(value, places))
        for column, places in digits.items()
        if column in frame.columns
    }
    return frame.assign(**columns)


def method_label(method: str) -> str:
    """Підпис методу пояснення."""
    return METHOD_LABELS.get(method, method)


# ---------------------------------------------------------------- ЛР4 / ваги


def handoff_table(benches: Iterable[CategoryBench]) -> pd.DataFrame:
    """Звірка з ЛР4: рядок на категорію."""
    rows = [
        {"category": bench.category}
        | asdict(bench.check)
        | {"threshold_image": bench.assets.config.threshold_image}
        for bench in benches
    ]
    frame = pd.DataFrame(rows)
    # Розбіжності порядку 1e-6 читаються лише в науковому записі; поріг — звичайне число.
    for column in ("max_score_diff", "end_to_end_diff"):
        frame[column] = frame[column].map(lambda value: f"{value:.1e}")
    return frame.rename(
        columns={
            "category": "Категорія",
            "n_val": "Зображень val",
            "n_test": "Зображень test",
            "max_score_diff": "max |Δ| оцінки test (кеш)",
            "end_to_end_diff": "max |Δ| наскрізна модель",
            "threshold_image": "Поріг зображення ЛР4",
        }
    )


def handoff_metrics(benches: Iterable[CategoryBench]) -> dict[str, dict]:
    """``metrics.handoff``: звірка і ключові параметри детектора ЛР4 по категоріях."""
    return {
        bench.category: asdict(bench.check)
        | {
            "threshold_image": bench.assets.config.threshold_image,
            "threshold_pixel": bench.assets.config.threshold_pixel,
            "backbone": bench.assets.config.backbone,
            "patch_size": bench.assets.config.patch_size,
            "layers": list(bench.assets.config.layers),
        }
        for bench in benches
    }


def latent_table(rows: Sequence[dict]) -> pd.DataFrame:
    """Латентний простір і лінійний еталон по категоріях."""
    table = pd.DataFrame(rows).rename(
        columns={
            "category": "Категорія",
            "participation_ratio": "Ефект. розмірність латенту",
            "fae_pixel_auroc": "Pixel AUROC FAE",
            "mahalanobis_pixel_auroc": "Pixel AUROC Махаланобіса",
            "pca_pixel_auroc": "Pixel AUROC PCA-64",
            "fae_pca_spearman": "ρ(FAE, PCA)",
            "angle_cos_mean": "cos головних кутів (сер.)",
        }
    )
    return formatted(table, {"Ефект. розмірність латенту": 1})


def jacobian_table(rows: Sequence[dict]) -> pd.DataFrame:
    """Спектри якобіанів."""
    table = pd.DataFrame(rows).rename(
        columns={
            "category": "Категорія",
            "group": "Патчі",
            "n": "n",
            "rank": "Напрямів σ > 0,5 (медіана)",
            "sigma_max": "σ_max (медіана)",
            "projection_error": "‖J²−J‖/‖J‖ (медіана)",
        }
    )
    return formatted(table, {"Напрямів σ > 0,5 (медіана)": 0})


# ---------------------------------------------------------------- XAI


def baseline_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Бази IG на val."""
    frame = summary.assign(baseline=summary["baseline"].map(lambda kind: BASELINE_LABELS[kind]))
    table = frame.rename(
        columns={
            "baseline": "База",
            "category": "Категорія",
            "baseline_score": "F(b), медіана",
            "delta": "F(x) − F(b), медіана",
            "pass_share": "Частка ε ≤ 0,05",
            "steps": "Кроків m, медіана",
            "auroc": "AUROC |IG|",
        }
    )
    return formatted(table, {"Частка ε ≤ 0,05": 2, "Кроків m, медіана": 0})


def cam_layer_table(table: pd.DataFrame) -> pd.DataFrame:
    """Вибір шару CAM на val: варіант × шар × категорія."""
    wide = table.pivot_table(index=["variant", "layer"], columns="category", values="auroc")
    wide["усі"] = wide.mean(axis=1)
    wide = wide.reset_index().rename(columns={"variant": "Варіант", "layer": "Шар"})
    layers = {"shallow": "крок 8", "deep": "крок 16", "fused": "злиття"}
    wide["Шар"] = wide["Шар"].map(layers)
    wide["Варіант"] = wide["Варіант"].map(method_label)
    return wide


def completeness_frame(
    benches: Sequence[CategoryBench], explanations: Mapping[str, TestExplanations]
) -> pd.DataFrame:
    """``ig_completeness.csv``: усі IG на test (зображення, m, ε, F(x), F(b))."""
    return pd.DataFrame(
        [
            row
            for bench in benches
            for row in completeness_table(
                bench.category,
                explanations[bench.category],
                bench.assets.predictions["image"].to_numpy(),
            )
        ]
    )


def xai_test_metrics(
    cam_layers: Mapping[str, str],
    completeness: pd.DataFrame,
    tolerance: float,
    localization: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    """Частина ``metrics.xai`` з кроку 4: шари CAM, повнота IG, метрики локалізації."""
    errors = completeness["completeness_error"]
    return {
        "gradcam_layer": cam_layers["gradcam"],
        "hirescam_layer": cam_layers["hirescam"],
        "ig_steps_used": {
            "median": float(completeness["steps"].median()),
            "max": int(completeness["steps"].max()),
        },
        "ig_completeness_pass_share": float((errors <= tolerance).mean()),
        "ig_completeness_error_median": float(errors.median()),
        "n_ig_images": len(completeness),
    } | {key: localization_metrics(frame) for key, frame in localization.items()}


def localization_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Метрики локалізації з ДІ (≤ 7 колонок)."""
    return pd.DataFrame(
        {
            "Категорія": summary["category"],
            "Метод": summary["method"].map(method_label),
            "n": summary["n"],
            "Pixel AUROC [95 % ДІ]": [
                _interval(row.pixel_auroc, row.pixel_auroc_low, row.pixel_auroc_high)
                for row in summary.itertuples()
            ],
            "Pointing game": summary["pointing"],
            "Маса в масці / площа": summary["mass_ratio"],
            "ρ з картою моделі": summary["spearman_model"],
        }
    )


def localization_metrics(summary: pd.DataFrame) -> dict[str, dict]:
    """``metrics.xai.localization.<метод>.<кат>``."""
    result: dict[str, dict] = {}
    for row in summary.itertuples():
        result.setdefault(row.method, {})[row.category] = {
            "n": int(row.n),
            "pixel_auroc": row.pixel_auroc,
            "pixel_auroc_ci": [row.pixel_auroc_low, row.pixel_auroc_high],
            "pointing_game": row.pointing,
            "mass_in_mask_ratio": row.mass_ratio,
            "iou_topk": row.iou_topk,
            "spearman_model_map": row.spearman_model,
        }
    return result


# ---------------------------------------------------------------- R3


def perturbation_table(summary: pd.DataFrame, group: str) -> pd.DataFrame:
    """ABPC і Δ проти випадкового (відгук — оцінка зображення)."""
    frame = summary[(summary["group"] == group) & (summary["response"] == "score_rel")]
    return pd.DataFrame(
        {
            "Категорія": frame["category"],
            "Режим": frame["mode"].map(_MODE_LABELS),
            "Метод": frame["method"].map(method_label),
            "AUC MoRF": frame["auc_morf"],
            "AUC випадкового": frame["auc_random"],
            "ABPC [95 % ДІ]": [
                _interval(row.abpc, row.abpc_low, row.abpc_high) for row in frame.itertuples()
            ],
            "Δ проти випадк. [95 % ДІ]": [
                _interval(row.delta_vs_random, row.delta_low, row.delta_high)
                for row in frame.itertuples()
            ],
        }
    )


def perturbation_metrics(summary: pd.DataFrame) -> dict[str, dict]:
    """``metrics.sanity.perturbation.<група>.<режим>.<відгук>.<метод>.<кат>``."""
    result: dict[str, dict] = {}
    for row in summary.itertuples():
        node = result.setdefault(row.group, {}).setdefault(row.mode, {})
        node = node.setdefault(row.response, {}).setdefault(row.method, {})
        node[row.category] = {
            "n": int(row.n),
            "auc_morf": row.auc_morf,
            "auc_lerf": row.auc_lerf,
            "auc_random": row.auc_random,
            "abpc": row.abpc,
            "abpc_ci": [row.abpc_low, row.abpc_high],
            "delta_vs_random": row.delta_vs_random,
            "delta_vs_random_ci": [row.delta_low, row.delta_high],
        }
    return result


def similarity_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Середні ρ і SSIM по рівнях і методах (+ «усі» категорії).

    Для рандомізації додається контроль — Pixel AUROC самого рандомізованого детектора на test
    (для «усі» — середнє по категоріях).
    """
    pooled = pd.concat([frame, frame.assign(category="усі")], ignore_index=True)
    aggregations = {
        "n": ("image", "size"),
        "spearman": ("spearman", "mean"),
        "ssim": ("ssim", "mean"),
    }
    if "detector_pixel_auroc" in frame.columns:
        aggregations["detector_pixel_auroc"] = ("detector_pixel_auroc", "mean")
    grouped = (
        pooled.groupby(["level", "method", "category"], sort=False)
        .agg(**aggregations)
        .reset_index()
    )
    return grouped.assign(
        level=grouped["level"].map(_LEVEL_LABELS), method=grouped["method"].map(method_label)
    ).rename(
        columns={
            "level": "Зміна",
            "method": "Метод",
            "category": "Категорія",
            "spearman": "Spearman ρ",
            "ssim": "SSIM (ранги)",
            "detector_pixel_auroc": "Pixel AUROC детектора",
        }
    )


def similarity_metrics(frame: pd.DataFrame) -> dict[str, dict]:
    """Блок ``metrics.sanity.randomization`` / ``stability``: середні ρ і SSIM.

    Структура: ``{рівень: {метод: {spearman, ssim, spearman_by_category: {кат: ρ},
    ssim_by_category: {кат: SSIM}}, detector_pixel_auroc: {кат: AUROC}}}``.
    """
    grouped = frame.groupby(["level", "method"])[["spearman", "ssim"]].mean()
    result: dict[str, dict] = {}
    for (level, method), row in grouped.iterrows():
        result.setdefault(level, {})[method] = {"spearman": row["spearman"], "ssim": row["ssim"]}
    # Ті самі середні окремо для кожної категорії: висновок тесту рандомізації робиться
    # покатегорійно (лише там, де рандомізований детектор справді втратив локалізацію).
    by_category = frame.groupby(["level", "method", "category"])[["spearman", "ssim"]].mean()
    for (level, method, category), row in by_category.iterrows():
        node = result[level][method]
        node.setdefault("spearman_by_category", {})[category] = row["spearman"]
        node.setdefault("ssim_by_category", {})[category] = row["ssim"]
    if "detector_pixel_auroc" in frame.columns:
        control = frame.groupby(["level", "category"])["detector_pixel_auroc"].first()
        for (level, category), value in control.items():
            result[level].setdefault("detector_pixel_auroc", {})[category] = value
        trained = frame.groupby("category")["trained_pixel_auroc"].first().to_dict()
        result["trained_pixel_auroc"] = trained
    return result


# ---------------------------------------------------------------- R4


def cases_table(cases: pd.DataFrame) -> pd.DataFrame:
    """Повна таблиця зображень test (сирі дані для звіту; колонки — англійською)."""

    def label(cause: str) -> str:
        return CAUSE_LABELS.get(cause, cause)

    return cases.assign(
        cause=cases["cause"].map(label), cause_plan_order=cases["cause_plan_order"].map(label)
    )


def groups_table(groups: pd.DataFrame) -> pd.DataFrame:
    """Порівняння груп (≤ 7 колонок)."""
    labels = DESCRIPTOR_LABELS | DEFECT_LABELS
    return pd.DataFrame(
        {
            "Категорія": groups["category"],
            "Порівняння": groups["comparison"],
            "Дескриптор": groups["descriptor"].map(labels),
            "Медіана 1-ї групи": groups["median_first"],
            "Медіана 2-ї групи": groups["median_second"],
            "r (ранг.-біс.)": groups["r"],
            "p (Голм)": groups["p_holm"],
        }
    )


def causes_table(causes: pd.DataFrame) -> pd.DataFrame:
    """Частоти причин."""
    return pd.DataFrame(
        {
            "Категорія": causes["category"],
            "Помилка": causes["outcome"],
            "Причина (правило)": causes["cause"].map(CAUSE_LABELS),
            "n": causes["n"],
            "Частка": causes["share"],
        }
    )


def defect_type_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Пропуски за типами дефектів."""
    return frame.rename(
        columns={
            "category": "Категорія",
            "defect_type": "Тип дефекту",
            "n": "n",
            "fn": "FN",
            "median_area": "Медіана площі",
            "median_contrast": "Медіана контрасту",
        }
    )


def overlap_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Перетин помилок FAE з PaDiM / PatchCore."""
    return frame.rename(
        columns={
            "category": "Категорія",
            "outcome": "Помилка FAE",
            "n_fae": "n FAE",
            "also_PaDiM": "з них і PaDiM",
            "also_PatchCore": "з них і PatchCore",
            "all_methods": "усі три методи",
        }
    )


def border_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Крайові артефакти."""
    return frame.drop(columns=["corner_area_share"]).rename(
        columns={
            "category": "Категорія",
            "n_fp": "FP",
            "fp_in_band": "Частка argmax FP у смузі",
            "band_area_share": "Площа смуги",
            "fp_in_corners": "Частка argmax FP у кутах",
            "ring1_to_center": "Помилка кільця 1 / центр",
            "ring2_to_center": "Помилка кільця 2 / центр",
        }
    )


def exclusion_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Діагностика «оцінка без рамки»."""
    return frame.drop(columns=["threshold"]).rename(
        columns={
            "category": "Категорія",
            "band": "Смуга, пікс.",
            "image_auroc": "Image AUROC",
            "f1": "Image F1",
            "fpr": "FPR",
            "tpr": "TPR",
            "masks_touching_band": "Масок у смузі",
        }
    )


# ---------------------------------------------------------------- дрейф


def drift_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Дрейф (≤ 7 колонок)."""
    return pd.DataFrame(
        {
            "Категорія": frame["category"],
            "Зміна": frame["family"].map(FAMILY_LABELS),
            "Рівень": frame["value"],
            "Pixel AUROC": frame["pixel_auroc"],
            "Image AUROC": frame["image_auroc"],
            "FPR при порозі ЛР4": frame["fpr"],
            "AUROC монітора дрейфу": frame["monitor_auroc"],
        }
    )


def drift_limits(frame: pd.DataFrame, config: Lab5Config) -> pd.DataFrame:
    """Перший рівень кожної родини, де якість виходить за апріорні допуски."""
    rows = []
    families = list(dict.fromkeys(level.family for level in config.drift_levels))
    for family in families:
        row = {"Зміна": FAMILY_LABELS[family]}
        for category, group in frame.groupby("category", sort=False):
            clean = group[group["family"] == "clean"].iloc[0]
            levels = group[group["family"] == family].sort_values("severity")
            limit = first_failing_level(
                list(zip(levels["value"], levels["pixel_auroc"], levels["fpr"], strict=True)),
                (clean["pixel_auroc"], clean["fpr"]),
                config.drift_auroc_tolerance,
                config.drift_fpr_tolerance,
            )
            row[category] = "не досягнуто" if limit is None else f"{limit:g}"
        rows.append(row)
    return pd.DataFrame(rows)


def drift_metrics(frame: pd.DataFrame, limits: pd.DataFrame) -> dict[str, object]:
    """``metrics.drift``."""
    result: dict[str, object] = {}
    for row in frame.itertuples():
        key = "clean" if row.family == "clean" else f"{row.value:g}"
        result.setdefault(row.family, {}).setdefault(key, {})[row.category] = {
            "pixel_auroc": row.pixel_auroc,
            "pixel_ap": row.pixel_ap,
            "image_auroc": row.image_auroc,
            "fpr": row.fpr,
            "tpr": row.tpr,
            "score_shift": row.score_shift,
            "ks": row.ks,
            "monitor_auroc": row.monitor_auroc,
            "pointing_game": row.pointing,
        }
    label_to_family = {label: family for family, label in FAMILY_LABELS.items()}
    result["limits"] = {
        label_to_family[row["Зміна"]]: {key: value for key, value in row.items() if key != "Зміна"}
        for row in limits.to_dict("records")
    }
    return result


# ---------------------------------------------------------------- edge


def edge_table(frame: pd.DataFrame) -> pd.DataFrame:
    """TFLite-варіанти, середнє по категоріях (≤ 7 колонок).

    «× fp32» — відношення latency (1 потік) до TFLite fp32 ТОГО САМОГО прогону: абсолютні
    мілісекунди залежать від пристрою й навантаження, співвідношення переносяться краще.
    Image F1 за порогом ЛР4 — у ``edge_variants_full.csv``.
    """
    averaged = frame.groupby("variant", sort=False).mean(numeric_only=True).reset_index()
    latency = averaged.get("latency_ms_1", pd.Series(np.nan, index=averaged.index))
    reference = float(latency[averaged["variant"] == "fp32"].iloc[0])
    table = pd.DataFrame(
        {
            "Варіант": averaged["variant"].map(lambda key: VARIANT_LABELS.get(key, key)),
            "Розмір, МБ": averaged["size_mb"],
            "Latency 1 потік, мс": latency,
            "× fp32 (1 потік)": latency / reference,
            "Latency усі потоки, мс": averaged.get("latency_ms_усі", np.nan),
            "Pixel AUROC": averaged["pixel_auroc"],
            "Image F1 (поріг з val)": averaged["image_f1_recalibrated"],
        }
    )
    # Рядок «Keras fp32» не має розміру файлу й latency TFLite — там буде «—».
    return formatted(table, _EDGE_DIGITS | {"× fp32 (1 потік)": 2})


def resolution_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Роздільність входу, середнє по категоріях."""
    averaged = frame.groupby("input_size", sort=False).mean(numeric_only=True).reset_index()
    reference = float(averaged.loc[averaged["input_size"].idxmax(), "latency_ms_1"])
    table = pd.DataFrame(
        {
            "Сторона входу, пікс.": averaged["input_size"],
            "Latency 1 потік, мс": averaged["latency_ms_1"],
            "× найбільший вхід": averaged["latency_ms_1"] / reference,
            "Latency усі потоки, мс": averaged["latency_ms_усі"],
            "Pixel AUROC": averaged["pixel_auroc"],
            "Image AUROC": averaged["image_auroc"],
        }
    )
    return formatted(table, _EDGE_DIGITS | {"× найбільший вхід": 2})


def edge_limits_display(edge: pd.DataFrame) -> pd.DataFrame:
    """``limits_edge.csv``: таблиця :func:`edge_limits_table` з точністю для читання."""
    return formatted(edge, _EDGE_DIGITS)


def lr4_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    """``edge_lr4_summary.csv``: зведення backbone ЛР4 з точністю, як у таблицях ЛР4."""
    digits = {column: 1 for column in frame.columns if column.startswith("Latency")}
    return formatted(frame, digits | {"GFLOPs (backbone + FAE)": 3})


def lr4_metrics(frame: pd.DataFrame) -> dict[str, dict]:
    """``metrics.edge.lr4_backbones``: числа зведення ЛР4 з ключами для звіту."""
    keys = {
        "GFLOPs (backbone + FAE)": "gflops",
        "Latency Keras, 1 потік, мс": "latency_keras_ms_1",
        "Latency Keras, усі потоки, мс": "latency_keras_ms_усі",
        "Pixel AUROC (val, macaroni2)": "pixel_auroc_val_macaroni2",
    }
    renamed = frame.rename(columns=keys).set_index("Backbone")
    return {backbone: row.to_dict() for backbone, row in renamed[list(keys.values())].iterrows()}


def edge_metrics(
    variants: pd.DataFrame, resolution: pd.DataFrame, postprocess: Mapping[str, float]
) -> dict[str, object]:
    """``metrics.edge``.

    Крім чисел «варіант × категорія» зберігаються ті самі зведення, що й у таблиці звіту:
    ``mean`` — середнє по категоріях, ``relative_to_fp32`` — відношення середніх розміру й
    latency до TFLite fp32 того самого прогону, ``postprocess_ms`` — latency самої
    пост-обробки і її частка в latency повної моделі fp32.

    Args:
        variants: Рядки «категорія × варіант» (:meth:`EdgeStudy.variants`).
        resolution: Рядки «категорія × сторона входу» (:meth:`EdgeStudy.resolution`).
        postprocess: Latency пост-обробки (:meth:`EdgeStudy.postprocess_latency`).
    """
    result: dict[str, object] = {}
    reference = variants[variants["variant"] == "keras"].set_index("category")["pixel_auroc"]
    for row in variants.to_dict("records"):
        values = {key: value for key, value in row.items() if key not in ("category", "variant")}
        values["pixel_auroc_delta"] = row["pixel_auroc"] - reference[row["category"]]
        result.setdefault(row["variant"], {})[row["category"]] = values
    resolution_metrics: dict[str, dict] = {}
    for row in resolution.to_dict("records"):
        values = {key: value for key, value in row.items() if key not in ("category", "input_size")}
        resolution_metrics.setdefault(str(row["input_size"]), {})[row["category"]] = values
    result["resolution"] = resolution_metrics
    tflite = variants[variants["variant"] != "keras"].assign(
        pixel_auroc_delta=lambda frame: (
            frame["pixel_auroc"] - frame["category"].map(reference).to_numpy()
        )
    )
    averaged = tflite.groupby("variant", sort=False).mean(numeric_only=True)
    result["mean"] = {variant: row.dropna().to_dict() for variant, row in averaged.iterrows()}
    costs = [column for column in averaged.columns if column.startswith(("latency_ms_", "size_mb"))]
    # Ділення таблиці на рядок (axis="columns"): кожна колонка ділиться на своє значення fp32.
    relative = averaged[costs].div(averaged.loc["fp32", costs], axis="columns")
    result["relative_to_fp32"] = {variant: row.to_dict() for variant, row in relative.iterrows()}
    shares = {
        key.replace("latency_ms_", "share_of_fp32_"): value / averaged.loc["fp32", key]
        for key, value in postprocess.items()
        if key in costs
    }
    result["postprocess_ms"] = dict(postprocess) | shares
    return result


def edge_limits_table(
    variants: pd.DataFrame, resolution: pd.DataFrame, config: Lab5Config
) -> pd.DataFrame:
    """Межі придатності edge-варіантів для реального часу (апріорні правила з ``config``).

    Варіант придатний для бюджету кадру, якщо медіанна latency (1 потік — найгірший
    випадок для слабкого edge-процесора) не перевищує бюджет, а падіння Pixel AUROC
    відносно Keras fp32 — не більше ``quantization_auroc_tolerance``.
    """
    fast, slow = sorted(config.frame_budgets_ms)
    tolerance = config.quantization_auroc_tolerance
    reference = variants[variants["variant"] == "keras"].set_index("category")["pixel_auroc"]
    candidates = []
    for variant, group in variants[variants["variant"] != "keras"].groupby("variant", sort=False):
        delta = (group.set_index("category")["pixel_auroc"] - reference).mean()
        candidates.append(
            (VARIANT_LABELS[variant], group["size_mb"].mean(), group["latency_ms_1"].mean(), delta)
        )
    base = resolution[resolution["input_size"] == max(config.edge_input_sizes)]
    base_auroc = base.set_index("category")["pixel_auroc"]
    for size, group in resolution.groupby("input_size", sort=False):
        if size == max(config.edge_input_sizes):
            continue
        delta = (group.set_index("category")["pixel_auroc"] - base_auroc).mean()
        candidates.append(
            (f"TFLite fp32, вхід {size}×{size}", np.nan, group["latency_ms_1"].mean(), delta)
        )
    rows = []
    for label, size_mb, latency, delta in candidates:
        quality_ok = delta >= -tolerance
        if not quality_ok:
            verdict = "непридатний: втрата якості понад допуск"
        elif latency <= fast:
            verdict = f"придатний для {1000 / fast:.0f} FPS"
        elif latency <= slow:
            verdict = f"лише {1000 / slow:.0f} FPS"
        else:
            verdict = "непридатний для реального часу (1 потік)"
        rows.append(
            {
                "Варіант": label,
                "Розмір, МБ": size_mb,
                "Latency 1 потік, мс": latency,
                "ΔPixel AUROC": delta,
                f"≤ {fast:g} мс": latency <= fast,
                f"≤ {slow:g} мс": latency <= slow,
                "Висновок": verdict,
            }
        )
    return pd.DataFrame(rows)


def limits_table(
    edge: pd.DataFrame, drift_limits: pd.DataFrame, cases: pd.DataFrame, config: Lab5Config
) -> pd.DataFrame:
    """Зведені межі придатності (D9): edge-варіанти, межі дрейфу й частоти помилок.

    Таблиця довга (рядок = умова) і текстова: вона для звіту, числа — у ``limits_edge.csv``,
    ``drift_limits.csv`` і ``error_cases.csv``.
    """
    rows = [
        {
            "Аспект": "edge",
            "Умова": row["Варіант"],
            "Значення": (
                f"{row['Latency 1 потік, мс']:.1f} мс (1 потік); "
                f"ΔPixel AUROC {row['ΔPixel AUROC']:+.4f}"
            ).replace(".", ","),
            "Висновок": row["Висновок"],
        }
        for row in edge.to_dict("records")
    ]
    categories = [column for column in drift_limits.columns if column != "Зміна"]
    for row in drift_limits.to_dict("records"):
        reached = [f"{category}: {row[category]}" for category in categories]
        safe = all(row[category] == "не досягнуто" for category in categories)
        rows.append(
            {
                "Аспект": "дрейф",
                "Умова": row["Зміна"],
                "Значення": "; ".join(reached).replace(".", ","),
                "Висновок": "у межах допусків на всіх рівнях"
                if safe
                else "перший рівень, де якість виходить за допуск",
            }
        )
    for category, frame in cases.groupby("category", sort=False):
        counts = frame["outcome"].value_counts()
        normals, anomalies = int((frame["label"] == 0).sum()), int((frame["label"] == 1).sum())
        fpr = counts.get("FP", 0) / normals
        fnr = counts.get("FN", 0) / anomalies
        rows.append(
            {
                "Аспект": "помилки (test, поріг ЛР4)",
                "Умова": category,
                "Значення": (
                    f"FP {counts.get('FP', 0)}/{normals} (FPR {fpr:.2f}); "
                    f"FN {counts.get('FN', 0)}/{anomalies} (FNR {fnr:.2f})"
                ).replace(".", ","),
                "Висновок": "рішення на рівні зображення ненадійне"
                if max(fpr, fnr) > config.error_rate_unreliable
                else "прийнятно лише з контролем людиною"
                if max(fpr, fnr) > config.error_rate_supervised
                else "прийнятно",
            }
        )
    return pd.DataFrame(rows)


def limits_metrics(table: pd.DataFrame) -> dict[str, dict]:
    """``metrics.limits``: ``{аспект: {умова: {значення, висновок}}}``."""
    result: dict[str, dict] = {}
    for row in table.to_dict("records"):
        result.setdefault(row["Аспект"], {})[row["Умова"]] = {
            "value": row["Значення"],
            "verdict": row["Висновок"],
        }
    return result


def config_metrics(config: Lab5Config, seed: int) -> dict[str, object]:
    """``metrics.config``: усі параметри (StrEnum → рядки, DriftLevel → словники)."""
    values = asdict(config)
    values["ig_baselines"] = [str(kind) for kind in config.ig_baselines]
    values["tflite_variants"] = [str(variant) for variant in config.tflite_variants]
    values["seed"] = seed
    return values


def mean_by_category(frame: pd.DataFrame, column: str) -> Mapping[str, float]:
    """Середнє колонки по категоріях."""
    return frame.groupby("category")[column].mean().to_dict()
