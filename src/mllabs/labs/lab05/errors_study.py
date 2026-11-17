"""Сценарій аналізу FP/FN (R4): дескриптори, порівняння груп, причини, крайові артефакти.

Пороги правил причин обчислюються НЕ з помилок test: «глобальна помилка» — з норм val,
«сила країв» — з норм val; лише межі «дрібний дефект» і «низький контраст» — квантилі
аномалій test (описова характеристика самої вибірки, а не налаштування моделі).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from mllabs.labs.lab05.config import Lab5Config
from mllabs.labs.lab05.error_cases import (
    DEFECT_LABELS,
    DESCRIPTOR_LABELS,
    CauseRules,
    DescriptorCalculator,
    DescriptorSettings,
    band_shares,
    border_exclusion,
    compare_groups,
    edge_distance_profile,
    ring_errors,
)
from mllabs.labs.lab05.workbench import CategoryBench

#: Результати класифікації зображення (матриця помилок).
_OUTCOMES = ("FP", "FN", "TP", "TN")
#: Колонки таблиці перетину помилок, що не є лічильниками.
_OVERLAP_KEYS = ("category", "outcome")


@dataclass(frozen=True)
class BorderData:
    """Дані для рисунків крайових артефактів однієї категорії.

    Attributes:
        mean_normal_map: Середня карта норм val + test.
        profile: Значення карти від відстані до краю.
        mean_image: Середнє зображення test (тло для положень argmax).
        fp_peaks: ``(n, 2)`` положення argmax FP (рядок, стовпець).
    """

    mean_normal_map: np.ndarray
    profile: pd.DataFrame
    mean_image: np.ndarray
    fp_peaks: np.ndarray


@dataclass(frozen=True)
class ErrorAnalysis:
    """Результати R4.

    Attributes:
        cases: Усі зображення test з дескрипторами, результатом і причиною.
        groups: Порівняння FP↔TN, FN↔TP.
        causes: Частоти причин (порядок правил з config).
        causes_plan_order: Те саме за порядком правил FP з плану (таблиця чутливості).
        defect_types: Пропуски за типами дефектів.
        overlap: Чи помиляються PaDiM/PatchCore на тих самих зображеннях.
        border: Крайові артефакти по категоріях.
        exclusion: Діагностика «оцінка без рамки».
        border_data: Дані рисунків по категоріях.
        rules: Правила причин по категоріях.
    """

    cases: pd.DataFrame
    groups: pd.DataFrame
    causes: pd.DataFrame
    causes_plan_order: pd.DataFrame
    defect_types: pd.DataFrame
    overlap: pd.DataFrame
    border: pd.DataFrame
    exclusion: pd.DataFrame
    border_data: dict[str, BorderData]
    rules: dict[str, CauseRules]


class ErrorStudy:
    """Аналіз помилок детектора на test.

    Args:
        config: Параметри ЛР5.
    """

    def __init__(self, config: Lab5Config) -> None:
        self._config = config
        self._calculator = DescriptorCalculator(
            DescriptorSettings(
                window=config.descriptor_window,
                saturation_level=config.saturation_level,
                shadow_level=config.shadow_level,
                brightness_cells=config.brightness_cells,
                ring_px=config.contrast_ring_px,
                pointing_px=config.pointing_tolerance_px,
                border_px=config.border_px,
            )
        )

    def run(self, benches: Sequence[CategoryBench]) -> ErrorAnalysis:
        """Повний аналіз для всіх категорій."""
        frames, rules, border_rows, exclusion_rows, border_data = [], {}, [], [], {}
        for bench in benches:
            cases, category_rules = self._cases(bench)
            frames.append(cases)
            rules[bench.category] = category_rules
            row, data = self._border(bench, cases)
            border_rows.append(row)
            border_data[bench.category] = data
            exclusion_rows += self._exclusion(bench)
        cases = pd.concat(frames, ignore_index=True)
        return ErrorAnalysis(
            cases=cases,
            groups=self._groups(cases),
            causes=self._causes(cases, "cause"),
            causes_plan_order=self._causes(cases, "cause_plan_order"),
            defect_types=self._defect_types(cases),
            overlap=self._overlap(benches),
            border=pd.DataFrame(border_rows),
            exclusion=pd.DataFrame(exclusion_rows),
            border_data=border_data,
            rules=rules,
        )

    def _cases(self, bench: CategoryBench) -> tuple[pd.DataFrame, CauseRules]:
        config = self._config
        assets = bench.assets
        test_rows, val_rows = bench.test_rows, bench.val_rows
        test = self._calculator.compute(
            np.asarray(assets.images[test_rows]),
            bench.test.maps,
            bench.test.patch_scores,
            assets.masks[test_rows],
        )
        val_normal = assets.labels(val_rows) == 0
        val = self._calculator.compute(
            np.asarray(assets.images[val_rows[val_normal]]),
            bench.val.maps[val_normal],
            bench.val.patch_scores[val_normal],
            assets.masks[val_rows[val_normal]],
        )
        predictions = assets.predictions
        cases = pd.concat(
            [
                predictions[
                    ["category", "image", "label", "defect_types", "score", "threshold", "outcome"]
                ].reset_index(drop=True),
                test,
            ],
            axis=1,
        )
        anomalies = cases[cases["label"] == 1]
        rules = CauseRules(
            # Медіанна помилка НОРМ val (не train: на train помилка занижена — FAE їх бачила).
            global_error=float(np.quantile(val["global_error"], config.global_error_quantile)),
            glare_share=config.glare_share,
            edge_strength=float(np.quantile(val["edge_strength"], config.edge_quantile)),
            border_px=config.border_px,
            small_area=float(np.quantile(anomalies["defect_area"], config.small_area_quantile)),
            low_contrast=float(
                np.quantile(anomalies["defect_contrast"], config.low_contrast_quantile)
            ),
            off_map_auroc=config.off_map_auroc,
            fp_order=config.fp_cause_order,
            fn_order=config.fn_cause_order,
        )
        plan_rules = replace(rules, fp_order=config.fp_cause_order_plan)
        return (
            cases.assign(cause=rules.assign(cases), cause_plan_order=plan_rules.assign(cases)),
            rules,
        )

    def _groups(self, cases: pd.DataFrame) -> pd.DataFrame:
        frames = []
        scopes = [(category, frame) for category, frame in cases.groupby("category", sort=False)]
        scopes.append(("усі", cases))
        for category, frame in scopes:
            for first, second, descriptors in (
                ("FP", "TN", list(DESCRIPTOR_LABELS)),
                ("FN", "TP", [*DESCRIPTOR_LABELS, *DEFECT_LABELS]),
            ):
                table = compare_groups(
                    frame, descriptors, first, second, self._config.min_group_size
                )
                frames.append(table.assign(category=category))
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _causes(cases: pd.DataFrame, column: str) -> pd.DataFrame:
        """Частоти причин за колонкою ``column`` (порядок правил з config або з плану)."""
        errors = cases[cases["outcome"].isin(["FP", "FN"])].rename(columns={column: "cause_key"})
        counts = (
            errors.groupby(["category", "outcome", "cause_key"])
            .size()
            .rename("n")
            .reset_index()
            .rename(columns={"cause_key": "cause"})
        )
        totals = counts.groupby(["category", "outcome"])["n"].transform("sum")
        return counts.assign(share=counts["n"] / totals)

    @staticmethod
    def _defect_types(cases: pd.DataFrame) -> pd.DataFrame:
        anomalies = cases[cases["label"] == 1].assign(
            defect_type=lambda frame: frame["defect_types"].str.split(",")
        )
        # explode: рядок з кількома типами дефекту → по рядку на тип.
        exploded = anomalies.explode("defect_type")
        exploded["defect_type"] = exploded["defect_type"].str.strip()
        return (
            exploded.groupby(["category", "defect_type"])
            .agg(
                n=("image", "size"),
                fn=("outcome", lambda outcome: int((outcome == "FN").sum())),
                median_area=("defect_area", "median"),
                median_contrast=("defect_contrast", "median"),
            )
            .reset_index()
        )

    def _overlap(self, benches: Sequence[CategoryBench]) -> pd.DataFrame:
        rows = []
        for bench in benches:
            table = bench.assets.all_predictions.pivot_table(
                index="image", columns="method", values="outcome", aggfunc="first"
            )
            fae = table[self._config.prediction_method]
            for outcome in ("FP", "FN"):
                selected = table[fae == outcome]
                others = [m for m in table.columns if m != self._config.prediction_method]
                row = {"category": bench.category, "outcome": outcome, "n_fae": len(selected)}
                for method in others:
                    row[f"also_{method}"] = int((selected[method] == outcome).sum())
                row["all_methods"] = int((selected[others] == outcome).all(axis=1).sum())
                rows.append(row)
        return pd.DataFrame(rows)

    def _border(self, bench: CategoryBench, cases: pd.DataFrame) -> tuple[dict, BorderData]:
        config = self._config
        val_normal = bench.assets.labels(bench.val_rows) == 0
        test_normal = cases["label"].to_numpy() == 0
        normal_maps = np.concatenate([bench.val.maps[val_normal], bench.test.maps[test_normal]])
        mean_map = normal_maps.mean(axis=0)
        side = mean_map.shape[0]
        fp = cases[cases["outcome"] == "FP"]
        band_area, corner_area = band_shares(side, config.border_px)
        rings = ring_errors(bench.train.patch_scores)
        row = {
            "category": bench.category,
            "n_fp": len(fp),
            "fp_in_band": float((fp["border_distance"] < config.border_px).mean())
            if len(fp)
            else np.nan,
            "band_area_share": band_area,
            "fp_in_corners": float((fp["border_edges"] >= 2).mean()) if len(fp) else np.nan,
            "corner_area_share": corner_area,
            "ring1_to_center": rings["ring1"] / rings["center"],
            "ring2_to_center": rings["ring2"] / rings["center"],
        }
        mean_image = np.asarray(bench.assets.images[bench.test_rows], np.float32).mean(axis=0)
        data = BorderData(
            mean_normal_map=mean_map,
            profile=edge_distance_profile(mean_map, side // 2 - 1),
            mean_image=mean_image.astype(np.uint8),
            fp_peaks=fp[["peak_row", "peak_col"]].to_numpy(),
        )
        return row, data

    def _exclusion(self, bench: CategoryBench) -> list[dict]:
        val_labels = bench.assets.labels(bench.val_rows)
        test_labels = bench.assets.labels(bench.test_rows)
        test_masks = bench.assets.masks[bench.test_rows]
        return [
            {"category": bench.category}
            | asdict(
                border_exclusion(
                    bench.val.maps, val_labels, bench.test.maps, test_labels, test_masks, band
                )
            )
            for band in self._config.border_exclusion_px
        ]


def significant_descriptors(groups: pd.DataFrame, alpha: float) -> list[dict]:
    """Дескриптори з p_Holm < α (для ``metrics.errors.significant_descriptors``)."""
    significant = groups[groups["p_holm"] < alpha]
    return [
        {
            "category": row.category,
            "comparison": row.comparison,
            "descriptor": row.descriptor,
            "r": row.r,
            "p_holm": row.p_holm,
        }
        for row in significant.itertuples()
    ]


def group_effects(groups: pd.DataFrame) -> dict[str, dict]:
    """Розміри ефекту порівнянь груп: ``{категорія: {fp_vs_tn | fn_vs_tp: {дескриптор: …}}}``.

    На відміну від :func:`significant_descriptors` (список) це словник з адресованими
    ключами — звіт може послатися на конкретне ``r``. Порівняння, для яких тест не робився
    (група менша за ``min_group_size``), пропускаються.
    """
    result: dict[str, dict] = {}
    for row in groups[groups["r"].notna()].itertuples():
        comparison = row.comparison.lower().replace(" ", "_")  # «FP vs TN» → «fp_vs_tn»
        node = result.setdefault(row.category, {}).setdefault(comparison, {})
        node[row.descriptor] = {
            "r": row.r,
            "p_holm": row.p_holm,
            "n_first": int(row.n_first),
            "n_second": int(row.n_second),
        }
    return result


def overlap_metrics(overlap: pd.DataFrame) -> dict[str, dict]:
    """Перетин помилок з іншими методами ЛР4: ``{категорія: {FP | FN: {лічильники}}}``."""
    result: dict[str, dict] = {}
    for row in overlap.to_dict("records"):
        counts = {key: int(value) for key, value in row.items() if key not in _OVERLAP_KEYS}
        result.setdefault(row["category"], {})[row["outcome"]] = counts
    return result


def error_metrics(
    analysis: ErrorAnalysis, categories: Sequence[str], alpha: float
) -> dict[str, object]:
    """Блок ``metrics.errors``."""
    result: dict[str, object] = {}
    cases, border = analysis.cases, analysis.border.set_index("category")
    overlap = overlap_metrics(analysis.overlap)
    for category in categories:
        frame = cases[cases["category"] == category]
        counts = frame["outcome"].value_counts()
        causes = analysis.causes[analysis.causes["category"] == category]
        plan = analysis.causes_plan_order[analysis.causes_plan_order["category"] == category]

        def shares(outcome: str, table: pd.DataFrame = causes) -> dict[str, float]:
            selected = table[table["outcome"] == outcome]
            return dict(zip(selected["cause"], selected["share"], strict=True))

        def cause_counts(outcome: str, table: pd.DataFrame = causes) -> dict[str, int]:
            selected = table[table["outcome"] == outcome]
            return {
                cause: int(n) for cause, n in zip(selected["cause"], selected["n"], strict=True)
            }

        result[category] = {
            "n_fp": int(counts.get("FP", 0)),
            "n_fn": int(counts.get("FN", 0)),
            "n_tp": int(counts.get("TP", 0)),
            "n_tn": int(counts.get("TN", 0)),
            "fp_cause_shares": shares("FP"),
            "fn_cause_shares": shares("FN"),
            "fp_cause_shares_plan_order": shares("FP", plan),
            "fp_cause_counts": cause_counts("FP"),
            "fn_cause_counts": cause_counts("FN"),
            "fp_cause_counts_plan_order": cause_counts("FP", plan),
            "overlap": overlap.get(category, {}),
            "fp_argmax_border_share": border.loc[category, "fp_in_band"],
            "border_area_share": border.loc[category, "band_area_share"],
            "fp_argmax_corner_share": border.loc[category, "fp_in_corners"],
            "corner_area_share": border.loc[category, "corner_area_share"],
            "ring_to_center_error_ratio": border.loc[category, "ring1_to_center"],
        }
    exclusion: dict[str, dict] = {}
    for row in analysis.exclusion.itertuples():
        exclusion.setdefault(str(row.band), {})[row.category] = {
            "image_auroc": row.image_auroc,
            "f1": row.f1,
            "fpr": row.fpr,
            "tpr": row.tpr,
            "masks_touching_band": row.masks_touching_band,
            "threshold": row.threshold,
        }
    result["border_exclusion"] = exclusion
    totals = cases["outcome"].value_counts()
    result["total"] = {f"n_{key.lower()}": int(totals.get(key, 0)) for key in _OUTCOMES}
    result["group_effects"] = group_effects(analysis.groups)
    result["significant_descriptors"] = significant_descriptors(analysis.groups, alpha)
    return result


def panel_positions(cases: pd.DataFrame, category: str, outcome: str, count: int) -> list[int]:
    """Позиції test найвпевненіших помилок (за |оцінка − поріг| / поріг)."""
    frame = cases[cases["category"] == category].reset_index(drop=True)
    margin = ((frame["score"] - frame["threshold"]) / frame["threshold"]).abs()
    selected = frame.index[frame["outcome"] == outcome]
    ordered = margin.loc[selected].sort_values(ascending=False)
    return [int(position) for position in ordered.index[:count]]


def cause_label_map(rules: Mapping[str, CauseRules]) -> dict[str, dict]:
    """Пороги правил по категоріях (для metrics і звіту)."""
    return {category: asdict(rule) for category, rule in rules.items()}
