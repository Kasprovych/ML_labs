"""Графіки лабораторної №5.

Кожна функція ПРИЙМАЄ вже пораховані результати й ПОВЕРТАЄ ``Figure``: нічого не рахує
(крім тривіальних середніх для показу) і не зберігає. Ширина — 16 см або 8,5 см, шрифти
≥ 8 pt, ряди розрізняються не лише кольором, а й маркером / типом лінії / штрихуванням
(ч/б друк). Карти пояснень нормовано на їхній максимум (шкала 0–1, «magma»); знакові
IG — розбіжна шкала з нулем посередині.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter

from mllabs.plotting import (
    DIVERGING,
    FULL_WIDTH,
    GRAY,
    HALF_WIDTH,
    INK,
    LIGHT_GRAY,
    ORANGE,
    SERIES_COLORS,
    SERIES_HATCHES,
    SERIES_LINESTYLES,
    SERIES_MARKERS,
    styled,
)

_CM = 1 / 2.54
_MAP_CMAP = "magma"
_MASK_COLOR = "#2ee6d6"  # бірюзовий контур маски — видно на темній і світлій карті
_SMALL = 8
#: σ згладжування знакової IG на панелях (лише для показу, метрики — на своїх картах).
_SIGNED_DISPLAY_SIGMA = 2.0
#: Підписи точок рисунка «якість ↔ latency»: у яких межах (частки розмаху даних по x і по y)
#: сусідня точка праворуч заважає підпису; ближче за третє число точки вважаються однією.
_LABEL_REACH = (0.5, 0.1, 0.02)


def _comma(value: float, digits: int = 2) -> str:
    """Число з десятковою комою."""
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def _decimal_axis(axis: Axes, which: str = "y", digits: int | None = None) -> None:
    """Підписи поділок з десятковою комою."""
    formatter = FuncFormatter(
        lambda value, _: (
            (f"{value:.{digits}f}" if digits is not None else f"{value:g}")
            .replace(".", ",")
            .replace("-", "−")
        )
    )
    (axis.yaxis if which == "y" else axis.xaxis).set_major_formatter(formatter)


def _plain_log(axis: Axes, which: str = "y") -> None:
    """Лог-вісь з підписами «0,1; 1; 10» замість «10⁻¹» і без підписів проміжних поділок."""
    target = axis.yaxis if which == "y" else axis.xaxis
    target.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
    target.set_minor_formatter(FuncFormatter(lambda value, _: ""))


def _hide(axis: Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _contour(axis: Axes, mask: np.ndarray, color: str = _MASK_COLOR) -> None:
    if mask is not None and mask.any():
        axis.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=0.8)


def _unit(image_map: np.ndarray) -> np.ndarray:
    peak = float(np.max(image_map))
    return image_map / peak if peak > 0 else image_map


def _show_map(
    axis: Axes, image_map: np.ndarray, mask: np.ndarray | None, vmax: float = 1.0
) -> AxesImage:
    image = axis.imshow(image_map, cmap=_MAP_CMAP, vmin=0, vmax=vmax)
    _contour(axis, mask)
    _hide(axis)
    return image


def _show_signed(axis: Axes, signed: np.ndarray, mask: np.ndarray | None) -> AxesImage:
    """Знакова IG: легке згладжування лише для показу (поодинокі пікселі інакше не видно)."""
    smooth = gaussian_filter(signed, _SIGNED_DISPLAY_SIGMA)
    limit = float(np.quantile(np.abs(smooth), 0.995)) or 1.0
    image = axis.imshow(smooth, cmap=DIVERGING, vmin=-limit, vmax=limit)
    _contour(axis, mask, INK)
    _hide(axis)
    return image


def _show_image(axis: Axes, image: np.ndarray, mask: np.ndarray | None) -> None:
    axis.imshow(np.asarray(image, np.uint8))
    _contour(axis, mask)
    _hide(axis)


# ---------------------------------------------------------------- R1


@styled
def plot_error_by_layer(table: pd.DataFrame) -> Figure:
    """Частка помилки шару 8 / шару 16 по групах патчів і категоріях (стовпці з накопиченням)."""
    categories = list(dict.fromkeys(table["Категорія"]))
    fig, axes = plt.subplots(1, len(categories), figsize=(FULL_WIDTH, 5.5 * _CM), sharey=True)
    for axis, category in zip(np.atleast_1d(axes), categories, strict=True):
        frame = table[table["Категорія"] == category]
        x = np.arange(len(frame))
        shallow = frame["Частка шару 8"].to_numpy()
        axis.bar(x, shallow, color=SERIES_COLORS[0], label="шар 8 (крок 8, текстура)")
        axis.bar(
            x,
            1 - shallow,
            bottom=shallow,
            color=SERIES_COLORS[1],
            hatch=SERIES_HATCHES[1],
            edgecolor="white",
            label="шар 16 (крок 16, контекст)",
        )
        axis.set_xticks(x, [label.replace(" ", "\n", 1) for label in frame["Група патчів"]])
        axis.set_title(category)
        axis.set_ylim(0, 1)
        _decimal_axis(axis)
    np.atleast_1d(axes)[0].set_ylabel("Частка помилки відновлення")
    fig.legend(
        *np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="outside lower center", ncols=2
    )
    return fig


@dataclass(frozen=True)
class LayerMapsRow:
    """Рядок рисунка карт по шарах.

    Attributes:
        category: Категорія.
        image: Зображення.
        mask: Маска.
        full: Карта моделі.
        shallow: Лише шар 8.
        deep: Лише шар 16.
    """

    category: str
    image: np.ndarray
    mask: np.ndarray
    full: np.ndarray
    shallow: np.ndarray
    deep: np.ndarray


@styled
def plot_layer_error_maps(rows: Sequence[LayerMapsRow]) -> Figure:
    """Зображення, карта моделі й її розклад на внески двох шарів backbone."""
    titles = ("зображення і маска", "карта моделі", "лише шар 8", "лише шар 16")
    fig, axes = plt.subplots(
        len(rows), 4, figsize=(FULL_WIDTH, 4.2 * _CM * len(rows)), squeeze=False
    )
    for line, row in zip(axes, rows, strict=True):
        vmax = float(row.full.max())
        _show_image(line[0], row.image, row.mask)
        line[0].set_ylabel(row.category, fontsize=_SMALL)
        for axis, image_map in zip(line[1:], (row.full, row.shallow, row.deep), strict=True):
            image = _show_map(axis, image_map, row.mask, vmax)
        fig.colorbar(image, ax=line[-1], fraction=0.05, label="помилка")
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title, fontsize=_SMALL)
    return fig


@styled
def plot_top_channels(table: pd.DataFrame) -> Figure:
    """Pixel AUROC test від кількості top-k каналів (порядок каналів обрано на val)."""
    fig, axis = plt.subplots(figsize=(HALF_WIDTH, 6 * _CM))
    for index, (category, frame) in enumerate(table.groupby("Категорія", sort=False)):
        axis.plot(
            frame["k каналів"],
            frame["Pixel AUROC"],
            color=SERIES_COLORS[index],
            linestyle=SERIES_LINESTYLES[index],
            marker=SERIES_MARKERS[index],
            label=category,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(
        sorted(table["k каналів"].unique()), [str(k) for k in sorted(table["k каналів"].unique())]
    )
    axis.set_xlabel("Кількість каналів k (top-k за реакцією на val)")
    axis.set_ylabel("Pixel AUROC (test)")
    _decimal_axis(axis, digits=3)
    axis.legend()
    return fig


@styled
def plot_weight_spectra(spectra: Mapping[str, Sequence[object]]) -> Figure:
    """Сингулярні числа матриць ``Dense`` FAE (лог-шкала), по категоріях."""
    layers = [spectrum.layer for spectrum in next(iter(spectra.values()))]
    fig, axes = plt.subplots(1, len(layers), figsize=(FULL_WIDTH, 5.2 * _CM))
    for column, (axis, layer) in enumerate(zip(axes, layers, strict=True)):
        for index, (category, items) in enumerate(spectra.items()):
            spectrum = items[column]
            axis.semilogy(
                np.arange(1, len(spectrum.singular_values) + 1),
                spectrum.singular_values,
                color=SERIES_COLORS[index],
                linestyle=SERIES_LINESTYLES[index],
                label=f"{category} (еф. ранг {spectrum.effective_rank:.0f})".replace(".", ","),
            )
        axis.set_title(f"{layer}\n{spectrum.shape[0]}×{spectrum.shape[1]}", fontsize=_SMALL)
        axis.set_xlabel("Номер σ")
        _plain_log(axis)
    axes[0].set_ylabel("Сингулярне число σ")
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, list(spectra), loc="outside lower center", ncols=len(spectra))
    return fig


@styled
def plot_latent_pca(scatter: Mapping[str, Mapping[str, np.ndarray]]) -> Figure:
    """PCA-2D латентних кодів: норма train (сірі точки) і дефектні патчі test."""
    fig, axes = plt.subplots(1, len(scatter), figsize=(FULL_WIDTH, 5.5 * _CM))
    for axis, (category, points) in zip(np.atleast_1d(axes), scatter.items(), strict=True):
        train, defect = points["train"], points["defect"]
        axis.scatter(train[:, 0], train[:, 1], s=2, color=LIGHT_GRAY, label="норма (train)")
        axis.scatter(
            defect[:, 0],
            defect[:, 1],
            s=5,
            marker="x",
            color=ORANGE,
            linewidths=0.6,
            label="дефектні патчі (test)",
        )
        axis.set_title(category)
        axis.set_xlabel("ГК 1")
    np.atleast_1d(axes)[0].set_ylabel("ГК 2")
    fig.legend(
        *np.atleast_1d(axes)[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncols=2,
        markerscale=3,
    )
    return fig


@styled
def plot_jacobian_spectrum(
    jacobians: Mapping[str, Mapping[str, object]], threshold: float
) -> Figure:
    """Медіанні спектри якобіана FAE: нормальні vs дефектні патчі."""
    labels = {"normal": "нормальні патчі", "defect": "дефектні патчі"}
    fig, axes = plt.subplots(1, len(jacobians), figsize=(FULL_WIDTH, 5.5 * _CM), sharey=True)
    for axis, (category, groups) in zip(np.atleast_1d(axes), jacobians.items(), strict=True):
        for index, (group, summary) in enumerate(groups.items()):
            values = np.median(summary.singular_values, axis=0)
            axis.semilogy(
                np.arange(1, len(values) + 1),
                values,
                color=SERIES_COLORS[index],
                linestyle=SERIES_LINESTYLES[index],
                label=labels[group],
            )
        axis.axhline(threshold, color=GRAY, linewidth=0.8, linestyle=":")
        axis.set_title(category)
        axis.set_xlabel("Номер σ")
        axis.set_ylim(1e-3, None)
        axis.set_xlim(0, min(len(values), 160))  # далі — лише нулі (ранг ≤ ширини латенту)
        _plain_log(axis)
    np.atleast_1d(axes)[0].set_ylabel("σ якобіана ∂ẑ/∂z (медіана)")
    fig.legend(
        *np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="outside lower center", ncols=2
    )
    return fig


@styled
def plot_linear_vs_fae(table: pd.DataFrame) -> Figure:
    """Pixel AUROC і AP: FAE, Махаланобіс у латенті, PCA-64 по категоріях."""
    maps = ["FAE (повна)", "Махаланобіс у латенті", "PCA-64 (лінійний АЕ)"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 5.5 * _CM))
    categories = list(dict.fromkeys(table["Категорія"]))
    x = np.arange(len(categories))
    width = 0.8 / len(maps)
    for axis, metric in zip(axes, ("Pixel AUROC", "Pixel AP"), strict=True):
        for index, name in enumerate(maps):
            values = table[table["Карта"] == name].set_index("Категорія").loc[categories, metric]
            axis.bar(
                x + (index - 1) * width,
                values,
                width,
                color=SERIES_COLORS[index],
                hatch=SERIES_HATCHES[index],
                edgecolor="white",
                label=name,
            )
        axis.set_xticks(x, categories)
        axis.set_ylabel(f"{metric} (test)")
        _decimal_axis(axis)
    low = table["Pixel AUROC"].min()
    axes[0].set_ylim(max(0.0, low - 0.05), 1.0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3)
    return fig


# ---------------------------------------------------------------- R2


@styled
def plot_ig_baselines(example: object, summary: pd.DataFrame, labels: Mapping[str, str]) -> Figure:
    """Бази IG на одному зображенні val і середні AUROC / повнота по всіх зображеннях val."""
    kinds = list(example.results)
    fig = plt.figure(figsize=(FULL_WIDTH, 11 * _CM))
    grid = fig.add_gridspec(3, len(kinds), height_ratios=(1, 1, 1.1))
    for column, kind in enumerate(kinds):
        result = example.results[kind]
        top = fig.add_subplot(grid[0, column])
        _show_image(top, np.clip(example.baselines[kind], 0, 255), None)
        short = labels[kind].split(" (")[0].replace(" ", "\n", 1)
        top.set_title(f"{short}\nF(b) = {_comma(result.baseline_score)}", fontsize=_SMALL)
        middle = fig.add_subplot(grid[1, column])
        _show_map(middle, _unit(result.map), example.mask)
        middle.set_title(f"|IG|, ε = {_comma(result.completeness_error, 3)}", fontsize=_SMALL)
    pooled = summary[summary["category"] == "усі"].set_index("baseline").loc[kinds]
    bar = fig.add_subplot(grid[2, : len(kinds) // 2 + 1])
    x = np.arange(len(kinds))
    bar.bar(x, pooled["auroc"], color=SERIES_COLORS[0])
    bar.set_xticks(x, [labels[kind].split(" (")[0] for kind in kinds], rotation=20, ha="right")
    bar.set_ylabel("AUROC |IG| (val)")
    bar.set_ylim(max(0.0, pooled["auroc"].min() - 0.1), 1.0)
    _decimal_axis(bar)
    share = fig.add_subplot(grid[2, len(kinds) // 2 + 1 :])
    share.bar(
        x, pooled["pass_share"], color=SERIES_COLORS[1], hatch=SERIES_HATCHES[1], edgecolor="white"
    )
    share.set_xticks(x, [labels[kind].split(" (")[0] for kind in kinds], rotation=20, ha="right")
    share.set_ylabel("Частка ε ≤ 0,05")
    share.set_ylim(0, 1.05)
    _decimal_axis(share)
    fig.suptitle(
        f"Зображення val ({example.category}), F(x) = "
        f"{_comma(example.results[kinds[0]].score)}; контур — маска дефекту",
        fontsize=_SMALL,
    )
    return fig


@dataclass(frozen=True)
class XaiRow:
    """Рядок панелі пояснень.

    Attributes:
        title: Підпис рядка (площа дефекту, оцінка).
        image: Зображення.
        mask: Маска.
        model_map: Карта моделі (у частках порога пікселя).
        ig_signed: Знакова IG.
        gradcam: Grad-CAM.
        hirescam: HiResCAM.
        caption: Підпис під зображенням (оцінка, причина помилки).
    """

    title: str
    image: np.ndarray
    mask: np.ndarray
    model_map: np.ndarray
    ig_signed: np.ndarray | None
    gradcam: np.ndarray
    hirescam: np.ndarray
    caption: str = ""


@styled
def plot_xai_panels(rows: Sequence[XaiRow], cam_titles: Mapping[str, str]) -> Figure:
    """Сітка «зображення / карта моделі / IG / Grad-CAM / HiResCAM» (аномалії, FP, FN)."""
    titles = (
        "зображення і маска",
        "карта моделі / поріг",
        "IG (знак)",
        cam_titles["gradcam"],
        cam_titles["hirescam"],
    )
    fig, axes = plt.subplots(
        len(rows), 5, figsize=(FULL_WIDTH, 3.4 * _CM * len(rows)), squeeze=False
    )
    for line, row in zip(axes, rows, strict=True):
        _show_image(line[0], row.image, row.mask)
        line[0].set_ylabel(row.title, fontsize=_SMALL)
        line[0].set_xlabel(row.caption, fontsize=7)
        _show_map(line[1], row.model_map, row.mask, vmax=2.0)
        if row.ig_signed is None:
            _hide(line[2])
            line[2].text(0.5, 0.5, "IG не рахувався", ha="center", va="center", fontsize=_SMALL)
        else:
            _show_signed(line[2], row.ig_signed, row.mask)
        _show_map(line[3], _unit(row.gradcam), row.mask)
        _show_map(line[4], _unit(row.hirescam), row.mask)
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title, fontsize=_SMALL)
    return fig


@styled
def plot_localization_summary(
    summary: pd.DataFrame, methods: Sequence[str], labels: Mapping[str, str]
) -> Figure:
    """Pixel AUROC, pointing game і маса в масці з 95 % ДІ: методи × категорії."""
    metrics = (
        ("pixel_auroc", "Pixel AUROC"),
        ("pointing", "Pointing game"),
        ("mass_ratio", "Маса в масці / площа"),
    )
    categories = list(dict.fromkeys(summary["category"]))
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 6.5 * _CM))
    x = np.arange(len(categories))
    width = 0.8 / len(methods)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for index, method in enumerate(methods):
            frame = summary[summary["method"] == method].set_index("category").reindex(categories)
            values = frame[metric].to_numpy()
            errors = np.vstack([values - frame[f"{metric}_low"], frame[f"{metric}_high"] - values])
            axis.bar(
                x + (index - (len(methods) - 1) / 2) * width,
                values,
                width,
                yerr=errors,
                capsize=1.5,
                error_kw={"linewidth": 0.6},
                color=SERIES_COLORS[index % 4] if index < 4 else GRAY,
                hatch=SERIES_HATCHES[index % 4],
                edgecolor="white",
                label=labels[method],
            )
        axis.set_xticks(x, categories, rotation=20)
        axis.set_title(title, fontsize=_SMALL)
        _decimal_axis(axis)
    axes[0].set_ylim(0.4, 1.0)
    axes[2].set_yscale("log")
    _plain_log(axes[2])
    axes[2].axhline(1.0, color=GRAY, linewidth=0.6, linestyle=":")
    axes[2].set_ylabel("відношення (лог)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=len(methods))
    return fig


# ---------------------------------------------------------------- R3


@styled
def plot_perturbation_curves(
    curves: pd.DataFrame, methods: Sequence[str], labels: Mapping[str, str]
) -> Figure:
    """MoRF-криві нормованої оцінки зображення: пояснення, оракул, випадковий порядок."""
    frame = curves[curves["group"] == "anomaly"]
    modes = (("replace", "заміна на норму"), ("noise", "зашумлення σ = 32"))
    categories = list(dict.fromkeys(frame["category"]))
    fig, axes = plt.subplots(
        len(modes), len(categories), figsize=(FULL_WIDTH, 10 * _CM), sharex=True, squeeze=False
    )
    styles = {method: index for index, method in enumerate(methods)}
    for row, (mode, mode_label) in enumerate(modes):
        for column, category in enumerate(categories):
            axis = axes[row, column]
            subset = frame[(frame["mode"] == mode) & (frame["category"] == category)]
            for method in (*methods, "random"):
                line = subset[(subset["method"] == method) & (subset["direction"] == "morf")]
                k = line["k"].to_numpy() + 1  # +1: лог-вісь з нулем вилучень
                if method == "random":
                    axis.plot(
                        k,
                        line["score_rel"],
                        color=GRAY,
                        linewidth=1.6,
                        linestyle="-",
                        label=labels[method],
                    )
                    continue
                index = styles[method]
                axis.plot(
                    k,
                    line["score_rel"],
                    color=SERIES_COLORS[index % 4] if index < 4 else INK,
                    linestyle=SERIES_LINESTYLES[index % 4],
                    marker=SERIES_MARKERS[index % 4] if index < 4 else "v",
                    markersize=2.5,
                    linewidth=0.9,
                    label=labels[method],
                )
            axis.set_xscale("log")
            axis.set_xticks([1, 3, 9, 33, 129], ["0", "2", "8", "32", "128"])
            axis.axhline(1.0, color=LIGHT_GRAY, linewidth=0.6)
            if row == 0:
                axis.set_title(category)
            if column == 0:
                axis.set_ylabel(f"{mode_label}\nоцінка / оцінка без змін", fontsize=_SMALL)
            if row == len(modes) - 1:
                axis.set_xlabel("Вилучено патчів (MoRF)")
            _decimal_axis(axis)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=len(methods) + 1
    )
    return fig


@styled
def plot_perturbation_example(
    states: Sequence[np.ndarray], steps: Sequence[int], scores: Sequence[float], threshold: float
) -> Figure:
    """Послідовність станів зображення при вилученні top-k патчів за IG (заміна на норму)."""
    fig, axes = plt.subplots(1, len(states), figsize=(FULL_WIDTH, 4.0 * _CM))
    for axis, state, k, score in zip(axes, states, steps, scores, strict=True):
        _show_image(axis, state, None)
        verdict = "аномалія" if score >= threshold else "норма"
        axis.set_title(f"k = {k}\nоцінка {_comma(score)}\n({verdict})", fontsize=_SMALL)
    return fig


@styled
def plot_sanity(example: object, rows: pd.DataFrame, labels: Mapping[str, str]) -> Figure:
    """Пояснення навченої та рандомізованих моделей; ρ із поясненням навченої.

    Над стовпцем — Pixel AUROC самого детектора (середнє категорій): висновок «пояснення не
    залежить від моделі» має сенс лише для рівнів, де детектор втратив локалізацію.
    """
    levels = (
        ("trained", "навчена\nмодель"),
        ("head", "випадкова головка\n(μ/σ з train)"),
        ("head_no_norm", "випадкова головка\nбез нормалізації"),
        ("full", "випадкова\nвся модель"),
    )
    methods = ("ig", "gradcam", "hirescam")
    fig, axes = plt.subplots(
        len(methods), len(levels) + 1, figsize=(FULL_WIDTH, 11 * _CM), squeeze=False
    )
    means = rows.groupby(["level", "method"])["spearman"].mean()
    control = rows.groupby("level")["detector_pixel_auroc"].mean()
    control["trained"] = rows.groupby("category")["trained_pixel_auroc"].first().mean()
    for line, method in zip(axes, methods, strict=True):
        _show_image(line[0], example.image, example.mask)
        line[0].set_ylabel(labels[method], fontsize=_SMALL)
        for axis, (level, title) in zip(line[1:], levels, strict=True):
            _show_map(axis, _unit(example.maps[(level, method)]), example.mask)
            if level != "trained":
                axis.set_xlabel(f"сер. ρ = {_comma(means[(level, method)])}", fontsize=_SMALL)
            if method == methods[0]:
                axis.set_title(f"{title}\nAUROC дет. {_comma(control[level], 3)}", fontsize=7)
    axes[0, 0].set_title(f"зображення\n({example.category})", fontsize=7)
    return fig


# ---------------------------------------------------------------- R4


@styled
def plot_error_descriptors(cases: pd.DataFrame, descriptors: Mapping[str, str]) -> Figure:
    """Box-plot дескрипторів FP vs TN і FN vs TP (усі категорії; ранги в межах категорії)."""
    pairs = (("FP", "TN"), ("FN", "TP"))
    fig, axes = plt.subplots(
        len(pairs), len(descriptors), figsize=(FULL_WIDTH, 9.5 * _CM), squeeze=False
    )
    for row, (first, second) in enumerate(pairs):
        for column, (descriptor, label) in enumerate(descriptors.items()):
            axis = axes[row, column]
            # Процентильний ранг у межах категорії: категорії мають різні шкали.
            ranks = cases.groupby("category")[descriptor].rank(pct=True)
            data = [ranks[cases["outcome"] == outcome].dropna() for outcome in (first, second)]
            box = axis.boxplot(data, widths=0.6, patch_artist=True, medianprops={"color": INK})
            for patch, color, hatch in zip(
                box["boxes"], (ORANGE, LIGHT_GRAY), ("///", ""), strict=True
            ):
                patch.set_facecolor(color)
                patch.set_hatch(hatch)
            axis.set_xticks(
                [1, 2], [f"{first}\n(n={len(data[0])})", f"{second}\n(n={len(data[1])})"]
            )
            if row == 0:
                axis.set_title(textwrap.fill(label, 18), fontsize=_SMALL)
            axis.set_ylim(0, 1.02)
            _decimal_axis(axis)
        axes[row, 0].set_ylabel("ранг у категорії (0–1)")
    return fig


@styled
def plot_fp_locations(data: Mapping[str, object], border_px: int) -> Figure:
    """Положення argmax карти на FP поверх середнього зображення test; смуга краю кадру."""
    fig, axes = plt.subplots(1, len(data), figsize=(FULL_WIDTH, 5.8 * _CM))
    for axis, (category, item) in zip(np.atleast_1d(axes), data.items(), strict=True):
        axis.imshow(item.mean_image, alpha=0.6)
        side = item.mean_image.shape[0]
        axis.add_patch(
            Rectangle(
                (border_px, border_px),
                side - 2 * border_px,
                side - 2 * border_px,
                fill=False,
                edgecolor=_MASK_COLOR,
                linestyle="--",
                linewidth=0.8,
            )
        )
        if len(item.fp_peaks):
            axis.scatter(
                item.fp_peaks[:, 1],
                item.fp_peaks[:, 0],
                s=14,
                marker="x",
                color=ORANGE,
                linewidths=1.0,
                clip_on=False,
            )
        axis.set_xlim(-0.5, side - 0.5)
        axis.set_ylim(side - 0.5, -0.5)
        axis.set_title(f"{category}: FP = {len(item.fp_peaks)}", fontsize=_SMALL)
        _hide(axis)
    return fig


@styled
def plot_border_artifacts(data: Mapping[str, object]) -> Figure:
    """Середня карта норм (рамка, кути) і профіль «відстань до краю → значення карти»."""
    fig = plt.figure(figsize=(FULL_WIDTH, 10 * _CM))
    grid = fig.add_gridspec(2, len(data), height_ratios=(1.1, 1))
    profile_axis = fig.add_subplot(grid[1, :])
    for column, (category, item) in enumerate(data.items()):
        axis = fig.add_subplot(grid[0, column])
        image = axis.imshow(item.mean_normal_map, cmap=_MAP_CMAP)
        _hide(axis)
        axis.set_title(category, fontsize=_SMALL)
        fig.colorbar(image, ax=axis, fraction=0.046, label="сер. карта норм")
        profile = item.profile
        relative = profile["value"] / profile["value"].iloc[-10:].mean()
        profile_axis.plot(
            profile["distance"],
            relative,
            color=SERIES_COLORS[column],
            linestyle=SERIES_LINESTYLES[column],
            label=category,
        )
    profile_axis.axhline(1.0, color=LIGHT_GRAY, linewidth=0.6)
    profile_axis.set_xlabel("Відстань пікселя до краю кадру, пікс.")
    profile_axis.set_ylabel("Карта норм / центр")
    _decimal_axis(profile_axis)
    profile_axis.legend()
    return fig


@styled
def plot_border_exclusion(frame: pd.DataFrame) -> Figure:
    """Image AUROC і FPR test від ширини виключеної смуги (поріг перекалібровано на val)."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 5.5 * _CM))
    for index, (category, group) in enumerate(frame.groupby("category", sort=False)):
        style = {
            "color": SERIES_COLORS[index],
            "linestyle": SERIES_LINESTYLES[index],
            "marker": SERIES_MARKERS[index],
            "label": category,
        }
        axes[0].plot(group["band"], group["image_auroc"], **style)
        axes[1].plot(group["band"], group["fpr"], **style)
    axes[0].set_ylabel("Image AUROC (test)")
    axes[1].set_ylabel("FPR (test)")
    for axis in axes:
        axis.set_xlabel("Виключена смуга краю, пікс.")
        _decimal_axis(axis)
    axes[0].legend()
    return fig


# ---------------------------------------------------------------- дрейф


@styled
def plot_drift_examples(example: object, families: Mapping[str, str]) -> Figure:
    """Одне зображення під усіма змінами (найсильніший рівень) і його карти."""
    keys = list(example.images)
    fig, axes = plt.subplots(2, len(keys), figsize=(FULL_WIDTH, 5.4 * _CM), squeeze=False)
    vmax = max(float(example.maps[key].max()) for key in keys)
    for column, key in enumerate(keys):
        _show_image(axes[0, column], example.images[key], None)
        axes[0, column].set_title(textwrap.fill(families[key].split(" (")[0], 11), fontsize=7)
        _show_map(axes[1, column], example.maps[key], None, vmax)
    axes[0, 0].set_ylabel("зображення", fontsize=_SMALL)
    axes[1, 0].set_ylabel("карта", fontsize=_SMALL)
    return fig


@styled
def plot_drift_curves(
    frame: pd.DataFrame, families: Sequence[str], labels: Mapping[str, str]
) -> Figure:
    """Pixel AUROC і FPR при порозі ЛР4 від рівня кожної зміни (лінія — категорія)."""
    fig, axes = plt.subplots(
        len(families), 2, figsize=(FULL_WIDTH, 3.4 * _CM * len(families)), squeeze=False
    )
    for row, family in enumerate(families):
        for index, (category, group) in enumerate(frame.groupby("category", sort=False)):
            clean = group[group["family"] == "clean"]
            levels = group[group["family"] == family].sort_values("severity")
            x = np.arange(len(levels) + 1)
            for column, metric in enumerate(("pixel_auroc", "fpr")):
                values = np.r_[clean[metric].to_numpy(), levels[metric].to_numpy()]
                axes[row, column].plot(
                    x,
                    values,
                    color=SERIES_COLORS[index],
                    linestyle=SERIES_LINESTYLES[index],
                    marker=SERIES_MARKERS[index],
                    markersize=3,
                    label=category,
                )
                axes[row, column].set_xticks(
                    x, ["0", *[f"{value:g}".replace(".", ",") for value in levels["value"]]]
                )
        axes[row, 0].set_ylabel(labels[family].replace(" (", "\n("), fontsize=7)
        for axis in axes[row]:
            _decimal_axis(axis)
    axes[0, 0].set_title("Pixel AUROC (test)", fontsize=_SMALL)
    axes[0, 1].set_title("FPR при порозі ЛР4 (test)", fontsize=_SMALL)
    axes[-1, 0].set_xlabel("Рівень зміни (0 — без змін)")
    axes[-1, 1].set_xlabel("Рівень зміни (0 — без змін)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=3)
    return fig


@styled
def plot_drift_monitor(
    frame: pd.DataFrame, families: Sequence[str], labels: Mapping[str, str]
) -> Figure:
    """Найсильніший рівень кожної зміни: зсув медіани оцінок норм і AUROC монітора без міток."""
    strongest = (
        frame[frame["family"] != "clean"]
        .sort_values("severity")
        .groupby(["category", "family"], sort=False)
        .tail(1)
    )
    categories = list(dict.fromkeys(frame["category"]))
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 6.5 * _CM), sharey=True)
    y = np.arange(len(families))
    height = 0.8 / len(categories)
    for index, category in enumerate(categories):
        subset = strongest[strongest["category"] == category].set_index("family").reindex(families)
        style = {
            "color": SERIES_COLORS[index],
            "hatch": SERIES_HATCHES[index],
            "edgecolor": "white",
            "label": category,
        }
        offset = y + (index - (len(categories) - 1) / 2) * height
        axes[0].barh(offset, subset["score_shift"], height, **style)
        axes[1].barh(offset, subset["monitor_auroc"], height, **style)
    axes[0].set_yticks(y, [labels[family].split(" (")[0] for family in families])
    axes[0].axvline(0, color=INK, linewidth=0.6)
    axes[0].set_xlabel("Зсув медіани оцінок норм")
    axes[1].axvline(0.5, color=GRAY, linewidth=0.6, linestyle=":")
    axes[1].set_xlabel("AUROC монітора (медіана помилки кадру)")
    axes[1].set_xlim(0.0, 1.0)
    for axis in axes:
        _decimal_axis(axis, "x")
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3)
    return fig


# ---------------------------------------------------------------- edge


def _crowded_on_right(points: pd.DataFrame, latency: float, auroc: float) -> bool:
    """Чи є праворуч від точки, на близькій висоті, інша точка рисунка «якість ↔ latency».

    «Близько» задано частками розмаху даних (``_LABEL_REACH``): підпис завдовжки з
    «динамічне int8» займає до половини розмаху по горизонталі. Точки, що практично
    збігаються з даною, не рахуються — їхні підписи розводяться по вертикалі.
    """
    x_span = max(float(np.ptp(points["latency_ms"])), 1e-9)  # ptp = max − min («peak to peak»)
    y_span = max(float(np.ptp(points["pixel_auroc"])), 1e-9)
    dx = (points["latency_ms"] - latency) / x_span
    dy = (points["pixel_auroc"] - auroc).abs() / y_span
    reach_x, reach_y, same = _LABEL_REACH
    return bool(((dx > same) & (dx < reach_x) & (dy < reach_y)).any())


@styled
def plot_edge_tradeoff(points: pd.DataFrame, budgets: Sequence[float]) -> Figure:
    """Pixel AUROC проти latency (1 потік): TFLite-варіанти, роздільності; бюджети кадру."""
    fig, axis = plt.subplots(figsize=(FULL_WIDTH, 7 * _CM))
    for index, (group, frame) in enumerate(points.groupby("group", sort=False)):
        axis.scatter(
            frame["latency_ms"],
            frame["pixel_auroc"],
            marker=SERIES_MARKERS[index % 4],
            color=SERIES_COLORS[index % 4],
            s=28,
            label=group,
        )
        for number, row in enumerate(frame.itertuples()):
            # Чергування зсуву по вертикалі: точки fp32 і fp16 майже збігаються. Якщо ж
            # праворуч на тій самій висоті є інша точка, підпис ставиться ЛІВОРУЧ від
            # маркера (ha="right"), щоб не накластися на неї та її підпис.
            leftward = _crowded_on_right(points, row.latency_ms, row.pixel_auroc)
            axis.annotate(
                row.label,
                (row.latency_ms, row.pixel_auroc),
                fontsize=7,
                xytext=(-5 if leftward else 5, 4 if number % 2 == 0 else -10),
                textcoords="offset points",
                ha="right" if leftward else "left",
            )
    for budget in budgets:
        axis.axvline(budget, color=GRAY, linestyle="--", linewidth=0.8)
        axis.text(
            budget, axis.get_ylim()[0], f" {budget:g} мс", fontsize=7, va="bottom", color=GRAY
        )
    axis.set_xlabel("Latency, мс на кадр (1 потік, медіана)")
    axis.set_ylabel("Pixel AUROC (test, середнє 3 категорій)")
    _decimal_axis(axis, digits=3)
    axis.legend(loc="lower right")
    return fig


@styled
def plot_quantization_maps(example: Mapping[str, np.ndarray], labels: Mapping[str, str]) -> Figure:
    """Карти однієї аномалії: Keras і чотири варіанти TFLite (спільна шкала)."""
    keys = [key for key in example if key not in ("image", "mask")]
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(FULL_WIDTH, 3.6 * _CM))
    _show_image(axes[0], example["image"], example["mask"])
    axes[0].set_title("зображення", fontsize=_SMALL)
    vmax = float(example["keras"].max())
    for axis, key in zip(axes[1:], keys, strict=True):
        image = _show_map(axis, example[key], example["mask"], vmax)
        axis.set_title(labels[key].replace("TFLite ", "").replace(" (еталон)", ""), fontsize=7)
    fig.colorbar(image, ax=axes[-1], fraction=0.05, label="оцінка")
    return fig
