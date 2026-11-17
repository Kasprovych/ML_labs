"""Графіки лабораторної №4.

Кожна функція ПРИЙМАЄ вже пораховані результати й ПОВЕРТАЄ ``Figure``; нічого не рахує і
не зберігає. Ширина — 16 см (уся сторінка A4) або 8,5 см; шрифти ≥ 8 pt; ряди
розрізняються не лише кольором, а й маркером / типом лінії / штрихуванням (ч/б друк).
Карти аномалій — у шкалі «оцінка / поріг»: 1 — поріг пікселя, обраний на val.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from mllabs.plotting import (
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

_CM = 1 / 2.54  # сантиметр у дюймах
_SMALL = 8
#: Колірна шкала карт: монотонна яскравість — читається і в ч/б.
_MAP_CMAP = "magma"
#: Межа шкали карт у одиницях порога.
_MAP_VMAX = 2.0


def _comma(value: float, digits: int | None = None) -> str:
    """Число з десятковою комою (українська норма)."""
    text = f"{value:.{digits}f}" if digits is not None else f"{round(value, 6):g}"
    return text.replace(".", ",").replace("-", "−")


def _decimal_comma(axis: Axes, which: str = "y") -> None:
    """Підписи поділок осі з десятковою комою."""
    target = axis.yaxis if which == "y" else axis.xaxis
    target.set_major_formatter(FuncFormatter(lambda value, _: _comma(value)))


def _log_ticks_with_comma(axis: Axes, which: str = "y") -> None:
    """Лог-вісь з простими підписами (0,05; 0,1; 20) замість «2×10⁻¹»."""
    target = axis.yaxis if which == "y" else axis.xaxis
    low, high = axis.get_ylim() if which == "y" else axis.get_xlim()
    candidates = np.array([1, 2, 3, 5])[None, :] * 10.0 ** np.arange(-4, 4)[:, None]
    ticks = [t for t in np.sort(candidates.ravel()) if low <= t <= high]
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(FuncFormatter(lambda value, _: _comma(value)))
    target.set_minor_formatter(NullFormatter())


def _hide_axes(axis: Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _contour(axis: Axes, mask: np.ndarray, color: str = "#ff2a1a", width: float = 1.0) -> None:
    """Контур маски дефекту поверх зображення."""
    if mask.any():
        axis.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=width)


# --------------------------------------------------------------------------------- дані


@dataclass(frozen=True)
class Example:
    """Приклад для сітки зображень.

    Attributes:
        image: ``(S, S, 3)`` ``uint8``.
        mask: ``(S, S)`` ``bool``.
        title: Підпис.
    """

    image: np.ndarray
    mask: np.ndarray
    title: str


@styled
def plot_samples_grid(examples: Mapping[str, Sequence[Example]]) -> Figure:
    """По 2 нормальних і 2 аномальних зображення кожної категорії; контур — маска дефекту."""
    n_rows, n_cols = len(examples), max(len(items) for items in examples.values())
    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=(FULL_WIDTH, 4.1 * _CM * n_rows), squeeze=False
    )
    for row, (category, items) in enumerate(examples.items()):
        for col, example in enumerate(items):
            axis = axes[row, col]
            axis.imshow(example.image)
            _contour(axis, example.mask)
            _hide_axes(axis)
            axis.set_title(example.title, fontsize=_SMALL)
        axes[row, 0].set_ylabel(category, fontsize=9)
    return figure


@styled
def plot_background_variance(table: pd.DataFrame) -> Figure:
    """Дві міри неоднорідності тла по 12 категоріях VisA; обрані — оранжеві зі штрихуванням."""
    figure, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 7 * _CM), sharey=True)
    ordered = table.sort_values("σ між кадрами")
    positions = np.arange(len(ordered))
    colors = [ORANGE if chosen else LIGHT_GRAY for chosen in ordered["Обрана"]]
    hatches = ["///" if chosen else "" for chosen in ordered["Обрана"]]
    for axis, column, label in (
        (axes[0], "σ між кадрами", "σ яскравості між кадрами (0–255)"),
        (axes[1], "σ яскравості в кадрі", "σ яскравості в кадрі, сітка 16×16 (0–255)"),
    ):
        bars = axis.barh(positions, ordered[column], color=colors, edgecolor=INK, linewidth=0.5)
        for bar, hatch in zip(bars, hatches, strict=True):
            bar.set_hatch(hatch)
        axis.set_xlabel(label)
        _decimal_comma(axis, "x")
    axes[0].set_yticks(positions, ordered["Категорія"])
    axes[0].legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor=ORANGE, hatch="///", edgecolor=INK, label="обрана"),
            Rectangle((0, 0), 1, 1, facecolor=LIGHT_GRAY, edgecolor=INK, label="інша"),
        ],
        loc="lower right",
    )
    return figure


@styled
def plot_defect_area_hist(areas: pd.DataFrame) -> Figure:
    """Розподіл частки площі дефекту на аномальних зображеннях (лог-шкала)."""
    figure, axis = plt.subplots(figsize=(HALF_WIDTH, 6.5 * _CM))
    positive = areas[areas["area_share"] > 0]
    bins = np.logspace(
        np.log10(positive["area_share"].min() * 100),
        np.log10(positive["area_share"].max() * 100),
        25,
    )
    for index, (category, frame) in enumerate(positive.groupby("category", sort=False)):
        axis.hist(
            frame["area_share"] * 100,
            bins=bins,
            histtype="step",
            linewidth=1.3,
            linestyle=SERIES_LINESTYLES[index % 4],
            color=SERIES_COLORS[index % 4],
            label=category,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Площа дефекту, % зображення (лог. шкала)")
    axis.set_ylabel("Кількість зображень")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _comma(value)))
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.5)
    return figure


@styled
def plot_patch_grid(image: np.ndarray, grid: int, patch_size: int, tile_grid: int) -> Figure:
    """Сітка патчів ознак, вікно агрегації ``p×p`` і межі тайлів на прикладі зображення."""
    figure, axis = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH))
    side = image.shape[0]
    step = side / grid
    axis.imshow(image)
    for line in np.arange(0, side + 1, step):
        axis.axhline(line - 0.5, color="white", linewidth=0.25, alpha=0.7)
        axis.axvline(line - 0.5, color="white", linewidth=0.25, alpha=0.7)
    centre = grid // 3
    corner = (centre - patch_size // 2) * step - 0.5
    axis.add_patch(
        Rectangle(
            (corner, corner),
            patch_size * step,
            patch_size * step,
            fill=False,
            edgecolor=ORANGE,
            linewidth=1.6,
        )
    )
    axis.add_patch(
        Rectangle((centre * step - 0.5, centre * step - 0.5), step, step, color=ORANGE, alpha=0.8)
    )
    for tile in range(1, tile_grid):
        position = tile * side / tile_grid - 0.5
        axis.axhline(position, color="#39d353", linewidth=1.6, linestyle="--")
        axis.axvline(position, color="#39d353", linewidth=1.6, linestyle="--")
    _hide_axes(axis)
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                color=GRAY,
                linewidth=1.5,
                label=f"патч {int(step)}×{int(step)} пікс. (сітка {grid}×{grid})",
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                fill=False,
                edgecolor=ORANGE,
                linewidth=1.6,
                label=f"вікно агрегації p = {patch_size}",
            ),
            Line2D(
                [],
                [],
                color="#39d353",
                linewidth=1.6,
                linestyle="--",
                label=f"тайли {tile_grid}×{tile_grid}",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=1,
        fontsize=_SMALL,
    )
    return figure


@styled
def plot_model_diagram(
    image_size: int,
    grid: int,
    channels: int,
    feature_dim: int,
    latent: int,
    hidden: int,
    sigma: float,
) -> Figure:
    """Схема конвеєра «зображення → backbone → патчі → FAE → карта» з розмірностями."""
    figure, axis = plt.subplots(figsize=(FULL_WIDTH, 6 * _CM))
    axis.set_xlim(0, 16)
    axis.set_ylim(-0.4, 5.0)
    _hide_axes(axis)
    boxes = [
        ("Зобра-\nження", f"{image_size}×{image_size}×3"),
        ("Backbone\nImageNet\n(замор.)", "шари з\nкроком 8, 16"),
        ("Злиття\nшарів,\nзведення\nканалів", f"{grid}×{grid}×{channels}\n→ ×{feature_dim}"),
        ("Агрегація\np×p", f"{grid}×{grid}×{feature_dim}"),
        ("FAE", f"{feature_dim}→{hidden}→\n{latent}→{hidden}→\n{feature_dim}"),
        ("Помилка\nвідновл.\n(MSE)", f"{grid}×{grid}"),
        ("Збільш.\n+ гаус\nσ = " + _comma(sigma), f"{image_size}×{image_size},\nmax → оцінка"),
    ]
    width, gap, height, base = 1.95, 0.32, 2.5, 1.7
    for index, (title, shape) in enumerate(boxes):
        left = 0.05 + index * (width + gap)
        trainable = title == "FAE"
        axis.add_patch(
            Rectangle(
                (left, base),
                width,
                height,
                facecolor="#f7c9b3" if trainable else "#f3f2ef",
                edgecolor=INK,
                linewidth=1.2 if trainable else 0.8,
                hatch="///" if trainable else "",
            )
        )
        axis.text(
            left + width / 2,
            base + height / 2,
            title,
            ha="center",
            va="center",
            fontsize=_SMALL,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1} if trainable else None,
        )
        axis.text(
            left + width / 2,
            base - 0.2,
            shape,
            ha="center",
            va="top",
            fontsize=_SMALL,
            color="#45443f",
        )
        if index < len(boxes) - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (left + width, base + height / 2),
                    (left + width + gap, base + height / 2),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    color=INK,
                    shrinkA=0,
                    shrinkB=0,
                )
            )
    fae_centre = 0.05 + 4 * (width + gap) + width / 2
    axis.text(
        fae_centre,
        base + height + 0.2,
        "навчається\n(лише норма train)",
        ha="center",
        va="bottom",
        fontsize=_SMALL,
    )
    return figure


# --------------------------------------------------------------------------- дослідження


def _bar_panel(
    axis: Axes,
    labels: Sequence[str],
    values: Sequence[float],
    highlight: int | None,
    ylabel: str,
    digits: int = 3,
) -> None:
    """Стовпці з виділеним (штриховим) обраним варіантом і підписами значень."""
    positions = np.arange(len(labels))
    colors = [ORANGE if index == highlight else SERIES_COLORS[0] for index in positions]
    bars = axis.bar(positions, values, color=colors, edgecolor=INK, linewidth=0.5)
    if highlight is not None:
        bars[highlight].set_hatch("///")
    for bar, value in zip(bars, values, strict=True):
        below = value < 0  # підпис від'ємного стовпця — під його кінцем
        axis.annotate(
            _comma(value, digits),
            (bar.get_x() + bar.get_width() / 2, bar.get_height() if not below else value),
            xytext=(0, -2 if below else 2),
            textcoords="offset points",
            ha="center",
            va="top" if below else "bottom",
            fontsize=_SMALL,
        )
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel(ylabel)
    _decimal_comma(axis)


def _zoom(axis: Axes, values: Sequence[float]) -> None:
    """Вісь y навколо значень (різниці в тисячних інакше не видно); межі вказано на осі."""
    low, high = min(values), max(values)
    margin = max((high - low) * 0.6, 0.005)
    axis.set_ylim(max(low - margin, 0), min(high + margin, 1.0))


@styled
def plot_study_backbones(table: pd.DataFrame, selected: str) -> Figure:
    """Pixel AUROC на val і latency backbone для трьох екстракторів."""
    figure, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 6.5 * _CM))
    labels = table["Варіант"].tolist()
    highlight = labels.index(selected)
    _bar_panel(axes[0], labels, table["Pixel AUROC"].tolist(), highlight, "Pixel AUROC (val)")
    _zoom(axes[0], table["Pixel AUROC"].tolist())
    _bar_panel(
        axes[1],
        labels,
        table["Latency backbone, мс"].tolist(),
        highlight,
        "Latency backbone, мс/зобр. (CPU, пакет 1)",
        digits=1,
    )
    return figure


@styled
def plot_study_two_metrics(
    table: pd.DataFrame,
    selected: str,
    second: str,
    second_label: str,
    second_digits: int = 3,
    second_scale: float = 1.0,
) -> Figure:
    """Pixel AUROC на val і друга метрика по варіантах дослідження."""
    figure, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 7 * _CM))
    labels = table["Варіант"].tolist()
    highlight = labels.index(selected) if selected in labels else None
    _bar_panel(axes[0], labels, table["Pixel AUROC"].tolist(), highlight, "Pixel AUROC (val)")
    _zoom(axes[0], table["Pixel AUROC"].tolist())
    values = (table[second] * second_scale).tolist()  # масштаб, щоб підписи були короткими
    _bar_panel(axes[1], labels, values, highlight, second_label, second_digits)
    if (table[second] < 0).any():
        axes[1].axhline(0, color=INK, linewidth=0.8)
    return figure


@styled
def plot_learning_curves_study(histories: Mapping[str, pd.DataFrame], selected: str) -> Figure:
    """Втрати val_normal і Pixel AUROC на val по епохах для всіх навчань досліджень."""
    figure, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 6.5 * _CM))
    for run_id, history in histories.items():
        is_selected = run_id == selected
        style = {
            "color": ORANGE if is_selected else GRAY,
            "linewidth": 1.8 if is_selected else 0.7,
            "zorder": 3 if is_selected else 2,
        }
        axes[0].plot(history["epoch"], history["val_mse"], **style)
        axes[1].plot(history["epoch"], history["val_pixel_auroc"], **style)
    axes[0].set_yscale("log")
    _log_ticks_with_comma(axes[0])
    axes[0].set_ylabel("Втрата MSE на val_normal (лог.)")
    axes[1].set_ylabel("Pixel AUROC (val, гістогр.)")
    for axis in axes:
        axis.set_xlabel("Епоха")
    _decimal_comma(axes[1])
    axes[1].legend(
        handles=[
            Line2D([], [], color=ORANGE, linewidth=1.8, label="обрана конфігурація"),
            Line2D([], [], color=GRAY, linewidth=0.7, label="інші навчання досліджень"),
        ],
        loc="lower right",
    )
    return figure


@styled
def plot_learning_curves(
    histories: Mapping[str, pd.DataFrame], best_epochs: Mapping[str, int]
) -> Figure:
    """Фінальні моделі: втрати train / val_normal і метрики val по епохах (стовпець — категорія)."""
    figure, axes = plt.subplots(
        2, len(histories), figsize=(FULL_WIDTH, 10 * _CM), squeeze=False, sharex="col"
    )
    for col, (category, history) in enumerate(histories.items()):
        top, bottom = axes[0, col], axes[1, col]
        # Чотири ряди з різними кольором, типом лінії І маркером (читається в ч/б).
        series = (
            (top, "train_mse", "MSE train", SERIES_COLORS[0], "-", "o"),
            (top, "val_mse", "MSE val_normal", SERIES_COLORS[1], "--", "s"),
            (bottom, "val_pixel_auroc", "Pixel AUROC (val)", SERIES_COLORS[3], "-.", "D"),
            (bottom, "val_image_auroc", "Image AUROC (val)", SERIES_COLORS[2], ":", "^"),
        )
        for axis, column, label, color, linestyle, marker in series:
            axis.plot(
                history["epoch"],
                history[column],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=2.5,
                linewidth=1.0,
                label=label,
            )
        for axis in (top, bottom):
            axis.axvline(best_epochs[category], color=INK, linewidth=0.7, linestyle=(0, (1, 2)))
        top.set_title(category, fontsize=9)
        top.set_yscale("log")
        _log_ticks_with_comma(top)
        bottom.set_xlabel("Епоха")
        _decimal_comma(bottom)
    axes[0, 0].set_ylabel("MSE відновлення (лог.)")
    axes[1, 0].set_ylabel("AUROC")
    handles = [
        *axes[0, 0].get_legend_handles_labels()[0],
        *axes[1, 0].get_legend_handles_labels()[0],
    ]
    handles.append(
        Line2D([], [], color=INK, linewidth=0.7, linestyle=(0, (1, 2)), label="найкраща епоха")
    )
    figure.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=_SMALL)
    return figure


# ------------------------------------------------------------------------------ test


@styled
def plot_roc_pr(curves: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]) -> Figure:
    """Піксельні ROC- (верхній ряд) і PR-криві (нижній) на test: категорія × метод."""
    categories = list(curves)
    figure, axes = plt.subplots(2, len(categories), figsize=(FULL_WIDTH, 10.5 * _CM), squeeze=False)
    for col, category in enumerate(categories):
        for index, (method, curve) in enumerate(curves[category].items()):
            style = {
                "color": SERIES_COLORS[index % 4],
                "linestyle": SERIES_LINESTYLES[index % 4],
                "linewidth": 1.2,
            }
            axes[0, col].plot(curve["fpr"], curve["tpr"], label=method, **style)
            axes[1, col].plot(curve["tpr"], curve["precision"], label=method, **style)
        axes[0, col].plot([0, 1], [0, 1], color=LIGHT_GRAY, linewidth=0.7)
        axes[0, col].set_title(category, fontsize=9)
        axes[0, col].set_xlabel("FPR (частка норм. пікселів)")
        axes[1, col].set_xlabel("Повнота (recall)")
        for axis in axes[:, col]:
            _decimal_comma(axis)
            _decimal_comma(axis, "x")
    axes[0, 0].set_ylabel("TPR (повнота)")
    axes[1, 0].set_ylabel("Точність (precision)")
    axes[0, 0].legend(loc="lower right")
    return figure


@dataclass(frozen=True)
class MapExample:
    """Рядок сітки карт: зображення, маска і карти методів у одиницях порога.

    Attributes:
        image: Зображення.
        mask: Маска дефекту.
        maps: Метод → карта, поділена на поріг пікселя цього методу.
        title: Підпис рядка (категорія, тип, рішення FAE).
    """

    image: np.ndarray
    mask: np.ndarray
    maps: Mapping[str, np.ndarray]
    title: str


@styled
def plot_anomaly_maps(examples: Sequence[MapExample]) -> Figure:
    """Сітка: зображення з маскою + карти трьох методів (1 = поріг пікселя з val)."""
    methods = list(examples[0].maps)
    n_rows, n_cols = len(examples), 1 + len(methods)
    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=(FULL_WIDTH, 3.55 * _CM * n_rows + 1.2 * _CM), squeeze=False
    )
    image_handle = None
    for row, example in enumerate(examples):
        axes[row, 0].imshow(example.image)
        _contour(axes[row, 0], example.mask)
        axes[row, 0].set_ylabel(example.title, fontsize=_SMALL)
        for col, method in enumerate(methods, start=1):
            image_handle = axes[row, col].imshow(
                example.maps[method], cmap=_MAP_CMAP, vmin=0, vmax=_MAP_VMAX
            )
            _contour(axes[row, col], example.mask, color="#22e0ff", width=0.8)
            if row == 0:
                axes[row, col].set_title(method, fontsize=9)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
    axes[0, 0].set_title("Зображення і маска", fontsize=9)
    colorbar = figure.colorbar(
        image_handle, ax=axes[-1, :], orientation="horizontal", fraction=0.6, aspect=50, pad=0.04
    )
    colorbar.set_label(
        "Оцінка пікселя / поріг пікселя (1 — межа рішення); контур — справжній дефект"
    )
    colorbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _comma(value)))
    return figure


@styled
def plot_sota_benchmark(table: pd.DataFrame) -> Figure:
    """Pixel AUROC і Image AUROC з 95 % ДІ: три методи × три категорії."""
    figure, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 6.8 * _CM))
    categories = list(dict.fromkeys(table["category"]))
    methods = list(dict.fromkeys(table["method"]))
    width = 0.8 / len(methods)
    for axis, metric, label in (
        (axes[0], "pixel_auroc", "Pixel AUROC (test)"),
        (axes[1], "image_auroc", "Image AUROC (test)"),
    ):
        for index, method in enumerate(methods):
            frame = table[table["method"] == method].set_index("category").loc[categories]
            positions = np.arange(len(categories)) + (index - (len(methods) - 1) / 2) * width
            errors = np.vstack(
                [frame[metric] - frame[f"{metric}_low"], frame[f"{metric}_high"] - frame[metric]]
            )
            axis.bar(
                positions,
                frame[metric],
                width,
                color=SERIES_COLORS[index % 4],
                hatch=SERIES_HATCHES[index % 4],
                edgecolor=INK,
                linewidth=0.5,
                label=method,
                yerr=errors,
                capsize=2,
                error_kw={"linewidth": 0.8},
            )
        axis.set_xticks(np.arange(len(categories)), categories)
        axis.set_ylabel(label)
        low = float((table[f"{metric}_low"]).min())
        axis.set_ylim(max(0.0, low - 0.05), 1.0)
        _decimal_comma(axis)
    axes[1].axhline(0.5, color=GRAY, linewidth=0.7, linestyle=":")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    return figure


@styled
def plot_accuracy_vs_cost(table: pd.DataFrame) -> Figure:
    """Pixel AUROC (val) проти latency і GFLOPs; фронт Парето (за latency) — лінія."""
    figure, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 7 * _CM), sharey=True)
    heads = list(dict.fromkeys(table["Головка"]))
    short = {"ResNet50": "R50", "MobileNetV2": "MNv2", "EfficientNetB0": "EB0"}
    for axis, column, label in (
        (axes[0], "Latency, мс", "Latency, мс/зобр. (CPU, пакет 1, лог.)"),
        (axes[1], "GFLOPs", "GFLOPs на зображення (лог.)"),
    ):
        for index, head in enumerate(heads):
            frame = table[table["Головка"] == head]
            axis.scatter(
                frame[column],
                frame["Pixel AUROC (val)"],
                marker=SERIES_MARKERS[index % 4],
                color=SERIES_COLORS[index % 4],
                s=36,
                label=head,
                zorder=3,
                edgecolors=INK,
                linewidths=0.4,
            )
            for _, row in frame.iterrows():
                axis.annotate(
                    short.get(row["Backbone"], row["Backbone"]),
                    (row[column], row["Pixel AUROC (val)"]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=_SMALL,
                )
        axis.set_xscale("log")
        axis.set_xlabel(label)
        _log_ticks_with_comma(axis, "x")
    for axis, column in ((axes[0], "Latency, мс"), (axes[1], "GFLOPs")):
        front = table[table[f"Парето ({'latency' if axis is axes[0] else 'FLOPs'})"]]
        front = front.sort_values(column)
        axis.step(
            front[column],
            front["Pixel AUROC (val)"],
            where="post",
            color=INK,
            linewidth=0.9,
            linestyle="--",
            label="фронт Парето",
        )
    axes[0].set_ylabel("Pixel AUROC (val, macaroni2)")
    _decimal_comma(axes[0])
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4)
    return figure
