"""Графіки лабораторної №1.

Кожна функція ПРИЙМАЄ вже пораховані дані й ПОВЕРТАЄ ``Figure``; нічого не
рахує і не зберігає (зберігає ``ArtifactStore``). Графіки розраховані на
вставку у звіт на всю ширину сторінки A4 і на чорно-білий друк: ряди
відрізняються не лише кольором, а й штрихуванням або формою маркера.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy import stats

from mllabs.labs.lab01.augmentation import POSITIVE_SHARE_COLUMN
from mllabs.plotting import (
    BLUE,
    DIVERGING,
    FULL_WIDTH,
    GRAY,
    GREEN,
    HALF_WIDTH,
    INK,
    LIGHT_GRAY,
    ORANGE,
    SERIES_COLORS,
    SERIES_HATCHES,
    styled,
)


@styled
def plot_target_distribution(class_balance: pd.DataFrame) -> Figure:
    """Стовпчикова діаграма розподілу класів цільової змінної.

    Args:
        class_balance: Таблиця «Клас», «Кількість», «Частка, %».
    """
    figure, axis = plt.subplots(figsize=(HALF_WIDTH, 2.6))
    labels = [f"{int(value)}" for value in class_balance["Клас"]]
    bars = axis.bar(labels, class_balance["Кількість"], color=BLUE, width=0.55)
    captions = [
        f"{count} ({share:.1f} %)"
        for count, share in zip(class_balance["Кількість"], class_balance["Частка, %"], strict=True)
    ]
    axis.bar_label(bars, labels=captions, padding=3)
    axis.set_xlabel("Клас IsBadBuy (0 — вдала покупка, 1 — «погана»)")
    axis.set_ylabel("Кількість авто")
    axis.grid(axis="x", visible=False)
    axis.margins(y=0.12)
    return figure


@styled
def plot_missing_values(missing_report: pd.DataFrame, drop_threshold: float) -> Figure:
    """Точкова діаграма часток явних і прихованих пропусків по ознаках.

    Шкала логарифмічна: частки відрізняються на чотири порядки (0,01 % – 95 %).

    Args:
        missing_report: Таблиця «Ознака», «Явні», «Приховані», «Частка, %».
        drop_threshold: Поріг частки пропусків (0..1), вище якого колонка видаляється.
    """
    report = missing_report.iloc[::-1]  # зріз [::-1] — зворотний порядок: найбільші зверху
    total = report["Явні"] + report["Приховані"]
    # Кількість рядків датасету відновлюється з кількості та частки пропусків.
    n_rows = (total / (report["Частка, %"] / 100)).max()
    explicit_share = (report["Явні"] / n_rows * 100).where(report["Явні"] > 0)
    hidden_share = (report["Приховані"] / n_rows * 100).where(report["Приховані"] > 0)

    figure, axis = plt.subplots(figsize=(FULL_WIDTH, 0.24 * len(report) + 1.2))
    axis.scatter(explicit_share, report["Ознака"], s=42, color=BLUE, label="явні (NaN)", zorder=3)
    axis.scatter(
        hidden_share,
        report["Ознака"],
        s=42,
        marker="s",
        facecolors="white",
        edgecolors=ORANGE,
        linewidths=1.6,
        label="приховані (ціна ≤ 1, NOT AVAIL)",
        zorder=3,
    )
    axis.axvline(drop_threshold * 100, color=INK, linestyle="--", linewidth=1)
    axis.annotate(
        f"поріг видалення {drop_threshold:.0%}",
        xy=(drop_threshold * 100, 0),
        xytext=(-4, 2),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )
    axis.set_xscale("log")
    axis.set_xlabel("Частка пропусків у колонці, % (логарифмічна шкала)")
    axis.set_ylabel("Ознака")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    return figure


@styled
def plot_correlation_heatmap(correlation: pd.DataFrame, coefficient_name: str) -> Figure:
    """Теплова карта матриці кореляцій із підписаними значеннями.

    Args:
        correlation: Квадратна матриця кореляцій.
        coefficient_name: Назва коефіцієнта в родовому відмінку («Пірсона»).
    """
    figure, axis = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.86))
    sns.heatmap(
        correlation,
        ax=axis,
        cmap=DIVERGING,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6.5},
        linewidths=0.6,
        linecolor="white",
        square=True,
        cbar_kws={"label": f"Коефіцієнт кореляції {coefficient_name}", "shrink": 0.8},
    )
    axis.grid(visible=False)
    axis.tick_params(axis="x", rotation=45)
    plt.setp(axis.get_xticklabels(), ha="right", rotation_mode="anchor")
    return figure


@styled
def plot_numeric_distributions(numeric: pd.DataFrame, n_columns: int = 4) -> Figure:
    """Сітка гістограм числових ознак із кривою щільності (KDE).

    Args:
        numeric: Числові ознаки (колонки вже з короткими підписами).
        n_columns: Кількість графіків у рядку сітки.
    """
    figure, axes = _grid(len(numeric.columns), n_columns, row_height=1.55)
    for axis, name in zip(axes, numeric.columns, strict=False):
        values = numeric[name].dropna()
        is_discrete = values.nunique() <= 12
        sns.histplot(
            values,
            ax=axis,
            bins=30 if not is_discrete else None,
            discrete=is_discrete,
            kde=not is_discrete,
            color=BLUE,
            edgecolor="white",
            linewidth=0.3,
            line_kws={"linewidth": 1.2},
        )
        _panel(axis, name)
    figure.supxlabel("Значення ознаки", fontsize=9)
    figure.supylabel("Кількість авто", fontsize=9)
    return figure


@styled
def plot_qq(numeric_sample: pd.DataFrame, n_columns: int = 2) -> Figure:
    """Сітка Q-Q графіків: квантилі вибірки проти квантилів нормального розподілу.

    Якщо ознака нормальна, точки лежать на прямій; вигин на краях означає
    важкі хвости або скошеність. Рисунок половинної ширини (8,5 см): у звіт
    він іде на ~55 % ширини сторінки, тому панелей мало (6–8 ознак).

    Args:
        numeric_sample: Підвибірка ознак, які треба показати.
        n_columns: Кількість графіків у рядку сітки.
    """
    figure, axes = _grid(len(numeric_sample.columns), n_columns, row_height=1.6, width=HALF_WIDTH)
    for axis, name in zip(axes, numeric_sample.columns, strict=False):
        values = numeric_sample[name].dropna()
        standardized = (values - values.mean()) / values.std()
        # probplot повертає пари квантилів і параметри прямої найкращого наближення.
        (theoretical, observed), (slope, intercept, _) = stats.probplot(standardized, dist="norm")
        axis.scatter(theoretical, observed, s=3, color=BLUE, rasterized=True)
        axis.plot(theoretical, slope * theoretical + intercept, color=INK, linewidth=1)
        _panel(axis, name)
    figure.supxlabel("Теоретичні квантилі нормального розподілу", fontsize=9)
    figure.supylabel("Квантилі вибірки (стандартизовані значення)", fontsize=9)
    return figure


@styled
def plot_outlier_overlap(overlap: pd.DataFrame) -> Figure:
    """Теплова карта коефіцієнтів Жаккара між множинами викидів різних методів."""
    figure, axis = plt.subplots(figsize=(FULL_WIDTH * 0.72, FULL_WIDTH * 0.55))
    sns.heatmap(
        overlap,
        ax=axis,
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.6,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Коефіцієнт Жаккара", "shrink": 0.85},
    )
    axis.grid(visible=False)
    axis.tick_params(axis="x", rotation=20)
    axis.tick_params(axis="y", rotation=0)
    return figure


@styled
def plot_outliers_pca(
    projection: pd.DataFrame, masks: pd.DataFrame, explained_variance: tuple[float, float]
) -> Figure:
    """Викиди кожного методу на площині двох головних компонент.

    Args:
        projection: Координати ``pc1``, ``pc2`` підвибірки рядків.
        masks: Булева таблиця «рядок × метод» (індекс сумісний із ``projection``).
        explained_variance: Частки дисперсії, пояснені компонентами.
    """
    methods = list(masks.columns)
    figure, axes = plt.subplots(
        2, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.85), sharex=True, sharey=True
    )
    for axis, method in zip(axes.flat, methods, strict=False):
        is_outlier = masks.loc[projection.index, method]
        normal, outliers = projection[~is_outlier], projection[is_outlier]
        axis.scatter(normal["pc1"], normal["pc2"], s=4, color=GRAY, alpha=0.5, label="звичайні")
        axis.scatter(
            outliers["pc1"],
            outliers["pc2"],
            s=16,
            marker="x",
            color=ORANGE,
            linewidths=0.9,
            label="викиди",
        )
        _panel(axis, f"{method}: {is_outlier.mean():.1%} точок")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=2, markerscale=1.8)
    figure.supxlabel(f"Головна компонента 1 ({explained_variance[0]:.0%} дисперсії)", fontsize=9)
    figure.supylabel(f"Головна компонента 2 ({explained_variance[1]:.0%} дисперсії)", fontsize=9)
    return figure


@styled
def plot_imputation_rmse(rmse_by_feature: pd.DataFrame) -> Figure:
    """Згруповані стовпчики RMSE відновлення прихованих значень по ознаках.

    Args:
        rmse_by_feature: Рядки — ознаки, колонки — методи імп'ютації.
    """
    table = rmse_by_feature.iloc[::-1]
    figure, axis = plt.subplots(figsize=(FULL_WIDTH, 0.42 * len(table) + 1.0))
    _grouped_horizontal_bars(axis, table)
    axis.set_xlabel("RMSE відновлення, у стандартних відхиленнях ознаки (менше — краще)")
    axis.set_ylabel("Ознака")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=len(table.columns))
    return figure


@styled
def plot_imputation_distributions(restored: pd.DataFrame, features: Sequence[str]) -> Figure:
    """Розподіл справжніх прихованих значень і значень, якими їх замінив кожен метод.

    Args:
        restored: Довга таблиця ``feature``, ``method``, ``true``, ``imputed``.
        features: Які ознаки показати (по одній на панель).
    """
    figure, axes = plt.subplots(1, len(features), figsize=(FULL_WIDTH, 2.7))
    line_styles = {"KNN": (ORANGE, "-"), "MICE": (GREEN, "--")}
    for axis, feature in zip(axes, features, strict=True):
        rows = restored[restored["feature"] == feature]
        first_method = rows["method"].iloc[0]
        truth = rows.loc[rows["method"] == first_method, "true"]
        bins = np.histogram_bin_edges(truth, bins=30)
        axis.hist(truth, bins=bins, color=LIGHT_GRAY, label="справжні значення")
        for method, (color, line_style) in line_styles.items():
            imputed = rows.loc[rows["method"] == method, "imputed"]
            axis.hist(
                imputed,
                bins=bins,
                histtype="step",
                color=color,
                linestyle=line_style,
                linewidth=1.4,
                label=method,
            )
        # Медіана замінює всі пропуски одним числом — це не розподіл, а одна лінія.
        median_value = rows.loc[rows["method"] == "Медіана", "imputed"].iloc[0]
        axis.axvline(median_value, color=BLUE, linestyle=":", linewidth=1.8, label="Медіана")
        _panel(axis, feature)
        axis.tick_params(axis="x", rotation=30)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=4)
    figure.supxlabel("Значення ознаки (вихідні одиниці)", fontsize=9)
    figure.supylabel("Кількість прихованих значень", fontsize=9)
    return figure


@styled
def plot_encoding_comparison(comparison: pd.DataFrame) -> Figure:
    """Якість проксі-моделі та розмір матриці ознак для кожного кодування.

    Дві величини різного масштабу — дві панелі зі спільною віссю методів.

    Args:
        comparison: Таблиця «Кодування», «Ознак», «PR-AUC», «σ PR-AUC».
    """
    table = comparison.iloc[::-1]
    figure, (quality_axis, size_axis) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4), sharey=True)
    quality_axis.errorbar(
        table["PR-AUC"],
        table["Кодування"],
        xerr=table["σ PR-AUC"],
        fmt="o",
        color=BLUE,
        ecolor=INK,
        elinewidth=1,
        capsize=3,
        markersize=6,
    )
    for value, name in zip(table["PR-AUC"], table["Кодування"], strict=True):
        quality_axis.annotate(
            f"{value:.3f}", (value, name), xytext=(0, 7), textcoords="offset points", ha="center"
        )
    quality_axis.set_xlabel("PR-AUC проксі-моделі\n(середнє ± σ за фолдами)")
    quality_axis.set_ylabel("Кодування")
    quality_axis.margins(x=0.15, y=0.25)

    bars = size_axis.barh(table["Кодування"], table["Ознак"], color=BLUE, height=0.55)
    size_axis.bar_label(bars, padding=3)
    size_axis.set_xlabel("Кількість ознак\nпісля кодування")
    size_axis.grid(axis="y", visible=False)
    size_axis.margins(x=0.12)
    return figure


@styled
def plot_scaling_before_after(
    transformed: Mapping[str, pd.DataFrame],
    skewness: pd.DataFrame,
    features: Sequence[str],
    methods: Sequence[str],
) -> Figure:
    """Гістограми ознак до і після перетворень: рядок — ознака, колонка — метод.

    Args:
        transformed: Словник «метод → перетворена таблиця».
        skewness: Асиметрія: рядки — ознаки, колонки — методи.
        features: Які ознаки показати.
        methods: Які методи показати (у порядку колонок).
    """
    figure, axes = plt.subplots(
        len(features),
        len(methods),
        figsize=(HALF_WIDTH, 1.5 * len(features) + 0.7),
        squeeze=False,
    )
    for row, feature in enumerate(features):  # enumerate дає пари (номер, елемент)
        for column, method in enumerate(methods):
            axis = axes[row, column]
            axis.hist(transformed[method][feature], bins=30, color=BLUE)
            axis.set_title(f"асим. {skewness.loc[feature, method]:.2f}", fontsize=8, pad=2)
            # На вузькій панелі лишаються 3 позначки по x і жодної по y (форма
            # розподілу видна й без чисел кількостей).
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.xaxis.set_major_formatter(FuncFormatter(_compact_number))
            axis.set_yticks([])
            axis.tick_params(labelsize=8)
            axis.grid(axis="x", visible=False)
            if row == 0:
                axis.annotate(
                    method,
                    xy=(0.5, 1.25),
                    xycoords="axes fraction",
                    ha="center",
                    fontsize=8.5,
                    fontweight="bold",
                )
            if column == 0:
                axis.set_ylabel(feature)
    figure.supxlabel("Значення ознаки після перетворення (k — тисячі)", fontsize=9)
    figure.supylabel("Кількість авто", fontsize=9)
    return figure


@styled
def plot_augmentation_class_balance(class_balance: pd.DataFrame) -> Figure:
    """Склад навчальної вибірки до і після кожного методу аугментації.

    Args:
        class_balance: Таблиця «Метод», «Клас 0», «Клас 1», «Синтетичних» і
            частка позитивного класу (``POSITIVE_SHARE_COLUMN``).
    """
    table = class_balance.iloc[::-1]
    real_minority = table["Клас 1"] - table["Синтетичних"]
    segments = (
        ("клас 0 (реальні)", table["Клас 0"], LIGHT_GRAY, ""),
        ("клас 1 (реальні)", real_minority, BLUE, ""),
        ("клас 1 (синтетичні)", table["Синтетичних"], ORANGE, "///"),
    )
    figure, axis = plt.subplots(figsize=(FULL_WIDTH, 2.6))
    left = np.zeros(len(table))
    for label, values, color, hatch in segments:
        axis.barh(
            table["Метод"],
            values,
            left=left,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=1,
            height=0.6,
            label=label,
        )
        left += values.to_numpy()
    for total, share, method in zip(
        table["Рядків"], table[POSITIVE_SHARE_COLUMN], table["Метод"], strict=True
    ):
        axis.annotate(
            f"клас 1: {share:.1f} %",
            (total, method),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    axis.set_xlabel("Кількість рядків навчальної вибірки")
    axis.set_ylabel("Метод аугментації")
    axis.grid(axis="y", visible=False)
    axis.margins(x=0.16)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    return figure


@styled
def plot_augmentation_pca(
    projection: pd.DataFrame,
    explained_variance: tuple[float, float],
    kinds: Sequence[str],
) -> Figure:
    """Реальні й синтетичні точки на площині PCA — по панелі на метод.

    Args:
        projection: Таблиця ``method``, ``kind``, ``pc1``, ``pc2``.
        explained_variance: Частки дисперсії, пояснені компонентами.
        kinds: Назви груп точок у порядку: мажоритарний, міноритарний, синтетичні.
    """
    majority, minority, synthetic = kinds
    styles = {
        majority: {"s": 4, "color": GRAY, "alpha": 0.45, "marker": "o"},
        minority: {"s": 7, "color": BLUE, "alpha": 0.7, "marker": "o"},
        synthetic: {"s": 12, "color": ORANGE, "alpha": 0.7, "marker": "+", "linewidths": 0.7},
    }
    methods = list(projection["method"].unique())
    figure, axes = plt.subplots(
        2, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.85), sharex=True, sharey=True
    )
    for axis, method in zip(axes.flat, methods, strict=False):
        rows = projection[projection["method"] == method]
        for kind, style in styles.items():
            points = rows[rows["kind"] == kind]
            # **style розпаковує словник в іменовані аргументи: s=..., color=...
            axis.scatter(points["pc1"], points["pc2"], label=kind, rasterized=True, **style)
        _panel(axis, method)
    # Легенда береться з панелі, де є всі три групи (на першій синтетичних немає).
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=3, markerscale=2.2)
    figure.supxlabel(f"Головна компонента 1 ({explained_variance[0]:.0%} дисперсії)", fontsize=9)
    figure.supylabel(f"Головна компонента 2 ({explained_variance[1]:.0%} дисперсії)", fontsize=9)
    return figure


@styled
def plot_feature_mutual_information(
    mutual_information: pd.DataFrame, original_group: str, top: int = 15
) -> Figure:
    """Взаємна інформація ознак із ціллю: нові ознаки поруч із вихідними.

    Args:
        mutual_information: Таблиця «Ознака», «Група», «MI», відсортована за спаданням MI.
        original_group: Назва групи вихідних ознак у колонці «Група».
        top: Скільки найінформативніших ознак показати (рисунок половинної
            ширини; більше рядків на 8,5 см не вміщується читабельно).
    """
    table = mutual_information.head(top).iloc[::-1]
    is_new = (table["Група"] != original_group).to_numpy()
    positions = np.arange(len(table))
    figure, axis = plt.subplots(figsize=(HALF_WIDTH, 0.24 * len(table) + 1.2))
    for mask, color, hatch, label in (
        (~is_new, LIGHT_GRAY, "", "вихідна ознака"),
        (is_new, BLUE, "///", "згенерована ознака"),
    ):
        axis.barh(
            positions[mask],
            table["MI"].to_numpy()[mask],
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.8,
            height=0.7,
            label=label,
        )
    axis.set_yticks(positions, table["Ознака"])
    axis.set_ylim(-0.6, len(table) - 0.4)
    axis.set_xlabel("Взаємна інформація з ціллю, нат")
    axis.set_ylabel("")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.grid(axis="y", visible=False)
    # Легенда над областю графіка: на вузькому рисунку всередині вона закрила б стовпчики.
    figure.legend(loc="outside upper center", ncols=2)
    return figure


def _grid(
    n_panels: int, n_columns: int, row_height: float, width: float = FULL_WIDTH
) -> tuple[Figure, list[Axes]]:
    """Створює сітку панелей заданої ширини й ховає зайві клітинки."""
    n_rows = -(-n_panels // n_columns)  # ділення з округленням угору (як Math.ceil)
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(width, row_height * n_rows + 0.5), squeeze=False
    )
    flat = list(axes.flat)
    for unused in flat[n_panels:]:
        unused.set_visible(False)
    return figure, flat[:n_panels]


def _compact_number(value: float, _position: int) -> str:
    """Підпис позначки осі: тисячі — як «15k», щоб на вузькій панелі не злипалися."""
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def _panel(axis: Axes, title: str) -> None:
    """Оформлює одну панель сітки: назва над панеллю, без власних підписів осей."""
    axis.set_title(title, fontsize=8, pad=3)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(labelsize=8)


def _grouped_horizontal_bars(axis: Axes, table: pd.DataFrame) -> None:
    """Малює згруповані горизонтальні стовпчики: рядок таблиці — група, колонка — ряд."""
    n_series = len(table.columns)
    bar_height = 0.8 / n_series
    positions = np.arange(len(table))
    for index, series_name in enumerate(table.columns):
        # Перший ряд — угорі групи, тому зсув зменшується зі зростанням номера ряду.
        offset = (n_series - 1) / 2 - index
        axis.barh(
            positions + offset * bar_height,
            table[series_name],
            height=bar_height,
            color=SERIES_COLORS[index],
            hatch=SERIES_HATCHES[index],
            edgecolor="white",
            linewidth=0.6,
            label=series_name,
        )
    axis.set_yticks(positions, table.index)
    axis.grid(axis="y", visible=False)
