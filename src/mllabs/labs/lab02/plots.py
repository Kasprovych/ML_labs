"""Графіки лабораторної №2.

Кожна функція ПРИЙМАЄ вже пораховані дані й ПОВЕРТАЄ ``Figure``; нічого не
рахує і не зберігає (зберігає ``ArtifactStore``). Розміри — під ширину A4
(16 см) або половину (8,5 см); ряди розрізняються не лише кольором, а й
штрихуванням, типом лінії чи маркером — для чорно-білого друку.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter

from mllabs.labs.lab02 import augmentation_analysis as aug
from mllabs.plotting import (
    FULL_WIDTH,
    GRAY,
    HALF_WIDTH,
    INK,
    SERIES_COLORS,
    SERIES_HATCHES,
    SERIES_LINESTYLES,
    SERIES_MARKERS,
    styled,
)

_CM = 1 / 2.54  # сантиметр у дюймах (matplotlib задає розмір рисунка в дюймах)


def _decimal_comma(axis: Axes, which: str = "y", step: float | None = None) -> None:
    """Підписи осі з десятковою комою (українська норма).

    Args:
        axis: Вісь графіка.
        which: ``"x"`` або ``"y"``.
        step: Крок поділок; ``None`` — як обере matplotlib.
    """
    target = axis.yaxis if which == "y" else axis.xaxis
    if step is not None:
        target.set_major_locator(MultipleLocator(step))
    # Формат %g показує стільки знаків, скільки потрібно (0,25, а не округлене 0,2).
    target.set_major_formatter(
        FuncFormatter(lambda value, _: f"{round(value, 6):g}".replace(".", ","))
    )


@styled
def plot_cv_pr_auc(
    table: pd.DataFrame,
    models: Mapping[str, str],
    scenarios: Mapping[str, str],
    chance_level: float,
) -> Figure:
    """Згруповані стовпці PR-AUC CV (± σ) для моделей і сценаріїв.

    Args:
        table: Колонки ``model``, ``scenario``, ``pr_auc``, ``pr_auc_std``.
        models: Ключ моделі → підпис (у порядку відображення).
        scenarios: Ключ сценарію → підпис (у порядку відображення).
        chance_level: Частка класу 1 — PR-AUC випадкового класифікатора.
    """
    figure, axis = plt.subplots(figsize=(FULL_WIDTH, 6.0 * _CM))
    positions = np.arange(len(models))
    width = 0.8 / len(scenarios)
    indexed = table.set_index(["model", "scenario"])
    for index, (scenario, label) in enumerate(scenarios.items()):
        values = [indexed.loc[(model, scenario), "pr_auc"] for model in models]
        errors = [indexed.loc[(model, scenario), "pr_auc_std"] for model in models]
        bars = axis.bar(
            positions + (index - (len(scenarios) - 1) / 2) * width,
            values,
            width,
            yerr=errors,
            capsize=2,
            ecolor=INK,
            color=SERIES_COLORS[index],
            hatch=SERIES_HATCHES[index],
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.3f}".replace(".", ",") for value in values],
            padding=2,
            fontsize=7,
            rotation=90,
        )
    axis.axhline(
        chance_level,
        color=INK,
        linestyle="--",
        linewidth=0.9,
        label="випадковий класифікатор",
    )
    axis.set_xticks(positions, list(models.values()))
    axis.set_ylabel("PR-AUC (CV, 5 фолдів)")
    axis.set_ylim(0, 0.6)
    axis.grid(axis="x", visible=False)
    _decimal_comma(axis, step=0.1)
    # Легенда під графіком: над високими стовпцями вона перекривала б підписи значень.
    axis.legend(ncols=len(scenarios) + 1, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return figure


@styled
def plot_test_heatmap(matrix: pd.DataFrame) -> Figure:
    """Теплова карта метрик test: рядки — конфігурації, стовпці — метрики.

    Колір нормовано В МЕЖАХ КОЖНОЇ метрики (найгірше значення — світле,
    найкраще — темне): Accuracy ≈ 0,8 і Recall ≈ 0,5 мають різні шкали.

    Args:
        matrix: Індекс — конфігурації; колонки — метрики (при OOF-порозі).
    """
    figure, axis = plt.subplots(figsize=(FULL_WIDTH, 10.0 * _CM))
    values = matrix.to_numpy(dtype=float)
    low, high = values.min(axis=0), values.max(axis=0)
    # Broadcasting: вектори low/high (по одному на колонку) застосовуються до кожного рядка.
    normalized = (values - low) / np.where(high > low, high - low, 1.0)
    axis.imshow(normalized, cmap="Blues", aspect="auto", vmin=-0.15, vmax=1.15)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:.3f}".replace(".", ","),
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if normalized[row, column] > 0.6 else INK,
            )
    axis.set_xticks(range(values.shape[1]), list(matrix.columns))
    axis.set_yticks(range(values.shape[0]), list(matrix.index))
    axis.xaxis.tick_top()
    axis.tick_params(length=0)
    axis.grid(visible=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    # Тонкі лінії між групами моделей (по 4 сценарії на модель).
    n_groups = values.shape[0] // 4
    for group in range(1, n_groups):
        axis.axhline(group * 4 - 0.5, color="white", linewidth=2)
    return figure


def _curve_panels(
    curves: pd.DataFrame,
    models: Mapping[str, str],
    scenarios: Mapping[str, str],
    x_column: str,
    y_column: str,
) -> tuple[Figure, np.ndarray]:
    """Спільна частина графіків кривих: панель на модель, лінія на сценарій."""
    figure, axes = plt.subplots(
        1, len(models), figsize=(FULL_WIDTH, 6.2 * _CM), sharex=True, sharey=True
    )
    for axis, (model, model_label) in zip(axes, models.items(), strict=True):
        for index, (scenario, label) in enumerate(scenarios.items()):
            curve = curves[(curves["model"] == model) & (curves["scenario"] == scenario)]
            axis.plot(
                curve[x_column],
                curve[y_column],
                color=SERIES_COLORS[index],
                linestyle=SERIES_LINESTYLES[index],
                linewidth=1.1,
                label=label,
            )
        axis.set_title(model_label)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal")
        _decimal_comma(axis, "x", step=0.2)
        _decimal_comma(axis, "y", step=0.2)
    return figure, axes


@styled
def plot_test_pr_curves(
    curves: pd.DataFrame,
    models: Mapping[str, str],
    scenarios: Mapping[str, str],
    chance_level: float,
) -> Figure:
    """Криві precision–recall на test: панель на модель, лінія на сценарій.

    Args:
        curves: Колонки ``model``, ``scenario``, ``recall``, ``precision``.
        models: Ключ моделі → підпис.
        scenarios: Ключ сценарію → підпис.
        chance_level: Частка класу 1 на test (precision випадкового класифікатора).
    """
    figure, axes = _curve_panels(curves, models, scenarios, "recall", "precision")
    for axis in axes:
        axis.axhline(chance_level, color=GRAY, linestyle="--", linewidth=0.8)
        axis.set_xlabel("Recall (повнота)")
    axes[0].set_ylabel("Precision (точність)")
    axes[0].legend(loc="upper right", fontsize=7)
    return figure


@styled
def plot_test_roc_curves(
    curves: pd.DataFrame, models: Mapping[str, str], scenarios: Mapping[str, str]
) -> Figure:
    """ROC-криві на test: панель на модель, лінія на сценарій.

    Args:
        curves: Колонки ``model``, ``scenario``, ``fpr``, ``tpr``.
        models: Ключ моделі → підпис.
        scenarios: Ключ сценарію → підпис.
    """
    figure, axes = _curve_panels(curves, models, scenarios, "fpr", "tpr")
    for axis in axes:
        axis.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=0.8)
        axis.set_xlabel("Частка хибних тривог (FPR)")
    axes[0].set_ylabel("Recall (TPR)")
    axes[0].legend(loc="lower right", fontsize=7)
    return figure


@styled
def plot_confusion_matrices(matrices: Mapping[str, np.ndarray], model_label: str) -> Figure:
    """Матриці помилок на test (кількість і частка від справжнього класу).

    Args:
        matrices: Підпис сценарію → матриця 2×2 (рядки — справжній клас).
        model_label: Назва моделі для підпису осі.
    """
    figure, axes = plt.subplots(1, len(matrices), figsize=(FULL_WIDTH * 0.75, 5.2 * _CM))
    for axis, (label, matrix) in zip(np.atleast_1d(axes), matrices.items(), strict=True):
        # keepdims=True залишає вимір, щоб ділення «по рядках» спрацювало через broadcasting.
        shares = matrix / matrix.sum(axis=1, keepdims=True)
        axis.imshow(shares, cmap="Blues", vmin=0, vmax=1.2)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]}\n({shares[row, column] * 100:.1f} %)".replace(".", ","),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if shares[row, column] > 0.6 else INK,
                )
        axis.set_xticks([0, 1], ["0", "1"])
        axis.set_yticks([0, 1], ["0", "1"])
        axis.set_xlabel("Передбачений клас")
        axis.set_title(f"{model_label}, сценарій {label}")
        axis.grid(visible=False)
    np.atleast_1d(axes)[0].set_ylabel("Справжній клас")
    return figure


@styled
def plot_threshold_shift(
    curves: Mapping[str, pd.DataFrame], thresholds: Mapping[str, float], model_label: str
) -> Figure:
    """Precision і recall як функції порога (OOF train) для двох сценаріїв.

    Args:
        curves: Підпис сценарію → таблиця ``threshold``, ``precision``, ``recall``.
        thresholds: Підпис сценарію → поріг з максимальним F1.
        model_label: Назва моделі.
    """
    figure, axes = plt.subplots(1, len(curves), figsize=(FULL_WIDTH, 5.8 * _CM), sharey=True)
    for axis, (label, curve) in zip(axes, curves.items(), strict=True):
        axis.plot(curve["threshold"], curve["precision"], color=SERIES_COLORS[0], label="Precision")
        axis.plot(
            curve["threshold"],
            curve["recall"],
            color=SERIES_COLORS[1],
            linestyle="--",
            label="Recall",
        )
        axis.axvline(0.5, color=GRAY, linestyle=":", linewidth=1)
        axis.axvline(thresholds[label], color=INK, linestyle="-.", linewidth=1)
        axis.annotate(
            "0,5",
            xy=(0.5, 1.0),
            xytext=(2, -2),
            textcoords="offset points",
            fontsize=7,
            va="top",
        )
        axis.annotate(
            f"поріг F1: {thresholds[label]:.2f}".replace(".", ","),
            xy=(thresholds[label], 0.04),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=7,
        )
        axis.set_title(f"{model_label}, сценарій {label}")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Поріг класифікації")
        _decimal_comma(axis, "x", step=0.2)
        _decimal_comma(axis, "y", step=0.2)
    axes[0].set_ylabel("Значення метрики (OOF)")
    axes[0].legend(loc="center right")
    return figure


@styled
def plot_ratio_sweep(frame: pd.DataFrame, models: Mapping[str, str]) -> Figure:
    """PR-AUC CV і розриви train − CV від частоти аугментації SMOTE.

    Args:
        frame: Колонки ``model``, ``ratio``, ``cv_pr_auc``, ``cv_pr_auc_std``,
            ``train_pr_auc``, ``augmented_pr_auc``.
        models: Ключ моделі → підпис.
    """
    figure, (quality, gap) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 6.6 * _CM))
    for index, (model, label) in enumerate(models.items()):
        rows = frame[frame[aug.COLUMN_MODEL] == model].sort_values(aug.COLUMN_RATIO)
        color, marker = SERIES_COLORS[index], SERIES_MARKERS[index]
        quality.errorbar(
            rows[aug.COLUMN_RATIO],
            rows[aug.COLUMN_CV],
            yerr=rows[aug.COLUMN_CV_STD],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.1,
            capsize=2,
            label=label,
        )
        augmented = rows.dropna(subset=[aug.COLUMN_AUGMENTED])
        gap.plot(
            augmented[aug.COLUMN_RATIO],
            augmented[aug.COLUMN_AUGMENTED] - augmented[aug.COLUMN_CV],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.1,
        )
        gap.plot(
            rows[aug.COLUMN_RATIO],
            rows[aug.COLUMN_TRAIN] - rows[aug.COLUMN_CV],
            color=color,
            marker=marker,
            markersize=4,
            markerfacecolor="white",
            linestyle=":",
            linewidth=0.9,
        )
    quality.set_ylabel("PR-AUC (CV) ± σ")
    quality.legend(loc="center right")
    gap.set_ylabel("Розрив PR-AUC: train − CV")
    # Колір лінії вже пояснено легендою лівої панелі; тут легенда пояснює тип лінії.
    # Line2D([], []) — «порожня» лінія лише для легенди (proxy artist).
    gap.legend(
        handles=[
            Line2D([], [], color=GRAY, marker="o", markersize=4, label="train + синтетичні"),
            Line2D(
                [],
                [],
                color=GRAY,
                marker="o",
                markersize=4,
                markerfacecolor="white",
                linestyle=":",
                label="лише реальні train",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncols=2,
    )
    for axis in (quality, gap):
        axis.set_xlabel("Частота SMOTE (клас 1 / клас 0)")
        _decimal_comma(axis, "x", step=0.25)
        _decimal_comma(axis)
    return figure


@styled
def plot_synthetic_memorization(frame: pd.DataFrame, models: Mapping[str, str]) -> Figure:
    """Розподіл ймовірності класу 1 на синтетичних і реальних позитивних рядках.

    Args:
        frame: Колонки ``model``, ``kind``, ``proba`` (сценарій В-SMOTE).
        models: Ключ моделі → підпис.
    """
    kinds = (aug.KIND_SYNTHETIC, aug.KIND_TRAIN_POSITIVE, aug.KIND_VALID_POSITIVE)
    figure, axes = plt.subplots(1, len(models), figsize=(FULL_WIDTH, 6.2 * _CM))
    bins = np.linspace(0, 1, 26)
    for axis, (model, label) in zip(axes, models.items(), strict=True):
        rows = frame[frame[aug.COLUMN_MODEL] == model]
        means = []
        for index, kind in enumerate(kinds):
            values = rows.loc[rows[aug.COLUMN_KIND] == kind, aug.COLUMN_PROBA]
            means.append(f"{values.mean():.2f}".replace(".", ","))
            # weights: кожен рядок має вагу 1/n — стовпці показують частку рядків групи.
            axis.hist(
                values,
                bins=bins,
                weights=np.full(len(values), 1 / len(values)),
                histtype="step",
                color=SERIES_COLORS[index],
                linestyle=SERIES_LINESTYLES[index],
                linewidth=1.3,
                label=kind,
            )
        axis.text(
            0.04,
            0.96,
            "середні:\n" + " / ".join(means),
            transform=axis.transAxes,  # координати в частках панелі, а не в даних
            va="top",
            fontsize=7,
        )
        axis.set_ylim(0, axis.get_ylim()[1] * 1.35)
        axis.set_title(label)
        axis.set_xlabel("Ймовірність класу 1")
        _decimal_comma(axis, "x", step=0.5)
        _decimal_comma(axis)
    axes[0].set_ylabel("Частка рядків групи")
    # Одна спільна легенда під панелями: групи однакові для всіх моделей.
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=len(kinds))
    return figure


@styled
def plot_learning_curves(
    frame: pd.DataFrame, models: Mapping[str, str], scenarios: Mapping[str, str]
) -> Figure:
    """Криві навчання: PR-AUC CV (суцільна) і train (пунктир) від розміру навчальної частини.

    Args:
        frame: Колонки ``model``, ``scenario``, ``train_rows``, ``cv_pr_auc``,
            ``cv_pr_auc_std``, ``train_pr_auc``.
        models: Ключ моделі → підпис.
        scenarios: Ключ сценарію → підпис (два сценарії).
    """
    figure, axes = plt.subplots(1, len(models), figsize=(FULL_WIDTH, 6.4 * _CM), sharey=True)
    sizes = sorted(frame[aug.COLUMN_SIZE].unique())
    for axis, (model, model_label) in zip(axes, models.items(), strict=True):
        for index, (scenario, label) in enumerate(scenarios.items()):
            rows = frame[
                (frame[aug.COLUMN_MODEL] == model) & (frame[aug.COLUMN_SCENARIO] == scenario)
            ].sort_values(aug.COLUMN_SIZE)
            color, marker = SERIES_COLORS[index], SERIES_MARKERS[index]
            axis.errorbar(
                rows[aug.COLUMN_SIZE],
                rows[aug.COLUMN_CV],
                yerr=rows[aug.COLUMN_CV_STD],
                color=color,
                marker=marker,
                markersize=3.5,
                capsize=2,
                linewidth=1.1,
                label=f"{label}: CV",
            )
            axis.plot(
                rows[aug.COLUMN_SIZE],
                rows[aug.COLUMN_TRAIN],
                color=color,
                marker=marker,
                markersize=3.5,
                markerfacecolor="white",
                linestyle="--",
                linewidth=0.9,
                label=f"{label}: train",
            )
        axis.set_xscale("log")
        # Поділки рівно в точках вимірювань, підписи — у тисячах рядків.
        axis.set_xticks(sizes, [f"{size / 1000:.3g}".replace(".", ",") for size in sizes])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.tick_params(axis="x", which="minor", length=0)
        axis.set_title(model_label)
        axis.set_xlabel("Рядків у навчанні, тис.")
        axis.set_ylim(0, 1.05)
        _decimal_comma(axis, "y", step=0.2)
    axes[0].set_ylabel("PR-AUC")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=4)
    return figure


@styled
def plot_fit_time(
    frame: pd.DataFrame, models: Mapping[str, str], scenarios: Mapping[str, str]
) -> Figure:
    """Середній час навчання одного фолда (логарифмічна шкала).

    Args:
        frame: Колонки ``model``, ``scenario``, ``fit_seconds``.
        models: Ключ моделі → підпис.
        scenarios: Ключ сценарію → підпис.
    """
    figure, axis = plt.subplots(figsize=(HALF_WIDTH, 6.0 * _CM))
    positions = np.arange(len(models))
    width = 0.8 / len(scenarios)
    indexed = frame.set_index(["model", "scenario"])
    for index, (scenario, label) in enumerate(scenarios.items()):
        values = [indexed.loc[(model, scenario), "fit_seconds"] for model in models]
        axis.bar(
            positions + (index - (len(scenarios) - 1) / 2) * width,
            values,
            width,
            color=SERIES_COLORS[index],
            hatch=SERIES_HATCHES[index],
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks(positions, list(models.values()))
    axis.set_ylabel("Час навчання фолда, с")
    axis.grid(axis="x", visible=False)
    figure.legend(ncols=2, loc="outside lower center")
    return figure
