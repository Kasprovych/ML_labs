"""Графіки лабораторної №3.

Кожна функція ПРИЙМАЄ вже пораховані результати й ПОВЕРТАЄ ``Figure``; нічого
не рахує і не зберігає (зберігає ``ArtifactStore``). Ширина — 16 см (уся
сторінка A4) або 8,5 см; шрифти ≥ 8 pt; ряди розрізняються не лише кольором,
а й маркером / типом лінії / штрихуванням — для чорно-білого друку.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from mllabs.labs.lab03.baseline import BaselineResult
from mllabs.labs.lab03.leakage import (
    VARIANT_LABELS,
    HoldoutLeakageResult,
    LeakageResult,
    LeakageVariant,
)
from mllabs.labs.lab03.search import SearchOutcome
from mllabs.labs.lab03.search_space import SearchSpace
from mllabs.labs.lab03.summary import BASELINE_LABEL, MethodReport, compare_leakage
from mllabs.modeling import mean_and_std
from mllabs.plotting import (
    FULL_WIDTH,
    GRAY,
    INK,
    LIGHT_GRAY,
    SERIES_COLORS,
    SERIES_HATCHES,
    SERIES_LINESTYLES,
    SERIES_MARKERS,
    styled,
)

_CM = 1 / 2.54  # сантиметр у дюймах (matplotlib задає розмір рисунка в дюймах)
_ANNOTATION_SIZE = 8


def _comma(value: float, digits: int | None = None) -> str:
    """Число з десятковою комою (українська норма)."""
    text = f"{value:.{digits}f}" if digits is not None else f"{round(value, 6):g}"
    return text.replace(".", ",").replace("-", "\u2212")  # типографський мінус


def _decimal_comma(axis: Axes, which: str = "y") -> None:
    """Підписи поділок осі з десятковою комою."""
    target = axis.yaxis if which == "y" else axis.xaxis
    target.set_major_formatter(FuncFormatter(lambda value, _: _comma(value)))


@styled
def plot_quality_vs_time(baseline: BaselineResult, reports: Sequence[MethodReport]) -> Figure:
    """«Якість моделі / час обчислень» (R20).

    x — реальний час самостійного пошуку (лог-шкала); y — PR-AUC. Порожній
    маркер — Best Score (Inner CV), заповнений — Nested CV ± σ; пунктир зі
    смугою — baseline на тих самих зовнішніх фолдах.
    """
    figure, axis = plt.subplots(figsize=(FULL_WIDTH, 7.5 * _CM))
    baseline_mean, baseline_std = baseline.outer_mean_std
    axis.axhspan(
        baseline_mean - baseline_std, baseline_mean + baseline_std, color=LIGHT_GRAY, alpha=0.6
    )
    axis.axhline(baseline_mean, color=INK, linestyle="--", linewidth=0.9)
    for index, report in enumerate(reports):
        outcome = report.outcome
        nested_mean, nested_std = report.nested.mean_std
        color, marker = SERIES_COLORS[index % 4], SERIES_MARKERS[index % 4]
        # Невеликий зсув по x, щоб порожній і заповнений маркери не накладалися.
        x_inner, x_nested = outcome.wall_seconds / 1.04, outcome.wall_seconds * 1.04
        axis.plot([x_inner, x_nested], [outcome.best_score, nested_mean], color=GRAY, lw=0.7)
        axis.scatter(
            x_inner,
            outcome.best_score,
            s=40,
            marker=marker,
            facecolors="white",
            edgecolors=color,
            linewidths=1.3,
            zorder=3,
        )
        axis.errorbar(
            x_nested,
            nested_mean,
            yerr=nested_std,
            fmt=marker,
            color=color,
            markersize=6,
            capsize=3,
            zorder=3,
        )
        axis.annotate(
            f"{outcome.n_candidates} / {outcome.n_fits}",
            (outcome.wall_seconds, max(outcome.best_score, nested_mean + nested_std)),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=_ANNOTATION_SIZE,
        )
    axis.set_xscale("log")
    # Поділки — фактичні часи методів: стандартні підписи лог-осі мали б вигляд «2×10¹».
    times = sorted(report.outcome.wall_seconds for report in reports)
    axis.set_xticks(times, [_comma(value, 0) for value in times])
    axis.minorticks_off()
    axis.set_xlabel("Час самостійного пошуку, с (логарифмічна шкала)")
    axis.set_ylabel("PR-AUC")
    _decimal_comma(axis)
    method_handles = [
        Line2D(
            [],
            [],
            marker=SERIES_MARKERS[i % 4],
            color=SERIES_COLORS[i % 4],
            linestyle="",
            label=report.label,
        )
        for i, report in enumerate(reports)
    ]
    kind_handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=INK,
            markerfacecolor="white",
            linestyle="",
            label="Best Score (Inner CV)",
        ),
        Line2D([], [], marker="o", color=INK, linestyle="", label="Nested CV ± σ"),
        Line2D([], [], color=INK, linestyle="--", label="Baseline (5 фолдів) ± σ"),
    ]
    axis.legend(
        handles=method_handles + kind_handles,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
    )
    return figure


@styled
def plot_search_convergence(outcomes: Sequence[SearchOutcome], optuna_startup: int) -> Figure:
    """Крива збіжності: найкращий внутрішній PR-AUC серед перших k оцінок (P5).

    Для Grid порядок оцінок — порядок ``ParameterGrid`` (лексикографічний), тож
    його крива сходинкова й залежить від порядку параметрів; порівнювати слід
    кінцеві точки. Вертикальна риска — межа стартових випадкових спроб TPE.
    """
    figure, axis = plt.subplots(figsize=(FULL_WIDTH, 7.0 * _CM))
    for index, outcome in enumerate(outcomes):
        curve = outcome.best_so_far()
        axis.plot(
            np.arange(1, len(curve) + 1),
            curve,
            color=SERIES_COLORS[index % 4],
            linestyle=SERIES_LINESTYLES[index % 4],
            linewidth=1.4,
            # steps-post: значення тримається до наступної оцінки — best-so-far
            # змінюється стрибком, а не плавно між кандидатами.
            drawstyle="steps-post",
            label=outcome.label,
        )
    axis.axvline(optuna_startup + 0.5, color=GRAY, linestyle=":", linewidth=0.9)
    axis.annotate(
        "кінець випадкового\nстарту TPE",
        (optuna_startup + 0.5, 0.02),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=_ANNOTATION_SIZE,
        color=INK,
    )
    axis.set_xscale("log")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_xlabel("Номер оцінки (кандидата), логарифмічна шкала")
    axis.set_ylabel("Найкращий PR-AUC (Inner CV)")
    _decimal_comma(axis)
    axis.legend(loc="lower right")
    return figure


@styled
def plot_nested_inner_vs_outer(baseline: BaselineResult, reports: Sequence[MethodReport]) -> Figure:
    """«Гантелі» по зовнішніх фолдах: внутрішня оцінка → зовнішній PR-AUC (R17).

    Перша група — baseline: його 3-фолдова оцінка в кожному зовнішньому train
    без жодного вибору. Її «inner − outer» — зсув, що є і без відбору (менший
    обсяг навчання у внутрішніх фолдах, різна складність фолдів). Над кожною
    групою — сирий оптимізм і оптимізм відбору (сирий мінус зсув baseline).
    """
    figure, axis = plt.subplots(figsize=(FULL_WIDTH, 9.0 * _CM))
    shift_mean, shift_std = baseline.fold_shift_mean_std
    groups = [
        (
            _wrap(BASELINE_LABEL),
            list(baseline.inner_in_outer_scores),
            list(baseline.outer_scores),
            INK,
            "X",
            f"сирий\n{_comma(shift_mean, 3)} ± {_comma(shift_std, 3)}\nвідбору\n—",
        )
    ]
    for index, report in enumerate(reports):
        raw_mean, raw_std = report.nested.optimism()
        selection_mean, selection_std = report.nested.selection_optimism(baseline.fold_shift)
        groups.append(
            (
                _wrap(report.label),
                [fold.inner_best for fold in report.nested.folds],
                report.nested.fold_scores.tolist(),
                SERIES_COLORS[index % 4],
                SERIES_MARKERS[index % 4],
                f"сирий\n{_comma(raw_mean, 3)} ± {_comma(raw_std, 3)}\n"
                f"відбору\n{_comma(selection_mean, 3)} ± {_comma(selection_std, 3)}",
            )
        )
    for position, (_, inner, outer, color, marker, text) in enumerate(groups):
        offsets = position + np.linspace(-0.3, 0.3, len(inner))
        for offset, inner_value, outer_value in zip(offsets, inner, outer, strict=True):
            axis.plot([offset, offset], [outer_value, inner_value], color=GRAY, lw=0.8)
        axis.scatter(
            offsets,
            inner,
            marker=marker,
            s=28,
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            zorder=3,
        )
        axis.scatter(offsets, outer, marker=marker, s=28, color=color, zorder=3)
        axis.annotate(
            text,
            (position, 1.0),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=_ANNOTATION_SIZE - 1,
        )
    axis.set_xticks(range(len(groups)), [group[0] for group in groups])
    axis.set_ylabel("PR-AUC")
    axis.set_xlim(-0.6, len(groups) - 0.4)
    low, high = axis.get_ylim()
    axis.set_ylim(low, high + (high - low) * 0.7)  # місце для 4-рядкових підписів оптимізму
    axis.grid(axis="x", visible=False)
    _decimal_comma(axis)
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                color=INK,
                markerfacecolor="white",
                linestyle="",
                label="внутрішня оцінка (3 фолди зовнішнього train)",
            ),
            Line2D(
                [], [], marker="o", color=INK, linestyle="", label="PR-AUC на зовнішньому фолді"
            ),
        ],
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
    )
    return figure


def _wrap(label: str) -> str:
    """Переносить довгу назву методу на кілька рядків (перед дужкою і після слеша)."""
    return label.replace(" (", "\n(").replace(" / ", " /\n")


@styled
def plot_leakage_comparison(
    results: Mapping[LeakageVariant, LeakageResult], holdout: HoldoutLeakageResult
) -> Figure:
    """PR-AUC CV ± σ для варіантів витоку і PR-AUC на test для двох головних (R19)."""
    variants = list(results)
    comparisons = compare_leakage(results)
    figure, axis = plt.subplots(figsize=(FULL_WIDTH, 7.0 * _CM))
    positions = np.arange(len(variants))
    for position, variant in zip(positions, variants, strict=True):
        mean, spread = mean_and_std(results[variant].valid_pr_auc)
        leaky = variant in comparisons
        axis.barh(
            position,
            mean,
            xerr=spread,
            height=0.6,
            capsize=3,
            ecolor=INK,
            color=SERIES_COLORS[1] if leaky else SERIES_COLORS[0],
            hatch=SERIES_HATCHES[1] if leaky else SERIES_HATCHES[0],
            edgecolor="white",
        )
        text = f"{_comma(mean, 3)}"
        if leaky:
            comparison = comparisons[variant]
            text += f"   Δ = {_comma(comparison.delta, 3)} ± {_comma(comparison.spread, 3)}"
        axis.annotate(
            text,
            (mean + spread, position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=_ANNOTATION_SIZE,
        )
    for variant, value in (
        (LeakageVariant.CORRECT, holdout.correct_pr_auc),
        (LeakageVariant.LEAK_FIT_ALL, holdout.leaky_pr_auc),
    ):
        axis.scatter(
            value,
            variants.index(variant),
            marker="D",
            s=36,
            color=INK,
            zorder=4,
            facecolors="white",
            linewidths=1.3,
        )
    axis.set_yticks(positions, [VARIANT_LABELS[variant] for variant in variants])
    axis.invert_yaxis()
    axis.set_xlabel("PR-AUC (5 фолдів повного train; Δ — завищення проти відповідного коректного)")
    axis.set_xlim(0, 0.8)
    axis.grid(axis="y", visible=False)
    _decimal_comma(axis, "x")
    axis.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=SERIES_COLORS[0], label="коректний (CV ± σ)"),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=SERIES_COLORS[1],
                hatch=SERIES_HATCHES[1],
                edgecolor="white",
                label="з витоком (CV ± σ)",
            ),
            Line2D(
                [],
                [],
                marker="D",
                color=INK,
                markerfacecolor="white",
                linestyle="",
                label="той самий пайплайн на test",
            ),
        ],
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
    )
    return figure


@styled
def plot_leakage_by_size(
    by_size: Mapping[int, Mapping[LeakageVariant, LeakageResult]],
) -> Figure:
    """Коректний і некоректний PR-AUC та завищення Δ ± s від обсягу даних (R14)."""
    sizes = np.array(sorted(by_size))
    figure, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 6.5 * _CM))
    for index, variant in enumerate((LeakageVariant.CORRECT, LeakageVariant.LEAK_FIT_ALL)):
        stats = [mean_and_std(by_size[size][variant].valid_pr_auc) for size in sizes]
        left.errorbar(
            sizes,
            [mean for mean, _ in stats],
            yerr=[spread for _, spread in stats],
            color=SERIES_COLORS[index],
            marker=SERIES_MARKERS[index],
            linestyle=SERIES_LINESTYLES[index],
            capsize=3,
            label=VARIANT_LABELS[variant],
        )
    deltas = [compare_leakage(by_size[size])[LeakageVariant.LEAK_FIT_ALL] for size in sizes]
    right.errorbar(
        sizes,
        [item.delta for item in deltas],
        yerr=[item.spread for item in deltas],
        color=INK,
        marker="o",
        capsize=3,
    )
    right.axhline(0, color=GRAY, linewidth=0.8)
    for axis in (left, right):
        axis.set_xscale("log")
        axis.set_xticks(sizes, [f"{size:,}".replace(",", " ") for size in sizes])
        axis.minorticks_off()
        axis.set_xlabel("Рядків у train (логарифмічна шкала)")
        _decimal_comma(axis)
    left.set_ylabel("PR-AUC (CV, 5 фолдів)")
    left.legend(loc="lower right")
    right.set_ylabel("Завищення Δ ± s")
    return figure


@styled
def plot_search_space_samples(outcomes: Sequence[SearchOutcome], space: SearchSpace) -> Figure:
    """Які значення кожного параметра перебрали методи і який PR-AUC отримали (R5, R21).

    Вісь x — значення параметра, y — внутрішній PR-AUC кандидата. Видно: Grid
    бачить лише 3 значення кожного параметра, Random розкиданий по всьому
    діапазону, TPE згущується біля добрих значень.
    """
    n_rows = (len(space.ranges) + 1) // 2  # по два параметри в рядку
    figure, axes = plt.subplots(
        n_rows, 2, figsize=(FULL_WIDTH, 5.5 * n_rows * _CM), sharey=True, squeeze=False
    )
    for axis, item in zip(axes.flat, space.ranges, strict=False):
        for index, outcome in enumerate(outcomes):
            values = [record.params[item.name] for record in outcome.history]
            scores = [record.mean_score for record in outcome.history]
            axis.scatter(
                values,
                scores,
                marker=SERIES_MARKERS[index % 4],
                s=16,
                facecolors="none" if index == 0 else SERIES_COLORS[index % 4],
                edgecolors=SERIES_COLORS[index % 4],
                linewidths=0.9,
                alpha=0.8,
                label=outcome.label,
            )
        if item.log:
            axis.set_xscale("log")
            axis.set_xticks(list(item.grid), [_comma(value) for value in item.grid])
            axis.minorticks_off()
        else:
            _decimal_comma(axis, "x")
        axis.set_xlabel(item.name)
        _decimal_comma(axis)
    for axis in axes[:, 0]:
        axis.set_ylabel("PR-AUC (Inner CV)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncols=len(outcomes), loc="outside lower center")
    return figure
