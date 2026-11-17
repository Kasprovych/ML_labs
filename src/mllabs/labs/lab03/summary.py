"""Таблиці для звіту та словники для ``metrics.json``.

Модуль нічого не рахує «по-справжньому» — лише складає вже отримані
результати в таблиці (≤ 7 колонок, короткі українські заголовки) і словники.
Таблиця методів оптимізації має рівно п'ять колонок із заголовками,
дослівно взятими із завдання.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mllabs.data import DatasetSplit
from mllabs.labs.lab03.baseline import BaselineResult
from mllabs.labs.lab03.config import Lab3Config
from mllabs.labs.lab03.holdout import BASELINE_KEY, HoldoutResult
from mllabs.labs.lab03.leakage import (
    REFERENCE_VARIANT,
    VARIANT_LABELS,
    VARIANT_LEAKS,
    HoldoutLeakageResult,
    LeakageResult,
    LeakageVariant,
)
from mllabs.labs.lab03.nested import NestedResult
from mllabs.labs.lab03.search import SearchOutcome
from mllabs.modeling import BootstrapInterval, mean_and_std
from mllabs.preprocessing import paired_difference

#: Заголовки таблиці методів — ДОСЛІВНО з розділу «Вимоги до звіту» завдання.
METHOD_COLUMNS = (
    "Метод оптимізації",
    "Кількість ітерацій / оцінок",
    "Час виконання (сек)",
    "Best Score (Inner CV)",
    "Unbiased Score (Nested CV)",
)
#: Підпис baseline у таблицях.
BASELINE_LABEL = "Baseline (ЛР2, без пошуку)"


def seconds(value: float) -> str:
    """Час до десятих секунди рядком: спільний формат таблиці (``%.4f``) дописав би нулі."""
    return f"{value:.1f}"


def plus_minus(mean: float, spread: float, digits: int = 4) -> str:
    """«0.4812 ± 0.0190» — середнє і розкид в одній клітинці."""
    return f"{mean:.{digits}f} ± {spread:.{digits}f}"


@dataclass(frozen=True)
class MethodReport:
    """Усе, що відомо про один метод пошуку після всіх етапів.

    Attributes:
        outcome: Самостійний пошук на dev (етап 2).
        nested: Вкладена CV (етап 4).
    """

    outcome: SearchOutcome
    nested: NestedResult

    @property
    def key(self) -> str:
        """Ключ методу."""
        return self.outcome.key

    @property
    def label(self) -> str:
        """Назва методу."""
        return self.outcome.label


@dataclass(frozen=True)
class HoldoutSummary:
    """Результати test для таблиць: метрики і бутстреп-різниці з baseline.

    Attributes:
        holdout: Прогнози й метрики всіх моделей на test.
        intervals: Ключ методу → «метод − baseline» (PR-AUC) з ДІ.
    """

    holdout: HoldoutResult
    intervals: Mapping[str, BootstrapInterval]


# ---------------------------------------------------------------------- дані й baseline


def data_state_frame(samples: Mapping[str, pd.Series], positive_label: int) -> pd.DataFrame:
    """Обсяг і частка класу 1 у кожній вибірці."""
    rows = []
    for name, target in samples.items():
        positives = int((target == positive_label).sum())
        rows.append(
            {
                "Вибірка": name,
                "Рядків": len(target),
                "Позитивних (клас 1)": positives,
                "Частка класу 1": positives / len(target),
            }
        )
    return pd.DataFrame(rows)


def baseline_frame(
    params: Mapping[str, Any], result: BaselineResult, lab2_gap: float | None
) -> pd.DataFrame:
    """Параметри baseline і його оцінки на dev (таблиця «показник — значення»)."""
    inner_mean, inner_std = result.inner_mean_std
    outer_mean, outer_std = result.outer_mean_std
    rows = [(f"Параметр {name}", str(value)) for name, value in params.items()]
    rows += [
        ("PR-AUC, 3 внутрішні фолди", plus_minus(inner_mean, inner_std)),
        ("PR-AUC, 5 зовнішніх фолдів", plus_minus(outer_mean, outer_std)),
        ("ROC-AUC, 5 зовнішніх фолдів", f"{np.mean(result.outer_roc_auc):.4f}"),
        ("PR-AUC на train-фолдах", f"{result.train_mean:.4f}"),
        ("Розрив train − CV", f"{result.train_mean - outer_mean:.4f}"),
    ]
    if lab2_gap is not None:
        rows.append(("Розрив train − CV у ЛР2 (повний train)", f"{lab2_gap:.4f}"))
    return pd.DataFrame(rows, columns=["Показник", "Значення"])


def preparation_metrics(
    config: Lab3Config,
    seed: int,
    dev_target: pd.Series,
    full: DatasetSplit,
    positive_label: int,
    params: Mapping[str, Any],
    baseline: BaselineResult,
    lab2_gap: float | None,
) -> dict[str, Any]:
    """Розділи ``data``, ``metric``, ``model``, ``search_space``, ``baseline``, ``config``.

    Args:
        config: Параметри ЛР3.
        seed: Зерно запуску.
        dev_target: Мітки dev-вибірки.
        full: Повний train і test.
        positive_label: Мітка позитивного класу.
        params: Параметри baseline-моделі.
        baseline: Оцінки baseline.
        lab2_gap: Розрив train − CV у ЛР2 (або ``None``).
    """
    is_positive = dev_target == positive_label
    inner_mean, inner_std = baseline.inner_mean_std
    outer_mean, outer_std = baseline.outer_mean_std
    shift_mean, shift_std = baseline.fold_shift_mean_std
    return {
        "data": {
            "dev_rows": len(dev_target),
            "dev_positives": int(is_positive.sum()),
            "dev_minority_share": float(is_positive.mean()),
            "full_train_rows": len(full.y_train),
            "test_rows": len(full.y_test),
            "manifest_checked": True,
        },
        "metric": {"name": "pr_auc", "scoring": config.scoring, "secondary": "roc_auc"},
        "model": {
            "kind": config.model.value,
            "params": dict(params),
            "lab2_train_cv_gap": lab2_gap,
        },
        "search_space": config.search_space.to_metrics(),
        "baseline": {
            "inner_pr_auc": inner_mean,
            "inner_pr_auc_std": inner_std,
            "inner_wall_seconds": baseline.inner_wall_seconds,
            "outer_pr_auc": outer_mean,
            "outer_pr_auc_std": outer_std,
            "outer_roc_auc": float(np.mean(baseline.outer_roc_auc)),
            "train_pr_auc": baseline.train_mean,
            "train_cv_gap": baseline.train_mean - outer_mean,
            "inner_in_outer_pr_auc": float(np.mean(baseline.inner_in_outer_scores)),
            "inner_minus_outer_mean": shift_mean,
            "inner_minus_outer_std": shift_std,
        },
        "config": {
            "seed": seed,
            "dev_sample_size": config.dev_sample_size,
            "inner_folds": config.inner_folds,
            "outer_folds": config.outer_folds,
            "random_iterations": list(config.random_iterations),
            "optuna_trials": config.optuna_trials,
            "optuna_startup_trials": config.optuna_startup_trials,
            "search_n_jobs": config.search_n_jobs,
            "search_threads_per_worker": config.search_threads_per_worker,
            "optuna_fold_n_jobs": config.optuna_fold_n_jobs,
            "optuna_threads_per_worker": config.optuna_threads_per_worker,
            "main_process_threads": config.main_process_threads,
        },
    }


# ---------------------------------------------------------------------- методи пошуку


def methods_table(baseline: BaselineResult, reports: Sequence[MethodReport]) -> pd.DataFrame:
    """Таблиця завдання: рівно 5 колонок із дослівними заголовками.

    Baseline не шукає параметрів: «1 / 3» — одна конфігурація на 3 фолдах, час —
    3-фолдова оцінка, а його оцінка на 5 зовнішніх фолдах уже незміщена.
    """
    inner_mean, inner_std = baseline.inner_mean_std
    outer_mean, outer_std = baseline.outer_mean_std
    rows = [
        (
            BASELINE_LABEL,
            f"1 / {len(baseline.inner_scores)}",
            seconds(baseline.inner_wall_seconds),
            plus_minus(inner_mean, inner_std),
            plus_minus(outer_mean, outer_std),
        )
    ]
    for report in reports:
        outcome = report.outcome
        rows.append(
            (
                report.label,
                f"{outcome.n_candidates} / {outcome.n_fits}",
                seconds(outcome.wall_seconds),
                plus_minus(outcome.best_score, outcome.best_score_std),
                plus_minus(*report.nested.mean_std),
            )
        )
    return pd.DataFrame(rows, columns=list(METHOD_COLUMNS))


def cost_table(reports: Sequence[MethodReport]) -> pd.DataFrame:
    """Обчислювальна вартість методів: навчання, режим паралелізму, час, швидкість.

    Порівнюється реальний (wall-clock) час: усі методи займають ≈ усі ядра.
    «Сумарний час навчань» між методами НЕ порівнянний — Grid/Random вчать
    кожну модель в 1 потік (10 процесів), Optuna — у 3 потоки (3 процеси), тож
    навчання Optuna коротше, але займає втричі більше ядер. Optuna повільніша
    за RS20 при тих самих 60 навчаннях, бо її спроби строго послідовні (TPE
    пропонує точку за попередніми результатами) і паралельні лише 3 фолди.
    """
    rows = [
        {
            "Метод": report.label,
            "Навчань моделі": report.outcome.n_fits,
            "Процесів × потоків": f"{report.outcome.workers} × {report.outcome.threads_per_fit}",
            "Час пошуку, с": seconds(report.outcome.wall_seconds),
            "Сумарний час навчань, с": seconds(report.outcome.compute_seconds),
            "Навчань за секунду": f"{report.outcome.n_fits / report.outcome.wall_seconds:.2f}",
            "Час Nested CV, с": seconds(report.nested.wall_seconds),
        }
        for report in reports
    ]
    return pd.DataFrame(rows)


def quality_table(baseline: BaselineResult, reports: Sequence[MethodReport]) -> pd.DataFrame:
    """Якість методів: внутрішня і вкладена оцінки, оптимізм у двох формах, приріст.

    Оптимізм подано двічі (усе — по зовнішніх фолдах, середнє ± s):

    * **сирий** «inner − outer» — містить і зсув без відбору (внутрішні моделі
      вчаться на 2/3 зовнішнього train, фолди різні); рядок baseline показує
      саме цей зсув;
    * **оптимізм відбору** — сирий мінус зсув baseline на тому самому фолді:
      лише ефект «найкращий із N шумних оцінок».

    Тож від'ємний сирий оптимізм не означає, що «внутрішня CV песимістична».
    """
    shift_mean, shift_std = baseline.fold_shift_mean_std
    inner_mean, inner_std = baseline.inner_mean_std
    rows = [
        {
            "Метод": BASELINE_LABEL,
            "Best Score ± σ": plus_minus(inner_mean, inner_std),
            "Nested ± σ": plus_minus(*baseline.outer_mean_std),
            "Оптимізм сирий ± s": plus_minus(shift_mean, shift_std),
            "Оптимізм відбору ± s": "— (точка відліку)",
            "Приріст над baseline Δ ± s": "—",
            "Δ > s?": "—",
        }
    ]
    for report in reports:
        gain = paired_difference(baseline.outer_scores, report.nested.fold_scores.tolist())
        rows.append(
            {
                "Метод": report.label,
                "Best Score ± σ": plus_minus(
                    report.outcome.best_score, report.outcome.best_score_std
                ),
                "Nested ± σ": plus_minus(*report.nested.mean_std),
                "Оптимізм сирий ± s": plus_minus(*report.nested.optimism()),
                "Оптимізм відбору ± s": plus_minus(
                    *report.nested.selection_optimism(baseline.fold_shift)
                ),
                "Приріст над baseline Δ ± s": plus_minus(gain.delta, gain.spread),
                "Δ > s?": "так" if gain.complex_wins else "ні",
            }
        )
    return pd.DataFrame(rows)


def pairwise_table(baseline: BaselineResult, reports: Sequence[MethodReport]) -> pd.DataFrame:
    """Попарні різниці зовнішніх PR-AUC усіх методів і baseline (правило D9: Δ > s)."""
    scores = {BASELINE_LABEL: list(baseline.outer_scores)}
    scores.update({report.label: report.nested.fold_scores.tolist() for report in reports})
    rows = []
    # combinations — усі невпорядковані пари без повторів (як два вкладені цикли j > i).
    for first, second in itertools.combinations(scores, 2):
        difference = paired_difference(scores[first], scores[second])
        rows.append(
            {
                "Метод A": first,
                "Метод B": second,
                "Δ = B − A": round(difference.delta, 4),
                "s": round(difference.spread, 4),
                "|Δ| > s?": "так" if abs(difference.delta) > difference.spread else "ні",
            }
        )
    return pd.DataFrame(rows)


def best_params_frame(
    baseline_params: Mapping[str, int | float],
    reports: Sequence[MethodReport],
    names: Sequence[str],
) -> pd.DataFrame:
    """Найкращі параметри самостійного пошуку кожного методу + параметри baseline."""
    rows = [{"Метод": BASELINE_LABEL, **{name: baseline_params[name] for name in names}}]
    rows += [
        {"Метод": report.label, **{name: report.outcome.best_params[name] for name in names}}
        for report in reports
    ]
    return pd.DataFrame(rows)


def search_history_frame(outcomes: Sequence[SearchOutcome]) -> pd.DataFrame:
    """Усі оцінки всіх методів (довга таблиця; для додатка і графіків)."""
    frames = []
    for outcome in outcomes:
        best_so_far = outcome.best_so_far()
        frames.append(
            pd.DataFrame(
                [
                    {
                        "method": outcome.key,
                        "evaluation": record.index + 1,
                        **record.params,
                        "pr_auc_mean": record.mean_score,
                        "pr_auc_std": record.std_score,
                        "compute_seconds": record.compute_seconds,
                        "best_so_far": best_so_far[position],
                    }
                    for position, record in enumerate(outcome.history)
                ]
            )
        )
    return pd.concat(frames, ignore_index=True)


def nested_folds_frame(baseline: BaselineResult, reports: Sequence[MethodReport]) -> pd.DataFrame:
    """Зовнішні фолди вкладеної CV для кожного методу і baseline (≤ 7 колонок)."""
    rows = [
        {
            "Метод": BASELINE_LABEL,
            "Фолд": fold + 1,
            # Для baseline «Inner best» — його 3-фолдова оцінка в зовнішньому train
            # (без вибору), а «Оптимізм» — зсув inner − outer без відбору.
            "Inner best": inner,
            "Outer PR-AUC": score,
            "Outer ROC-AUC": roc_auc,
            "Оптимізм": inner - score,
            "Параметри": "ЛР2",
        }
        for fold, (inner, score, roc_auc) in enumerate(
            zip(
                baseline.inner_in_outer_scores,
                baseline.outer_scores,
                baseline.outer_roc_auc,
                strict=True,
            )
        )
    ]
    for report in reports:
        rows += [
            {
                "Метод": report.label,
                "Фолд": fold.fold + 1,
                "Inner best": fold.inner_best,
                "Outer PR-AUC": fold.outer_pr_auc,
                "Outer ROC-AUC": fold.outer_roc_auc,
                "Оптимізм": fold.optimism,
                "Параметри": format_params(fold.best_params),
            }
            for fold in report.nested.folds
        ]
    return pd.DataFrame(rows)


def format_params(params: Mapping[str, int | float]) -> str:
    """Компактний запис параметрів: ``lr=0.031; leaves=24; ...``."""
    short = {
        "learning_rate": "lr",
        "max_leaf_nodes": "leaves",
        "min_samples_leaf": "min_leaf",
        "max_features": "feat",
    }
    return "; ".join(f"{short.get(name, name)}={value:.3g}" for name, value in params.items())


def methods_metrics(
    baseline: BaselineResult, reports: Sequence[MethodReport], test: HoldoutSummary
) -> dict[str, Any]:
    """Розділи ``methods`` і ``methods_summary`` для ``metrics.json``."""
    methods = {}
    for report in reports:
        outcome, nested = report.outcome, report.nested
        nested_mean, nested_std = nested.mean_std
        optimism_mean, optimism_std = nested.optimism()
        selection_mean, selection_std = nested.selection_optimism(baseline.fold_shift)
        gain = paired_difference(baseline.outer_scores, nested.fold_scores.tolist())
        interval = test.intervals[report.key]
        methods[report.key] = {
            "label": report.label,
            "n_candidates": outcome.n_candidates,
            "n_fits": outcome.n_fits,
            "wall_seconds": outcome.wall_seconds,
            # compute_seconds між методами непорівнянний: різна кількість потоків на
            # навчання (workers × threads_per_fit). Методи порівнюються за wall_seconds.
            "compute_seconds": outcome.compute_seconds,
            "workers": outcome.workers,
            "threads_per_fit": outcome.threads_per_fit,
            "evals_per_second": outcome.n_fits / outcome.wall_seconds,
            "best_score_inner": outcome.best_score,
            "best_score_inner_std": outcome.best_score_std,
            "best_params": dict(outcome.best_params),
            "nested_mean": nested_mean,
            "nested_std": nested_std,
            "nested_roc_auc": nested.roc_auc,
            "nested_inner_mean": nested.inner_mean,
            "optimism_raw_mean": optimism_mean,
            "optimism_raw_std": optimism_std,
            "optimism_selection_mean": selection_mean,
            "optimism_selection_std": selection_std,
            "optimism_selection_exceeds_spread": selection_mean > selection_std,
            "best_minus_nested": outcome.best_score - nested_mean,
            "gain_vs_baseline": gain.delta,
            "gain_vs_baseline_std": gain.spread,
            "gain_exceeds_spread": gain.complex_wins,
            "nested_wall_seconds": nested.wall_seconds,
            "test_pr_auc": test.holdout.pr_auc[report.key],
            "test_roc_auc": test.holdout.roc_auc[report.key],
            "test_delta_vs_baseline": interval.delta,
            "test_delta_low": interval.low,
            "test_delta_high": interval.high,
            "test_delta_excludes_zero": interval.excludes_zero,
        }
    return {"methods": methods, "methods_summary": _methods_summary(baseline, reports)}


def _methods_summary(baseline: BaselineResult, reports: Sequence[MethodReport]) -> dict[str, Any]:
    """Зведення для тексту висновків за правилом D9.

    «Найкращого за якістю» методу тут свідомо немає: метод кращий за інший лише
    тоді, коли парна різниця зовнішніх PR-AUC перевищує розкид (Δ > s). Тому
    зберігається лише кількість розрізнюваних пар — окремо «метод проти
    baseline» і «метод проти методу». Час порівнюється за wall-clock.
    """
    pairs = pairwise_table(baseline, reports)
    distinguishable = pairs["|Δ| > s?"] == "так"
    versus_baseline = pairs["Метод A"] == BASELINE_LABEL
    shift_mean, shift_std = baseline.fold_shift_mean_std
    return {
        "time_basis": "wall_seconds",
        "fastest": min(reports, key=lambda report: report.outcome.wall_seconds).key,
        "slowest": max(reports, key=lambda report: report.outcome.wall_seconds).key,
        "wall_ratio_slowest_to_fastest": max(r.outcome.wall_seconds for r in reports)
        / min(r.outcome.wall_seconds for r in reports),
        "pairs_total": len(pairs),
        "pairs_distinguishable": int(distinguishable.sum()),
        "pairs_vs_baseline_total": int(versus_baseline.sum()),
        "pairs_vs_baseline_distinguishable": int((distinguishable & versus_baseline).sum()),
        "pairs_between_methods_total": int((~versus_baseline).sum()),
        "pairs_between_methods_distinguishable": int((distinguishable & ~versus_baseline).sum()),
        "baseline_shift_mean": shift_mean,
        "baseline_shift_std": shift_std,
    }


# ---------------------------------------------------------------------- test


def holdout_results_frame(test: HoldoutSummary, labels: Mapping[str, str]) -> pd.DataFrame:
    """PR-AUC/ROC-AUC на test і бутстреп-різниця з baseline."""
    rows = []
    for key, label in labels.items():
        interval = test.intervals.get(key)
        rows.append(
            {
                "Модель": label,
                "PR-AUC test": test.holdout.pr_auc[key],
                "ROC-AUC test": test.holdout.roc_auc[key],
                "Δ PR-AUC проти baseline": interval.delta if interval else np.nan,
                "95 % ДІ": f"[{interval.low:.4f}; {interval.high:.4f}]" if interval else "—",
                "ДІ не містить 0": ("так" if interval.excludes_zero else "ні") if interval else "—",
            }
        )
    return pd.DataFrame(rows)


def baseline_test_metrics(test: HoldoutSummary) -> dict[str, float]:
    """Метрики baseline на test для розділу ``baseline``."""
    return {
        "test_pr_auc": test.holdout.pr_auc[BASELINE_KEY],
        "test_roc_auc": test.holdout.roc_auc[BASELINE_KEY],
    }


# ---------------------------------------------------------------------- витік


@dataclass(frozen=True)
class LeakageComparison:
    """Некоректний варіант проти свого коректного: парна різниця по фолдах."""

    variant: LeakageVariant
    reference: LeakageVariant
    delta: float
    spread: float

    @property
    def exceeds_spread(self) -> bool:
        """Чи завищення перевищує розкид (правило Δ > s)."""
        return self.delta > self.spread


def compare_leakage(
    results: Mapping[LeakageVariant, LeakageResult],
) -> dict[LeakageVariant, LeakageComparison]:
    """Δ = некоректний − коректний (PR-AUC на реальних val-рядках) для кожного варіанта з парою."""
    comparisons = {}
    for variant, reference in REFERENCE_VARIANT.items():
        if variant in results and reference in results:
            difference = paired_difference(
                results[reference].valid_pr_auc, results[variant].valid_pr_auc
            )
            comparisons[variant] = LeakageComparison(
                variant, reference, difference.delta, difference.spread
            )
    return comparisons


def leakage_comparison_frame(results: Mapping[LeakageVariant, LeakageResult]) -> pd.DataFrame:
    """Головна таблиця витоку (≤ 7 колонок)."""
    comparisons = compare_leakage(results)
    rows = []
    for variant, result in results.items():
        comparison = comparisons.get(variant)
        rows.append(
            {
                "Пайплайн": VARIANT_LABELS[variant],
                "Що протікає": VARIANT_LEAKS[variant],
                "PR-AUC CV ± σ": plus_minus(*mean_and_std(result.valid_pr_auc)),
                "ROC-AUC CV": round(float(np.mean(result.valid_roc_auc)), 4),
                "PR-AUC на train": round(float(np.mean(result.train_pr_auc)), 4),
                "Завищення Δ ± s": plus_minus(comparison.delta, comparison.spread)
                if comparison
                else "—",
                "Δ > s?": ("так" if comparison.exceeds_spread else "ні") if comparison else "—",
            }
        )
    return pd.DataFrame(rows)


def leakage_folds_frame(results: Mapping[LeakageVariant, LeakageResult]) -> pd.DataFrame:
    """Варіант × фолд (для SMOTE до розбиття — ще «наївні» оцінки з синтетикою у val)."""
    rows = []
    for variant, result in results.items():
        for fold in range(len(result.valid_pr_auc)):
            rows.append(
                {
                    "variant": variant.value,
                    "fold": fold + 1,
                    "pr_auc": result.valid_pr_auc[fold],
                    "roc_auc": result.valid_roc_auc[fold],
                    "train_pr_auc": result.train_pr_auc[fold],
                    "naive_pr_auc": result.naive_pr_auc[fold] if result.naive_pr_auc else np.nan,
                    "naive_roc_auc": result.naive_roc_auc[fold] if result.naive_roc_auc else np.nan,
                }
            )
    return pd.DataFrame(rows)


def leakage_holdout_frame(
    results: Mapping[LeakageVariant, LeakageResult], holdout: HoldoutLeakageResult
) -> pd.DataFrame:
    """CV проти test для коректного і некоректного пайплайнів: величина завищення."""
    rows = []
    for variant, test_pr_auc, test_roc_auc in (
        (LeakageVariant.CORRECT, holdout.correct_pr_auc, holdout.correct_roc_auc),
        (LeakageVariant.LEAK_FIT_ALL, holdout.leaky_pr_auc, holdout.leaky_roc_auc),
    ):
        cv_mean = float(np.mean(results[variant].valid_pr_auc))
        rows.append(
            {
                "Пайплайн": VARIANT_LABELS[variant],
                "PR-AUC CV": cv_mean,
                "PR-AUC test": test_pr_auc,
                "CV − test": cv_mean - test_pr_auc,
                "ROC-AUC test": test_roc_auc,
            }
        )
    return pd.DataFrame(rows)


def leakage_by_size_frame(
    by_size: Mapping[int, Mapping[LeakageVariant, LeakageResult]], positives: Mapping[int, int]
) -> pd.DataFrame:
    """Завищення від обсягу даних: коректний, некоректний, Δ ± s."""
    rows = []
    for size, results in by_size.items():
        comparison = compare_leakage(results)[LeakageVariant.LEAK_FIT_ALL]
        rows.append(
            {
                "Рядків": size,
                "Позитивних": positives[size],
                "Коректний ± σ": plus_minus(
                    *mean_and_std(results[LeakageVariant.CORRECT].valid_pr_auc)
                ),
                "Некоректний ± σ": plus_minus(
                    *mean_and_std(results[LeakageVariant.LEAK_FIT_ALL].valid_pr_auc)
                ),
                "Завищення Δ ± s": plus_minus(comparison.delta, comparison.spread),
                "Δ > s?": "так" if comparison.exceeds_spread else "ні",
            }
        )
    return pd.DataFrame(rows)


def leakage_metrics(
    results: Mapping[LeakageVariant, LeakageResult],
    holdout: HoldoutLeakageResult,
    by_size: Mapping[int, Mapping[LeakageVariant, LeakageResult]],
) -> dict[str, Any]:
    """Розділ ``leakage`` для ``metrics.json``."""
    comparisons = compare_leakage(results)
    payload: dict[str, Any] = {}
    for variant, result in results.items():
        mean, spread = mean_and_std(result.valid_pr_auc)
        entry: dict[str, Any] = {
            "pr_auc": mean,
            "pr_auc_std": spread,
            "roc_auc": float(np.mean(result.valid_roc_auc)),
            "train_pr_auc": float(np.mean(result.train_pr_auc)),
        }
        if variant in comparisons:
            comparison = comparisons[variant]
            entry.update(
                delta=comparison.delta,
                delta_std=comparison.spread,
                delta_exceeds_spread=comparison.exceeds_spread,
            )
        if result.naive_pr_auc:
            entry.update(
                naive_pr_auc=float(np.mean(result.naive_pr_auc)),
                naive_roc_auc=float(np.mean(result.naive_roc_auc)),
            )
        payload[variant.value] = entry
    headline = comparisons[LeakageVariant.LEAK_FIT_ALL]
    incorrect_cv = payload[LeakageVariant.LEAK_FIT_ALL.value]["pr_auc"]
    payload["headline"] = {
        "correct_cv": payload[LeakageVariant.CORRECT.value]["pr_auc"],
        "incorrect_cv": incorrect_cv,
        "inflation_cv": headline.delta,
        "inflation_cv_std": headline.spread,
        "correct_test": holdout.correct_pr_auc,
        "incorrect_test": holdout.leaky_pr_auc,
        "inflation_vs_test": incorrect_cv - holdout.leaky_pr_auc,
        "test_delta": holdout.interval.delta,
        "test_delta_low": holdout.interval.low,
        "test_delta_high": holdout.interval.high,
        "test_delta_excludes_zero": holdout.interval.excludes_zero,
    }
    payload["by_size"] = {}
    for size, size_results in by_size.items():
        comparison = compare_leakage(size_results)[LeakageVariant.LEAK_FIT_ALL]
        payload["by_size"][str(size)] = {
            "correct": float(np.mean(size_results[LeakageVariant.CORRECT].valid_pr_auc)),
            "incorrect": float(np.mean(size_results[LeakageVariant.LEAK_FIT_ALL].valid_pr_auc)),
            "delta": comparison.delta,
            "delta_std": comparison.spread,
            "delta_exceeds_spread": comparison.exceeds_spread,
        }
    return payload
