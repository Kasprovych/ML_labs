"""Аналіз аугментації: чи покращує вона якість і коли спричиняє перенавчання.

Три осі умов:

1. **Інтенсивність аугментації** — частка синтетичних рядків
   (``sampling_strategy`` SMOTE: 0 — без аугментації, 1.0 — класи вирівняно);
2. **Місткість моделі** — лінійна модель (мала), бустинг (середня),
   ліс глибоких дерев (велика); це вже три моделі сітки;
3. **Обсяг даних** — криві навчання: навчальна частина фолда зменшується, а
   валідація лишається повною.

Для кожної точки вимірюється не лише якість на валідації, а й те, як модель
поводиться на синтетичних рядках: якщо на них вона майже впевнена (і PR-AUC
на «навчальних + синтетичних» близький до 1), а на реальних позитивних
валідації — ні, модель запам'ятала вигадані точки.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mllabs.labs.lab02.comparison import GridResult
from mllabs.labs.lab02.runner import ConfigurationKey, ConfigurationResult, ConfigurationRunner
from mllabs.labs.lab02.scenarios import ScenarioCatalog, ScenarioId
from mllabs.modeling import BootstrapInterval, MetricsCalculator, ModelFactory, ModelKind

#: Назви колонок «довгих» таблиць для графіків.
COLUMN_MODEL = "model"
COLUMN_RATIO = "ratio"
COLUMN_SCENARIO = "scenario"
COLUMN_SIZE = "train_rows"
COLUMN_CV = "cv_pr_auc"
COLUMN_CV_STD = "cv_pr_auc_std"
COLUMN_TRAIN = "train_pr_auc"
COLUMN_AUGMENTED = "augmented_pr_auc"
COLUMN_KIND = "kind"
COLUMN_PROBA = "proba"

#: Групи рядків на графіку «запам'ятовування синтетики».
KIND_SYNTHETIC = "синтетичні (train)"
KIND_TRAIN_POSITIVE = "реальні клас 1 (train)"
KIND_VALID_POSITIVE = "реальні клас 1 (валідація)"


@dataclass(frozen=True)
class SweepResult:
    """Результати перебору частоти аугментації.

    Attributes:
        results: Підсумок за ключем ``(модель, частота)``; частота 0 — без аугментації.
    """

    results: Mapping[tuple[ModelKind, float], ConfigurationResult]

    def ratios(self) -> list[float]:
        """Усі частоти в порядку зростання."""
        return sorted({ratio for _, ratio in self.results})


class RatioSweep:
    """Перебір частоти аугментації SMOTE для кожної моделі (вісь 1 і 2).

    Точки «без аугментації» і типова частота не перераховуються: вони беруться
    з основної сітки (ті самі фолди, ті самі пайплайни).

    Args:
        runner: Виконавець крос-валідації.
        models: Моделі.
        extra_ratios: Додаткові частоти, яких немає в основній сітці.
        default_ratio: Типова частота сценарію В-SMOTE.
    """

    def __init__(
        self,
        runner: ConfigurationRunner,
        models: Sequence[ModelKind],
        extra_ratios: Sequence[float],
        default_ratio: float,
    ) -> None:
        self._runner = runner
        self._models = tuple(models)
        self._extra_ratios = tuple(extra_ratios)
        self._default_ratio = default_ratio

    def run(self, grid: GridResult) -> SweepResult:
        """Оцінює додаткові частоти й об'єднує їх із точками основної сітки."""
        results: dict[tuple[ModelKind, float], ConfigurationResult] = {}
        for model in self._models:
            results[(model, 0.0)] = grid.get(model, ScenarioId.B)
            results[(model, self._default_ratio)] = grid.get(model, ScenarioId.C_SMOTE)
            for ratio in self._extra_ratios:
                key = ConfigurationKey(model, ScenarioId.C_SMOTE, sampling_strategy=ratio)
                results[(model, ratio)] = self._runner.run(key)
        return SweepResult(results)

    def table(self, sweep: SweepResult, calculator: MetricsCalculator) -> pd.DataFrame:
        """Якість і «запам'ятовування синтетики» від частоти аугментації."""
        rows = []
        for (model, ratio), result in _in_model_order(sweep.results, self._models):
            rows.append(
                {
                    "Модель": ModelFactory.short_name(model),
                    "Частота": ratio,
                    "PR-AUC CV": result.pr_auc,
                    "PR-AUC train": float(result.train_pr_auc.mean()),
                    "PR-AUC train+синт.": _mean_or_nan(result.augmented_train_pr_auc),
                    "Ймовірність синт.": _mean_or_nan(result.synthetic_proba),
                    "Ймовірність клас 1 CV": float(result.valid_positive_proba(calculator).mean()),
                }
            )
        return pd.DataFrame(rows)

    def calibration_table(self, sweep: SweepResult) -> pd.DataFrame:
        """Зсув ймовірностей і порога від частоти аугментації (OOF train)."""
        rows = [
            {
                "Модель": ModelFactory.short_name(model),
                "Частота": ratio,
                "Brier": result.cv_mean("brier"),
                "Сер. ймовірність": result.cv_mean("mean_proba"),
                "Поріг": result.threshold,
                "F1 (0,5)": result.cv_mean("f1"),
                "F1 (поріг)": result.cv_mean("f1", tuned=True),
            }
            for (model, ratio), result in _in_model_order(sweep.results, self._models)
        ]
        return pd.DataFrame(rows)

    def plot_frame(self, sweep: SweepResult) -> pd.DataFrame:
        """Довга таблиця для графіка: модель, частота, PR-AUC CV ± σ, розриви."""
        rows = [
            {
                COLUMN_MODEL: model.value,
                COLUMN_RATIO: ratio,
                COLUMN_CV: result.pr_auc,
                COLUMN_CV_STD: result.cv_std("pr_auc"),
                COLUMN_TRAIN: float(result.train_pr_auc.mean()),
                COLUMN_AUGMENTED: _mean_or_nan(result.augmented_train_pr_auc),
            }
            for (model, ratio), result in _in_model_order(sweep.results, self._models)
        ]
        return pd.DataFrame(rows)

    def to_metrics(self, sweep: SweepResult, calculator: MetricsCalculator) -> dict[str, Any]:
        """Розділ ``augmentation.overfitting`` (частина від перебору частот)."""
        metrics: dict[str, Any] = {}
        for model in self._models:
            points = {ratio: sweep.results[(model, ratio)] for ratio in sweep.ratios()}
            # Розрив «train + синтетичні − CV» існує лише там, де є синтетичні рядки.
            gaps = {
                f"{ratio:g}": result.augmented_gap
                for ratio, result in points.items()
                if result.augmented_gap is not None
            }
            default = points[self._default_ratio]
            metrics[model.value] = {
                "pr_auc_by_ratio": {
                    f"{ratio:g}": result.pr_auc for ratio, result in points.items()
                },
                "train_gap_by_ratio": {
                    f"{ratio:g}": result.train_gap for ratio, result in points.items()
                },
                "gap_by_ratio": gaps,
                "brier_by_ratio": {
                    f"{ratio:g}": result.cv_mean("brier") for ratio, result in points.items()
                },
                "threshold_by_ratio": {
                    f"{ratio:g}": result.threshold for ratio, result in points.items()
                },
                "max_gap_ratio": float(max(gaps, key=gaps.__getitem__)),
                "best_ratio": max(points, key=lambda ratio: points[ratio].pr_auc),
                "synthetic_mean_proba": _mean_or_nan(default.synthetic_proba),
                "train_positive_mean_proba": float(default.train_positive_proba.mean()),
                "real_val_positive_mean_proba": float(
                    default.valid_positive_proba(calculator).mean()
                ),
            }
        return metrics


class LearningCurveStudy:
    """Криві навчання Б проти В-SMOTE: як ефект аугментації залежить від обсягу даних (вісь 3).

    Args:
        runner: Виконавець крос-валідації.
        models: Моделі.
        sizes: Розміри навчальної частини фолда (менші за повний).
        sampling_strategy: Частота аугментації сценарію В-SMOTE.
    """

    #: Сценарії, що порівнюються на кривих навчання.
    SCENARIOS = (ScenarioId.B, ScenarioId.C_SMOTE)

    def __init__(
        self,
        runner: ConfigurationRunner,
        models: Sequence[ModelKind],
        sizes: Sequence[int],
        sampling_strategy: float,
    ) -> None:
        self._runner = runner
        self._models = tuple(models)
        self._sizes = tuple(sizes)
        self._sampling_strategy = sampling_strategy

    def run(self, grid: GridResult) -> dict[tuple[ModelKind, ScenarioId, int], ConfigurationResult]:
        """Оцінює менші розміри й додає повний розмір з основної сітки."""
        results: dict[tuple[ModelKind, ScenarioId, int], ConfigurationResult] = {}
        for model in self._models:
            for scenario in self.SCENARIOS:
                full = grid.get(model, scenario)
                results[(model, scenario, full.train_rows)] = full
                ratio = self._sampling_strategy if scenario is ScenarioId.C_SMOTE else 0.0
                for size in self._sizes:
                    key = ConfigurationKey(model, scenario, ratio, train_size=size)
                    results[(model, scenario, size)] = self._runner.run(key)
        return results

    def table(
        self,
        results: Mapping[tuple[ModelKind, ScenarioId, int], ConfigurationResult],
        catalog: ScenarioCatalog,
    ) -> pd.DataFrame:
        """PR-AUC CV і train від розміру навчальної частини."""
        rows = [
            {
                "Модель": ModelFactory.short_name(model),
                "Сценарій": catalog.label(scenario),
                "Рядків": result.train_rows,
                "Клас 1": result.train_positives,
                "PR-AUC CV": result.pr_auc,
                "σ": result.cv_std("pr_auc"),
                "PR-AUC train": float(result.train_pr_auc.mean()),
            }
            for (model, scenario, _), result in _in_model_order(results, self._models)
        ]
        return pd.DataFrame(rows)

    def plot_frame(
        self, results: Mapping[tuple[ModelKind, ScenarioId, int], ConfigurationResult]
    ) -> pd.DataFrame:
        """Довга таблиця для графіка кривих навчання."""
        rows = [
            {
                COLUMN_MODEL: model.value,
                COLUMN_SCENARIO: scenario.value,
                COLUMN_SIZE: result.train_rows,
                COLUMN_CV: result.pr_auc,
                COLUMN_CV_STD: result.cv_std("pr_auc"),
                COLUMN_TRAIN: float(result.train_pr_auc.mean()),
            }
            for (model, scenario, _), result in _in_model_order(results, self._models)
        ]
        return pd.DataFrame(rows)

    def to_metrics(
        self, results: Mapping[tuple[ModelKind, ScenarioId, int], ConfigurationResult]
    ) -> dict[str, Any]:
        """Ефект SMOTE (Δ PR-AUC В − Б) на найменшому та повному розмірі."""
        metrics: dict[str, Any] = {}
        for model in self._models:
            sizes = sorted({size for key_model, _, size in results if key_model is model})
            deltas = {
                size: results[(model, ScenarioId.C_SMOTE, size)].pr_auc
                - results[(model, ScenarioId.B, size)].pr_auc
                for size in sizes
            }
            metrics[model.value] = {
                "delta_by_size": {str(size): delta for size, delta in deltas.items()},
                "smallest_size": sizes[0],
                "smallest_size_delta": deltas[sizes[0]],
                "full_size": sizes[-1],
                "full_size_delta": deltas[sizes[-1]],
                "gap_smallest_size_b": results[(model, ScenarioId.B, sizes[0])].train_gap,
                "gap_smallest_size_c": results[(model, ScenarioId.C_SMOTE, sizes[0])].train_gap,
            }
        return metrics


def memorization_frame(
    grid: GridResult, models: Sequence[ModelKind], calculator: MetricsCalculator
) -> pd.DataFrame:
    """Ймовірності класу 1 на синтетичних, реальних навчальних і валідаційних позитивних.

    Сценарій В-SMOTE з типовою частотою. Повертає довгу таблицю
    ``model, kind, proba`` для графіка розподілів.
    """
    frames = []
    for model in models:
        result = grid.get(model, ScenarioId.C_SMOTE)
        groups = {
            KIND_SYNTHETIC: result.synthetic_proba,
            KIND_TRAIN_POSITIVE: result.train_positive_proba,
            KIND_VALID_POSITIVE: result.valid_positive_proba(calculator),
        }
        for kind, values in groups.items():
            frames.append(
                pd.DataFrame({COLUMN_MODEL: model.value, COLUMN_KIND: kind, COLUMN_PROBA: values})
            )
    return pd.concat(frames, ignore_index=True)


class AugmentationEffect:
    """Зведення «чи покращила аугментація якість»: CV (парно) і test (бутстреп).

    Args:
        catalog: Сценарії.
        models: Моделі.
    """

    #: Сценарії аугментації, що порівнюються з Б.
    AUGMENTED = (ScenarioId.C_SMOTE, ScenarioId.C_ADASYN)

    def __init__(self, catalog: ScenarioCatalog, models: Sequence[ModelKind]) -> None:
        self._catalog = catalog
        self._models = tuple(models)

    def table(
        self,
        grid: GridResult,
        test_intervals: Mapping[tuple[ModelKind, ScenarioId], BootstrapInterval],
    ) -> pd.DataFrame:
        """Δ PR-AUC В − Б: на CV (парно, правило Δ > s) і на test (95 % ДІ)."""
        rows = []
        for model in self._models:
            for scenario in self.AUGMENTED:
                difference = grid.paired(model, ScenarioId.B, scenario)
                interval = test_intervals[(model, scenario)]
                rows.append(
                    {
                        "Модель": ModelFactory.short_name(model),
                        "Метод": self._catalog.spec(scenario).sampler_kind.value.upper(),
                        "Δ PR-AUC CV": difference.delta,
                        "s": difference.spread,
                        "Покращує (Δ > s)": "так" if difference.complex_wins else "ні",
                        "Δ PR-AUC test": interval.delta,
                        "95 % ДІ test": f"[{interval.low:.4f}; {interval.high:.4f}]",
                    }
                )
        return pd.DataFrame(rows)

    def threshold_table(self, grid: GridResult) -> pd.DataFrame:
        """Чи «покращення» F1 від аугментації — лише зсув порога (CV, OOF)."""
        rows = []
        for model in self._models:
            base = grid.get(model, ScenarioId.B)
            for scenario in self.AUGMENTED:
                augmented = grid.get(model, scenario)
                rows.append(
                    {
                        "Модель": ModelFactory.short_name(model),
                        "Метод": self._catalog.spec(scenario).sampler_kind.value.upper(),
                        "Δ F1 (0,5)": augmented.cv_mean("f1") - base.cv_mean("f1"),
                        "Δ F1 (поріг)": augmented.cv_mean("f1", tuned=True)
                        - base.cv_mean("f1", tuned=True),
                        "Δ Brier": augmented.cv_mean("brier") - base.cv_mean("brier"),
                        "Поріг Б": base.threshold,
                        "Поріг В": augmented.threshold,
                    }
                )
        return pd.DataFrame(rows)

    def to_metrics(
        self,
        grid: GridResult,
        test_intervals: Mapping[tuple[ModelKind, ScenarioId], BootstrapInterval],
        test_metrics: Mapping[tuple[ModelKind, ScenarioId], Mapping[str, float]],
    ) -> dict[str, Any]:
        """Розділ ``augmentation.effect`` для ``metrics.json``."""
        metrics: dict[str, Any] = {}
        for model in self._models:
            base = grid.get(model, ScenarioId.B)
            base_test = test_metrics[(model, ScenarioId.B)]
            per_method: dict[str, Any] = {}
            for scenario in self.AUGMENTED:
                augmented = grid.get(model, scenario)
                augmented_test = test_metrics[(model, scenario)]
                difference = grid.paired(model, ScenarioId.B, scenario)
                interval = test_intervals[(model, scenario)]
                per_method[self._catalog.spec(scenario).sampler_kind.value] = {
                    "cv_delta": difference.delta,
                    "cv_spread": difference.spread,
                    "cv_improves": difference.complex_wins,
                    "cv_roc_auc_delta": augmented.cv_mean("roc_auc") - base.cv_mean("roc_auc"),
                    "test_delta": interval.delta,
                    "test_ci_low": interval.low,
                    "test_ci_high": interval.high,
                    "test_ci_excludes_zero": interval.excludes_zero,
                    "f1_delta_at_05": augmented.cv_mean("f1") - base.cv_mean("f1"),
                    "f1_delta_tuned": augmented.cv_mean("f1", tuned=True)
                    - base.cv_mean("f1", tuned=True),
                    "brier_delta": augmented.cv_mean("brier") - base.cv_mean("brier"),
                    "threshold_base": base.threshold,
                    "threshold_augmented": augmented.threshold,
                    "test_f1_delta_at_05": augmented_test["f1_at_05"] - base_test["f1_at_05"],
                    "test_f1_delta_tuned": augmented_test["f1"] - base_test["f1"],
                    "test_recall_delta_at_05": augmented_test["recall_at_05"]
                    - base_test["recall_at_05"],
                }
            metrics[model.value] = per_method
        return metrics


def _mean_or_nan(values: np.ndarray | None) -> float:
    """Середнє масиву або NaN, якщо масиву немає (сценарій без аугментації)."""
    return float(np.mean(values)) if values is not None else float("nan")


def merge_overfitting_metrics(
    sweep_metrics: Mapping[str, Mapping[str, Any]],
    curve_metrics: Mapping[str, Mapping[str, Any]],
    models: Sequence[ModelKind],
) -> dict[str, dict[str, Any]]:
    """Розділ ``augmentation.overfitting``: перебір частот і криві навчання разом, по моделях."""
    return {
        model.value: {
            **sweep_metrics.get(model.value, {}),
            **curve_metrics.get(model.value, {}),
        }
        for model in models
    }


def _in_model_order[K: tuple, V](
    results: Mapping[K, V], models: Sequence[ModelKind]
) -> list[tuple[K, V]]:
    """Пари «ключ → результат», упорядковані за порядком моделей у конфігурації.

    Ключ — кортеж, перший елемент якого модель; решта елементів (частота,
    сценарій, розмір) упорядковуються звичайним порівнянням кортежів.
    ``[K: tuple, V]`` — параметри типу функції (як generic-метод ``<K, V>`` у Java).
    """
    return sorted(results.items(), key=lambda item: (models.index(item[0][0]), *item[0][1:]))
