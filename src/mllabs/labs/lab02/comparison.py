"""Основна сітка «модель × сценарій» у крос-валідації та її таблиці.

Усе тут рахується ЛИШЕ на навчальній вибірці. Сценарії порівнюються парно:
на тих самих фолдах d_k = PR-AUC(складніший) − PR-AUC(простіший),
Δ = mean(d), s = std(d, ddof=1); «краще», лише якщо Δ > s — те саме правило,
за яким у ЛР1 обирався кожен елемент препроцесингу.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from mllabs.labs.lab02.runner import ConfigurationKey, ConfigurationResult, ConfigurationRunner
from mllabs.labs.lab02.scenarios import ScenarioCatalog, ScenarioId
from mllabs.modeling import ModelFactory, ModelKind
from mllabs.preprocessing import PairedDifference, paired_difference

#: Діагностичні варіанти та з чим кожен порівнюється парно.
DECOMPOSITION_STEPS: tuple[tuple[ScenarioId, ScenarioId | None], ...] = (
    (ScenarioId.A, None),
    (ScenarioId.A_MISSING, ScenarioId.A),
    (ScenarioId.B, ScenarioId.A_MISSING),
    (ScenarioId.B_MICE, ScenarioId.B),
)

#: Парні порівняння сценаріїв завдання: (простіший, складніший).
SCENARIO_PAIRS: tuple[tuple[ScenarioId, ScenarioId], ...] = (
    (ScenarioId.A, ScenarioId.B),
    (ScenarioId.B, ScenarioId.C_SMOTE),
    (ScenarioId.B, ScenarioId.C_ADASYN),
)


@dataclass(frozen=True)
class GridResult:
    """Результати крос-валідації основної сітки та діагностичних варіантів.

    Attributes:
        results: Підсумок кожної конфігурації за ключем ``(модель, сценарій)``.
    """

    results: Mapping[tuple[ModelKind, ScenarioId], ConfigurationResult]

    def get(self, model: ModelKind, scenario: ScenarioId) -> ConfigurationResult:
        """Підсумок конфігурації."""
        return self.results[(model, scenario)]

    def has(self, model: ModelKind, scenario: ScenarioId) -> bool:
        """Чи оцінювалася конфігурація."""
        return (model, scenario) in self.results

    def paired(
        self, model: ModelKind, simpler: ScenarioId, complex_: ScenarioId
    ) -> PairedDifference:
        """Парна різниця PR-AUC «складніший − простіший» по фолдах."""
        return paired_difference(
            self.get(model, simpler).fold_pr_auc, self.get(model, complex_).fold_pr_auc
        )

    def best(self, scenarios: Iterable[ScenarioId]) -> ConfigurationResult:
        """Конфігурація з найвищим середнім PR-AUC CV серед указаних сценаріїв."""
        allowed = set(scenarios)
        candidates = [result for key, result in self.results.items() if key[1] in allowed]
        return max(candidates, key=lambda result: result.pr_auc)


class ScenarioComparison:
    """Запускає сітку конфігурацій і будує таблиці для звіту.

    Args:
        runner: Виконавець крос-валідації конфігурацій.
        catalog: Сценарії (для підписів).
        models: Моделі, що порівнюються.
        decomposition_models: Моделі для діагностичних варіантів.
        sampling_strategy: Типова частота аугментації сценарію В.
    """

    def __init__(
        self,
        runner: ConfigurationRunner,
        catalog: ScenarioCatalog,
        models: Sequence[ModelKind],
        decomposition_models: Sequence[ModelKind],
        sampling_strategy: float,
    ) -> None:
        self._runner = runner
        self._catalog = catalog
        self._models = tuple(models)
        self._decomposition_models = tuple(decomposition_models)
        self._sampling_strategy = sampling_strategy

    def run(self) -> GridResult:
        """Оцінює всі моделі на сценаріях завдання і діагностичні варіанти."""
        keys = [
            self._key(model, spec.scenario_id)
            for model in self._models
            for spec in self._catalog.main()
        ]
        keys += [
            self._key(model, scenario)
            for model in self._decomposition_models
            for scenario in (ScenarioId.A_MISSING, ScenarioId.B_MICE)
        ]
        return GridResult({(key.model, key.scenario): self._runner.run(key) for key in keys})

    def _key(self, model: ModelKind, scenario: ScenarioId) -> ConfigurationKey:
        augmented = self._catalog.spec(scenario).is_augmented
        return ConfigurationKey(model, scenario, self._sampling_strategy if augmented else 0.0)

    # ------------------------------------------------------------------ таблиці

    def cv_results_table(self, grid: GridResult) -> pd.DataFrame:
        """Головні метрики CV сценаріїв завдання: середнє по 5 фолдах."""
        rows = [
            {
                "Модель": ModelFactory.short_name(model),
                "Сценарій": spec.label,
                "PR-AUC": result.pr_auc,
                "σ PR-AUC": result.cv_std("pr_auc"),
                "ROC-AUC": result.cv_mean("roc_auc"),
                "F1 (0,5)": result.cv_mean("f1"),
                "F1 (поріг)": result.cv_mean("f1", tuned=True),
            }
            for model in self._models
            for spec in self._catalog.main()
            for result in [grid.get(model, spec.scenario_id)]
        ]
        return pd.DataFrame(rows)

    def plot_frame(self, grid: GridResult) -> pd.DataFrame:
        """Довга таблиця для графіка PR-AUC CV: ``model, scenario, pr_auc, pr_auc_std``."""
        rows = [
            {
                "model": model.value,
                "scenario": spec.scenario_id.value,
                "pr_auc": grid.get(model, spec.scenario_id).pr_auc,
                "pr_auc_std": grid.get(model, spec.scenario_id).cv_std("pr_auc"),
            }
            for model in self._models
            for spec in self._catalog.main()
        ]
        return pd.DataFrame(rows)

    def thresholds_table(self, grid: GridResult) -> pd.DataFrame:
        """Поріг на OOF-прогнозах і precision/recall при 0,5 та при ньому."""
        rows = [
            {
                "Модель": ModelFactory.short_name(model),
                "Сценарій": spec.label,
                "Поріг": result.threshold,
                "Precision (0,5)": result.cv_mean("precision"),
                "Recall (0,5)": result.cv_mean("recall"),
                "Precision (поріг)": result.cv_mean("precision", tuned=True),
                "Recall (поріг)": result.cv_mean("recall", tuned=True),
            }
            for model in self._models
            for spec in self._catalog.main()
            for result in [grid.get(model, spec.scenario_id)]
        ]
        return pd.DataFrame(rows)

    def folds_table(self, grid: GridResult) -> pd.DataFrame:
        """PR-AUC і ROC-AUC кожного фолда кожної конфігурації (для парних різниць)."""
        frames = []
        for (model, scenario), result in grid.results.items():
            folds = result.folds_default[["pr_auc", "roc_auc"]].reset_index()
            frames.append(
                folds.assign(model=model.value, scenario=scenario.value)[
                    ["model", "scenario", "fold", "pr_auc", "roc_auc"]
                ]
            )
        table = pd.concat(frames, ignore_index=True)
        return table.rename(
            columns={
                "model": "Модель",
                "scenario": "Сценарій",
                "fold": "Фолд",
                "pr_auc": "PR-AUC",
                "roc_auc": "ROC-AUC",
            }
        )

    def decomposition_table(self, grid: GridResult) -> pd.DataFrame:
        """Покроковий перехід А → А_missing → Б → Б_MICE з парними різницями."""
        rows = []
        for model in self._decomposition_models:
            for scenario, reference in DECOMPOSITION_STEPS:
                result = grid.get(model, scenario)
                difference = (
                    grid.paired(model, reference, scenario) if reference is not None else None
                )
                rows.append(
                    {
                        "Модель": ModelFactory.short_name(model),
                        "Варіант": self._catalog.label(scenario),
                        "PR-AUC": result.pr_auc,
                        "σ": result.cv_std("pr_auc"),
                        "Порівняно з": self._catalog.label(reference) if reference else "—",
                        "Δ": difference.delta if difference else float("nan"),
                        "s": difference.spread if difference else float("nan"),
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ metrics.json

    def to_metrics(self, grid: GridResult) -> dict[str, Any]:
        """Розділ ``cv`` для ``metrics.json``."""
        metrics: dict[str, Any] = {}
        for model in self._models:
            metrics[model.value] = {
                spec.scenario_id.value: _configuration_metrics(grid.get(model, spec.scenario_id))
                for spec in self._catalog.main()
            }
        best = grid.best(spec.scenario_id for spec in self._catalog.main())
        metrics["best"] = {
            "model": best.key.model.value,
            "scenario": best.key.scenario.value,
            "pr_auc": best.pr_auc,
        }
        metrics["decomposition"] = {
            model.value: {
                scenario.value: {
                    "pr_auc": grid.get(model, scenario).pr_auc,
                    "pr_auc_std": grid.get(model, scenario).cv_std("pr_auc"),
                    "convergence_warnings": grid.get(model, scenario).convergence_warnings,
                    **(
                        {
                            "reference": reference.value,
                            "delta": grid.paired(model, reference, scenario).delta,
                            "spread": grid.paired(model, reference, scenario).spread,
                            "improves": grid.paired(model, reference, scenario).complex_wins,
                        }
                        if reference is not None
                        else {}
                    ),
                }
                for scenario, reference in DECOMPOSITION_STEPS
            }
            for model in self._decomposition_models
        }
        metrics["paired"] = {
            model.value: {
                f"{complex_.value}_vs_{simpler.value}": {
                    "delta": grid.paired(model, simpler, complex_).delta,
                    "spread": grid.paired(model, simpler, complex_).spread,
                    "improves": grid.paired(model, simpler, complex_).complex_wins,
                }
                for simpler, complex_ in SCENARIO_PAIRS
            }
            for model in self._models
        }
        return metrics


def _configuration_metrics(result: ConfigurationResult) -> dict[str, Any]:
    """Ключові числа однієї конфігурації для ``metrics.json``."""
    return {
        "pr_auc": result.pr_auc,
        "pr_auc_std": result.cv_std("pr_auc"),
        "roc_auc": result.cv_mean("roc_auc"),
        "roc_auc_std": result.cv_std("roc_auc"),
        "accuracy": result.cv_mean("accuracy"),
        "precision": result.cv_mean("precision"),
        "recall": result.cv_mean("recall"),
        "f1": result.cv_mean("f1"),
        "f1_tuned": result.cv_mean("f1", tuned=True),
        "precision_tuned": result.cv_mean("precision", tuned=True),
        "recall_tuned": result.cv_mean("recall", tuned=True),
        "accuracy_tuned": result.cv_mean("accuracy", tuned=True),
        "brier": result.cv_mean("brier"),
        "mean_proba": result.cv_mean("mean_proba"),
        "threshold": result.threshold,
        "train_pr_auc": float(result.train_pr_auc.mean()),
        "gap": result.train_gap,
        "fit_seconds": result.fit_seconds,
        "predict_seconds": result.predict_seconds,
        "n_features": result.n_features,
    }
