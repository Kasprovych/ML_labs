"""Підсумкові таблиці: порівняльна таблиця для звіту, профіль моделей, передача в ЛР3.

Нічого не навчає — лише зводить уже пораховані результати крос-валідації
й тесту.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from mllabs.labs.lab02.comparison import GridResult
from mllabs.labs.lab02.runner import ConfigurationResult
from mllabs.labs.lab02.scenarios import ScenarioCatalog, ScenarioId, ScenarioSpec
from mllabs.labs.lab02.test_evaluation import HoldoutResult
from mllabs.modeling import ModelFactory, ModelKind

#: Колонки довгої таблиці часу навчання (для графіка).
COLUMN_MODEL = "model"
COLUMN_SCENARIO = "scenario"
COLUMN_FIT_SECONDS = "fit_seconds"


class ComparisonTableBuilder:
    """Компактна порівняльна таблиця «модель × сценарій»: CV і test поруч (R9).

    Args:
        catalog: Сценарії.
        models: Моделі.
    """

    def __init__(self, catalog: ScenarioCatalog, models: Sequence[ModelKind]) -> None:
        self._catalog = catalog
        self._models = tuple(models)

    def build(self, grid: GridResult, holdout: HoldoutResult) -> pd.DataFrame:
        """PR-AUC, ROC-AUC і F1 (при OOF-порозі) у CV і на test."""
        rows = []
        for model in self._models:
            for spec in self._catalog.main():
                cv = grid.get(model, spec.scenario_id)
                test = holdout.get(model, spec.scenario_id).at_threshold
                rows.append(
                    {
                        "Конфігурація": f"{ModelFactory.short_name(model)} · {spec.label}",
                        "PR-AUC CV": cv.pr_auc,
                        "PR-AUC test": test.pr_auc,
                        "ROC-AUC CV": cv.cv_mean("roc_auc"),
                        "ROC-AUC test": test.roc_auc,
                        "F1 CV": cv.cv_mean("f1", tuned=True),
                        "F1 test": test.f1,
                    }
                )
        return pd.DataFrame(rows)


class ModelProfileBuilder:
    """Профіль кожної моделі з вимірювань — основа аналізу сильних і слабких сторін (R7).

    Args:
        catalog: Сценарії.
        models: Моделі.
    """

    def __init__(self, catalog: ScenarioCatalog, models: Sequence[ModelKind]) -> None:
        self._catalog = catalog
        self._models = tuple(models)

    def build(self, grid: GridResult) -> pd.DataFrame:
        """Таблиця профілю: якість, чутливість до препроцесингу й аугментації, перенавчання, час.

        «Найкращий PR-AUC» — з найкращого сценарію моделі; колонки з позначкою «(Б)» —
        зі сценарію Б, щоб моделі порівнювалися на однакових даних.
        """
        rows = []
        for model in self._models:
            best = self._best(grid, model)
            base = grid.get(model, ScenarioId.B)
            rows.append(
                {
                    "Модель": ModelFactory.short_name(model),
                    "Найкращий PR-AUC": best.pr_auc,
                    "Δ Б − А": grid.paired(model, ScenarioId.A, ScenarioId.B).delta,
                    "Δ В − Б": grid.paired(model, ScenarioId.B, ScenarioId.C_SMOTE).delta,
                    "Розрив train − CV (Б)": base.train_gap,
                    "Навчання, с (Б)": base.fit_seconds,
                    "σ PR-AUC (Б)": base.cv_std("pr_auc"),
                }
            )
        return pd.DataFrame(rows)

    def fit_time_frame(self, grid: GridResult) -> pd.DataFrame:
        """Середній час навчання фолда за моделями й сценаріями (для графіка)."""
        rows = [
            {
                COLUMN_MODEL: model.value,
                COLUMN_SCENARIO: spec.scenario_id.value,
                COLUMN_FIT_SECONDS: grid.get(model, spec.scenario_id).fit_seconds,
            }
            for model in self._models
            for spec in self._catalog.main()
        ]
        return pd.DataFrame(rows)

    def to_metrics(self, grid: GridResult) -> dict[str, Any]:
        """Розділ ``profile`` для ``metrics.json``."""
        metrics = {}
        for model in self._models:
            best = self._best(grid, model)
            base = grid.get(model, ScenarioId.B)
            metrics[model.value] = {
                "best_pr_auc": best.pr_auc,
                "best_scenario": best.key.scenario.value,
                "preprocessing_gain": grid.paired(model, ScenarioId.A, ScenarioId.B).delta,
                "augmentation_gain": grid.paired(model, ScenarioId.B, ScenarioId.C_SMOTE).delta,
                "train_cv_gap": base.train_gap,
                "fit_seconds": base.fit_seconds,
                "predict_seconds": base.predict_seconds,
                "pr_auc_std": base.cv_std("pr_auc"),
                "n_features": base.n_features,
                "convergence_warnings_by_scenario": {
                    spec.scenario_id.value: grid.get(model, spec.scenario_id).convergence_warnings
                    for spec in self._catalog.main()
                },
            }
        return metrics

    def models_metrics(self, factory: ModelFactory, grid: GridResult) -> dict[str, Any]:
        """Розділ ``models`` для ``metrics.json``: параметри й попередження про незбіжність LR."""
        return {
            "settings": factory.settings.to_dict(),
            **{model.value: factory.describe(model) for model in self._models},
            "logreg_convergence_warnings": {
                spec.scenario_id.value: grid.get(
                    ModelKind.LOGREG, spec.scenario_id
                ).convergence_warnings
                for spec in self._catalog.main()
            },
        }

    def _best(self, grid: GridResult, model: ModelKind) -> ConfigurationResult:
        """Найкраща за CV конфігурація цієї моделі серед сценаріїв завдання."""
        main: list[ScenarioSpec] = self._catalog.main()
        return max(
            (grid.get(model, spec.scenario_id) for spec in main), key=lambda result: result.pr_auc
        )


def lab3_handoff(grid: GridResult, cv_folds: int, models: Sequence[ModelKind]) -> dict[str, Any]:
    """Що передається в ЛР3: сценарій, фолди та «точка відліку» деревних моделей."""
    handoff: dict[str, Any] = {"scenario": "full", "cv_folds": cv_folds}
    for model in models:
        if model is ModelKind.LOGREG:
            continue
        base = grid.get(model, ScenarioId.B)
        handoff[model.value] = {
            "cv_pr_auc": base.pr_auc,
            "cv_pr_auc_std": base.cv_std("pr_auc"),
            "train_pr_auc": float(base.train_pr_auc.mean()),
            "train_cv_gap": base.train_gap,
            "fit_seconds_per_fold": base.fit_seconds,
        }
    return handoff


def models_table(factory: ModelFactory, models: Sequence[ModelKind]) -> pd.DataFrame:
    """Таблиця моделей: сімейство і ключові гіперпараметри."""
    rows = [
        {
            "Модель": ModelFactory.display_name(model),
            "Скорочення": ModelFactory.short_name(model),
            "Сімейство": ModelFactory.family(model),
            "Ключові параметри": ", ".join(
                f"{key}={value}" for key, value in factory.describe(model).items()
            ),
        }
        for model in models
    ]
    return pd.DataFrame(rows)
