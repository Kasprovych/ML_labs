"""Baseline: HistGradientBoosting з параметрами ЛР2, без підбору гіперпараметрів.

Baseline — точка відліку: що дає тюнінг понад «стандартні» параметри. Щоб
порівняння було чесним, він оцінюється на ТИХ САМИХ розбиттях dev-вибірки:

* на 3 внутрішніх фолдах — порівнянно з Best Score (Inner CV) методів пошуку;
* на 5 зовнішніх фолдах — порівнянно з Unbiased Score (Nested CV). Baseline
  нічого не обирає, тож його зовнішня оцінка вже незміщена;
* на 3 внутрішніх фолдах КОЖНОГО зовнішнього train — так само, як вкладена CV
  оцінює кандидатів. Різниця «inner − outer» baseline показує, наскільки
  внутрішня оцінка відрізняється від зовнішньої БЕЗ жодного відбору: внутрішні
  моделі вчаться на 2/3 зовнішнього train (~10,7 тис. рядків), а зовнішня — на
  всьому (~16 тис.), і фолди різні за складністю. Віднявши цей зсув від
  «inner − outer» методу пошуку, отримуємо чистий оптимізм ВІДБОРУ.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from mllabs.modeling import CrossValidator, MetricsCalculator, mean_and_std

#: Ключ сценарію Б (повний препроцесинг ЛР1) у розділі ``test`` метрик ЛР2.
_LAB2_SCENARIO_KEY = "B"


@dataclass(frozen=True)
class BaselineResult:
    """Оцінки baseline на внутрішніх і зовнішніх розбиттях.

    Attributes:
        inner_scores: PR-AUC на 3 внутрішніх фолдах.
        inner_wall_seconds: Реальний час 3-фолдової оцінки, с.
        inner_compute_seconds: Сумарний час навчання + прогнозу 3 фолдів, с.
        outer_scores: PR-AUC на 5 зовнішніх фолдах.
        outer_roc_auc: ROC-AUC на 5 зовнішніх фолдах.
        outer_train_scores: PR-AUC на власних навчальних рядках зовнішніх фолдів.
        inner_in_outer_scores: Середній PR-AUC 3 внутрішніх фолдів усередині
            кожного зовнішнього train (аналог ``inner_best`` вкладеної CV, але
            для однієї конфігурації — без вибору найкращої).
    """

    inner_scores: tuple[float, ...]
    inner_wall_seconds: float
    inner_compute_seconds: float
    outer_scores: tuple[float, ...]
    outer_roc_auc: tuple[float, ...]
    outer_train_scores: tuple[float, ...]
    inner_in_outer_scores: tuple[float, ...]

    @property
    def inner_mean_std(self) -> tuple[float, float]:
        """Середній PR-AUC на внутрішніх фолдах і σ."""
        return mean_and_std(self.inner_scores)

    @property
    def outer_mean_std(self) -> tuple[float, float]:
        """Середній PR-AUC на зовнішніх фолдах і σ."""
        return mean_and_std(self.outer_scores)

    @property
    def train_mean(self) -> float:
        """Середній PR-AUC на навчальних рядках (для розриву train − CV)."""
        return float(np.mean(self.outer_train_scores))

    @property
    def fold_shift(self) -> np.ndarray:
        """«Inner − outer» baseline по зовнішніх фолдах: зсув без відбору."""
        return np.array(self.inner_in_outer_scores) - np.array(self.outer_scores)

    @property
    def fold_shift_mean_std(self) -> tuple[float, float]:
        """Середній зсув «inner − outer» baseline і його σ по фолдах."""
        return mean_and_std(self.fold_shift)


class BaselineEvaluator:
    """Оцінює baseline-пайплайн на внутрішніх і зовнішніх розбиттях dev.

    Args:
        inner_cv: ``CrossValidator`` внутрішньої петлі (ті самі 3 фолди, що в пошуку).
        outer_cv: ``CrossValidator`` зовнішньої петлі (ті самі 5 фолдів, що у вкладеній CV).
        calculator: Калькулятор метрик.
    """

    def __init__(
        self, inner_cv: CrossValidator, outer_cv: CrossValidator, calculator: MetricsCalculator
    ) -> None:
        self._inner_cv = inner_cv
        self._outer_cv = outer_cv
        self._calculator = calculator

    def evaluate(
        self, pipeline: BaseEstimator, features: pd.DataFrame, target: pd.Series
    ) -> BaselineResult:
        """3-фолдова і 5-фолдова CV baseline + 3-фолдова всередині кожного зовнішнього train."""
        started = time.perf_counter()
        inner = self._inner_cv.evaluate(pipeline, features, target)
        inner_wall = time.perf_counter() - started
        outer = self._outer_cv.evaluate(pipeline, features, target)
        roc_auc = outer.fold_metrics(self._calculator)["roc_auc"]
        # Ті самі внутрішні розбиття, що бачить пошук у вкладеній CV: splits()
        # залежить лише від міток зовнішнього train і зерна.
        inner_in_outer = [
            float(
                self._inner_cv.evaluate(pipeline, features.iloc[train], target.iloc[train])
                .valid_pr_auc(self._calculator)
                .mean()
            )
            for train, _ in self._outer_cv.splits(features, target)
        ]
        return BaselineResult(
            inner_scores=tuple(inner.valid_pr_auc(self._calculator).tolist()),
            inner_wall_seconds=inner_wall,
            inner_compute_seconds=float(
                sum(fold.fit_seconds + fold.predict_seconds for fold in inner.folds)
            ),
            outer_scores=tuple(outer.valid_pr_auc(self._calculator).tolist()),
            outer_roc_auc=tuple(roc_auc.tolist()),
            outer_train_scores=tuple(outer.train_pr_auc(self._calculator).tolist()),
            inner_in_outer_scores=tuple(inner_in_outer),
        )


@dataclass(frozen=True)
class Lab2Reference:
    """Довідкові числа HGB (сценарій Б) з ЛР2 на ПОВНОМУ train — не для прямого порівняння.

    Attributes:
        cv_pr_auc: PR-AUC 5-кратної CV ЛР2.
        train_cv_gap: Розрив «PR-AUC на train − CV» у ЛР2.
        test_pr_auc: PR-AUC на test у ЛР2 (модель навчена на 58 тис. рядків).
    """

    cv_pr_auc: float
    train_cv_gap: float
    test_pr_auc: float


def read_lab2_reference(metrics_file: Path, model_key: str) -> Lab2Reference | None:
    """Читає довідку ЛР2 з її ``metrics.json``; ``None``, якщо файлу чи ключів немає."""
    if not metrics_file.is_file():
        return None
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    try:
        handoff = metrics["lab3_handoff"]
        return Lab2Reference(
            cv_pr_auc=float(handoff[model_key]["cv_pr_auc"]),
            train_cv_gap=float(handoff[model_key]["train_cv_gap"]),
            test_pr_auc=float(metrics["test"][model_key][_LAB2_SCENARIO_KEY]["pr_auc"]),
        )
    except KeyError:
        return None
