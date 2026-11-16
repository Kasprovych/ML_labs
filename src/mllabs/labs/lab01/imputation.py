"""Порівняння методів заповнення пропусків (імп'ютації)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

# IterativeImputer — «експериментальний» клас sklearn: потребує імпорту-перемикача.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from mllabs.labs.lab01.timing import FOLD_TIME_COLUMN, approximate_seconds
from mllabs.preprocessing import (
    CategoricalImputation,
    CvScores,
    ImputerKind,
    PreprocessingOptions,
    PreprocessorFactory,
    ProxyModelEvaluator,
)

_IMPUTER_LABELS = {
    ImputerKind.MEDIAN: "Медіана",
    ImputerKind.KNN: "KNN",
    ImputerKind.MICE: "MICE",
}
_CATEGORICAL_LABELS = {
    CategoricalImputation.MODE: "Мода",
    CategoricalImputation.MISSING_CATEGORY: "Категорія MISSING",
}


@dataclass(frozen=True)
class MaskingResult:
    """Результат експерименту з маскуванням.

    Attributes:
        rmse_by_feature: RMSE відновлення по ознаках (рядки) і методах (колонки),
            в одиницях стандартного відхилення ознаки.
        rmse_overall: RMSE по всіх прихованих значеннях разом, по методах.
        restored: Довга таблиця прихованих значень у вихідних одиницях:
            ``feature``, ``method``, ``true``, ``imputed``.
    """

    rmse_by_feature: pd.DataFrame
    rmse_overall: dict[str, float]
    restored: pd.DataFrame


class ImputationBenchmark:
    """Порівнює медіану, KNN і MICE.

    Проблема: якість імп'ютації не можна виміряти на справжніх пропусках —
    невідомо, що там було. Тому в рядках БЕЗ пропусків частина значень штучно
    ховається (маскування MCAR — «missing completely at random»), імп'ютер їх
    відновлює, а ми порівнюємо з правдою.

    Args:
        mask_share: Частка значень, що ховаються.
        sample_size: Розмір підвибірки повних рядків (KNN квадратичний за N).
        knn_neighbors: Кількість сусідів ``KNNImputer``.
        mice_max_iter: Максимум ітерацій ``IterativeImputer``.
        random_state: Зерно для підвибірки, маски та MICE.
    """

    def __init__(
        self,
        mask_share: float,
        sample_size: int,
        knn_neighbors: int,
        mice_max_iter: int,
        random_state: int,
    ) -> None:
        self._mask_share = mask_share
        self._sample_size = sample_size
        self._knn_neighbors = knn_neighbors
        self._mice_max_iter = mice_max_iter
        self._random_state = random_state

    def run_masking(self, numeric: pd.DataFrame) -> MaskingResult:
        """Ховає частину значень у повних рядках і міряє похибку відновлення.

        Args:
            numeric: Числові ознаки навчальної вибірки.
        """
        complete = numeric.dropna()
        sample = complete.sample(
            n=min(self._sample_size, len(complete)), random_state=self._random_state
        )
        # Стандартизація: KNN рахує відстані, тож ознаки мають бути в одному масштабі;
        # заодно RMSE різних ознак стає порівнянним (1.0 ≈ «вгадування середнім»).
        mean, std = sample.mean(), sample.std()
        truth = (sample - mean) / std

        generator = np.random.default_rng(self._random_state)
        hidden = generator.random(truth.shape) < self._mask_share  # булева матриця маски
        masked = truth.mask(hidden)

        imputers = {
            ImputerKind.MEDIAN: SimpleImputer(strategy="median"),
            ImputerKind.KNN: KNNImputer(n_neighbors=self._knn_neighbors),
            ImputerKind.MICE: IterativeImputer(
                max_iter=self._mice_max_iter, random_state=self._random_state
            ),
        }
        rmse_by_feature: dict[str, pd.Series] = {}
        rmse_overall: dict[str, float] = {}
        restored_parts: list[pd.DataFrame] = []
        for kind, imputer in imputers.items():
            imputed = pd.DataFrame(
                imputer.fit_transform(masked), index=truth.index, columns=truth.columns
            )
            # where(hidden) лишає тільки приховані клітинки (решта стає NaN і
            # не бере участі в середньому): похибка рахується лише там, де ховали.
            squared_error = ((imputed - truth) ** 2).where(hidden)
            rmse_by_feature[_IMPUTER_LABELS[kind]] = np.sqrt(squared_error.mean())
            rmse_overall[kind.value] = float(np.sqrt(squared_error.stack().mean()))
            restored_parts.append(
                self._restored_values(truth, imputed, hidden, mean, std).assign(
                    method=_IMPUTER_LABELS[kind]
                )
            )
        return MaskingResult(
            rmse_by_feature=pd.DataFrame(rmse_by_feature),
            rmse_overall=rmse_overall,
            # concat склеює таблиці одна під одною (як UNION ALL).
            restored=pd.concat(restored_parts, ignore_index=True),
        )

    @staticmethod
    def _restored_values(
        truth: pd.DataFrame,
        imputed: pd.DataFrame,
        hidden: np.ndarray,
        mean: pd.Series,
        std: pd.Series,
    ) -> pd.DataFrame:
        """Збирає приховані значення (справжні й відновлені) у вихідних одиницях."""
        # melt «розгортає» широку таблицю в довгу: (ознака, значення) у кожному рядку.
        true_long = (truth * std + mean).where(hidden).melt(var_name="feature", value_name="true")
        imputed_long = (imputed * std + mean).where(hidden).melt(value_name="imputed")
        return true_long.assign(imputed=imputed_long["imputed"]).dropna()

    def compare_on_proxy(
        self,
        factory: PreprocessorFactory,
        evaluator: ProxyModelEvaluator,
        base_options: PreprocessingOptions,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[dict[ImputerKind, CvScores], dict[CategoricalImputation, CvScores]]:
        """Міряє вплив способу імп'ютації на проксі-модель (крос-валідація на train).

        Змінюється лише один параметр за раз; решта препроцесингу — ``base_options``
        (у ЛР1 — остаточний сценарій FULL, щоб рішення оцінювалось у ньому).

        Returns:
            Пара словників: метрики для числових імп'ютерів і для способів
            заповнення категоріальних ознак.
        """
        numeric_scores = {
            kind: evaluator.evaluate(
                factory.build_from_options(replace(base_options, imputer=kind)), features, target
            )
            for kind in ImputerKind
        }
        categorical_scores = {
            kind: evaluator.evaluate(
                factory.build_from_options(replace(base_options, categorical_imputation=kind)),
                features,
                target,
            )
            for kind in CategoricalImputation
        }
        return numeric_scores, categorical_scores

    @staticmethod
    def comparison_table(
        masking: MaskingResult,
        numeric_scores: Mapping[ImputerKind, CvScores],
        categorical_scores: Mapping[CategoricalImputation, CvScores],
    ) -> pd.DataFrame:
        """Зводить RMSE відновлення та метрики проксі-моделі в одну таблицю."""
        rows = [
            {
                "Ознаки": "числові",
                "Метод": _IMPUTER_LABELS[kind],
                "RMSE відновлення": masking.rmse_overall[kind.value],
                "PR-AUC": scores.pr_auc,
                "ROC-AUC": scores.roc_auc,
                FOLD_TIME_COLUMN: approximate_seconds(scores.fit_seconds),
            }
            for kind, scores in numeric_scores.items()
        ]
        rows += [
            {
                "Ознаки": "категоріальні",
                "Метод": _CATEGORICAL_LABELS[kind],
                "RMSE відновлення": np.nan,  # для категорій RMSE не визначений
                "PR-AUC": scores.pr_auc,
                "ROC-AUC": scores.roc_auc,
                FOLD_TIME_COLUMN: approximate_seconds(scores.fit_seconds),
            }
            for kind, scores in categorical_scores.items()
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def features_to_show(masking: MaskingResult) -> list[str]:
        """Три ознаки для графіка розподілів: KNN відновлює їх найгірше, середньо й найкраще."""
        ordered = masking.rmse_by_feature[_IMPUTER_LABELS[ImputerKind.KNN]].sort_values()
        return [ordered.index[0], ordered.index[len(ordered) // 2], ordered.index[-1]]

    @staticmethod
    def positive_rate_by_missingness(column: pd.Series, is_positive: pd.Series) -> dict[str, float]:
        """Частка позитивного класу серед рядків із пропуском і без нього.

        Якщо частки сильно різняться — пропуск інформативний (не випадковий), і
        його варто кодувати окремою категорією, а не ховати за модою.
        """
        is_missing = column.isna()
        return {
            "missing": float(is_positive[is_missing].mean()),
            "present": float(is_positive[~is_missing].mean()),  # ~ — логічне «НЕ» для масок
        }

    @staticmethod
    def to_metrics(
        masking: MaskingResult,
        numeric_scores: Mapping[ImputerKind, CvScores],
        categorical_scores: Mapping[CategoricalImputation, CvScores],
        wheel_type_rates: Mapping[str, float],
    ) -> dict[str, object]:
        """Словник для ``metrics.json``."""
        metrics: dict[str, object] = {
            f"rmse_{name}": value for name, value in masking.rmse_overall.items()
        }
        # min(словник, key=словник.get) — ключ із найменшим значенням.
        metrics["best_method"] = min(masking.rmse_overall, key=masking.rmse_overall.get)
        metrics["pr_auc"] = {kind.value: scores.pr_auc for kind, scores in numeric_scores.items()}
        metrics["pr_auc_std"] = {
            kind.value: scores.pr_auc_std for kind, scores in numeric_scores.items()
        }
        # PR-AUC по фолдах — для парного порівняння варіантів за правилом вибору.
        metrics["pr_auc_folds"] = {
            kind.value: list(scores.fold_pr_auc) for kind, scores in numeric_scores.items()
        }
        # Кількість ознак на вході моделі — мірило складності варіанта для правила вибору.
        metrics["n_features"] = {
            kind.value: scores.n_features for kind, scores in numeric_scores.items()
        }
        metrics["n_features_categorical"] = {
            kind.value: scores.n_features for kind, scores in categorical_scores.items()
        }
        metrics["pr_auc_folds_categorical"] = {
            kind.value: list(scores.fold_pr_auc) for kind, scores in categorical_scores.items()
        }
        metrics["pr_auc_categorical"] = {
            kind.value: scores.pr_auc for kind, scores in categorical_scores.items()
        }
        metrics["wheel_type_missing_positive_rate"] = wheel_type_rates["missing"]
        metrics["wheel_type_present_positive_rate"] = wheel_type_rates["present"]
        return metrics
