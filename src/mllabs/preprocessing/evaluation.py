"""Оцінювання варіантів препроцесингу проксі-моделлю в крос-валідації.

«Проксі-модель» — проста модель, яка тут є вимірювальним приладом, а не метою:
її якість показує, наскільки корисним для навчання є той чи інший спосіб
підготовки даних. Порівняння самих моделей — тема ЛР2.

Це інструмент ЛР1, а не загальний оцінювач: модель зафіксована (логістична
регресія), повертаються лише середні по фолдах. Для ЛР2–3 (різні моделі,
підбір гіперпараметрів, оцінка на тесті) потрібен власний код оцінювання.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from mllabs.preprocessing.pipelines import MODEL_STEP, build_model_pipeline


@dataclass(frozen=True)
class CvScores:
    """Середні метрики крос-валідації (усі — для позитивного, рідкісного класу).

    Attributes:
        pr_auc: Площа під кривою precision–recall (головна метрика при
            дисбалансі: не залежить від порога і чутлива саме до рідкісного класу).
        pr_auc_std: Вибіркове стандартне відхилення PR-AUC між фолдами
            (``ddof=1``) — «шум» оцінки одного варіанта.
        fold_pr_auc: PR-AUC на кожному фолді. Фолди однакові для всіх варіантів,
            оцінених одним оцінювачем на тих самих даних, тому два варіанти
            порівнюються ПОПАРНО по фолдах (``mllabs.preprocessing.selection``).
        roc_auc: Площа під ROC-кривою.
        f1: F1 при стандартному порозі 0.5.
        recall: Повнота — частка знайдених «поганих» авто (поріг 0.5).
        precision: Точність — частка справді «поганих» серед позначених (поріг 0.5).
        n_features: Кількість ознак на вході моделі після препроцесингу.
        fit_seconds: Середній час навчання одного фолда, с.
    """

    pr_auc: float
    pr_auc_std: float
    roc_auc: float
    f1: float
    recall: float
    precision: float
    n_features: int
    fit_seconds: float
    fold_pr_auc: tuple[float, ...] = ()


class ProxyModelEvaluator:
    """Оцінює препроцесор (і, за бажання, семплер) логістичною регресією.

    Інструмент саме ЛР1: модель фіксована (``LogisticRegression`` зі
    стандартною L2-регуляризацією), результат — лише середні метрики по фолдах
    і σ PR-AUC. Він відповідає на питання «який препроцесинг кращий за інших
    рівних», а не «яка модель найкраща», тому НЕ призначений як загальний
    оцінювач для ЛР2–3.

    Для кожного фолда ``StratifiedKFold`` увесь ланцюжок «препроцесор →
    [семплер] → модель» навчається лише на навчальній частині фолда й
    оцінюється на відкладеній. Тестову вибірку цей клас не бачить узагалі.

    Args:
        n_splits: Кількість фолдів крос-валідації.
        max_iter: Ліміт ітерацій оптимізатора логістичної регресії.
        random_state: Зерно для перемішування фолдів і моделі.
        positive_label: Мітка позитивного (рідкісного) класу — з
            ``DatasetSchema.positive_label``; для неї рахуються PR-AUC, F1,
            recall і precision.
    """

    def __init__(
        self, n_splits: int, max_iter: int, random_state: int, positive_label: int
    ) -> None:
        self._n_splits = n_splits
        self._max_iter = max_iter
        self._random_state = random_state
        # make_scorer «загортає» функцію метрики в об'єкт, який cross_validate
        # викликає на кожному фолді; pos_label каже, який клас вважати позитивним.
        self._scoring = {
            "average_precision": make_scorer(
                average_precision_score, response_method="predict_proba", pos_label=positive_label
            ),
            "roc_auc": "roc_auc",  # симетрична щодо класів: pos_label не потрібен
            "f1": make_scorer(f1_score, pos_label=positive_label),
            "recall": make_scorer(recall_score, pos_label=positive_label),
            "precision": make_scorer(precision_score, pos_label=positive_label),
        }

    def evaluate(
        self,
        preprocessor: BaseEstimator,
        features: pd.DataFrame,
        target: pd.Series,
        sampler: BaseEstimator | None = None,
    ) -> CvScores:
        """Повертає середні метрики крос-валідації на навчальній вибірці.

        Args:
            preprocessor: Нефітований препроцесор (з ``PreprocessorFactory``).
            features: Ознаки навчальної вибірки (не тестової!).
            target: Цільова змінна навчальної вибірки.
            sampler: Нефітований семплер imblearn або ``None`` — без аугментації.
        """
        # Спільна функція збирає плоский ланцюжок «препроцесор → [семплер] → модель»;
        # ЛР2–3 будують свої пайплайни нею ж.
        pipeline = build_model_pipeline(
            preprocessor,
            LogisticRegression(max_iter=self._max_iter, random_state=self._random_state),
            sampler=sampler,
        )
        result = cross_validate(
            pipeline,
            features,
            target,
            cv=self.splits(features, target),
            scoring=self._scoring,
            return_estimator=True,
        )
        pr_auc = result["test_average_precision"]
        return CvScores(
            pr_auc=float(pr_auc.mean()),
            # ddof=1 — вибіркове σ (ділення на n − 1): на кількох фолдах ddof=0 його занижує.
            pr_auc_std=float(pr_auc.std(ddof=1)),
            roc_auc=float(result["test_roc_auc"].mean()),
            f1=float(result["test_f1"].mean()),
            recall=float(result["test_recall"].mean()),
            precision=float(result["test_precision"].mean()),
            n_features=int(result["estimator"][0].named_steps[MODEL_STEP].n_features_in_),
            fit_seconds=float(result["fit_time"].mean()),
            fold_pr_auc=tuple(float(value) for value in pr_auc),
        )

    def splits(
        self, features: pd.DataFrame, target: pd.Series
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Номери рядків «навчання / перевірка» для кожного фолда.

        Розбиття залежить лише від ``target``, кількості фолдів і зерна, тому
        для тих самих даних воно ОДНАКОВЕ в усіх викликах :meth:`evaluate` —
        це умова попарного порівняння варіантів по фолдах.
        """
        folds = StratifiedKFold(
            n_splits=self._n_splits, shuffle=True, random_state=self._random_state
        )
        return list(folds.split(features, target))
