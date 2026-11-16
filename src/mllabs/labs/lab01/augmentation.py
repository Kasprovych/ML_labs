"""Аугментація: порівняння SMOTE, ADASYN і гаусового шуму."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA

from mllabs.preprocessing import CvScores, ProxyModelEvaluator, SamplerFactory, SamplerKind

SAMPLER_LABELS = {
    SamplerKind.NONE: "Без аугментації",
    SamplerKind.SMOTE: "SMOTE",
    SamplerKind.ADASYN: "ADASYN",
    SamplerKind.GAUSSIAN_NOISE: "Гаусів шум",
}
# У підписах «клас 1» — позитивний клас схеми (для kick: IsBadBuy = 1, «погана» покупка);
# самі обчислення порівнюють мітки з ``positive_label``, а не з числом 1.
KIND_MAJORITY = "клас 0 (реальні)"
POSITIVE_SHARE_COLUMN = "Частка позитивного класу, %"
KIND_MINORITY = "клас 1 (реальні)"
KIND_SYNTHETIC = "клас 1 (синтетичні)"


@dataclass(frozen=True)
class ResamplingIllustration:
    """Що семплери роблять із навчальною вибіркою (ілюстрація, не оцінка).

    Attributes:
        class_balance: Таблиця по методах: кількість рядків, розподіл класів,
            кількість синтетичних рядків.
        projection: Точки на площині PCA: ``method``, ``kind``, ``pc1``, ``pc2``.
        explained_variance: Частки дисперсії двох головних компонент.
    """

    class_balance: pd.DataFrame
    projection: pd.DataFrame
    explained_variance: tuple[float, float]


class AugmentationBenchmark:
    """Порівнює методи синтезу рядків рідкісного класу.

    Дві незалежні частини:

    * :meth:`illustrate` — показує, ЩО роблять семплери: баланс класів до/після
      і де на площині PCA лягають синтетичні точки. Семплери застосовуються до
      всієї навчальної вибірки, але жодна метрика тут не рахується.
    * :meth:`compare` — міряє, ЧИ ДОПОМАГАЄ аугментація моделі. Тут семплер
      стоїть усередині пайплайна крос-валідації: синтетичні рядки додаються
      лише в навчальну частину кожного фолда, а оцінка — на реальних рядках.

    Args:
        sampler_factory: Фабрика семплерів з однаковими налаштуваннями.
        plot_sample_size: Скільки реальних точок (обох класів разом) показати на
            графіку; синтетичні беруться в тій самій пропорції.
        random_state: Зерно для підвибірки точок.
        positive_label: Мітка позитивного (рідкісного) класу — з
            ``DatasetSchema.positive_label``.
    """

    def __init__(
        self,
        sampler_factory: SamplerFactory,
        plot_sample_size: int,
        random_state: int,
        positive_label: int,
    ) -> None:
        self._sampler_factory = sampler_factory
        self._plot_sample_size = plot_sample_size
        self._random_state = random_state
        self._positive_label = positive_label

    def illustrate(
        self, preprocessor: BaseEstimator, features: pd.DataFrame, target: pd.Series
    ) -> ResamplingIllustration:
        """Застосовує кожен семплер до підготовленої навчальної вибірки.

        Args:
            preprocessor: Нефітований препроцесор; навчається тут на train.
            features: Ознаки навчальної вибірки.
            target: Цільова змінна навчальної вибірки.
        """
        matrix = clone(preprocessor).fit_transform(features, target)
        labels = target.to_numpy()
        pca = PCA(n_components=2).fit(matrix)  # площина будується лише за реальними точками
        generator = np.random.default_rng(self._random_state)
        # Однакова частка точок із кожної групи: на графіку зберігаються реальні
        # пропорції класів (рідкісний клас не «затоплює» мажоритарний).
        plot_share = min(1.0, self._plot_sample_size / len(labels))

        balance_rows = []
        projections = []
        for kind in SamplerKind:
            sampler = self._sampler_factory.create(kind)
            if sampler is None:
                resampled, resampled_labels = matrix, labels
            else:
                resampled, resampled_labels = sampler.fit_resample(matrix, labels)
            n_positive = int((resampled_labels == self._positive_label).sum())
            balance_rows.append(
                {
                    "Метод": SAMPLER_LABELS[kind],
                    "Рядків": len(resampled_labels),
                    "Клас 0": len(resampled_labels) - n_positive,
                    "Клас 1": n_positive,
                    "Синтетичних": len(resampled_labels) - len(labels),
                    POSITIVE_SHARE_COLUMN: n_positive / len(resampled_labels) * 100,
                }
            )
            # Семплери imblearn дописують синтетичні рядки в кінець: перші
            # len(labels) рядків — реальні, решта — згенеровані.
            is_positive = labels == self._positive_label
            groups = {
                KIND_MAJORITY: matrix[~is_positive],
                KIND_MINORITY: matrix[is_positive],
                KIND_SYNTHETIC: resampled[len(labels) :],
            }
            for group_name, rows in groups.items():
                projections.append(
                    self._project(pca, rows, plot_share, generator, kind, group_name)
                )
        explained = pca.explained_variance_ratio_
        return ResamplingIllustration(
            class_balance=pd.DataFrame(balance_rows),
            projection=pd.concat(projections, ignore_index=True),
            explained_variance=(float(explained[0]), float(explained[1])),
        )

    def _project(
        self,
        pca: PCA,
        rows: np.ndarray,
        share: float,
        generator: np.random.Generator,
        kind: SamplerKind,
        group_name: str,
    ) -> pd.DataFrame:
        """Проєктує випадкову підвибірку рядків (частку ``share``) на площину PCA."""
        n_points = round(share * len(rows))
        picked = generator.choice(len(rows), size=n_points, replace=False)
        coordinates = pca.transform(rows[picked]) if n_points else np.empty((0, 2))
        return pd.DataFrame(
            {
                "method": SAMPLER_LABELS[kind],
                "kind": group_name,
                "pc1": coordinates[:, 0],  # [:, 0] — усі рядки, перша колонка
                "pc2": coordinates[:, 1],
            }
        )

    def compare(
        self,
        evaluator: ProxyModelEvaluator,
        preprocessor: BaseEstimator,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> dict[SamplerKind, CvScores]:
        """Оцінює проксі-модель із кожним семплером у крос-валідації на train."""
        return {
            kind: evaluator.evaluate(
                preprocessor, features, target, sampler=self._sampler_factory.create(kind)
            )
            for kind in SamplerKind
        }

    @staticmethod
    def comparison_table(scores: Mapping[SamplerKind, CvScores]) -> pd.DataFrame:
        """Таблиця метрик проксі-моделі для кожного методу аугментації."""
        return pd.DataFrame(
            [
                {
                    "Метод": SAMPLER_LABELS[kind],
                    "PR-AUC": result.pr_auc,
                    "σ PR-AUC": result.pr_auc_std,
                    "ROC-AUC": result.roc_auc,
                    "F1": result.f1,
                    "Recall": result.recall,
                    "Precision": result.precision,
                }
                for kind, result in scores.items()
            ]
        )

    @staticmethod
    def to_metrics(
        illustration: ResamplingIllustration, scores: Mapping[SamplerKind, CvScores]
    ) -> dict[str, object]:
        """Словник для ``metrics.json``: по методах — розмір вибірки та метрики."""
        balance = illustration.class_balance.set_index("Метод")
        return {
            kind.value: {
                "train_rows": int(balance.loc[SAMPLER_LABELS[kind], "Рядків"]),
                "minority_share": balance.loc[SAMPLER_LABELS[kind], POSITIVE_SHARE_COLUMN] / 100,
                "pr_auc": result.pr_auc,
                "roc_auc": result.roc_auc,
                "f1": result.f1,
                "recall": result.recall,
                "precision": result.precision,
            }
            for kind, result in scores.items()
        }
