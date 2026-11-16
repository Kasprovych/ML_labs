"""Виявлення викидів кількома методами та їх порівняння.

Усі детектори реалізують один інтерфейс :class:`OutlierDetector`, тому
порівняння (:class:`OutlierComparison`) не знає, які саме методи порівнює —
класичний патерн «Стратегія».
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

_OUTLIER_LABEL = -1  # так sklearn позначає викиди у fit_predict (1 — звичайна точка)


class OutlierDetector(Protocol):
    """Детектор викидів: позначає рядки-аномалії."""

    name: str  # назва для таблиць і графіків
    key: str  # ідентифікатор для ключів metrics.json

    def detect(self, features: pd.DataFrame) -> pd.Series:
        """Повертає булеву серію: ``True`` — рядок є викидом."""
        ...


class UnivariateOutlierDetector(OutlierDetector, Protocol):
    """Одновимірний детектор: перевіряє кожну ознаку окремо."""

    def detect_by_feature(self, features: pd.DataFrame) -> pd.DataFrame:
        """Повертає булеву таблицю тієї самої форми: ``True`` — значення є викидом."""
        ...


class IqrDetector:
    """Правило міжквартильного розмаху (IQR, «вуса» Тьюкі).

    Викид — значення далі ніж ``factor * IQR`` від першого (Q1) чи третього (Q3)
    квартиля, де ``IQR = Q3 − Q1``. Метод не припускає нормальності розподілу,
    але на скошених ознаках позначає весь довгий хвіст.

    Args:
        factor: Множник розмаху; 1.5 — класичне правило.
    """

    name = "IQR"
    key = "iqr"

    def __init__(self, factor: float) -> None:
        self._factor = factor

    def bounds(self, features: pd.DataFrame) -> pd.DataFrame:
        """Нижня і верхня межі «нормальних» значень кожної ознаки."""
        first_quartile = features.quantile(0.25)
        third_quartile = features.quantile(0.75)
        spread = third_quartile - first_quartile
        return pd.DataFrame(
            {
                "lower": first_quartile - self._factor * spread,
                "upper": third_quartile + self._factor * spread,
            }
        )

    def detect_by_feature(self, features: pd.DataFrame) -> pd.DataFrame:
        """Позначає значення поза межами; пропуск викидом не вважається."""
        bounds = self.bounds(features)
        # Порівняння таблиці з Series вирівнюється за назвами колонок (broadcasting):
        # кожна колонка порівнюється зі «своєю» межею. NaN у порівнянні дає False.
        return (features < bounds["lower"]) | (features > bounds["upper"])

    def detect(self, features: pd.DataFrame) -> pd.Series:
        """Рядок — викид, якщо викидом є хоча б одне його значення."""
        return self.detect_by_feature(features).any(axis=1)


class ZScoreDetector:
    """Правило Z-оцінки («три сигми»).

    ``z = (x − середнє) / σ`` — на скільки стандартних відхилень значення
    віддалене від середнього. Для нормального розподілу ``|z| > 3`` має лише
    0,27 % значень. Метод припускає нормальність: на скошених даних середнє і
    σ самі зміщені викидами, тому частина аномалій «маскується».

    Args:
        threshold: Поріг |z|.
    """

    name = "Z-score"
    key = "zscore"

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def detect_by_feature(self, features: pd.DataFrame) -> pd.DataFrame:
        """Позначає значення з ``|z|`` вище порога; пропуск викидом не вважається."""
        z_scores = (features - features.mean()) / features.std()
        return z_scores.abs() > self._threshold

    def detect(self, features: pd.DataFrame) -> pd.Series:
        """Рядок — викид, якщо викидом є хоча б одне його значення."""
        return self.detect_by_feature(features).any(axis=1)


class IsolationForestDetector:
    """Isolation Forest — ансамбль випадкових дерев, що «ізолюють» точки.

    Кожне дерево ділить дані випадковими порогами. Аномальну точку вдається
    відокремити від решти за кілька поділів (короткий шлях у дереві), звичайну —
    за багато. Метод багатовимірний: бачить незвичні КОМБІНАЦІЇ значень, навіть
    якщо кожне окремо в нормі.

    Args:
        contamination: Очікувана частка викидів — стільки рядків і буде позначено.
        n_estimators: Кількість дерев.
        random_state: Зерно генератора.
    """

    name = "Isolation Forest"
    key = "isolation_forest"

    def __init__(self, contamination: float, n_estimators: int, random_state: int) -> None:
        self._pipeline = _complete_and_scale(
            IsolationForest(
                contamination=contamination, n_estimators=n_estimators, random_state=random_state
            )
        )

    def detect(self, features: pd.DataFrame) -> pd.Series:
        """Позначає рядки, які ліс ізолює найшвидше."""
        labels = self._pipeline.fit_predict(features)
        return pd.Series(labels == _OUTLIER_LABEL, index=features.index)


class LofDetector:
    """Local Outlier Factor — порівняння локальної щільності точки із сусідами.

    Точка — викид, якщо навколо неї значно «рідше», ніж навколо її ``k``
    найближчих сусідів. Метод ловить локальні аномалії (точку на краю щільного
    скупчення), які глобальні методи пропускають.

    Args:
        contamination: Очікувана частка викидів.
        n_neighbors: Кількість сусідів ``k``.
    """

    name = "LOF"
    key = "lof"

    def __init__(self, contamination: float, n_neighbors: int) -> None:
        self._pipeline = _complete_and_scale(
            LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors)
        )

    def detect(self, features: pd.DataFrame) -> pd.Series:
        """Позначає рядки з найнижчою відносною локальною щільністю."""
        labels = self._pipeline.fit_predict(features)
        return pd.Series(labels == _OUTLIER_LABEL, index=features.index)


def _complete_and_scale(detector: object) -> Pipeline:
    """Додає перед багатовимірним детектором заповнення пропусків і стандартизацію.

    Моделі sklearn не приймають NaN, а LOF рахує відстані, тому ознаки мають
    бути в одному масштабі. Усе навчається на тих самих (навчальних) даних.
    """
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), detector)


@dataclass(frozen=True)
class OutlierReport:
    """Результат порівняння детекторів.

    Attributes:
        masks: Булева таблиця «рядок × метод»: ``True`` — метод вважає рядок викидом.
        by_method: Підсумок по методах: кількість і частка викидів, частка
            позитивного класу серед них.
        by_feature: Частка викидів у кожній ознаці для одновимірних методів, %.
        overlap: Матриця коефіцієнтів Жаккара між методами.
        all_methods_share: Частка рядків, які викидами вважають усі методи одразу.
        keys: Словник «назва методу → ідентифікатор для metrics.json».
    """

    masks: pd.DataFrame
    by_method: pd.DataFrame
    by_feature: pd.DataFrame
    overlap: pd.DataFrame
    all_methods_share: float
    keys: dict[str, str]

    def to_metrics(self) -> dict[str, object]:
        """Словник для ``metrics.json``: частки викидів і частка класу 1 серед них."""
        summary = self.by_method.set_index("Метод")
        metrics: dict[str, object] = {
            f"{self.keys[name]}_share": summary.loc[name, "Частка, %"] / 100 for name in self.keys
        }
        metrics["all_methods_share"] = self.all_methods_share
        metrics["positive_rate_in_outliers"] = {
            self.keys[name]: summary.loc[name, "Клас 1 серед викидів, %"] / 100
            for name in self.keys
        }
        return metrics


class OutlierComparison:
    """Запускає кілька детекторів на тих самих даних і порівнює результати.

    Args:
        univariate: Одновимірні детектори (IQR, Z-score).
        multivariate: Багатовимірні детектори (Isolation Forest, LOF).
    """

    def __init__(
        self,
        univariate: Sequence[UnivariateOutlierDetector],
        multivariate: Sequence[OutlierDetector],
    ) -> None:
        self._univariate = list(univariate)
        self._multivariate = list(multivariate)

    def run(self, numeric: pd.DataFrame, is_positive: pd.Series) -> OutlierReport:
        """Порівнює детектори.

        Args:
            numeric: Числові ознаки навчальної вибірки.
            is_positive: Булева серія: чи належить рядок до позитивного класу.
        """
        detectors = [*self._univariate, *self._multivariate]
        masks = pd.DataFrame({detector.name: detector.detect(numeric) for detector in detectors})
        by_method = pd.DataFrame(
            {
                "Метод": masks.columns,
                "Викидів": masks.sum().to_numpy(),
                "Частка, %": (masks.mean() * 100).to_numpy(),
                # is_positive[маска] — позитивність лише рядків-викидів; mean() — їх частка.
                "Клас 1 серед викидів, %": [
                    is_positive[masks[name]].mean() * 100 for name in masks.columns
                ],
            }
        )
        by_feature = pd.DataFrame({"Ознака": numeric.columns})
        for detector in self._univariate:
            by_feature[f"{detector.name}, %"] = (
                detector.detect_by_feature(numeric).mean() * 100
            ).to_numpy()
        return OutlierReport(
            masks=masks,
            by_method=by_method,
            by_feature=by_feature,
            overlap=_jaccard_matrix(masks),
            all_methods_share=float(masks.all(axis=1).mean()),
            keys={detector.name: detector.key for detector in detectors},
        )


def _jaccard_matrix(masks: pd.DataFrame) -> pd.DataFrame:
    """Коефіцієнт Жаккара для кожної пари методів.

    ``J(A, B) = |A ∩ B| / |A ∪ B|`` — частка спільних викидів серед усіх,
    позначених хоча б одним із двох методів: 1 — множини збігаються, 0 — не
    перетинаються.
    """
    names = list(masks.columns)
    matrix = pd.DataFrame(1.0, index=names, columns=names)
    for first in names:
        for second in names:
            union = (masks[first] | masks[second]).sum()
            intersection = (masks[first] & masks[second]).sum()
            matrix.loc[first, second] = intersection / union if union else 0.0
    return matrix


def project_to_plane(
    numeric: pd.DataFrame, sample_size: int, random_state: int
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Проєктує числові ознаки на площину двох головних компонент (PCA).

    PCA шукає напрямки найбільшого розкиду даних; перші два дають найкращу
    «плоску фотографію» багатовимірної хмари точок — зручно для графіка.

    Args:
        numeric: Числові ознаки (пропуски заповнюються медіаною).
        sample_size: Скільки випадкових рядків повернути (щоб графік читався).
        random_state: Зерно для підвибірки.

    Returns:
        Пара: таблиця з колонками ``pc1``, ``pc2`` (індекс — як у вибраних
        рядків ``numeric``) і частки дисперсії, поясненої кожною компонентою.
    """
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=2)
    )
    coordinates = pd.DataFrame(
        pipeline.fit_transform(numeric), index=numeric.index, columns=["pc1", "pc2"]
    )
    explained = pipeline[-1].explained_variance_ratio_  # [-1] — останній крок пайплайна
    sample = coordinates.sample(n=min(sample_size, len(coordinates)), random_state=random_state)
    return sample, (float(explained[0]), float(explained[1]))
