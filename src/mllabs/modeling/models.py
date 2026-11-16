"""Моделі-класифікатори з фіксованими базовими гіперпараметрами.

Одна фабрика на ЛР2 і ЛР3: ЛР2 порівнює моделі з цими параметрами, ЛР3 бере
ту саму «точку відліку» і підбирає гіперпараметри. Фабрика повертає НЕФІТОВАНУ
модель — навчання виконує крос-валідація або код лабораторної.

Три моделі навмисно з різних сімейств:

* **логістична регресія** — лінійна модель: зважена сума ознак, пропущена через
  сигмоїду. Швидка й інтерпретовна, але бачить лише лінійні залежності й
  чутлива до масштабу ознак;
* **випадковий ліс** — бегінг: сотні глибоких дерев рішень, кожне навчене на
  бутстреп-вибірці з випадковою підмножиною ознак; прогноз — середнє дерев;
* **гістограмний градієнтний бустинг** — дерева будуються ПОСЛІДОВНО, кожне
  наступне виправляє помилки попередніх; числові ознаки попередньо
  розбиваються на ≤ 255 інтервалів (гістограм), що робить навчання швидким.
  Це той самий алгоритм, що в LightGBM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ModelKind(StrEnum):
    """Модель-класифікатор. Значення — короткі ключі для таблиць і ``metrics.json``."""

    LOGREG = "logreg"
    RANDOM_FOREST = "random_forest"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"


#: Підписи моделей для таблиць і графіків звіту.
_DISPLAY_NAMES = {
    ModelKind.LOGREG: "Логістична регресія",
    ModelKind.RANDOM_FOREST: "Випадковий ліс",
    ModelKind.HIST_GRADIENT_BOOSTING: "Градієнтний бустинг",
}

#: Короткі підписи — для легенд і вузьких колонок.
_SHORT_NAMES = {
    ModelKind.LOGREG: "LR",
    ModelKind.RANDOM_FOREST: "RF",
    ModelKind.HIST_GRADIENT_BOOSTING: "HGB",
}

_FAMILIES = {
    ModelKind.LOGREG: "лінійна",
    ModelKind.RANDOM_FOREST: "бегінг дерев",
    ModelKind.HIST_GRADIENT_BOOSTING: "бустинг дерев",
}


@dataclass(frozen=True)
class ModelSettings:
    """Базові гіперпараметри моделей (без підбору — підбір є темою ЛР3).

    Attributes:
        logreg_max_iter: Ліміт ітерацій оптимізатора ``lbfgs``. На
            немасштабованих ознаках (сценарій А) стандартних 100 не вистачає;
            2000 — із запасом. Якщо й цього мало, sklearn видає
            ``ConvergenceWarning`` — це факт про чутливість моделі до масштабу.
        forest_estimators: Кількість дерев лісу. Понад ~150 дерев якість лісу
            зазвичай уже не зростає, а час — зростає лінійно; 200 — компроміс.
        forest_n_jobs: Кількість потоків лісу (``-1`` — усі ядра): дерева
            незалежні й навчаються паралельно.
        boosting_max_iter: Кількість дерев (ітерацій) бустингу. Раннє зупинення
            вимкнено, тому кількість явна й однакова для всіх сценаріїв.
        boosting_learning_rate: Крок навчання бустингу — внесок кожного дерева;
            0.1 — стандарт sklearn.
    """

    logreg_max_iter: int = 2000
    forest_estimators: int = 200
    forest_n_jobs: int = -1
    boosting_max_iter: int = 200
    boosting_learning_rate: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        """Словник зі звичайних типів (для ``metrics.json``)."""
        return asdict(self)


class ModelFactory:
    """Створює нефітовані моделі з однаковими базовими параметрами.

    ``class_weight`` не задається жодній моделі: інакше ефект аугментації
    (SMOTE/ADASYN) змішався б з переважуванням класів.

    Args:
        settings: Базові гіперпараметри; за замовчуванням — стандартні.
        random_state: Зерно генераторів моделей.
    """

    def __init__(self, settings: ModelSettings | None = None, random_state: int = 0) -> None:
        self._settings = settings or ModelSettings()
        self._random_state = random_state

    @property
    def settings(self) -> ModelSettings:
        """Базові гіперпараметри, з якими фабрика створює моделі."""
        return self._settings

    def create(self, kind: ModelKind) -> ClassifierMixin:
        """Повертає нову нефітовану модель."""
        settings = self._settings
        match kind:
            case ModelKind.LOGREG:
                # L2-регуляризація і оптимізатор lbfgs — стандартні значення sklearn.
                return LogisticRegression(
                    max_iter=settings.logreg_max_iter, random_state=self._random_state
                )
            case ModelKind.RANDOM_FOREST:
                # Решта параметрів стандартні (max_depth=None, min_samples_leaf=1):
                # навмисно «сирий» ліс із глибокими деревами — ЛР3 його регуляризує.
                return RandomForestClassifier(
                    n_estimators=settings.forest_estimators,
                    n_jobs=settings.forest_n_jobs,
                    random_state=self._random_state,
                )
            case ModelKind.HIST_GRADIENT_BOOSTING:
                # early_stopping=False: з "auto" бустинг сам відрізав би випадкові 10 %
                # навчальних рядків для зупинки — а після SMOTE серед них були б
                # синтетичні, тобто зупинка валідувалася б на вигаданих точках.
                return HistGradientBoostingClassifier(
                    max_iter=settings.boosting_max_iter,
                    learning_rate=settings.boosting_learning_rate,
                    early_stopping=False,
                    random_state=self._random_state,
                )

    def describe(self, kind: ModelKind) -> dict[str, Any]:
        """Ключові гіперпараметри моделі (для таблиці моделей у звіті)."""
        model = self.create(kind)
        keys = {
            # l1_ratio=0 — чиста L2-регуляризація: нові версії sklearn задають
            # тип регуляризації саме ним (параметр penalty застарів).
            ModelKind.LOGREG: ("C", "l1_ratio", "solver", "max_iter"),
            ModelKind.RANDOM_FOREST: (
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "max_features",
            ),
            ModelKind.HIST_GRADIENT_BOOSTING: (
                "max_iter",
                "learning_rate",
                "max_leaf_nodes",
                "min_samples_leaf",
                "l2_regularization",
                "early_stopping",
            ),
        }[kind]
        params = model.get_params()
        return {key: params[key] for key in keys}

    @staticmethod
    def display_name(kind: ModelKind) -> str:
        """Повна українська назва моделі."""
        return _DISPLAY_NAMES[kind]

    @staticmethod
    def short_name(kind: ModelKind) -> str:
        """Коротка назва моделі (LR, RF, HGB)."""
        return _SHORT_NAMES[kind]

    @staticmethod
    def family(kind: ModelKind) -> str:
        """Сімейство алгоритмів."""
        return _FAMILIES[kind]
