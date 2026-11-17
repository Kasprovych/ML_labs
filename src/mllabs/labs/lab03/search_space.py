"""Простір гіперпараметрів — єдине джерело меж для всіх методів пошуку.

Три методи оптимізації по-різному «дивляться» на простір:

* **Grid Search** перебирає скінченну сітку (декартів добуток значень);
* **Random Search** вибирає точки з неперервних розподілів у тих самих межах;
* **Bayesian (Optuna TPE)** пропонує точки через ``trial.suggest_*`` у тих самих межах.

Щоб порівняння методів було чесним, межі мають бути однаковими. Тому сітка,
розподіли й виклики Optuna генеруються з одного об'єкта :class:`SearchSpace`,
а не задаються тричі вручну — розбіжність меж стає неможливою.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import optuna
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.utils import check_random_state


class Distribution(Protocol):
    """Розподіл, з якого ``RandomizedSearchCV`` бере значення: потрібен лише ``rvs``."""

    def rvs(self, size: int | None = None, random_state: object = None) -> object:
        """Вибирає значення (або масив значень)."""
        ...


class InvalidSearchSpaceError(ValueError):
    """Діапазон параметра задано суперечливо (сітка виходить за межі тощо)."""


@dataclass(frozen=True)
class ParameterRange:
    """Один гіперпараметр: межі, точки сітки і шкала.

    Attributes:
        name: Назва параметра моделі (без префікса кроку пайплайна).
        description: Що робить параметр — для таблиці у звіті.
        low: Нижня межа (включно).
        high: Верхня межа (включно).
        grid: Точки сітки для Grid Search; мають містити обидві межі.
        log: Чи вибирати значення в логарифмічній шкалі (однаково часто
            0,01–0,03 і 0,03–0,1) — природно для «множникових» параметрів.
        integer: Чи параметр цілий.
    """

    name: str
    description: str
    low: float
    high: float
    grid: tuple[float, ...]
    log: bool
    integer: bool

    def __post_init__(self) -> None:
        """Перевіряє узгодженість діапазону (аналог перевірки в конструкторі Java)."""
        if not self.low < self.high:
            raise InvalidSearchSpaceError(f"{self.name}: нижня межа має бути меншою за верхню")
        if self.log and self.low <= 0:
            raise InvalidSearchSpaceError(f"{self.name}: лог-шкала вимагає додатних меж")
        if any(not self.low <= value <= self.high for value in self.grid):
            raise InvalidSearchSpaceError(f"{self.name}: точки сітки виходять за межі")
        if self.low not in self.grid or self.high not in self.grid:
            raise InvalidSearchSpaceError(f"{self.name}: сітка має містити обидві межі")

    def cast(self, value: float) -> int | float:
        """Зводить значення (зокрема скаляр NumPy) до ``int`` або ``float`` Python."""
        return round(float(value)) if self.integer else float(value)

    def grid_values(self) -> list[int | float]:
        """Точки сітки у потрібному типі."""
        return [self.cast(value) for value in self.grid]

    def distribution(self) -> Distribution | None:
        """Розподіл для ``RandomizedSearchCV`` (об'єкт із методом ``rvs``)."""
        if self.integer:
            return IntLogUniform(int(self.low), int(self.high)) if self.log else None
        if self.log:
            return loguniform(self.low, self.high)
        # uniform(loc, scale) — рівномірний на [loc, loc + scale].
        return uniform(loc=self.low, scale=self.high - self.low)

    def suggest(self, trial: optuna.Trial) -> int | float:
        """Значення, яке пропонує Optuna в тих самих межах і шкалі."""
        if self.integer:
            return trial.suggest_int(self.name, int(self.low), int(self.high), log=self.log)
        return trial.suggest_float(self.name, self.low, self.high, log=self.log)


class IntLogUniform:
    """Цілий лог-рівномірний розподіл на ``[low, high]`` для ``RandomizedSearchCV``.

    У SciPy немає готового цілого лог-рівномірного розподілу, а sklearn вимагає
    від розподілу лише методу ``rvs(random_state=...)`` («качина типізація»:
    підходить будь-який об'єкт із таким методом, наслідувати нічого не треба).
    Значення береться з неперервного лог-рівномірного розподілу на
    ``[low, high + 1)`` і округлюється вниз — кожне ціле отримує частку,
    пропорційну довжині свого відрізка в лог-шкалі.

    Args:
        low: Найменше ціле значення.
        high: Найбільше ціле значення.
    """

    def __init__(self, low: int, high: int) -> None:
        if not 0 < low < high:
            raise InvalidSearchSpaceError("IntLogUniform: потрібно 0 < low < high")
        self._low = low
        self._high = high

    def rvs(
        self, size: int | None = None, random_state: int | np.random.RandomState | None = None
    ) -> int | np.ndarray:
        """Вибирає одне ціле (``size=None``) або масив цілих."""
        generator = check_random_state(random_state)
        continuous = loguniform(self._low, self._high + 1).rvs(size=size, random_state=generator)
        values = np.minimum(np.floor(continuous).astype(int), self._high)
        return int(values) if size is None else values

    def __repr__(self) -> str:
        """Подання для журналу та ``cv_results_``."""
        return f"IntLogUniform({self._low}, {self._high})"


@dataclass(frozen=True)
class SearchSpace:
    """Набір діапазонів гіперпараметрів.

    Attributes:
        ranges: Діапазони у фіксованому порядку (він же — порядок колонок таблиць).
    """

    ranges: tuple[ParameterRange, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Назви параметрів (без префікса)."""
        return tuple(item.name for item in self.ranges)

    @property
    def grid_size(self) -> int:
        """Кількість точок повної сітки (добуток розмірів по осях)."""
        return math.prod(len(item.grid) for item in self.ranges)

    def param_grid(self, prefix: str) -> dict[str, list[int | float]]:
        """Сітка для ``GridSearchCV``: ``{"model__learning_rate": [...], ...}``."""
        return {f"{prefix}__{item.name}": item.grid_values() for item in self.ranges}

    def distributions(self, prefix: str) -> dict[str, Distribution]:
        """Розподіли для ``RandomizedSearchCV`` у тих самих межах.

        Raises:
            InvalidSearchSpaceError: Якщо цілий параметр задано в лінійній шкалі
                (для нього немає розподілу; у цій лабораторній такого немає).
        """
        result = {}
        for item in self.ranges:
            distribution = item.distribution()
            if distribution is None:
                raise InvalidSearchSpaceError(f"{item.name}: цілий лінійний розподіл не підтримано")
            result[f"{prefix}__{item.name}"] = distribution
        return result

    def suggest(self, trial: optuna.Trial) -> dict[str, int | float]:
        """Параметри, запропоновані Optuna для однієї спроби (без префікса)."""
        return {item.name: item.suggest(trial) for item in self.ranges}

    def strip(self, params: Mapping[str, Any], prefix: str) -> dict[str, int | float]:
        """``{"model__x": v}`` → ``{"x": v}`` із приведенням типів."""
        by_name = {item.name: item for item in self.ranges}
        return {name: by_name[name].cast(params[f"{prefix}__{name}"]) for name in self.names}

    @staticmethod
    def prefixed(params: Mapping[str, Any], prefix: str) -> dict[str, Any]:
        """``{"x": v}`` → ``{"model__x": v}`` — для ``pipeline.set_params(**...)``."""
        return {f"{prefix}__{name}": value for name, value in params.items()}

    def describe(self) -> pd.DataFrame:
        """Таблиця простору для звіту."""
        rows = [
            {
                "Параметр": item.name,
                "Що регулює": item.description,
                "Сітка (Grid)": "; ".join(f"{value:g}" for value in item.grid),
                "Межі (Random, Optuna)": f"[{item.low:g}; {item.high:g}]",
                "Шкала": "лог" if item.log else "лінійна",
                "Тип": "ціле" if item.integer else "дробове",
            }
            for item in self.ranges
        ]
        return pd.DataFrame(rows)

    def to_metrics(self) -> dict[str, Any]:
        """Опис простору для ``metrics.json``."""
        return {
            **{
                item.name: {
                    "grid": item.grid_values(),
                    "low": item.low,
                    "high": item.high,
                    "log": item.log,
                    "integer": item.integer,
                }
                for item in self.ranges
            },
            "grid_size": self.grid_size,
        }


def hgb_search_space() -> SearchSpace:
    """Простір D3 для ``HistGradientBoostingClassifier``: 4 параметри, сітка 3⁴ = 81.

    Межі ``learning_rate`` звужено пробою плану на train-підвибірці (не на test):
    0,3 стабільно гірше, оптимум лежить нижче 0,1.
    """
    return SearchSpace(
        ranges=(
            ParameterRange(
                "learning_rate",
                "внесок кожного дерева (швидкість навчання)",
                0.01,
                0.1,
                (0.01, 0.03, 0.1),
                log=True,
                integer=False,
            ),
            ParameterRange(
                "max_leaf_nodes",
                "складність одного дерева (кількість листків)",
                15,
                63,
                (15, 31, 63),
                log=True,
                integer=True,
            ),
            ParameterRange(
                "min_samples_leaf",
                "мінімум рядків у листку (регуляризація)",
                20,
                500,
                (20, 100, 500),
                log=True,
                integer=True,
            ),
            ParameterRange(
                "max_features",
                "частка ознак, доступних кожному розбиттю",
                0.3,
                1.0,
                (0.3, 0.6, 1.0),
                log=False,
                integer=False,
            ),
        )
    )
