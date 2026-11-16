"""Базовий клас лабораторної роботи (патерн «Шаблонний метод»)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, final

from mllabs.core.context import LabContext
from mllabs.core.reproducibility import set_global_seed


@dataclass(frozen=True)
class Step:
    """Один крок лабораторної — відповідає одному пункту завдання.

    Attributes:
        title: Назва кроку для журналу, напр. ``"2. Аналіз кореляцій"``.
        action: Функція без аргументів, що виконує крок.
    """

    title: str
    action: Callable[[], None]


class Lab(ABC):
    """Абстрактна лабораторна робота.

    Нащадок оголошує ``NUMBER``, ``TITLE`` і перелік кроків у :meth:`steps`.
    Порядок виконання, логування, вимірювання часу й фіксація seed
    реалізовані тут один раз — у методі :meth:`run`.

    Приклад::

        class Lab01(Lab):
            NUMBER = 1
            TITLE = "Дослідницький аналіз даних"

            def steps(self) -> Sequence[Step]:
                return [
                    Step("1. Завантаження даних", self._load_data),
                    Step("2. Аналіз кореляцій", self._analyze_correlations),
                ]

    Args:
        context: Контекст виконання (шляхи, сховище артефактів, логер).
    """

    #: Номер лабораторної; обов'язково перевизначається в нащадку.
    NUMBER: ClassVar[int]
    #: Коротка назва лабораторної для журналу та команди ``mllabs list``.
    TITLE: ClassVar[str]

    def __init__(self, context: LabContext) -> None:
        self._context = context
        self._log = context.logger

    @abstractmethod
    def steps(self) -> Sequence[Step]:
        """Повертає кроки лабораторної в порядку виконання."""

    @final
    def run(self) -> None:
        """Виконує всі кроки по черзі (нащадки цей метод не перевизначають)."""
        set_global_seed(self._context.seed)
        self._log.info("=== Лабораторна робота №%d: %s ===", self.NUMBER, self.TITLE)
        started = time.perf_counter()

        for step in self.steps():
            self._run_step(step)

        self._log.info("=== Готово за %.1f с ===", time.perf_counter() - started)

    def _run_step(self, step: Step) -> None:
        """Виконує один крок і записує в журнал його тривалість."""
        self._log.info("--- %s ---", step.title)
        started = time.perf_counter()
        step.action()
        self._log.info("    завершено за %.1f с", time.perf_counter() - started)
