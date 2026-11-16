"""Контекст виконання лабораторної: усе, що потрібно коду під час запуску."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from mllabs.core.artifacts import ArtifactStore
from mllabs.core.paths import LabPaths, ProjectPaths
from mllabs.core.reproducibility import DEFAULT_SEED


@dataclass(frozen=True)
class LabContext:
    """Незмінний набір залежностей, що передається в кожну лабораторну.

    Аналог у Java: об'єкт, який Spring інжектить у конструктор. Завдяки
    цьому лабораторну легко тестувати — у тесті підставляється тимчасовий
    каталог замість справжнього.

    Attributes:
        number: Номер лабораторної.
        project: Шляхи всього проєкту (дані, конфігурація).
        seed: Зерно генераторів випадкових чисел.
        paths: Шляхи цієї лабораторної (обчислюється автоматично).
        artifacts: Сховище графіків, таблиць і метрик (обчислюється автоматично).
        logger: Логер з іменем лабораторної (обчислюється автоматично).
    """

    number: int
    project: ProjectPaths
    seed: int = DEFAULT_SEED
    paths: LabPaths = field(init=False)
    artifacts: ArtifactStore = field(init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        # frozen-датаклас забороняє звичайне присвоєння, тому похідні поля
        # заповнюються через object.__setattr__ (стандартний прийом).
        paths = self.project.lab(self.number)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "artifacts", ArtifactStore(paths))
        object.__setattr__(self, "logger", logging.getLogger(f"mllabs.lab{self.number:02d}"))
