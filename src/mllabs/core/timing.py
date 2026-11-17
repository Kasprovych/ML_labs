"""Вимірювання тривалості кроків лабораторної для ``metrics.timing``."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any


class StepTimer:
    """Запам'ятовує, скільки секунд тривав кожен крок.

    ``Lab.run`` лише пише тривалість кроків у журнал; тут вона ще й
    зберігається, щоб потрапити в ``metrics.json`` (бюджет часу лабораторної).
    """

    def __init__(self) -> None:
        self._seconds: dict[str, float] = {}
        self._started = time.perf_counter()

    def timed(self, key: str, action: Callable[[], None]) -> Callable[[], None]:
        """Повертає дію, що виконує ``action`` і записує її тривалість під ключем ``key``."""

        # Вкладена функція «замикає» key і action (closure) — як лямбда в Java.
        def wrapper() -> None:
            started = time.perf_counter()
            action()
            self._seconds[key] = time.perf_counter() - started

        return wrapper

    def to_metrics(self) -> dict[str, Any]:
        """Секунди кожного кроку, загальний час від створення таймера і кількість ядер."""
        return {
            "steps_seconds": dict(self._seconds),
            "total_seconds": time.perf_counter() - self._started,
            "n_cpu": os.cpu_count(),
        }
