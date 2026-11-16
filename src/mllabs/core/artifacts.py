"""Збереження результатів запуску: графіків, таблиць і метрик.

Ідея: код лабораторної НІКОЛИ не друкує цифри «в нікуди». Усе, що має
потрапити у звіт, зберігається через :class:`ArtifactStore`, а звіт
посилається на ці файли. Так цифри у звіті гарантовано збігаються з кодом.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mllabs.core.paths import LabPaths

if TYPE_CHECKING:  # імпорти лише для перевірки типів — не сповільнюють запуск
    import pandas as pd
    from matplotlib.figure import Figure

_logger = logging.getLogger(__name__)

#: Роздільна здатність графіків; 200 dpi достатньо для друку на A4.
_FIGURE_DPI = 200


class ArtifactStore:
    """Сховище артефактів однієї лабораторної роботи.

    Аналог у Java: репозиторій (Repository), що ховає деталі файлової системи.

    Args:
        paths: Шляхи лабораторної, у каталог ``artifacts/`` якої все пишеться.
    """

    def __init__(self, paths: LabPaths) -> None:
        self._paths = paths

    def save_figure(self, figure: Figure, name: str) -> Path:
        """Зберігає графік matplotlib у PNG і закриває його (звільняє пам'ять).

        Args:
            figure: Об'єкт ``matplotlib.figure.Figure``.
            name: Ім'я файлу без розширення, напр. ``"correlation_pearson"``.

        Returns:
            Шлях до збереженого файлу.
        """
        import matplotlib.pyplot as plt

        target = self._prepare(self._paths.figures_dir / f"{name}.png")
        figure.savefig(target, dpi=_FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)
        _logger.info("Графік збережено: %s", target.name)
        return target

    def save_table(
        self,
        table: pd.DataFrame,
        name: str,
        *,
        float_format: str = "%.4f",
        index: bool = False,
    ) -> Path:
        """Зберігає таблицю у CSV — у такому вигляді вона потрапить у звіт.

        Args:
            table: Таблиця ``pandas.DataFrame``.
            name: Ім'я файлу без розширення.
            float_format: Формат дробових чисел (стиль ``printf``).
            index: Чи записувати індекс рядків першою колонкою.

        Returns:
            Шлях до збереженого файлу.
        """
        target = self._prepare(self._paths.tables_dir / f"{name}.csv")
        table.to_csv(target, index=index, float_format=float_format)
        _logger.info("Таблицю збережено: %s", target.name)
        return target

    def save_metrics(self, metrics: dict[str, Any]) -> Path:
        """Додає числові результати у ``metrics.json`` (злиття з наявними).

        Значення з цього файлу підставляються у звіт через ``{{metrics.ключ}}``.

        Args:
            metrics: Словник результатів; допускаються вкладені словники.

        Returns:
            Шлях до ``metrics.json``.
        """
        target = self._prepare(self._paths.metrics_file)
        merged = self.load_metrics()
        merged.update(metrics)
        target.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2, default=_to_builtin),
            encoding="utf-8",
        )
        _logger.info("Метрики оновлено: %s", ", ".join(metrics))
        return target

    def load_metrics(self) -> dict[str, Any]:
        """Читає ``metrics.json``; повертає порожній словник, якщо файлу ще немає."""
        if not self._paths.metrics_file.is_file():
            return {}
        return json.loads(self._paths.metrics_file.read_text(encoding="utf-8"))

    @staticmethod
    def _prepare(target: Path) -> Path:
        """Створює батьківський каталог файлу, якщо його ще немає."""
        target.parent.mkdir(parents=True, exist_ok=True)
        return target


def _to_builtin(value: object) -> object:
    """Перетворює типи NumPy (``np.float64`` тощо) на звичайні типи Python.

    Стандартний ``json`` не вміє серіалізувати скаляри NumPy, а саме їх
    повертають sklearn і TensorFlow.
    """
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Неможливо серіалізувати у JSON: {type(value).__name__}")
