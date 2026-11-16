"""Єдине місце, де описано структуру каталогів проєкту.

Жоден інший модуль не повинен «склеювати» шляхи вручну — усі шляхи беруться
звідси. Аналог у Java: клас-конфігурація з константами шляхів, який
інжектиться в сервіси.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Файл-маркер, за яким визначається корінь проєкту.
_ROOT_MARKER = "pyproject.toml"


def find_project_root(start: Path | None = None) -> Path:
    """Знаходить корінь проєкту, піднімаючись угору до файлу ``pyproject.toml``.

    Args:
        start: Каталог, з якого починати пошук. За замовчуванням — поточний
            робочий каталог.

    Returns:
        Абсолютний шлях до кореня проєкту.

    Raises:
        FileNotFoundError: Якщо ``pyproject.toml`` не знайдено в жодному з
            батьківських каталогів.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    raise FileNotFoundError(
        f"Не знайдено {_ROOT_MARKER} вище від {current}. Запускайте команди з каталогу проєкту."
    )


@dataclass(frozen=True)
class LabPaths:
    """Каталоги та файли однієї лабораторної роботи (``labs/labNN/``).

    Attributes:
        root: Каталог лабораторної, напр. ``labs/lab01``.
    """

    root: Path

    @property
    def assignment_file(self) -> Path:
        """Текст завдання з методички (переписаний у Markdown)."""
        return self.root / "ASSIGNMENT.md"

    @property
    def plan_file(self) -> Path:
        """План виконання з чеклістом вимог (створює агент lab-planner)."""
        return self.root / "PLAN.md"

    @property
    def guide_file(self) -> Path:
        """Навчальний конспект для автора (створює агент tutor)."""
        return self.root / "GUIDE.md"

    @property
    def artifacts_dir(self) -> Path:
        """Усе, що генерує код під час запуску: графіки, таблиці, метрики."""
        return self.root / "artifacts"

    @property
    def figures_dir(self) -> Path:
        """Графіки у форматі PNG."""
        return self.artifacts_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        """Таблиці у форматі CSV."""
        return self.artifacts_dir / "tables"

    @property
    def models_dir(self) -> Path:
        """Збережені навчені моделі."""
        return self.artifacts_dir / "models"

    @property
    def metrics_file(self) -> Path:
        """JSON з числовими результатами, які підставляються у звіт."""
        return self.artifacts_dir / "metrics.json"

    @property
    def report_dir(self) -> Path:
        """Каталог звіту: вихідний Markdown і готовий PDF."""
        return self.root / "report"

    @property
    def report_source(self) -> Path:
        """Вихідний текст звіту."""
        return self.report_dir / "report.md"


@dataclass(frozen=True)
class ProjectPaths:
    """Структура каталогів усього проєкту.

    Attributes:
        root: Корінь проєкту (каталог із ``pyproject.toml``).
    """

    root: Path

    @classmethod
    def discover(cls) -> ProjectPaths:
        """Створює об'єкт, автоматично знайшовши корінь проєкту."""
        return cls(root=find_project_root())

    @property
    def config_dir(self) -> Path:
        """Конфігураційні файли (дані для титульної сторінки тощо)."""
        return self.root / "config"

    @property
    def data_raw_dir(self) -> Path:
        """Сирі датасети — як завантажені, без змін."""
        return self.root / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        """Оброблені датасети, які передаються між лабораторними."""
        return self.root / "data" / "processed"

    @property
    def labs_dir(self) -> Path:
        """Каталог із матеріалами лабораторних (завдання, звіти, артефакти)."""
        return self.root / "labs"

    @property
    def labs_package_dir(self) -> Path:
        """Каталог із кодом лабораторних (пакет ``mllabs.labs``)."""
        return self.root / "src" / "mllabs" / "labs"

    def lab(self, number: int) -> LabPaths:
        """Повертає шляхи лабораторної з номером ``number``."""
        return LabPaths(root=self.labs_dir / lab_slug(number))


def lab_slug(number: int) -> str:
    """Формує уніфіковане ім'я лабораторної: ``1`` -> ``"lab01"``.

    Raises:
        ValueError: Якщо номер не додатний.
    """
    if number < 1:
        raise ValueError(f"Номер лабораторної має бути >= 1, отримано {number}")
    return f"lab{number:02d}"
