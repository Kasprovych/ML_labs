"""Пошук класів лабораторних за номером.

Угода (convention over configuration): код лабораторної №N лежить у модулі
``mllabs.labs.labNN.lab`` і містить рівно один клас-нащадок :class:`Lab`.
Реєструвати його будь-де додатково не потрібно.
"""

from __future__ import annotations

import importlib
import inspect

from mllabs.core.lab import Lab
from mllabs.core.paths import ProjectPaths, lab_slug


class LabNotFoundError(LookupError):
    """Код лабораторної з таким номером ще не створено."""


def load_lab_class(number: int) -> type[Lab]:
    """Імпортує модуль лабораторної та повертає її клас.

    Args:
        number: Номер лабораторної.

    Returns:
        Клас (не екземпляр!) лабораторної.

    Raises:
        LabNotFoundError: Якщо модуля немає або в ньому немає нащадка ``Lab``.
    """
    module_name = f"mllabs.labs.{lab_slug(number)}.lab"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name is None or not module_name.startswith(error.name):
            raise  # бракує сторонньої бібліотеки — це інша помилка, не ховаємо її
        raise LabNotFoundError(
            f"Лабораторну №{number} ще не створено. Виконайте: uv run mllabs new {number}"
        ) from error

    candidates = [
        member
        for _, member in inspect.getmembers(module, inspect.isclass)
        if issubclass(member, Lab) and member is not Lab and member.__module__ == module_name
    ]
    if len(candidates) != 1:
        raise LabNotFoundError(
            f"У модулі {module_name} має бути рівно один нащадок Lab, знайдено {len(candidates)}"
        )
    return candidates[0]


def available_lab_numbers(project: ProjectPaths) -> list[int]:
    """Повертає номери лабораторних, для яких уже існує код."""
    numbers = []
    for package_dir in sorted(project.labs_package_dir.glob("lab[0-9][0-9]")):
        if (package_dir / "lab.py").is_file():
            numbers.append(int(package_dir.name.removeprefix("lab")))
    return numbers
