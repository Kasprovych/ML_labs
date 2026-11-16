"""Генератор каркаса лабораторної: пакет із кодом і заготовка звіту."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from string import Template

from mllabs.core.frontmatter import split_front_matter
from mllabs.core.paths import ProjectPaths, lab_slug

_logger = logging.getLogger(__name__)


class ScaffoldError(Exception):
    """Каркас створити неможливо (немає завдання або код уже існує)."""


class LabScaffolder:
    """Створює файли нової лабораторної з шаблонів ``scaffold/templates``.

    Тему, мету й коротку назву бере з шапки ``labs/labNN/ASSIGNMENT.md``,
    тож їх не треба вводити вдруге.

    Args:
        project: Шляхи проєкту.
    """

    def __init__(self, project: ProjectPaths) -> None:
        self._project = project

    def create(self, number: int) -> list[Path]:
        """Створює каркас лабораторної ``number``.

        Наявні файли не перезаписуються — щоб випадково не знищити роботу.

        Returns:
            Список створених файлів.

        Raises:
            ScaffoldError: Якщо немає ``ASSIGNMENT.md`` або код уже існує.
        """
        slug = lab_slug(number)
        lab = self._project.lab(number)
        package_dir = self._project.labs_package_dir / slug

        if (package_dir / "lab.py").exists():
            raise ScaffoldError(f"Код лабораторної №{number} уже існує: {package_dir}")
        values = {"number": str(number), "slug": slug, **self._read_assignment(lab.assignment_file)}

        targets = {
            "package_init.py.tmpl": package_dir / "__init__.py",
            "lab.py.tmpl": package_dir / "lab.py",
            "config.py.tmpl": package_dir / "config.py",
            "report.md.tmpl": lab.report_source,
        }
        created = []
        for template_name, target in targets.items():
            if target.exists():
                _logger.warning("Пропущено (вже існує): %s", target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self._render(template_name, values), encoding="utf-8")
            created.append(target)
        return created

    @staticmethod
    def _read_assignment(assignment_file: Path) -> dict[str, str]:
        """Читає тему, мету й коротку назву з шапки файлу завдання."""
        if not assignment_file.is_file():
            raise ScaffoldError(f"Не знайдено файл завдання: {assignment_file}")
        try:
            meta, _ = split_front_matter(assignment_file.read_text(encoding="utf-8"))
            topic = str(meta["topic"])
            return {
                "topic": topic,
                "purpose": str(meta["purpose"]),
                # Коротка назва йде в код (TITLE, docstring); повна тема — у звіт.
                "short_title": str(meta.get("short_title", topic)),
            }
        except (ValueError, KeyError) as error:
            raise ScaffoldError(
                f"{assignment_file.name} має містити шапку з полями topic і purpose: {error}"
            ) from error

    @staticmethod
    def _render(template_name: str, values: dict[str, str]) -> str:
        """Підставляє значення в шаблон (синтаксис ``${name}``)."""
        template_file = resources.files("mllabs.scaffold") / "templates" / template_name
        return Template(template_file.read_text(encoding="utf-8")).substitute(values)
