"""Тести генератора каркаса лабораторної."""

import pytest

from mllabs.core.paths import ProjectPaths
from mllabs.scaffold import LabScaffolder, ScaffoldError


def test_scaffold_creates_code_and_report_from_assignment(project: ProjectPaths):
    created = LabScaffolder(project).create(1)

    lab_module = project.labs_package_dir / "lab01" / "lab.py"
    assert lab_module in created
    assert "class Lab1(Lab):" in lab_module.read_text(encoding="utf-8")
    assert 'topic: "Тестова тема"' in project.lab(1).report_source.read_text(encoding="utf-8")
    compile(lab_module.read_text(encoding="utf-8"), str(lab_module), "exec")  # валідний Python


def test_scaffold_refuses_to_overwrite_existing_lab(project: ProjectPaths):
    scaffolder = LabScaffolder(project)
    scaffolder.create(1)

    with pytest.raises(ScaffoldError):
        scaffolder.create(1)


def test_scaffold_requires_assignment(project: ProjectPaths):
    with pytest.raises(ScaffoldError):
        LabScaffolder(project).create(2)
