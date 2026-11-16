"""Командний рядок проєкту: ``uv run mllabs <команда>``.

Команди:

* ``list``      — показати лабораторні та їхній стан;
* ``new N``     — створити каркас лабораторної №N;
* ``run N``     — виконати код лабораторної №N;
* ``report N``  — зібрати PDF-звіт лабораторної №N;
* ``doctor``    — перевірити, що середовище готове до роботи.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from collections.abc import Callable, Sequence

from mllabs.core.context import LabContext
from mllabs.core.logging_setup import configure_logging
from mllabs.core.paths import ProjectPaths
from mllabs.core.registry import LabNotFoundError, available_lab_numbers, load_lab_class
from mllabs.core.reproducibility import DEFAULT_SEED
from mllabs.reporting import ReportBuilder, ReportError
from mllabs.reporting.pdf import ChromePdfRenderer
from mllabs.reporting.settings import TitlePageSettings
from mllabs.scaffold import LabScaffolder, ScaffoldError

_logger = logging.getLogger("mllabs.cli")


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входу. Повертає код завершення процесу (0 — успіх)."""
    arguments = _build_parser().parse_args(argv)
    configure_logging(verbose=arguments.verbose)
    try:
        project = ProjectPaths.discover()
        handler: Callable[[argparse.Namespace, ProjectPaths], int] = arguments.handler
        return handler(arguments, project)
    except (LabNotFoundError, ReportError, ScaffoldError, FileNotFoundError) as error:
        # Очікувані помилки користувача: показуємо зрозуміле повідомлення без трасування.
        _logger.error("%s", error)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mllabs", description="Лабораторні роботи з ML")
    parser.add_argument("-v", "--verbose", action="store_true", help="детальний журнал")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="показати лабораторні").set_defaults(handler=_list)
    commands.add_parser("doctor", help="перевірити середовище").set_defaults(handler=_doctor)

    new = commands.add_parser("new", help="створити каркас лабораторної")
    new.add_argument("number", type=int, help="номер лабораторної")
    new.set_defaults(handler=_new)

    run = commands.add_parser("run", help="виконати лабораторну")
    run.add_argument("number", type=int, help="номер лабораторної")
    run.add_argument("--seed", type=int, default=DEFAULT_SEED, help="зерно випадковості")
    run.set_defaults(handler=_run)

    report = commands.add_parser("report", help="зібрати PDF-звіт")
    report.add_argument("number", type=int, help="номер лабораторної")
    report.set_defaults(handler=_report)
    return parser


def _list(_: argparse.Namespace, project: ProjectPaths) -> int:
    """Друкує таблицю стану лабораторних: код / конспект / звіт."""
    with_code = set(available_lab_numbers(project))
    assigned = {
        int(directory.name.removeprefix("lab"))
        for directory in project.labs_dir.glob("lab[0-9][0-9]")
    }
    print(f"{'№':<3} {'код':<5} {'GUIDE':<6} {'PDF':<4} назва")
    for number in sorted(assigned | with_code):
        lab = project.lab(number)
        title = load_lab_class(number).TITLE if number in with_code else "(каркас не створено)"
        has_pdf = any(lab.report_dir.glob("*.pdf"))
        marks = (_mark(number in with_code), _mark(lab.guide_file.is_file()), _mark(has_pdf))
        print(f"{number:<3} {marks[0]:<5} {marks[1]:<6} {marks[2]:<4} {title}")
    return 0


def _new(arguments: argparse.Namespace, project: ProjectPaths) -> int:
    for created in LabScaffolder(project).create(arguments.number):
        print(f"створено: {created.relative_to(project.root)}")
    return 0


def _run(arguments: argparse.Namespace, project: ProjectPaths) -> int:
    lab_class = load_lab_class(arguments.number)
    context = LabContext(number=arguments.number, project=project, seed=arguments.seed)
    lab_class(context).run()
    return 0


def _report(arguments: argparse.Namespace, project: ProjectPaths) -> int:
    result = ReportBuilder(project).build(arguments.number)
    for warning in result.warnings:
        _logger.warning("%s", warning)
    print(result.pdf_file)
    return 0


def _doctor(_: argparse.Namespace, project: ProjectPaths) -> int:
    """Перевіряє ключові залежності; повертає 1, якщо чогось бракує."""
    problems = 0
    print(f"Python        {platform.python_version()} ({platform.machine()})")

    try:
        import tensorflow as tf

        gpus = [device.name for device in tf.config.list_physical_devices("GPU")]
        print(f"TensorFlow    {tf.__version__}; GPU: {gpus or 'немає (обчислення на CPU)'}")
    except ImportError as error:
        problems += 1
        print(f"TensorFlow    ПОМИЛКА: {error}")

    try:
        print(f"Chrome (PDF)  {ChromePdfRenderer.find_browser()}")
        unfilled = TitlePageSettings.load(project.config_dir / "report.toml").unfilled_fields()
        status = f"не заповнено: {', '.join(unfilled)}" if unfilled else "заповнено"
        print(f"Титульна      {status}")
        problems += bool(unfilled)
    except ReportError as error:
        problems += 1
        print(f"Звіти         ПОМИЛКА: {error}")

    print("Середовище готове." if problems == 0 else f"Знайдено проблем: {problems}")
    return int(problems > 0)


def _mark(done: bool) -> str:
    return "так" if done else "—"


if __name__ == "__main__":
    sys.exit(main())
