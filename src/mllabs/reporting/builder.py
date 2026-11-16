"""Фасад підсистеми звітів: одна команда — готовий PDF."""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from mllabs.core.paths import ProjectPaths
from mllabs.reporting.html_renderer import HtmlReportRenderer
from mllabs.reporting.pdf import ChromePdfRenderer, PdfRenderer
from mllabs.reporting.settings import TitlePageSettings
from mllabs.reporting.source import ReportSource

_logger = logging.getLogger(__name__)

_SETTINGS_FILE_NAME = "report.toml"


@dataclass(frozen=True)
class ReportBuildResult:
    """Результат збирання звіту.

    Attributes:
        pdf_file: Шлях до готового PDF.
        warnings: Зауваження, які варто виправити перед здачею.
    """

    pdf_file: Path
    warnings: list[str] = field(default_factory=list)


class ReportBuilder:
    """Збирає PDF-звіт лабораторної з ``labs/labNN/report/report.md``.

    Args:
        project: Шляхи проєкту.
        pdf_renderer: Реалізація друку PDF; за замовчуванням — Chrome.
            Параметр потрібен, щоб у тестах підставити заглушку
            (dependency injection, як у Java).
    """

    def __init__(self, project: ProjectPaths, pdf_renderer: PdfRenderer | None = None) -> None:
        self._project = project
        self._pdf_renderer = pdf_renderer or ChromePdfRenderer()
        self._html_renderer = HtmlReportRenderer()

    def build(self, lab_number: int) -> ReportBuildResult:
        """Збирає звіт лабораторної ``lab_number``.

        Returns:
            Шлях до PDF і список попереджень.

        Raises:
            ReportError: Якщо звіт зібрати неможливо (немає файлу, рисунка,
                метрики тощо).
        """
        lab = self._project.lab(lab_number)
        settings = TitlePageSettings.load(self._project.config_dir / _SETTINGS_FILE_NAME)
        source = ReportSource.read(lab.report_source)
        metrics = (
            json.loads(lab.metrics_file.read_text(encoding="utf-8"))
            if lab.metrics_file.is_file()
            else {}
        )

        html_text = self._html_renderer.render(
            source,
            settings,
            lab_number=lab_number,
            year=datetime.datetime.now(tz=datetime.UTC).year,
            metrics=metrics,
        )

        build_dir = lab.report_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        html_file = build_dir / "report.html"
        html_file.write_text(html_text, encoding="utf-8")

        pdf_file = lab.report_dir / f"Звіт_ЛР{lab_number}.pdf"
        self._pdf_renderer.render(html_file, pdf_file)
        _logger.info("Звіт зібрано: %s", pdf_file)

        warnings = [
            f"У config/{_SETTINGS_FILE_NAME} не заповнено поле '{name}'"
            for name in settings.unfilled_fields()
        ]
        return ReportBuildResult(pdf_file=pdf_file, warnings=warnings)
