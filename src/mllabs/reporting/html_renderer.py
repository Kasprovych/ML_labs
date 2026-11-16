"""Складання повного HTML-документа звіту (титульна сторінка + текст)."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any

import markdown
from jinja2 import Environment, PackageLoader, StrictUndefined

from mllabs.reporting.preprocessor import ReportPreprocessor
from mllabs.reporting.settings import TitlePageSettings
from mllabs.reporting.source import ReportSource

_TEMPLATE_NAME = "report.html.j2"
_MARKDOWN_EXTENSIONS = ("tables", "fenced_code", "sane_lists")


class HtmlReportRenderer:
    """Перетворює вихідний звіт на самодостатній HTML-файл.

    Оформлення (поля, шрифт, інтервали) задане у ``templates/report.css`` і
    відтворює стилі шаблону кафедри; структура сторінок — у
    ``templates/report.html.j2``.
    """

    def __init__(self) -> None:
        self._environment = Environment(
            loader=PackageLoader("mllabs.reporting", "templates"),
            autoescape=True,
            undefined=StrictUndefined,  # помилка замість мовчазного порожнього рядка
        )

    def render(
        self,
        source: ReportSource,
        settings: TitlePageSettings,
        *,
        lab_number: int,
        year: int,
        metrics: Mapping[str, Any],
    ) -> str:
        """Повертає готовий HTML звіту.

        Args:
            source: Розібраний ``report.md``.
            settings: Дані титульної сторінки.
            lab_number: Номер лабораторної (друкується на титульній сторінці).
            year: Рік (друкується внизу титульної сторінки).
            metrics: Вміст ``metrics.json`` для підстановок у тексті.
        """
        preprocessor = ReportPreprocessor(source.base_dir, metrics)
        plain_markdown = preprocessor.preprocess(source.body)
        body_html = markdown.markdown(plain_markdown, extensions=list(_MARKDOWN_EXTENSIONS))
        body_html = preprocessor.restore(body_html)

        templates = resources.files("mllabs.reporting") / "templates"
        return self._environment.get_template(_TEMPLATE_NAME).render(
            title=settings,
            lab_number=lab_number,
            year=year,
            topic=source.topic,
            purpose=source.purpose,
            body_html=body_html,
            stylesheet=(templates / "report.css").read_text(encoding="utf-8"),
            logo_uri=_as_uri(templates / "title_logo.png"),
        )


def _as_uri(resource: Traversable) -> str:
    """Повертає ``file://``-адресу ресурсу пакета (для тега ``<img>``)."""
    with resources.as_file(resource) as path:
        return path.resolve().as_uri()
