"""Генерація звіту: Markdown -> HTML -> PDF за шаблоном кафедри.

Зовнішньому коду потрібен лише фасад :class:`ReportBuilder`.
"""

from mllabs.reporting.builder import ReportBuilder, ReportBuildResult
from mllabs.reporting.errors import ReportError

__all__ = ["ReportBuildResult", "ReportBuilder", "ReportError"]
