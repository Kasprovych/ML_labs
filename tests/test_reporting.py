"""Тести підсистеми звітів (без запуску браузера — він замінений заглушкою)."""

import json
from pathlib import Path

import pytest

from mllabs.core.paths import ProjectPaths
from mllabs.reporting import ReportBuilder, ReportError
from mllabs.reporting.preprocessor import ReportPreprocessor


class FakePdfRenderer:
    """Заглушка замість Chrome: просто створює порожній файл."""

    def render(self, html_file: Path, pdf_file: Path) -> None:
        pdf_file.write_bytes(b"%PDF-fake")


def _process(text: str, base_dir: Path, metrics: dict | None = None) -> str:
    preprocessor = ReportPreprocessor(base_dir, metrics or {})
    return preprocessor.restore(preprocessor.preprocess(text))


def test_figures_and_tables_are_numbered_and_referenced(tmp_path: Path):
    (tmp_path / "a.png").write_bytes(b"png")
    (tmp_path / "b.png").write_bytes(b"png")
    (tmp_path / "t.csv").write_text("model,f1\nRF,0.9\n", encoding="utf-8")
    text = (
        "Див. рис. {@second} і табл. {@cmp}.\n\n"
        "![Перший](a.png)\n\n![Другий](b.png){#second width=60%}\n\n![Порівняння](t.csv){#cmp}\n"
    )

    result = _process(text, tmp_path)

    assert "Див. рис. 2 і табл. 1." in result
    assert "Рис. 1. Перший" in result
    assert "Рис. 2. Другий" in result
    assert "width: 60%" in result
    assert "Таблиця 1. Порівняння" in result
    assert "<th>model</th>" in result
    assert "<td>0.9</td>" in result


def test_manual_and_csv_tables_are_numbered_in_document_order(tmp_path: Path):
    (tmp_path / "t.csv").write_text("model,f1\nRF,0.9\n", encoding="utf-8")
    text = "Таблиця: Ручна {#manual}\n\n| a |\n|---|\n| 1 |\n\n![З файлу](t.csv){#csv}\n"

    result = _process(text, tmp_path)

    assert "Таблиця 1. Ручна" in result
    assert "Таблиця 2. З файлу" in result


def test_metrics_are_substituted_with_format(tmp_path: Path):
    result = _process("F1 = {{metrics.rf.f1:.2f}}", tmp_path, {"rf": {"f1": 0.87654}})

    assert result == "F1 = 0.88"


def test_code_is_left_untouched(tmp_path: Path):
    text = "`{{metrics.x}}` і\n```\n$a$ {@nope}\n```\n"

    assert _process(text, tmp_path) == text


def test_inline_math_becomes_mathml(tmp_path: Path):
    assert "<math" in _process("Формула $x^2$ у тексті.", tmp_path)


@pytest.mark.parametrize(
    "text",
    ["![Немає](missing.png)", "{{metrics.absent}}", "рис. {@unknown}"],
)
def test_broken_report_fails_with_clear_error(tmp_path: Path, text: str):
    with pytest.raises(ReportError):
        _process(text, tmp_path)


def test_builder_produces_pdf_and_warns_about_unfilled_settings(project: ProjectPaths):
    lab = project.lab(1)
    lab.report_dir.mkdir()
    lab.report_source.write_text(
        '---\ntopic: "Тема"\npurpose: "Мета"\n---\n## 1. Розділ\n\nF1 = {{metrics.f1}}\n',
        encoding="utf-8",
    )
    lab.artifacts_dir.mkdir()
    lab.metrics_file.write_text(json.dumps({"f1": 0.75}), encoding="utf-8")

    result = ReportBuilder(project, pdf_renderer=FakePdfRenderer()).build(1)

    html_text = (lab.report_dir / "build" / "report.html").read_text(encoding="utf-8")
    assert result.pdf_file.name == "Звіт_ЛР1.pdf"
    assert result.pdf_file.is_file()
    assert "F1 = 0.75" in html_text
    assert "лабораторної роботи № 1" in html_text
    assert any("student_name" in warning for warning in result.warnings)
