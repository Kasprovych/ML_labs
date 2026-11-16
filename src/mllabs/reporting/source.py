"""Читання вихідного файлу звіту ``report.md``.

Файл складається з двох частин:

1. «Шапка» (front matter) у форматі YAML між рядками ``---`` — тема й мета.
2. Основний текст у Markdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mllabs.core.frontmatter import split_front_matter
from mllabs.reporting.errors import ReportError


@dataclass(frozen=True)
class ReportSource:
    """Розібраний вихідний файл звіту.

    Attributes:
        topic: Тема роботи (з методички).
        purpose: Мета роботи.
        body: Основний текст у Markdown (без шапки).
        base_dir: Каталог файлу — від нього відраховуються шляхи до рисунків.
    """

    topic: str
    purpose: str
    body: str
    base_dir: Path

    @classmethod
    def read(cls, source_file: Path) -> ReportSource:
        """Читає та перевіряє ``report.md``.

        Raises:
            ReportError: Якщо файлу немає, немає шапки або в ній бракує полів.
        """
        if not source_file.is_file():
            raise ReportError(f"Не знайдено вихідний файл звіту: {source_file}")

        try:
            meta, body = split_front_matter(source_file.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ReportError(
                f"{source_file.name}: {error}. Приклад:\n---\ntopic: ...\npurpose: ...\n---"
            ) from error

        missing = [key for key in ("topic", "purpose") if not meta.get(key)]
        if missing:
            raise ReportError(f"У шапці {source_file.name} бракує полів: {', '.join(missing)}")

        return cls(
            topic=str(meta["topic"]).strip(),
            purpose=str(meta["purpose"]).strip(),
            body=body,
            base_dir=source_file.parent,
        )
