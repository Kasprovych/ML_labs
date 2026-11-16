"""Перетворення HTML на PDF.

Друкує PDF браузер Chrome у «безголовому» (headless) режимі — тим самим
рушієм, що й «Друк -> Зберегти як PDF». Це не потребує ні Microsoft Word,
ні LaTeX.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from mllabs.reporting.errors import ReportError

_logger = logging.getLogger(__name__)

#: Змінна середовища для явного шляху до браузера.
CHROME_ENV_VARIABLE = "MLLABS_CHROME"

_CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome",
    "chromium",
)
_TIMEOUT_SECONDS = 120


class PdfRenderer(Protocol):
    """Інтерфейс «друкаря» PDF.

    ``Protocol`` — це аналог Java-інтерфейсу, але без явного ``implements``:
    підходить будь-який клас, що має метод із такою сигнатурою.
    """

    def render(self, html_file: Path, pdf_file: Path) -> None:
        """Друкує ``html_file`` у ``pdf_file``."""
        ...


class ChromePdfRenderer:
    """Друкує PDF через Chrome / Chromium / Edge у headless-режимі."""

    def render(self, html_file: Path, pdf_file: Path) -> None:
        """Друкує ``html_file`` у ``pdf_file``.

        Raises:
            ReportError: Якщо браузер не знайдено або друк завершився помилкою.
        """
        pdf_file.unlink(missing_ok=True)
        # Окремий тимчасовий профіль: headless-Chrome не конфліктує з відкритим
        # браузером користувача (інакше після оновлення Chrome друк періодично
        # падає на завершенні процесу, хоча PDF уже записано).
        with tempfile.TemporaryDirectory(prefix="mllabs-chrome-") as profile_dir:
            command = [
                self.find_browser(),
                "--headless",
                "--disable-gpu",
                "--no-first-run",
                f"--user-data-dir={profile_dir}",
                "--no-pdf-header-footer",  # без службових колонтитулів браузера
                f"--print-to-pdf={pdf_file}",
                html_file.resolve().as_uri(),
            ]
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, check=False
                )
            except subprocess.TimeoutExpired as error:
                raise ReportError(
                    f"Chrome не встиг надрукувати PDF за {_TIMEOUT_SECONDS} с"
                ) from error

        if not _is_complete_pdf(pdf_file):
            raise ReportError(f"Chrome не зміг створити PDF:\n{completed.stderr.strip()}")
        if completed.returncode != 0:
            # PDF записано повністю, а збій стався вже під час закриття браузера.
            _logger.warning("Chrome завершився з кодом %d, але PDF створено", completed.returncode)

    @staticmethod
    def find_browser() -> str:
        """Шукає виконуваний файл браузера.

        Raises:
            ReportError: Якщо жодного придатного браузера не знайдено.
        """
        candidates = (os.environ.get(CHROME_ENV_VARIABLE), *_CHROME_CANDIDATES)
        for candidate in candidates:
            if not candidate:
                continue
            if Path(candidate).is_file():
                return candidate
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        raise ReportError(
            "Не знайдено Google Chrome / Chromium для друку PDF. Встановіть Chrome "
            f"або вкажіть шлях у змінній середовища {CHROME_ENV_VARIABLE}."
        )


def _is_complete_pdf(pdf_file: Path) -> bool:
    """Перевіряє, що файл існує й закінчується маркером кінця PDF (``%%EOF``)."""
    if not pdf_file.is_file() or pdf_file.stat().st_size == 0:
        return False
    with pdf_file.open("rb") as stream:
        stream.seek(max(0, pdf_file.stat().st_size - 1024))
        return b"%%EOF" in stream.read()
