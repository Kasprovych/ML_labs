"""Дані титульної сторінки, що читаються з ``config/report.toml``."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from mllabs.reporting.errors import ReportError

#: Маркер незаповненого значення в конфігурації.
PLACEHOLDER_MARK = "???"


@dataclass(frozen=True)
class TitlePageSettings:
    """Текстові поля титульної сторінки.

    Attributes:
        ministry: Назва міністерства.
        university: Назва університету.
        institute: Назва інституту.
        department: Назва кафедри.
        city: Місто (друкується внизу сторінки разом із роком).
        work_kind: Вид роботи в родовому відмінку («лабораторної роботи»).
        course: Назва курсу.
        student_group: Академічна група.
        student_name: Прізвище та ініціали студента.
        teacher_position: Посада й науковий ступінь викладача.
        teacher_name: Прізвище та ініціали викладача.
    """

    ministry: str
    university: str
    institute: str
    department: str
    city: str
    work_kind: str
    course: str
    student_group: str
    student_name: str
    teacher_position: str
    teacher_name: str

    @classmethod
    def load(cls, config_file: Path) -> TitlePageSettings:
        """Читає налаштування з TOML-файлу.

        Raises:
            ReportError: Якщо файлу немає або в ньому бракує обов'язкового поля.
        """
        if not config_file.is_file():
            raise ReportError(f"Не знайдено файл налаштувань звіту: {config_file}")
        with config_file.open("rb") as stream:
            raw = tomllib.load(stream)
        try:
            return cls(
                ministry=raw["university"]["ministry"],
                university=raw["university"]["name"],
                institute=raw["university"]["institute"],
                department=raw["university"]["department"],
                city=raw["university"]["city"],
                work_kind=raw["work"]["kind"],
                course=raw["work"]["course"],
                student_group=raw["student"]["group"],
                student_name=raw["student"]["full_name"],
                teacher_position=raw["teacher"]["position"],
                teacher_name=raw["teacher"]["full_name"],
            )
        except KeyError as error:
            raise ReportError(f"У {config_file.name} бракує поля {error}") from error

    def unfilled_fields(self) -> list[str]:
        """Повертає назви полів, у яких досі стоїть маркер ``???``."""
        return [
            field.name for field in fields(self) if PLACEHOLDER_MARK in getattr(self, field.name)
        ]
