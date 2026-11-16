"""Розбір «шапки» (front matter) у Markdown-файлах.

Шапка — це блок YAML між двома рядками ``---`` на самому початку файлу::

    ---
    topic: Назва теми
    purpose: Мета роботи
    ---
    Далі звичайний текст Markdown...
"""

from __future__ import annotations

import re
from typing import Any

import yaml

_FRONT_MATTER = re.compile(r"\A---\s*\n(?P<meta>.*?)\n---\s*\n(?P<body>.*)\Z", re.DOTALL)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Відокремлює шапку від основного тексту.

    Args:
        text: Повний вміст Markdown-файлу.

    Returns:
        Пара ``(поля_шапки, основний_текст)``.

    Raises:
        ValueError: Якщо файл не починається з коректної шапки.
    """
    match = _FRONT_MATTER.match(text)
    if match is None:
        raise ValueError("файл має починатися з шапки між рядками '---'")
    meta = yaml.safe_load(match["meta"]) or {}
    if not isinstance(meta, dict):
        raise ValueError("шапка має бути набором пар 'ключ: значення'")
    return meta, match["body"]
