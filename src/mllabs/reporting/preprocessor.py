"""Розширення Markdown для звітів: рисунки, таблиці, формули, метрики.

Стандартний Markdown не вміє нумерувати рисунки, вставляти таблиці з CSV чи
формули. Цей модуль додає такі можливості (повний опис синтаксису —
у ``.claude/skills/lab-report/SKILL.md``):

=================================================  ============================
Запис у ``report.md``                              Результат у PDF
=================================================  ============================
``![Назва](../artifacts/figures/x.png){#roc}``     рисунок + «Рис. N. Назва»
``![Назва](../artifacts/tables/x.csv){#cmp}``      «Таблиця N. Назва» + таблиця
``Таблиця: Назва {#lbl}`` перед Markdown-таблицею  «Таблиця N. Назва»
``(рис. {@roc})``, ``(табл. {@cmp})``              номер за міткою
``{{metrics.f1:.3f}}``                             значення з ``metrics.json``
``$E = mc^2$`` та ``$$ ... $$``                    формула (LaTeX)
=================================================  ============================

Принцип роботи: кожна знайдена конструкція замінюється на унікальний
токен-заглушку, а готовий HTML відкладається у «схованку» (stash). Після
перетворення Markdown -> HTML токени замінюються назад. Так бібліотека
Markdown не може випадково зіпсувати згенерований HTML.
"""

from __future__ import annotations

import csv
import html
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from latex2mathml.converter import convert as latex_to_mathml

from mllabs.reporting.errors import ReportError

# Фрагменти коду (```блок``` або `рядок`) — у них нічого не замінюємо.
_CODE = re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]+`)", re.DOTALL)
# ![підпис](шлях){#мітка width=70%} — окремим рядком.
_EMBED = re.compile(
    r"^!\[(?P<caption>[^\]]*)\]\((?P<src>[^)\s]+)\)(?:\{(?P<attrs>[^}]*)\})?[ \t]*$",
    re.MULTILINE,
)
# «Таблиця: Назва {#мітка}» — підпис для звичайної Markdown-таблиці.
_TABLE_CAPTION = re.compile(
    r"^Таблиця:[ \t]*(?P<caption>[^{\n]+?)[ \t]*(?:\{(?P<attrs>[^}]*)\})?[ \t]*$",
    re.MULTILINE,
)
_METRIC = re.compile(r"\{\{\s*metrics\.(?P<key>[\w.]+)(?::(?P<spec>[^}\s]+))?\s*\}\}")
_MATH_BLOCK = re.compile(r"\$\$(?P<latex>.+?)\$\$", re.DOTALL)
_MATH_INLINE = re.compile(r"(?<![\\$\w])\$(?!\s)(?P<latex>[^$\n]+?)(?<!\s)\$(?![\d$])")
_REFERENCE = re.compile(r"\{@(?P<label>[\w:-]+)\}")
_TOKEN = re.compile(r"(?:<p>)?(MLLABSSTASH\d{5}X)(?:</p>)?")


class ReportPreprocessor:
    """Перетворює розширений Markdown на звичайний і назад підставляє HTML.

    Використання — у два етапи::

        preprocessor = ReportPreprocessor(base_dir, metrics)
        plain_markdown = preprocessor.preprocess(text)   # до бібліотеки Markdown
        final_html = preprocessor.restore(html_text)     # після неї

    Args:
        base_dir: Каталог ``report.md``; від нього відраховуються шляхи.
        metrics: Вміст ``metrics.json`` для підстановки ``{{metrics...}}``.
    """

    def __init__(self, base_dir: Path, metrics: Mapping[str, Any]) -> None:
        self._base_dir = base_dir
        self._metrics = metrics
        self._stash: dict[str, str] = {}
        self._labels: dict[str, int] = {}
        self._figure_count = 0
        self._table_count = 0

    def preprocess(self, text: str) -> str:
        """Замінює розширені конструкції на токени-заглушки."""
        text = self._outside_code(text, self._replace_extensions)
        # Посилання розв'язуємо окремим проходом: мітка може стояти нижче
        # за текстом, ніж посилання на неї.
        return self._outside_code(text, lambda part: _REFERENCE.sub(self._reference, part))

    def restore(self, html_text: str) -> str:
        """Повертає збережений HTML на місце токенів-заглушок."""
        return _TOKEN.sub(lambda match: self._stash[match[1]], html_text)

    # ------------------------------------------------------------------
    # Обхід тексту
    # ------------------------------------------------------------------

    @staticmethod
    def _outside_code(text: str, transform: Callable[[str], str]) -> str:
        """Застосовує ``transform`` до всього тексту, крім фрагментів коду."""
        parts = _CODE.split(text)
        # re.split із групою повертає [текст, код, текст, код, ...]:
        # парні індекси — звичайний текст, непарні — код.
        return "".join(part if i % 2 else transform(part) for i, part in enumerate(parts))

    def _replace_extensions(self, part: str) -> str:
        part = _METRIC.sub(self._metric, part)
        part = self._replace_numbered(part)
        part = _MATH_BLOCK.sub(self._math_block, part)
        return _MATH_INLINE.sub(self._math_inline, part)

    def _replace_numbered(self, part: str) -> str:
        """Замінює вставки й підписи таблиць за один прохід у порядку появи в тексті.

        Два окремі проходи нумерували б спершу всі CSV-таблиці, а потім усі
        ручні — і номери таблиць ішли б не за порядком.
        """
        handlers = ((_EMBED, self._embed), (_TABLE_CAPTION, self._table_caption))
        matches = sorted(
            ((match, handler) for pattern, handler in handlers for match in pattern.finditer(part)),
            key=lambda pair: pair[0].start(),
        )
        chunks: list[str] = []
        position = 0
        for match, handler in matches:
            chunks.append(part[position : match.start()])
            chunks.append(handler(match))
            position = match.end()
        chunks.append(part[position:])
        return "".join(chunks)

    def _stashed(self, html_fragment: str, *, block: bool) -> str:
        """Ховає HTML і повертає токен; блочний токен обрамляється порожніми рядками."""
        token = f"MLLABSSTASH{len(self._stash):05d}X"
        self._stash[token] = html_fragment
        return f"\n\n{token}\n\n" if block else token

    # ------------------------------------------------------------------
    # Рисунки та таблиці
    # ------------------------------------------------------------------

    def _embed(self, match: re.Match[str]) -> str:
        """Вставляє рисунок або CSV-таблицю залежно від розширення файлу."""
        source = (self._base_dir / match["src"]).resolve()
        if not source.is_file():
            raise ReportError(f"Файл для вставки у звіт не знайдено: {match['src']} ({source})")
        attrs = _parse_attrs(match["attrs"])
        caption = html.escape(match["caption"].strip())

        if source.suffix.lower() == ".csv":
            number = self._next_table(attrs.get("label"))
            fragment = _table_title(number, caption) + _csv_to_html(source)
        else:
            number = self._next_figure(attrs.get("label"))
            width = html.escape(attrs.get("width", "100%"))
            fragment = (
                '<figure class="figure">'
                f'<img src="{source.as_uri()}" style="width: {width}" alt="">'
                f"<figcaption>Рис. {number}. {caption}</figcaption>"
                "</figure>"
            )
        return self._stashed(fragment, block=True)

    def _table_caption(self, match: re.Match[str]) -> str:
        """Нумерує підпис звичайної Markdown-таблиці."""
        attrs = _parse_attrs(match["attrs"])
        number = self._next_table(attrs.get("label"))
        return self._stashed(_table_title(number, html.escape(match["caption"])), block=True)

    def _next_figure(self, label: str | None) -> int:
        self._figure_count += 1
        self._register(label, self._figure_count)
        return self._figure_count

    def _next_table(self, label: str | None) -> int:
        self._table_count += 1
        self._register(label, self._table_count)
        return self._table_count

    def _register(self, label: str | None, number: int) -> None:
        if label is None:
            return
        if label in self._labels:
            raise ReportError(f"Мітка {{#{label}}} використана у звіті двічі")
        self._labels[label] = number

    def _reference(self, match: re.Match[str]) -> str:
        label = match["label"]
        if label not in self._labels:
            raise ReportError(f"Посилання {{@{label}}} вказує на неіснуючу мітку")
        return str(self._labels[label])

    # ------------------------------------------------------------------
    # Метрики та формули
    # ------------------------------------------------------------------

    def _metric(self, match: re.Match[str]) -> str:
        """Підставляє значення з ``metrics.json``, напр. ``{{metrics.rf.f1:.3f}}``."""
        value: Any = self._metrics
        for key in match["key"].split("."):
            if not isinstance(value, Mapping) or key not in value:
                raise ReportError(
                    f"У metrics.json немає ключа '{match['key']}' (використано у звіті)"
                )
            value = value[key]
        return format(value, match["spec"] or "")

    def _math_block(self, match: re.Match[str]) -> str:
        mathml = latex_to_mathml(match["latex"].strip(), display="block")
        return self._stashed(f'<div class="formula">{mathml}</div>', block=True)

    def _math_inline(self, match: re.Match[str]) -> str:
        return self._stashed(latex_to_mathml(match["latex"]), block=False)


def _parse_attrs(raw: str | None) -> dict[str, str]:
    """Розбирає ``#мітка width=70%`` у словник ``{"label": ..., "width": ...}``."""
    attrs: dict[str, str] = {}
    for item in (raw or "").split():
        if item.startswith("#"):
            attrs["label"] = item[1:]
        elif "=" in item:
            key, _, value = item.partition("=")
            attrs[key] = value
        else:
            raise ReportError(f"Незрозумілий атрибут '{item}' у фігурних дужках")
    return attrs


def _table_title(number: int, escaped_caption: str) -> str:
    return f'<p class="table-title">Таблиця {number}. {escaped_caption}</p>'


def _csv_to_html(csv_file: Path) -> str:
    """Перетворює CSV на HTML-таблицю (перший рядок — заголовки)."""
    with csv_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ReportError(f"CSV-файл порожній: {csv_file}")

    def render_row(cells: list[str], tag: str) -> str:
        return "<tr>" + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells) + "</tr>"

    header = render_row(rows[0], "th")
    body = "".join(render_row(row, "td") for row in rows[1:])
    return f'<table class="data-table"><thead>{header}</thead><tbody>{body}</tbody></table>'
