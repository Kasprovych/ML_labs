"""Єдиний стиль графіків проєкту: розміри під сторінку A4, палітра, шрифти.

Графіки всіх лабораторних розраховані на вставку у звіт і на чорно-білий друк:
ряди відрізняються не лише кольором, а й штрихуванням, типом лінії або формою
маркера. Функції побудови графіків позначаються декоратором :func:`styled`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

#: Ширина рисунка: 16 см — корисна ширина сторінки A4 (matplotlib рахує в дюймах).
#: Такі рисунки у звіті вставляються на ВСЮ ширину: тоді шрифт 8–9 pt лишається 8–9 pt.
FULL_WIDTH = 16 / 2.54
#: «Половинна» ширина 8,5 см — для рисунків, які у звіті стоять на ~55 % ширини
#: сторінки. Рисунок одразу малюється в цьому розмірі, тож підписи не зменшуються
#: під час вставлення (на 16-сантиметровому рисунку, стиснутому вдвічі, вони стали б ~5 pt).
HALF_WIDTH = 8.5 / 2.54

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
PURPLE = "#8a5cc2"
GRAY = "#9a9994"
LIGHT_GRAY = "#d5d4ce"
INK = "#1f1f1e"

#: Кольори рядів у фіксованому порядку та їхні «ч/б-двійники»: штрихування
#: (для стовпців), тип лінії і маркер (для ліній).
SERIES_COLORS = (BLUE, ORANGE, GREEN, PURPLE)
SERIES_HATCHES = ("", "///", "...", "xx")
SERIES_LINESTYLES = ("-", "--", ":", "-.")
SERIES_MARKERS = ("o", "s", "^", "D")

#: Розбіжна шкала: два кольори-полюси та нейтральна середина (нуль).
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_orange", [BLUE, "#f3f2ef", ORANGE])

STYLE = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e3de",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.edgecolor": "#6b6a66",
    "text.color": INK,
    "axes.labelcolor": INK,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
}


def styled[**P](plot_function: Callable[P, Figure]) -> Callable[P, Figure]:
    """Декоратор: виконує функцію побудови графіка з єдиним стилем оформлення.

    Декоратор — функція, що «обгортає» іншу функцію (схоже на анотацію з
    AOP-перехоплювачем у Spring): запис ``@styled`` над ``def plot_x`` означає
    ``plot_x = styled(plot_x)``. Тут обгортка тимчасово вмикає налаштування
    ``STYLE`` і повертає їх назад після побудови, не чіпаючи глобальний стан.
    ``[**P]`` — параметр типу: обгортка має ту саму сигнатуру, що й функція.
    """

    @functools.wraps(plot_function)  # зберігає ім'я та docstring вихідної функції
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Figure:
        # *args / **kwargs — «усі позиційні та іменовані аргументи як є» (varargs).
        with plt.rc_context(STYLE):
            return plot_function(*args, **kwargs)

    return wrapper
