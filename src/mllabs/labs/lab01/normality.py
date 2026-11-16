"""Перевірка числових ознак на нормальність розподілу."""

from __future__ import annotations

import pandas as pd
from scipy import stats

_P_VALUE_FLOOR = 1e-4


class NormalityTester:
    """Перевіряє нормальність двома статистичними тестами.

    Нульова гіпотеза обох тестів: «дані взято з нормального розподілу».
    Якщо p-значення менше за ``alpha`` — гіпотезу відхиляємо.

    * **Шапіро-Уїлка** — найпотужніший тест для малих вибірок; у scipy його
      p-значення ненадійне для N > 5000, тому береться випадкова підвибірка.
    * **Д'Агостіно-Пірсона** (``normaltest``) — об'єднує перевірку асиметрії
      та ексцесу; працює на вибірках будь-якого розміру.

    На десятках тисяч рядків тести відхиляють нормальність за найменшого
    відхилення, тому поруч наводяться асиметрія та ексцес — вони показують,
    НАСКІЛЬКИ розподіл відрізняється від нормального.

    Args:
        alpha: Рівень значущості.
        shapiro_sample_size: Розмір підвибірки для тесту Шапіро-Уїлка.
        random_state: Зерно для підвибірки.
    """

    def __init__(self, alpha: float, shapiro_sample_size: int, random_state: int) -> None:
        self._alpha = alpha
        self._shapiro_sample_size = shapiro_sample_size
        self._random_state = random_state

    def test(self, numeric: pd.DataFrame) -> pd.DataFrame:
        """Тестує кожну числову ознаку.

        Args:
            numeric: Числові ознаки навчальної вибірки.

        Returns:
            Таблиця: «Ознака», «Асиметрія», «Ексцес», статистика W і p-значення
            Шапіро-Уїлка, p-значення Д'Агостіно-Пірсона, «Нормальна» (так/ні —
            за обома тестами одразу). Статистика K² Д'Агостіно не виводиться:
            таблиця має вміститися на сторінку звіту, а рішення приймається за p.
        """
        rows = [self._test_feature(name, numeric[name].dropna()) for name in numeric.columns]
        return pd.DataFrame(rows)

    def qq_sample(self, numeric: pd.DataFrame) -> pd.DataFrame:
        """Випадкова підвибірка рядків для Q-Q графіків (того ж розміру, що й для тесту).

        На десятках тисяч точок Q-Q графік зливається в лінію-«ковбасу»;
        кількох тисяч достатньо, щоб побачити форму хвостів.
        """
        return numeric.sample(
            n=min(self._shapiro_sample_size, len(numeric)), random_state=self._random_state
        )

    @staticmethod
    def features_for_qq(table: pd.DataFrame, count: int) -> list[str]:
        """Ознаки для Q-Q графіків: ``count`` найдальших від нормальності за |асиметрією|.

        Args:
            table: Результат :meth:`test`.
            count: Скільки ознак показати.
        """
        ordered = table.assign(_abs_skew=table["Асиметрія"].abs()).sort_values(
            "_abs_skew", ascending=False
        )
        return list(ordered["Ознака"].head(count))

    @staticmethod
    def to_metrics(table: pd.DataFrame) -> dict[str, object]:
        """Словник для ``metrics.json``: скільки ознак нормальні й яка найскошеніша.

        Args:
            table: Результат :meth:`test`.
        """
        # idxmax — індекс рядка з найбільшим значенням (тут: найбільша |асиметрія|).
        most_skewed = table.loc[table["Асиметрія"].abs().idxmax()]
        return {
            "n_features_tested": len(table),
            "n_normal_features": int((table["Нормальна"] == "так").sum()),
            "max_abs_skew": abs(most_skewed["Асиметрія"]),
            "most_skewed_feature": most_skewed["Ознака"],
        }

    def _test_feature(self, name: str, values: pd.Series) -> dict[str, object]:
        sample = values.sample(
            n=min(self._shapiro_sample_size, len(values)), random_state=self._random_state
        )
        shapiro = stats.shapiro(sample)
        dagostino = stats.normaltest(values)
        is_normal = shapiro.pvalue > self._alpha and dagostino.pvalue > self._alpha
        return {
            "Ознака": name,
            # Асиметрія: 0 — симетричний; > 0 — довгий правий хвіст.
            "Асиметрія": stats.skew(values),
            # Ексцес (надлишковий): 0 — як у нормального; > 0 — важчі хвости.
            "Ексцес": stats.kurtosis(values),
            "W (Ш-У)": shapiro.statistic,
            "p (Ш-У)": _format_p_value(shapiro.pvalue),
            "p (Д-П)": _format_p_value(dagostino.pvalue),
            "Нормальна": "так" if is_normal else "ні",
        }


def _format_p_value(p_value: float) -> str:
    """Форматує p-значення: дуже малі показує як ``<0.0001``, а не ``0.0000``."""
    return f"<{_P_VALUE_FLOOR:.4f}" if p_value < _P_VALUE_FLOOR else f"{p_value:.4f}"
