"""Детерміноване очищення сирої таблиці.

«Детерміноване» означає: правила нічого не «вчать» із даних (не рахують
середніх, не підбирають параметрів), а лише виправляють очевидні технічні
вади — типи, регістр, позначки пропусків. Тому їх можна безпечно застосувати
до всього датасету ще ДО розбиття на train/test: витоку даних це не створює.
Усе, що вчиться (імп'ютація, кодування, масштабування), живе в
``mllabs.preprocessing`` і виконується лише на train.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mllabs.data.schema import DatasetSchema

_KEPT = "залишено"
_DROPPED_DUPLICATE = "видалено: дублікат"
_DROPPED_SPARSE = "видалено: забагато пропусків"


@dataclass(frozen=True)
class CleaningResult:
    """Результат очищення.

    Attributes:
        data: Очищена таблиця (ознаки + ціль).
        schema: Схема без колонок, видалених під час очищення.
        log: Журнал правил: «Правило», «Колонок», «Змінено значень».
        missing_report: Пропуски по колонках: явні (NaN у сирих даних),
            приховані (стали NaN після правил), частка та рішення щодо колонки.
        hidden_missing_cells: Скільки значень правила перетворили на пропуски.
    """

    data: pd.DataFrame
    schema: DatasetSchema
    log: pd.DataFrame
    missing_report: pd.DataFrame
    hidden_missing_cells: int


class RawDataCleaner:
    """Застосовує до сирої таблиці правила очищення, описані схемою.

    Args:
        schema: Схема датасету: з неї беруться ролі колонок і параметри правил.
        max_missing_share: Колонки з більшою часткою пропусків видаляються —
            заповнювати більшість значень вигаданими числами безглуздо.

    Застереження щодо витоку: частка пропусків для цього правила рахується на
    ВСЬОМУ датасеті, до розбиття на train/test, — тобто строго кажучи, рішення
    спирається й на тестові рядки. Це прийнятно, бо (1) рішення бінарне і
    стосується колонки, а не значень: жодна статистика тесту не потрапляє в
    модель; (2) не використовується цільова змінна; (3) у ``kick`` під правило
    потрапляють лише ``PRIMEUNIT`` і ``AUCGUART`` з ~95 % пропусків — за
    стратифікованого розбиття їхня частка однакова в train і test, тож рішення
    на самому train було б тим самим. Для датасету, де частка пропусків
    близька до порога, правило слід застосовувати після розбиття, лише за train.
    """

    def __init__(self, schema: DatasetSchema, max_missing_share: float) -> None:
        self._schema = schema
        self._max_missing_share = max_missing_share

    def clean(self, raw: pd.DataFrame) -> CleaningResult:
        """Очищує сиру таблицю й повертає її разом зі звітом про зміни.

        Args:
            raw: Сира таблиця з джерела даних; не змінюється.

        Returns:
            Очищена таблиця, оновлена схема, журнал правил і звіт про пропуски.
        """
        schema = self._schema
        explicit_missing = raw.isna().sum()
        n_cast = len(schema.categorical) + 1 + (schema.date_column is not None)

        data = self._cast_types(raw)
        data, n_case = self._normalize_case(data)
        data, n_tokens = self._tokens_to_missing(data)
        data, n_prices = self._implausible_prices_to_missing(data)
        normalized_missing = data.isna().sum()

        duplicates = [column for column in schema.drop_columns if column in data.columns]
        missing_share = data.drop(columns=duplicates).isna().mean()
        # Булева маска як індекс: лишає тільки ті елементи Series, де умова істинна.
        sparse = list(missing_share[missing_share > self._max_missing_share].index)
        data = data.drop(columns=duplicates + sparse)

        log = pd.DataFrame(
            [
                ("Приведення типів (ціль, категорії, дата)", n_cast, n_cast * len(raw)),
                ("Уніфікація регістру категорій", len(schema.uppercase_columns), n_case),
                ("Текстові позначки пропуску → NaN", len(schema.categorical), n_tokens),
                ("Ціни ≤ 1 → NaN (прихований пропуск)", len(schema.price_columns), n_prices),
                ("Видалення колонок-дублікатів", len(duplicates), len(duplicates) * len(raw)),
                (
                    f"Видалення колонок із пропусками > {self._max_missing_share:.0%}",
                    len(sparse),
                    len(sparse) * len(raw),
                ),
            ],
            columns=["Правило", "Колонок", "Змінено значень"],
        )
        missing_report = self._missing_report(
            explicit_missing, normalized_missing, len(raw), duplicates, sparse
        )
        return CleaningResult(
            data=data,
            schema=schema.without(duplicates + sparse),
            log=log,
            missing_report=missing_report,
            hidden_missing_cells=n_tokens + n_prices,
        )

    def _cast_types(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Приводить колонки до робочих типів: ціль — int, категорії — рядки, дата — datetime."""
        schema = self._schema
        data = raw.copy()
        data[schema.target] = data[schema.target].astype(int)
        for column in (*schema.categorical, *schema.drop_columns):
            if column in data.columns:
                data[column] = _to_text(data[column])
        date = schema.date_column
        if date is not None and not pd.api.types.is_datetime64_any_dtype(data[date]):
            # У сирих даних дата — це unix-час: кількість секунд від 01.01.1970.
            data[date] = pd.to_datetime(data[date], unit="s")
        return data

    def _normalize_case(self, data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Переводить вказані схемою колонки у верхній регістр (``Manual`` → ``MANUAL``)."""
        result = data.copy()
        changed = 0
        for column in self._schema.uppercase_columns:
            upper = _to_text(result[column].str.upper())
            # fillna("") потрібен, бо NaN != NaN, і пропуски рахувались би як «змінені».
            changed += int((upper.fillna("") != result[column].fillna("")).sum())
            result[column] = upper
        return result, changed

    def _tokens_to_missing(self, data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Замінює текстові позначки на кшталт ``NOT AVAIL`` справжнім пропуском."""
        result = data.copy()
        changed = 0
        tokens = list(self._schema.missing_tokens)
        for column in self._schema.categorical:
            is_token = result[column].isin(tokens)
            changed += int(is_token.sum())
            # mask(умова) ставить NaN там, де умова істинна, решту лишає як є.
            result[column] = result[column].mask(is_token)
        return result, changed

    def _implausible_prices_to_missing(self, data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Ціни 0 або 1 долар — це не ціни, а «значення невідоме»; робимо їх пропусками."""
        result = data.copy()
        prices = list(self._schema.price_columns)
        is_implausible = result[prices] <= 1
        result[prices] = result[prices].mask(is_implausible)
        return result, int(is_implausible.sum().sum())

    def _missing_report(
        self,
        explicit: pd.Series,
        normalized: pd.Series,
        n_rows: int,
        duplicates: list[str],
        sparse: list[str],
    ) -> pd.DataFrame:
        """Збирає таблицю пропусків по колонках, відсортовану за спаданням частки."""
        decision = pd.Series(_KEPT, index=normalized.index)
        decision[duplicates] = _DROPPED_DUPLICATE
        decision[sparse] = _DROPPED_SPARSE
        report = pd.DataFrame(
            {
                "Ознака": [self._schema.label(column) for column in normalized.index],
                "Явні": explicit.to_numpy(),
                "Приховані": (normalized - explicit).to_numpy(),
                "Частка, %": (normalized / n_rows * 100).to_numpy(),
                "Рішення": decision.to_numpy(),
            }
        )
        has_missing = (report["Явні"] + report["Приховані"]) > 0
        return (
            report[has_missing | (report["Рішення"] != _KEPT)]
            .sort_values("Частка, %", ascending=False)
            .reset_index(drop=True)
        )


def _to_text(values: pd.Series) -> pd.Series:
    """Перетворює колонку на рядки (тип ``object``), зберігаючи пропуски як NaN.

    Навіть «числові» на вигляд коди (поштовий індекс, номер покупця) — це
    категорії: арифметика над ними не має змісту, тому зберігаємо їх текстом.
    """
    # map(..., na_action="ignore") застосовує функцію до кожного значення, крім NaN.
    text = values.astype(object).map(lambda value: str(value).strip(), na_action="ignore")
    # where(умова, інакше) лишає значення там, де умова істинна. Явний object замість
    # нового типу pandas "str" — його однаково розуміють усі трансформери sklearn.
    return text.astype(object).where(text.notna(), np.nan)
