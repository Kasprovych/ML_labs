"""Інженерія ознак: нові колонки, обчислені з наявних.

Два трансформери з принципово різною природою:

* :class:`RowFeatureEngineer` — БЕЗ стану: нова ознака рядка залежить лише від
  цього рядка (відношення, логарифм, частина дати). Її можна рахувати будь-де.
* :class:`GroupAggregateEncoder` — ЗІ станом: «середня ціна авто цієї моделі»
  рахується по всій вибірці, тому має навчатися (``fit``) лише на train,
  інакше тестові рядки вплинуть на власні ознаки (витік даних).

Назви колонок тут — із датасету Carvana ``kick`` (див. ``mllabs.data.schema``).

Чого тут свідомо НЕМАЄ — ознак, які були б точною (або майже точною) копією
інших колонок пайплайна й лише роздували б мультиколінеарність:

* різниці цін на кшталт ``AcqRetAvg − AcqAucAvg`` — точна лінійна комбінація
  ознак, які вже є в моделі (лінійна модель «побудує» її сама);
* ``ln(1 + WarrantyCost)`` — після ``PowerTransformer`` (Yeo-Johnson, близький
  до логарифма) вона збігалася з перетвореною ``WarrantyCost`` (VIF ≈ 7·10⁵);
* сама статистика групи поруч із відхиленням від неї: ``ціна − середнє`` разом
  із ``ціна`` і ``середнє`` — це три колонки, одна з яких є сумою двох інших.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_ACQUISITION_AUCTION = "MMRAcquisitionAuctionAveragePrice"
_CURRENT_AUCTION = "MMRCurrentAuctionAveragePrice"

_GROUP_NONLINEAR = "нелінійна комбінація"
_GROUP_DATE = "частина дати"
_GROUP_INDICATOR = "індикатор пропуску"
_GROUP_AGGREGATE = "агрегація за категорією"


@dataclass(frozen=True)
class FeatureDescription:
    """Опис однієї згенерованої ознаки для таблиці у звіті.

    Attributes:
        name: Назва колонки.
        group: До якої групи методів належить ознака.
        formula: Як вона обчислюється.
    """

    name: str
    group: str
    formula: str


class RowFeatureEngineer(TransformerMixin, BaseEstimator):
    """Додає ознаки, що обчислюються в межах одного рядка.

    Навіщо: лінійна модель не здатна сама «здогадатися» поділити ціну покупки
    на ринкову оцінку, а саме це відношення показує, переплатили за авто чи ні.

    Ділення на пропуск дає пропуск — його згодом заповнить імп'ютер пайплайна.
    Ділення на нуль неможливе: ціни ``<= 1`` уже перетворено на пропуски під
    час очищення.

    Застереження щодо мультиколінеарності: відношення ``a / b`` не є лінійною
    комбінацією ``a`` і ``b``, але після ``PowerTransformer`` (≈ логарифм для
    додатних цін) воно стає близьким до різниці ``log a − log b``. Тому VIF
    відношень у пайплайні високий (десятки), хоча точної залежності немає;
    логістична регресія з L2-регуляризацією з цим справляється.
    """

    #: Числові ознаки-комбінації (далі проходять імп'ютацію та масштабування).
    NUMERIC_FEATURES: tuple[str, ...] = (
        "cost_to_mmr",
        "mmr_trend",
        "miles_per_year",
        "warranty_share",
    )
    #: Частини дати — категорії без порядку (кодуються One-Hot).
    DATE_FEATURES: tuple[str, ...] = ("purchase_year", "purchase_month", "purchase_weekday")
    #: Індикатори 0/1 — у модель ідуть без змін.
    INDICATOR_FEATURES: tuple[str, ...] = ("wheel_type_missing", "mmr_missing")
    #: Індикатори пропуску ОДНІЄЇ категоріальної колонки: «індикатор → колонка».
    #: Якщо колонку кодує One-Hot з окремою категорією ``MISSING``, індикатор
    #: дослівно повторює колонку ``<колонка>_MISSING`` і в пайплайн не подається
    #: (див. ``PreprocessorFactory.indicator_columns``). ``MappingProxyType`` —
    #: незмінне «вікно» на словник, як ``Collections.unmodifiableMap`` у Java.
    CATEGORY_MISSING_INDICATORS: Mapping[str, str] = MappingProxyType(
        {"wheel_type_missing": "WheelType"}
    )

    def fit(self, features: pd.DataFrame, target: object = None) -> Self:
        """Нічого не вчить: метод потрібен лише інтерфейсу sklearn."""
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Повертає копію таблиці з доданими ознаками.

        Args:
            features: Очищена таблиця з колонками датасету ``kick``.
        """
        acquisition = features[_ACQUISITION_AUCTION]
        current = features[_CURRENT_AUCTION]
        purchase_date = features["PurchDate"]
        # assign повертає НОВУ таблицю з доданими колонками; вихідна не змінюється.
        return features.assign(
            cost_to_mmr=features["VehBCost"] / acquisition,
            mmr_trend=current / acquisition,
            miles_per_year=features["VehOdo"] / (features["VehicleAge"] + 1),
            warranty_share=features["WarrantyCost"] / features["VehBCost"],
            # .dt — доступ до «календарних» властивостей колонки з датами.
            purchase_year=purchase_date.dt.year,
            purchase_month=purchase_date.dt.month,
            purchase_weekday=purchase_date.dt.day_of_week,
            wheel_type_missing=features["WheelType"].isna().astype(int),
            mmr_missing=(acquisition.isna() | current.isna()).astype(int),
        )

    @classmethod
    def describe(cls) -> list[FeatureDescription]:
        """Перелік згенерованих ознак із формулами."""
        return [
            FeatureDescription("cost_to_mmr", _GROUP_NONLINEAR, "VehBCost / AcqAucAvg"),
            FeatureDescription("mmr_trend", _GROUP_NONLINEAR, "CurAucAvg / AcqAucAvg"),
            FeatureDescription("miles_per_year", _GROUP_NONLINEAR, "VehOdo / (VehicleAge + 1)"),
            FeatureDescription("warranty_share", _GROUP_NONLINEAR, "WarrantyCost / VehBCost"),
            FeatureDescription("purchase_year", _GROUP_DATE, "рік із PurchDate"),
            FeatureDescription("purchase_month", _GROUP_DATE, "місяць із PurchDate"),
            FeatureDescription("purchase_weekday", _GROUP_DATE, "день тижня з PurchDate"),
            FeatureDescription(
                "wheel_type_missing", _GROUP_INDICATOR, "1, якщо WheelType порожній"
            ),
            FeatureDescription(
                "mmr_missing", _GROUP_INDICATOR, "1, якщо AcqAucAvg або CurAucAvg порожня"
            ),
        ]


@dataclass(frozen=True)
class GroupAggregation:
    """Одна агрегація: статистика числової ознаки в межах категорії.

    Attributes:
        group_column: Колонка, за якою групуємо (напр. ``Model``).
        value_column: Числова колонка, яку агрегуємо (напр. ``VehBCost``).
        statistic: ``"mean"`` або ``"median"``.
    """

    group_column: str
    value_column: str
    statistic: str

    @property
    def aggregate_name(self) -> str:
        """Внутрішній ключ статистики групи: ``VehBCost_mean_by_Model``."""
        return f"{self.value_column}_{self.statistic}_by_{self.group_column}"

    @property
    def deviation_name(self) -> str:
        """Назва колонки з відхиленням рядка від своєї групи: ``VehBCost_dev_by_Model``."""
        return f"{self.value_column}_dev_by_{self.group_column}"


#: Агрегації для датасету ``kick``. Цільова змінна НЕ використовується — інакше
#: це було б Target Encoding з усіма його ризиками витоку.
KICK_AGGREGATIONS: tuple[GroupAggregation, ...] = (
    GroupAggregation("Model", "VehBCost", "mean"),
    GroupAggregation("Make", "VehBCost", "mean"),
    GroupAggregation("VehicleAge", "VehOdo", "median"),
)


class GroupAggregateEncoder(TransformerMixin, BaseEstimator):
    """Додає відхилення рядка від статистики своєї групи.

    Приклад: «наскільки це авто дорожче чи дешевше за середнє для своєї
    моделі». Підозріло дешева Impala — сигнал, якого не видно з самої ціни.

    Сама статистика групи («середня ціна моделі») у таблицю НЕ додається:
    разом із вихідною ознакою й відхиленням вона дала б точну лінійну
    залежність ``ознака = статистика + відхилення``, а медіанний пробіг за
    ``VehicleAge`` — ще й функцію від самого ``VehicleAge``. Лінійна модель
    за потреби «відновить» статистику як ``ознака − відхилення``.

    Статистики груп рахуються у ``fit`` (лише train) і зберігаються. Для групи,
    якої в train не було, береться статистика по всій навчальній вибірці.

    Args:
        aggregations: Які агрегації рахувати.

    Attributes:
        statistics_: Словник «ключ агрегації → Series (група → статистика)».
        fallbacks_: Словник «ключ агрегації → статистика по всьому train».
    """

    def __init__(self, aggregations: Sequence[GroupAggregation] = KICK_AGGREGATIONS) -> None:
        # Угода sklearn: конструктор лише запам'ятовує параметри під тими самими
        # іменами — так працюють clone() і get_params(), потрібні крос-валідації.
        self.aggregations = aggregations

    def fit(self, features: pd.DataFrame, target: object = None) -> Self:
        """Рахує статистику кожної групи на навчальній вибірці."""
        self.statistics_: dict[str, pd.Series] = {}
        self.fallbacks_: dict[str, float] = {}
        for aggregation in self.aggregations:
            values = features[aggregation.value_column]
            # groupby(...).agg(...) — як GROUP BY у SQL: одна статистика на групу.
            by_group = values.groupby(features[aggregation.group_column]).agg(aggregation.statistic)
            self.statistics_[aggregation.aggregate_name] = by_group
            self.fallbacks_[aggregation.aggregate_name] = float(values.agg(aggregation.statistic))
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Повертає копію таблиці з колонками відхилень від статистик груп."""
        new_columns: dict[str, pd.Series] = {}
        for aggregation in self.aggregations:
            name = aggregation.aggregate_name
            group_statistic = (
                features[aggregation.group_column]
                .map(self.statistics_[name])
                .astype(float)
                .fillna(self.fallbacks_[name])
            )
            new_columns[aggregation.deviation_name] = (
                features[aggregation.value_column] - group_statistic
            )
        return features.assign(**new_columns)

    @property
    def output_features(self) -> tuple[str, ...]:
        """Назви всіх колонок, які додає трансформер."""
        return tuple(aggregation.deviation_name for aggregation in self.aggregations)

    def describe(self) -> list[FeatureDescription]:
        """Перелік згенерованих ознак із формулами."""
        descriptions: list[FeatureDescription] = []
        for aggregation in self.aggregations:
            statistic = "середнє" if aggregation.statistic == "mean" else "медіана"
            aggregate = f"{statistic} {aggregation.value_column} у групі {aggregation.group_column}"
            descriptions.append(
                FeatureDescription(
                    aggregation.deviation_name,
                    _GROUP_AGGREGATE,
                    f"{aggregation.value_column} − {aggregate}",
                )
            )
        return descriptions
