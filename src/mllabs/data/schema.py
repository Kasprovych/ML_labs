"""Схема датасету: які колонки є і яку роль кожна відіграє.

Схема — це «паспорт» таблиці. Замість того щоб у десяти місцях писати рядки
на кшталт ``"VehBCost"``, код питає схему: «які ознаки числові?», «яка колонка —
ціль?». Аналог у Java — незмінний ``record`` з метаданими сутності.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class DatasetSchema:
    """Ролі колонок табличного датасету для задачі бінарної класифікації.

    Attributes:
        name: Коротке ім'я датасету; ним називається каталог у ``data/processed/``.
        target: Назва цільової колонки.
        positive_label: Значення цілі, яке вважається «позитивним» класом
            (зазвичай рідкісний клас, який модель має знаходити).
        numeric: Числові ознаки.
        categorical_low: Номінальні ознаки з малою кількістю категорій
            (кодуються One-Hot).
        categorical_high: Ознаки високої кардинальності — десятки й сотні
            категорій (кодуються Target або Frequency Encoding).
        binary: Ознаки рівно з двома значеннями (кодуються Ordinal: 0/1).
        date_column: Колонка з датою або ``None``; у модель іде не сама дата,
            а її частини (рік, місяць, день тижня).
        drop_columns: Колонки, що дублюють інші й видаляються під час очищення.
        collinear_columns: Ознаки, які майже повністю повторюють інші
            (мультиколінеарність); сценарій повного препроцесингу їх прибирає.
        price_columns: Ціни, у яких значення ``<= 1`` — це прихований пропуск
            (авто не може коштувати 0 чи 1 долар).
        uppercase_columns: Категоріальні колонки, де регістр літер не має
            значення (``Manual`` і ``MANUAL`` — одна категорія).
        missing_tokens: Текстові позначки відсутнього значення (``NOT AVAIL``).
        short_names: Скорочені підписи довгих назв колонок для графіків і
            таблиць звіту; колонки без запису підписуються повною назвою.
    """

    name: str
    target: str
    positive_label: int
    numeric: tuple[str, ...]
    categorical_low: tuple[str, ...]
    categorical_high: tuple[str, ...]
    binary: tuple[str, ...] = ()
    date_column: str | None = None
    drop_columns: tuple[str, ...] = ()
    collinear_columns: tuple[str, ...] = ()
    price_columns: tuple[str, ...] = ()
    uppercase_columns: tuple[str, ...] = ()
    missing_tokens: tuple[str, ...] = ()
    # dict — змінюваний тип, тому значення за замовчуванням задається фабрикою:
    # інакше всі екземпляри ділили б ОДИН словник (у Java такого підводного каменя немає).
    short_names: Mapping[str, str] = field(default_factory=dict)

    @property
    def categorical(self) -> tuple[str, ...]:
        """Усі категоріальні ознаки: низької, високої кардинальності та бінарні."""
        return self.categorical_low + self.categorical_high + self.binary

    @property
    def features(self) -> tuple[str, ...]:
        """Усі ознаки (без цілі) у порядку: числові, категоріальні, дата."""
        date = (self.date_column,) if self.date_column else ()
        return self.numeric + self.categorical + date

    def label(self, column: str) -> str:
        """Повертає підпис колонки для графіка чи таблиці (скорочений, якщо заданий)."""
        return self.short_names.get(column, column)

    def without(self, columns: Collection[str]) -> DatasetSchema:
        """Повертає копію схеми без указаних колонок.

        Потрібно після очищення: колонки, видалені через надлишок пропусків,
        мають зникнути й зі схеми.

        Args:
            columns: Назви колонок, яких у таблиці більше немає.
        """

        def keep(names: tuple[str, ...]) -> tuple[str, ...]:
            # Генератор у дужках — «ледачий» аналог stream().filter(...); tuple() його збирає.
            return tuple(name for name in names if name not in columns)

        # dataclasses.replace створює копію frozen-об'єкта зі зміненими полями
        # (як «wither»-метод у Java record).
        return replace(
            self,
            numeric=keep(self.numeric),
            categorical_low=keep(self.categorical_low),
            categorical_high=keep(self.categorical_high),
            binary=keep(self.binary),
            collinear_columns=keep(self.collinear_columns),
            price_columns=keep(self.price_columns),
            uppercase_columns=keep(self.uppercase_columns),
        )

    def to_dict(self) -> dict[str, Any]:
        """Перетворює схему на словник зі звичайних типів (для ``schema.json``)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DatasetSchema:
        """Відновлює схему зі словника, прочитаного з JSON.

        JSON не має кортежів, тому списки перетворюються назад на ``tuple``.
        """
        # {ключ: значення for ...} — dict comprehension, аналог Collectors.toMap.
        values = {
            key: tuple(value) if isinstance(value, list) else value for key, value in raw.items()
        }
        return cls(**values)  # ** розпаковує словник в іменовані аргументи конструктора


_MMR_PRICES = (
    "MMRAcquisitionAuctionAveragePrice",
    "MMRAcquisitionAuctionCleanPrice",
    "MMRAcquisitionRetailAveragePrice",
    "MMRAcquisitonRetailCleanPrice",  # одруківка «Acquisiton» — в оригінальних даних
    "MMRCurrentAuctionAveragePrice",
    "MMRCurrentAuctionCleanPrice",
    "MMRCurrentRetailAveragePrice",
    "MMRCurrentRetailCleanPrice",
)

#: Схема датасету Carvana «Don't Get Kicked!» (OpenML id 41162).
#:
#: Ціль ``IsBadBuy = 1`` — куплене на аукціоні авто виявилося «поганим» (kick).
#: Ціни MMR (Manheim Market Report) — ринкові оцінки авто: на момент придбання
#: (Acquisition) і поточні (Current), аукціонні (Auction) і роздрібні (Retail),
#: для середнього (Average) та відмінного (Clean) стану.
KICK_SCHEMA = DatasetSchema(
    name="kick",
    target="IsBadBuy",
    positive_label=1,
    numeric=("VehYear", "VehicleAge", "VehOdo", *_MMR_PRICES, "VehBCost", "WarrantyCost"),
    categorical_low=(
        "Auction",
        "Color",
        "WheelType",
        "Nationality",
        "Size",
        "TopThreeAmericanName",
        "PRIMEUNIT",
        "AUCGUART",
    ),
    categorical_high=("Make", "Model", "Trim", "SubModel", "BYRNO", "VNZIP1", "VNST"),
    binary=("Transmission", "IsOnlineSale"),
    date_column="PurchDate",
    drop_columns=("WheelTypeID",),
    collinear_columns=(
        "VehYear",
        "MMRAcquisitionAuctionCleanPrice",
        "MMRAcquisitonRetailCleanPrice",
        "MMRCurrentAuctionCleanPrice",
        "MMRCurrentRetailCleanPrice",
    ),
    price_columns=(*_MMR_PRICES, "VehBCost"),
    uppercase_columns=("Transmission",),
    missing_tokens=("NOT AVAIL",),
    short_names={
        "MMRAcquisitionAuctionAveragePrice": "AcqAucAvg",
        "MMRAcquisitionAuctionCleanPrice": "AcqAucClean",
        "MMRAcquisitionRetailAveragePrice": "AcqRetAvg",
        "MMRAcquisitonRetailCleanPrice": "AcqRetClean",
        "MMRCurrentAuctionAveragePrice": "CurAucAvg",
        "MMRCurrentAuctionCleanPrice": "CurAucClean",
        "MMRCurrentRetailAveragePrice": "CurRetAvg",
        "MMRCurrentRetailCleanPrice": "CurRetClean",
        "TopThreeAmericanName": "TopThreeName",
    },
)
