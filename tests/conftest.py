"""Спільні фікстури тестів.

Фікстура (fixture) у pytest — аналог ``@BeforeEach`` у JUnit: готує
оточення для тесту й передається в нього як аргумент за іменем.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mllabs.core.paths import ProjectPaths

_ASSIGNMENT = """---
number: 1
topic: "Тестова тема"
purpose: "Тестова мета"
---
# Завдання
"""

_REPORT_SETTINGS = """
[university]
ministry = "МІНІСТЕРСТВО"
name = "УНІВЕРСИТЕТ"
institute = "ІНСТИТУТ"
department = "кафедра"
city = "Львів"
[work]
kind = "лабораторної роботи"
course = "Курс"
[student]
group = "КН-1"
full_name = "???"
[teacher]
position = "доц."
full_name = "Викладач В. В."
"""


@pytest.fixture
def project(tmp_path: Path) -> ProjectPaths:
    """Мініатюрний проєкт у тимчасовому каталозі (справжні файли не чіпаються)."""
    paths = ProjectPaths(root=tmp_path)
    lab = paths.lab(1)
    lab.root.mkdir(parents=True)
    lab.assignment_file.write_text(_ASSIGNMENT, encoding="utf-8")
    paths.config_dir.mkdir()
    (paths.config_dir / "report.toml").write_text(_REPORT_SETTINGS, encoding="utf-8")
    return paths


def _make_raw_kick_frame(n_rows: int, seed: int) -> pd.DataFrame:
    """Крихітна синтетична копія сирого датасету ``kick`` у форматі OpenML.

    Відтворює все, що важливо для коду: ті самі колонки й типи (категорії —
    рядки, ціль — ``'0'``/``'1'``, дата — unix-секунди), приховані пропуски
    (нульові ціни, ``NOT AVAIL``), різний регістр (``Manual``), колонки з 95 %
    пропусків і ціль, що залежить від віку авто та пропуску ``WheelType``.
    """
    rng = np.random.default_rng(seed)

    def pick(options: list[str], probabilities: list[float] | None = None) -> np.ndarray:
        return rng.choice(options, size=n_rows, p=probabilities).astype(object)

    age = rng.integers(1, 10, size=n_rows)
    base_price = rng.lognormal(8.8, 0.3, size=n_rows)
    prices = {
        name: base_price * factor * rng.normal(1.0, 0.03, size=n_rows)
        for name, factor in (
            ("MMRAcquisitionAuctionAveragePrice", 1.0),
            ("MMRAcquisitionAuctionCleanPrice", 1.15),
            ("MMRAcquisitionRetailAveragePrice", 1.4),
            ("MMRAcquisitonRetailCleanPrice", 1.6),
            ("MMRCurrentAuctionAveragePrice", 1.02),
            ("MMRCurrentAuctionCleanPrice", 1.17),
            ("MMRCurrentRetailAveragePrice", 1.42),
            ("MMRCurrentRetailCleanPrice", 1.62),
        )
    }
    wheel_type = pick(["Alloy", "Covers", "Special"], [0.5, 0.45, 0.05])
    wheel_missing = rng.random(n_rows) < 0.05
    wheel_type[wheel_missing] = None
    logit = -3.0 + 0.3 * age + 2.5 * wheel_missing
    target = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)

    frame = pd.DataFrame(
        {
            "IsBadBuy": target.astype(str),
            "PurchDate": rng.integers(1_230_000_000, 1_290_000_000, size=n_rows).astype(float),
            "Auction": pick(["ADESA", "MANHEIM", "OTHER"]),
            "VehYear": (2010 - age).astype(float),
            "VehicleAge": age.astype(float),
            "Make": pick([f"MAKE{i}" for i in range(6)]),
            "Model": pick([f"MODEL{i}" for i in range(40)]),
            "Trim": pick([f"T{i}" for i in range(12)]),
            "SubModel": pick([f"SUB{i}" for i in range(30)]),
            "Color": pick(["SILVER", "WHITE", "BLUE", "NOT AVAIL"], [0.4, 0.3, 0.28, 0.02]),
            "Transmission": pick(["AUTO", "MANUAL", "Manual"], [0.9, 0.08, 0.02]),
            "WheelTypeID": wheel_type.copy(),
            "WheelType": wheel_type,
            "VehOdo": rng.normal(70_000, 15_000, size=n_rows),
            "Nationality": pick(["AMERICAN", "OTHER ASIAN", "TOP LINE ASIAN"]),
            "Size": pick(["MEDIUM", "LARGE", "COMPACT", "SUV"]),
            "TopThreeAmericanName": pick(["GM", "CHRYSLER", "FORD", "OTHER"]),
            **prices,
            "PRIMEUNIT": pick(["NO", "YES"]),
            "AUCGUART": pick(["GREEN", "RED"]),
            "BYRNO": pick([str(code) for code in range(10_000, 10_010)]),
            "VNZIP1": pick([str(code) for code in range(33_600, 33_615)]),
            "VNST": pick(["FL", "TX", "CA", "NC", "AZ", "CO"]),
            "VehBCost": base_price * rng.normal(1.0, 0.1, size=n_rows),
            "IsOnlineSale": pick(["0", "1"], [0.97, 0.03]),
            "WarrantyCost": rng.lognormal(7.0, 0.4, size=n_rows),
        }
    )
    # Приховані та явні пропуски, як у справжніх даних.
    frame.loc[rng.random(n_rows) < 0.02, "MMRAcquisitionAuctionAveragePrice"] = 0.0
    frame.loc[rng.random(n_rows) < 0.01, "MMRCurrentAuctionAveragePrice"] = np.nan
    frame.loc[rng.random(n_rows) < 0.03, "Trim"] = None
    frame.loc[rng.random(n_rows) < 0.95, ["PRIMEUNIT", "AUCGUART"]] = None
    categorical = frame.select_dtypes(exclude="number").columns
    return frame.astype(dict.fromkeys(categorical, "category"))


@pytest.fixture(scope="session")
def raw_kick_frame() -> pd.DataFrame:
    """Сира синтетична таблиця ``kick`` на 1200 рядків (одна на всю сесію тестів)."""
    return _make_raw_kick_frame(n_rows=1200, seed=0)


@pytest.fixture(scope="session")
def clean_kick_frame(raw_kick_frame: pd.DataFrame) -> pd.DataFrame:
    """Та сама таблиця після ``RawDataCleaner`` (ознаки + ціль)."""
    from mllabs.data import KICK_SCHEMA, RawDataCleaner

    return RawDataCleaner(KICK_SCHEMA, max_missing_share=0.5).clean(raw_kick_frame).data
