"""Датасет VisA (Visual Anomaly, Amazon, 2022): завантаження, перевірка, розпакування, опис.

VisA — 12 категорій промислових об'єктів (друковані плати, макарони, капсули…), для
кожної є нормальні зображення й аномальні з піксельними масками дефектів. Архів
один (1,93 ГБ), тому:

* архів качається потоково з докачуванням (HTTP ``Range``), перевіряється SHA-256;
* розпаковуються лише потрібні категорії (і невелика вибірка інших для виміру тла);
* архів — недовірені дані: шляхи членів перевіряються, ``tarfile`` працює з
  ``filter="data"`` (жоден файл не потрапить за межі цільового каталогу).
"""

from __future__ import annotations

import csv
import hashlib
import logging
import tarfile
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from mllabs.data import DatasetDownloadError

_logger = logging.getLogger(__name__)

#: Файл офіційного розбиття «1 клас» (train — лише норма, test — норма й аномалії).
SPLIT_MEMBER = "split_csv/1cls.csv"
#: Розмір шматка під час читання/запису архіву (8 МБ).
_CHUNK = 8 * 1024 * 1024
#: Маркер завершеного розпакування (щоб не читати архів щоразу).
_EXTRACT_MARKER = ".extracted.txt"
#: Маркер перевіреного SHA-256 (розмір і час зміни архіву на момент перевірки).
_VERIFIED_SUFFIX = ".sha256-verified"
#: Усі 12 категорій VisA (порядок як в архіві).
ALL_CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)


@dataclass(frozen=True)
class VisaSample:
    """Одне зображення VisA з офіційного розбиття.

    Attributes:
        category: Категорія, напр. ``"macaroni2"``.
        official_split: ``"train"`` або ``"test"`` з ``1cls.csv``.
        label: 0 — норма, 1 — аномалія.
        image: Шлях до зображення відносно каталогу VisA.
        mask: Шлях до маски дефекту (лише для аномалій), інакше ``""``.
        defect_types: Типи дефектів через кому з ``image_anno.csv`` (``""`` для норми).
    """

    category: str
    official_split: str
    label: int
    image: str
    mask: str
    defect_types: str


class VisaDatasetSource:
    """Отримує архів VisA, розпаковує потрібне й читає опис зображень.

    Args:
        root: Каталог VisA у ``data/raw/visa`` (туди кладуться архів і розпаковані файли).
        url: Пряме посилання на архів.
        sha256: Очікувана контрольна сума архіву.
        archive_name: Ім'я файлу архіву.
    """

    def __init__(self, root: Path, url: str, sha256: str, archive_name: str) -> None:
        self._root = root
        self._url = url
        self._sha256 = sha256
        self._archive = root / archive_name

    @property
    def root(self) -> Path:
        """Каталог, відносно якого задано шляхи ``VisaSample.image`` / ``mask``."""
        return self._root

    def ensure_archive(self) -> Path:
        """Гарантує наявність перевіреного архіву: качає/докачує й звіряє SHA-256.

        Raises:
            DatasetDownloadError: Якщо завантаження не вдалося або контрольна сума не збіглася.
        """
        if self._archive.is_file() and self._is_verified():
            return self._archive
        if not self._archive.is_file():
            self._download()
        self._verify()
        return self._archive

    def extract(
        self,
        categories: Sequence[str],
        sample_categories: Sequence[str],
        normal_per_sample_category: int,
    ) -> None:
        """Розпаковує потрібні члени архіву за ОДИН послідовний прохід.

        Args:
            categories: Категорії, що розпаковуються повністю.
            sample_categories: Категорії, з яких беруться лише перші нормальні зображення
                (для виміру дисперсії тла по всьому VisA).
            normal_per_sample_category: Скільки нормальних зображень брати з таких категорій.
        """
        request = f"{sorted(categories)}|{sorted(sample_categories)}|{normal_per_sample_category}"
        marker = self._root / _EXTRACT_MARKER
        if marker.is_file() and marker.read_text(encoding="utf-8") == request:
            return
        archive = self.ensure_archive()
        taken: dict[str, int] = {}
        _logger.info(
            "Розпакування VisA: %s (+ вибірка з %d категорій)", categories, len(sample_categories)
        )
        # Режим "r|" — потокове читання: архів проходиться один раз згори донизу.
        with tarfile.open(archive, mode="r|") as tar:
            for member in tar:
                if not member.isfile() or not is_safe_member(member.name):
                    continue
                if self._wanted(
                    member.name, categories, sample_categories, normal_per_sample_category, taken
                ):
                    tar.extract(member, self._root, filter="data")
        marker.write_text(request, encoding="utf-8")

    def read_samples(self, category: str) -> list[VisaSample]:
        """Читає зображення категорії з ``1cls.csv`` разом із типами дефектів."""
        defect_types = self._read_defect_types(category)
        samples = []
        with (self._root / SPLIT_MEMBER).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["object"] != category:
                    continue
                label = int(row["label"] == "anomaly")
                samples.append(
                    VisaSample(
                        category=category,
                        official_split=row["split"],
                        label=label,
                        image=row["image"],
                        mask=row["mask"] or "",
                        defect_types=defect_types.get(row["image"], "") if label else "",
                    )
                )
        return samples

    def normal_images(self, category: str) -> list[Path]:
        """Нормальні зображення категорії, що є на диску (у т.ч. з частково розпакованих)."""
        return sorted((self._root / category / "Data" / "Images" / "Normal").glob("*.JPG"))

    def _read_defect_types(self, category: str) -> dict[str, str]:
        """Словник «шлях зображення → типи дефектів» з ``<категорія>/image_anno.csv``."""
        path = self._root / category / "image_anno.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            return {row["image"]: row["label"] for row in csv.DictReader(handle)}

    @staticmethod
    def _wanted(
        name: str,
        categories: Sequence[str],
        sample_categories: Sequence[str],
        limit: int,
        taken: dict[str, int],
    ) -> bool:
        """Чи потрібен член архіву (див. :meth:`extract`); ``taken`` — лічильник вибірки."""
        if name == SPLIT_MEMBER:
            return True
        category = PurePosixPath(name).parts[0]
        if category in categories:
            return True
        if category not in sample_categories:
            return False
        if name.endswith("image_anno.csv"):
            return True
        if "/Images/Normal/" in name and taken.get(category, 0) < limit:
            taken[category] = taken.get(category, 0) + 1
            return True
        return False

    def _download(self) -> None:
        """Потокове завантаження у ``.part`` з докачуванням обірваного файлу."""
        partial = self._archive.with_suffix(self._archive.suffix + ".part")
        self._root.mkdir(parents=True, exist_ok=True)
        offset = partial.stat().st_size if partial.is_file() else 0
        request = urllib.request.Request(self._url, headers={"Range": f"bytes={offset}-"})
        _logger.info("Завантаження VisA з %s (з байта %d)", self._url, offset)
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("ab") as out:
                if offset and response.status != 206:  # сервер ігнорує Range — починаємо заново
                    out.truncate(0)
                while chunk := response.read(_CHUNK):  # «морж» := присвоює і перевіряє водночас
                    out.write(chunk)
        except OSError as error:
            raise DatasetDownloadError(f"Не вдалося завантажити VisA: {error}") from error
        partial.rename(self._archive)

    def _verify(self) -> None:
        """Звіряє SHA-256 архіву й запам'ятовує успішну перевірку маркером."""
        digest = file_sha256(self._archive)
        if digest != self._sha256:
            raise DatasetDownloadError(
                f"SHA-256 архіву VisA не збігається: {digest} ≠ {self._sha256}. "
                f"Видаліть {self._archive} і запустіть знову."
            )
        self._verified_marker().write_text(self._fingerprint(), encoding="utf-8")

    def _is_verified(self) -> bool:
        """Чи перевірено саме цей файл (розмір і час зміни не змінилися)."""
        marker = self._verified_marker()
        return marker.is_file() and marker.read_text(encoding="utf-8") == self._fingerprint()

    def _verified_marker(self) -> Path:
        return self._archive.with_suffix(self._archive.suffix + _VERIFIED_SUFFIX)

    def _fingerprint(self) -> str:
        stat = self._archive.stat()
        return f"{self._sha256}:{stat.st_size}:{stat.st_mtime_ns}"


def is_safe_member(name: str) -> bool:
    """Чи безпечний шлях члена архіву: відносний і без ``..`` (захист від path traversal)."""
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def file_sha256(path: Path) -> str:
    """SHA-256 файлу, прочитаного шматками (не тримає 2 ГБ у пам'яті)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def count_by_split(samples: Iterable[VisaSample]) -> dict[str, int]:
    """Кількості «train / test-норма / test-аномалія» — для звірки з офіційними числами."""
    counts = {"train": 0, "test_normal": 0, "test_anomalous": 0}
    for sample in samples:
        if sample.official_split == "train":
            counts["train"] += 1
        elif sample.label:
            counts["test_anomalous"] += 1
        else:
            counts["test_normal"] += 1
    return counts
