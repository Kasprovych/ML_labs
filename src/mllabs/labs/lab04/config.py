"""Параметри лабораторної роботи №4.

Усі числа експерименту зібрано тут (аналог ``record`` з налаштуваннями в Java). Поруч із
кожним — чому саме таке значення. Регулятори бюджету часу (розд. 5.1 плану):
``fae.vectors_per_epoch``, ``fae.max_epochs``, ``patchcore_coreset_ratio``, ``latency_runs``,
``study_tile_grid``. Вони мають значення з плану, крім ``fae``: 60 епох × 25 тис. векторів
замість 30 × 50 тис., щоб могло спрацювати раннє зупинення (див. опис атрибута ``fae``).
Бюджет вкладається у 30–40 хв (див. ``metrics.timing``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mllabs.anomaly.backbones import BackboneKind
from mllabs.anomaly.fae import FaeSettings
from mllabs.anomaly.visa import VISA_ARCHIVE, VISA_SHA256, VISA_URL
from mllabs.anomaly.workspace import WorkspaceSettings


@dataclass(frozen=True)
class PublishedResult:
    """Опубліковане число для довідкового порівняння (не пряме: інший протокол).

    Attributes:
        method: Метод.
        category: Категорія VisA або ``"mean"`` (середнє по 12 категоріях).
        metric: ``"image_auroc"``, ``"pixel_auroc"``, ``"pixel_ap"``.
        value: Значення (частка, 0–1).
        source: Стаття і таблиця.
        protocol: Коротко про протокол (backbone, роздільність, розбиття).
    """

    method: str
    category: str
    metric: str
    value: float
    source: str
    protocol: str


_SPD = "Zou et al., ECCV 2022 (arXiv 2207.14315)"
_SPD_PADIM = "1-class, ImageNet-supervised ResNet, повний офіційний test, колонка «Supervised»"
_CCAD = "CCAD (arXiv 2512.21459), Table 8, група «Single class based»"
_CCAD_PROTOCOL = "1-class, перезапуск авторами CCAD (WRN50), повний офіційний test"

#: Довідкові числа. Перевірено за HTML-версіями статей 2026-11-17. Середнє Image AU-ROC
#: «Supervised» у Table 7 статті SPD надруковано з опискою («7.80») — не використовується.
PUBLISHED_RESULTS: tuple[PublishedResult, ...] = (
    PublishedResult("PaDiM", "macaroni2", "image_auroc", 0.698, f"{_SPD}, Table 7", _SPD_PADIM),
    PublishedResult("PaDiM", "capsules", "image_auroc", 0.658, f"{_SPD}, Table 7", _SPD_PADIM),
    PublishedResult("PaDiM", "pcb4", "image_auroc", 0.985, f"{_SPD}, Table 7", _SPD_PADIM),
    PublishedResult("PaDiM", "macaroni2", "pixel_auroc", 0.956, f"{_SPD}, Table 8", _SPD_PADIM),
    PublishedResult("PaDiM", "capsules", "pixel_auroc", 0.838, f"{_SPD}, Table 8", _SPD_PADIM),
    PublishedResult("PaDiM", "pcb4", "pixel_auroc", 0.884, f"{_SPD}, Table 8", _SPD_PADIM),
    PublishedResult("PaDiM", "macaroni2", "pixel_ap", 0.008, f"{_SPD}, Table 8", _SPD_PADIM),
    PublishedResult("PaDiM", "capsules", "pixel_ap", 0.028, f"{_SPD}, Table 8", _SPD_PADIM),
    PublishedResult("PaDiM", "pcb4", "pixel_ap", 0.120, f"{_SPD}, Table 8", _SPD_PADIM),
    PublishedResult(
        "PatchCore",
        "mean",
        "image_auroc",
        0.924,
        f"{_SPD}, Table 6",
        "1-class, WRN50, середнє VisA",
    ),
    PublishedResult(
        "PatchCore",
        "mean",
        "pixel_auroc",
        0.984,
        f"{_SPD}, Table 6",
        "1-class, WRN50, середнє VisA",
    ),
    PublishedResult(
        "PatchCore", "mean", "pixel_ap", 0.384, f"{_SPD}, Table 6", "1-class, WRN50, середнє VisA"
    ),
    PublishedResult("PatchCore", "macaroni2", "image_auroc", 0.472, _CCAD, _CCAD_PROTOCOL),
    PublishedResult("PatchCore", "capsules", "image_auroc", 0.688, _CCAD, _CCAD_PROTOCOL),
    PublishedResult("PatchCore", "pcb4", "image_auroc", 0.697, _CCAD, _CCAD_PROTOCOL),
    PublishedResult("PatchCore", "macaroni2", "pixel_auroc", 0.936, _CCAD, _CCAD_PROTOCOL),
    PublishedResult("PatchCore", "capsules", "pixel_auroc", 0.974, _CCAD, _CCAD_PROTOCOL),
    PublishedResult("PatchCore", "pcb4", "pixel_auroc", 0.934, _CCAD, _CCAD_PROTOCOL),
)


@dataclass(frozen=True)
class Lab4Config:
    """Незмінні параметри експерименту.

    Attributes:
        visa_url: Офіційне посилання на архів VisA (S3, без облікових даних).
        visa_sha256: Контрольна сума архіву.
        visa_archive: Ім'я файлу архіву в ``data/raw/visa``.
        visa_license: Ліцензія (файл ``LICENSE-DATASET`` в архіві — CC BY 4.0).
        categories: Категорії роботи: неоднорідне тло, кілька об'єктів, складна структура.
        study_category: Категорія для досліджень гіперпараметрів (найважча за літературою).
        background_categories: Усі 12 категорій — для виміру дисперсії тла.
        background_sample: Нормальних зображень на категорію для виміру тла.
        brightness_cells: Сітка для σ яскравості (16×16 блоків по 16 пікс.).
        train_fraction: Частка офіційного train на навчання (решта — val_normal).
        val_fraction_of_test: Частка офіційного test у розмічений val.
        image_size: Сторона зображення для backbone (256 → сітка 32×32 з кроком 8).
        eval_size: Сторона карт і масок для метрик (однакова для всіх варіантів).
        blur_sigma: σ гаусового згладжування карти (як у PaDiM/PatchCore).
        feature_dim: Розмірність ознак ``D`` після зведення каналів.
        backbones: Екстрактори в порядку «від дешевшого»: при рівній якості перемагає
            перший (правило простоти).
        patch_sizes: Розміри агрегації ``p`` (поле зору ≈ 8·p пікс. понад рецептивне поле).
        default_patch_size: ``p`` базової конфігурації.
        study_tile_grid: ``g`` для варіанта з тайлами (2 → роздільність 512); 1 — вимкнено.
        extraction_batch_size: Пакет для проходу backbone.
        fae: Базові гіперпараметри FAE (D6 плану) з двома змінами після рев'ю: 60 епох
            по 25 тис. векторів замість 30 по 50 тис. За першого варіанта раннє зупинення
            не спрацювало в жодному з 13 навчань (найкраща епоха = 30 = максимум). Тепер
            епоха вдвічі коротша (стільки ж векторів за той самий час), а зупинка за
            ``val_mse`` (patience 5) має шанс спрацювати до максимуму.
        latent_grid: Альтернативні ширини латенту (k = 64 — база).
        dropout_grid: Варіанти Dropout.
        l2_grid: Варіанти L2.
        noise_grid: Варіанти шуму входу (σ у нормалізованих одиницях).
        selection_tolerance: Різниця Pixel AUROC, меншу за яку вважаємо нічиєю (обирається
            простіший варіант): ≈ півширини ДІ на val не менша, тож дрібніші різниці — шум.
        padim_channels: ``d`` каналів PaDiM (як у статті для ResNet18: 100).
        padim_epsilon: Регуляризація коваріації.
        patchcore_coreset_ratio: Частка банку (1 % — стандарт статті для MVTec).
        patchcore_projection_dim: Розмірність проєкції в coreset.
        patchcore_max_train_vectors: Межа векторів перед coreset (пам'ять 16 ГБ).
        scoring_batch_size: Пакет зображень при оцінюванні головками.
        bootstrap_repeats: Повторів бутстрепу.
        confidence_level: Рівень довіри ДІ.
        histogram_bins: Бінів для гістограмних метрик.
        latency_warmup: Прогрівальних запусків.
        latency_runs: Вимірювань для пакета 1.
        latency_runs_batch: Вимірювань для пакета 16 (кожен запуск у 16 разів довший).
        latency_batch: Розмір «пропускного» пакета.
        intra_op_threads: Потоки TF усередині операції в головному процесі (None — усі ядра).
        inter_op_threads: Паралельних операцій графа (2 — менше конкуренції за ядра).
        decode_latency_images: Скільки сирих JPG декодувати для виміру часу підготовки.
        published_results: Довідкові числа з літератури.
    """

    visa_url: str = VISA_URL
    visa_sha256: str = VISA_SHA256
    visa_archive: str = VISA_ARCHIVE
    visa_license: str = "CC BY 4.0 (LICENSE-DATASET в архіві; Anomalib вказує CC BY-NC-SA 4.0)"
    categories: tuple[str, ...] = ("macaroni2", "capsules", "pcb4")
    study_category: str = "macaroni2"
    background_categories: tuple[str, ...] = (
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
    background_sample: int = 50
    brightness_cells: int = 16
    train_fraction: float = 0.85
    val_fraction_of_test: float = 0.5
    image_size: int = 256
    eval_size: int = 256
    blur_sigma: float = 4.0
    feature_dim: int = 512
    backbones: tuple[BackboneKind, ...] = (
        BackboneKind.MOBILENET_V2,
        BackboneKind.EFFICIENTNET_B0,
        BackboneKind.RESNET50,
    )
    patch_sizes: tuple[int, ...] = (1, 3, 5)
    default_patch_size: int = 3
    study_tile_grid: int = 2
    extraction_batch_size: int = 16
    fae: FaeSettings = field(
        default_factory=lambda: FaeSettings(vectors_per_epoch=25_000, max_epochs=60)
    )
    latent_grid: tuple[int, ...] = (16, 256)
    dropout_grid: tuple[float, ...] = (0.2,)
    l2_grid: tuple[float, ...] = (1e-4,)
    noise_grid: tuple[float, ...] = (0.2,)
    selection_tolerance: float = 0.002
    padim_channels: int = 100
    padim_epsilon: float = 0.01
    patchcore_coreset_ratio: float = 0.01
    patchcore_projection_dim: int = 128
    patchcore_max_train_vectors: int = 800_000
    scoring_batch_size: int = 8
    bootstrap_repeats: int = 1000
    confidence_level: float = 0.95
    histogram_bins: int = 4096
    latency_warmup: int = 5
    latency_runs: int = 30
    latency_runs_batch: int = 10
    latency_batch: int = 16
    intra_op_threads: int | None = None
    inter_op_threads: int | None = 2
    decode_latency_images: int = 20
    published_results: tuple[PublishedResult, ...] = PUBLISHED_RESULTS

    def workspace_settings(self) -> WorkspaceSettings:
        """Параметри даних для спільного :class:`VisaWorkspace`."""
        return WorkspaceSettings(
            categories=self.categories,
            background_categories=self.background_categories,
            background_sample=self.background_sample,
            train_fraction=self.train_fraction,
            val_fraction_of_test=self.val_fraction_of_test,
            image_size=self.image_size,
            eval_size=self.eval_size,
            feature_dim=self.feature_dim,
            extraction_batch_size=self.extraction_batch_size,
        )
