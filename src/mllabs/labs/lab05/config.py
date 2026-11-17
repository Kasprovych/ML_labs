"""Параметри лабораторної роботи №5 (пояснюваність і межі надійності детектора ЛР4).

Усі числа експерименту оголошено тут, а не розкидано по коду. Аналог у Java — ``record``
із налаштуваннями (``@ConfigurationProperties``). Жоден параметр не підбирався на test:
усе, що «обирається», обирає код на розміченому val (база IG, шар Grad-CAM, пороги TFLite),
а допуски «меж придатності» задано апріорі.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BaselineKind(StrEnum):
    """Базові лінії (baseline) Integrated Gradients.

    ``StrEnum`` — перелік, значення якого є рядками (зручно писати в CSV і JSON).
    """

    BLUR = "blur"
    NEAREST_NORMAL = "nearest_normal"
    EXPECTED = "expected"
    MEAN = "mean"
    BLACK = "black"


class TfliteVariant(StrEnum):
    """Варіанти моделі TensorFlow Lite для edge-пристрою."""

    FP32 = "fp32"
    FP16 = "fp16"
    DYNAMIC_INT8 = "dynamic_int8"
    FULL_INT8 = "full_int8"


@dataclass(frozen=True)
class DriftLevel:
    """Одна трансформація дрейфу: родина і сила.

    Attributes:
        family: Ключ родини (``brightness_down``, ``noise`` …).
        value: Параметр (множник, амплітуда, σ, якість JPEG, зсув у пікселях).
        severity: Порядковий номер рівня в родині (1 — найслабший).
    """

    family: str
    value: float
    severity: int


def _levels(family: str, values: tuple[float, ...]) -> tuple[DriftLevel, ...]:
    """Рівні однієї родини в порядку зростання сили."""
    return tuple(DriftLevel(family, value, index + 1) for index, value in enumerate(values))


#: Контрольований дрейф (D8 плану). Яскравість розділено на дві родини (темніше / світліше):
#: «межа рівня» має сенс лише вздовж одного напряму зміни.
DRIFT_LEVELS: tuple[DriftLevel, ...] = (
    *_levels("brightness_down", (0.85, 0.7)),
    *_levels("brightness_up", (1.15, 1.3)),
    *_levels("illumination", (0.1, 0.2, 0.4)),
    *_levels("noise", (3.0, 6.0, 12.0)),
    *_levels("blur", (0.5, 1.0, 2.0)),
    *_levels("jpeg", (70.0, 40.0, 20.0)),
    *_levels("shift", (4.0, 8.0, 16.0)),
)


@dataclass(frozen=True)
class Lab5Config:
    """Незмінні параметри експерименту.

    Attributes:
        categories: Категорії VisA, для яких ЛР4 зберегла детектор.
        lab4_number: Номер лабораторної, артефакти якої аналізуються.
        prediction_method: Назва методу FAE у ``test_predictions.csv`` ЛР4.
        score_tolerance: Допустима розбіжність перерахованих оцінок test з оцінками ЛР4
            (інше — ознака, що модель і таблиці ЛР4 не відповідають одна одній).
        handoff_end_to_end_images: Скільки зображень test перевіряти наскрізною моделлю
            (з кешем ознак ЛР4 порівнюється float16-ознака — різниця очікувано ~1e-3).
        intra_op_threads: Потоки TF усередині операції (None — усі ядра).
        inter_op_threads: Паралельних операцій графа (2 — менше конкуренції за ядра).
        inference_batch: Пакет наскрізного проходу детектора (perturbation, дрейф).
        head_batch: Пакет головки FAE на кешованих ознаках.
        histogram_bins: Бінів для швидких піксельних AUROC/AP (похибка ~1e-4).
        train_vectors: Векторів train для PCA, латентних статистик, Махаланобіса.
        defect_patch_coverage: Частка пікселів маски в патчі, з якої патч «дефектний».
        top_k_channels: Кількості каналів для кривої «Pixel AUROC від top-k».
        pca_components: k лінійного автоенкодера (= ширина латенту FAE).
        jacobian_patches: Патчів кожної групи (норма / дефект) для якобіана.
        jacobian_batch: Пакет обчислення якобіанів.
        singular_threshold: σ_i якобіана, вище якого напрям вважається «збереженим».
        ig_anomalies_per_category: Аномалій test для IG (стратифіковано за площею).
        area_strata: Кількість страт площі дефекту (терцилі).
        max_fp_per_category: FP на категорію для панелей.
        perturbation_fp_per_category: Найвпевненіших FP на категорію для IG і perturbation.
        val_images_for_selection: Аномалій val на категорію для вибору бази IG.
        ig_steps: Початкова кількість кроків інтегрування m.
        ig_max_steps: Межа ескалації m (64 → 128 → 256) — однакова на val і test.
        ig_alpha_batch: Точок шляху в одному пакеті градієнта (m + 1 = 65 → 2 пакети).
        completeness_tolerance: Допустима похибка аксіоми повноти у частках порога зображення
            ЛР4: ``|ΣIG − (F(x) − F(b))| ≤ tol · threshold_image``. Шкала однакова для всіх
            баз: відносна похибка ``/|Δ|`` (як у плані) упереджено пропускала ненейтральні бази
            з великим |Δ| (рішення рев'ю, див. PLAN.md).
        min_completeness_pass_share: Мінімальна частка зображень з ε ≤ допуску, щоб базу
            можна було обрати.
        ig_baselines: Бази IG у порядку пріоритету при нічиїй (простіша — першою).
        baseline_blur_sigma: σ розмиття для бази «розмите зображення», пікс.
        reference_candidates: Скільки найближчих train-норм розглядати для бази.
        expected_gradients_k: Кількість баз Expected Gradients.
        selection_tolerance: Різниця AUROC, меншу за яку вважаємо нічиєю.
        attribution_blur_sigma: σ згладжування |IG| (= σ карти моделі — однакова роздільність).
        gradcam_layers: Порядок пріоритету шару Grad-CAM при нічиїй.
        pointing_tolerance_px: Розширення маски для pointing game (1 патч = 8 пікс.).
        correlation_grid: Сторона сітки, до якої зменшуються карти для Spearman ρ / SSIM.
        bootstrap_repeats: Повторів бутстрепу.
        confidence_level: Рівень довіри ДІ.
        perturbation_steps: Скільки патчів вилучено на кожному стані (лог-шкала).
        random_orders: Випадкових порядків патчів (нижня межа якості пояснення).
        noise_sigma: σ шуму режиму «зашумлення» (одиниці яскравості 0–255).
        sanity_images_per_category: Аномалій на категорію для рандомізації та стійкості.
        sanity_ig_steps: m для IG у перевірках (повнота тут не мета — порівнюються карти).
        stability_noise_sigma: σ малого шуму для перевірки стійкості пояснень.
        descriptor_window: Сторона вікна навколо argmax карти, пікс.
        saturation_level: Яскравість, з якої піксель «пересвічений».
        shadow_level: Яскравість, нижче якої піксель «тінь».
        border_px: Ширина смуги «край кадру» для правил причин, пікс.
        border_exclusion_px: Ширини виключеної смуги в діагностиці рамки.
        brightness_cells: Сітка σ яскравості (як у ЛР4).
        contrast_ring_px: Ширина кільця навколо маски для контрасту дефекту.
        glare_share: Частка пересвічених пікселів у вікні, з якої причина — «відблиск».
        edge_quantile: Квантиль сили країв у вікні серед норм val, вище якого — «край об'єкта».
        global_error_quantile: Квантиль медіанної помилки норм val для «глобальної» причини.
        small_area_quantile: Квантиль площ аномалій test для «дрібного дефекту».
        low_contrast_quantile: Квантиль контрасту для «низького контрасту».
        off_map_auroc: Піксельний AUROC зображення, з якого FN вважається «дефект поза max».
        fp_cause_order: Порядок перевірки правил FP: від найконкретнішої (положення максимуму
            біля краю кадру — відомий крайовий артефакт ЛР4) до найзагальнішої («уся сцена
            аномальніша за норму»). Змінено ПІСЛЯ першого прогону (FP з максимумом у куті
            кадру й підвищеною медіаною зараховувався до «глобальних») — тому поруч
            рахується таблиця чутливості з порядком плану (``fp_cause_order_plan``).
        fp_cause_order_plan: Порядок правил FP з PLAN.md (для таблиці чутливості).
        significance_level: Рівень значущості для переліку значущих дескрипторів (p_Holm).
        oracle_jitter: Масштаб випадкового зсуву, що розбиває нічиї в порядку «маски-оракула».
        mahalanobis_batch: Векторів у пакеті кодування латенту.
        mean_image_batch: Зображень у пакеті підрахунку середнього train.
        xai_panel_images: Аномалій на рисунку пояснень категорії.
        perturbation_example_steps: Стани рисунка-прикладу perturbation.
        error_panel_images: Помилок кожного типу на категорію на панелях FP/FN.
        error_rate_unreliable: FPR або FNR на test, вище якого рішення зображення «ненадійне».
        error_rate_supervised: FPR або FNR, вище якого потрібен контроль людиною.
        fn_cause_order: Порядок перевірки правил FN.
        min_group_size: Мінімальний розмір групи для статистичного тесту.
        drift_levels: Рівні контрольованого дрейфу.
        drift_auroc_tolerance: Допустиме падіння Pixel AUROC під дрейфом.
        drift_fpr_tolerance: Допустиме зростання FPR при порозі ЛР4.
        tflite_variants: Варіанти TFLite.
        calibration_images: Train-норм для калібрування повного int8.
        latency_warmup: Прогрівальних запусків.
        latency_runs: Вимірювань.
        latency_threads: Потоки інтерпретатора (None — усі ядра).
        edge_input_sizes: Сторони входу backbone для дослідження роздільності.
        frame_budgets_ms: Бюджети кадру: 30 FPS і 10 FPS.
        quantization_auroc_tolerance: Допустиме падіння Pixel AUROC від квантування.
    """

    categories: tuple[str, ...] = ("macaroni2", "capsules", "pcb4")
    lab4_number: int = 4
    prediction_method: str = "FAE"
    score_tolerance: float = 1e-3
    handoff_end_to_end_images: int = 8
    intra_op_threads: int | None = None
    inter_op_threads: int | None = 2
    inference_batch: int = 32
    head_batch: int = 32
    histogram_bins: int = 4096
    train_vectors: int = 100_000
    defect_patch_coverage: float = 0.25
    top_k_channels: tuple[int, ...] = (8, 16, 32, 64, 128, 512)
    pca_components: int = 64
    jacobian_patches: int = 256
    jacobian_batch: int = 32
    singular_threshold: float = 0.5
    ig_anomalies_per_category: int = 15
    area_strata: int = 3
    max_fp_per_category: int = 6
    perturbation_fp_per_category: int = 5
    val_images_for_selection: int = 5
    ig_steps: int = 64
    ig_max_steps: int = 256
    ig_alpha_batch: int = 33
    completeness_tolerance: float = 0.05
    min_completeness_pass_share: float = 0.8
    ig_baselines: tuple[BaselineKind, ...] = (
        BaselineKind.BLUR,
        BaselineKind.NEAREST_NORMAL,
        BaselineKind.EXPECTED,
        BaselineKind.MEAN,
        BaselineKind.BLACK,
    )
    baseline_blur_sigma: float = 10.0
    reference_candidates: int = 5
    expected_gradients_k: int = 4
    selection_tolerance: float = 0.01
    attribution_blur_sigma: float = 4.0
    gradcam_layers: tuple[str, ...] = ("fused", "shallow", "deep")
    pointing_tolerance_px: int = 8
    correlation_grid: int = 64
    bootstrap_repeats: int = 1000
    confidence_level: float = 0.95
    perturbation_steps: tuple[int, ...] = (0, 1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128)
    random_orders: int = 5
    noise_sigma: float = 32.0
    sanity_images_per_category: int = 5
    sanity_ig_steps: int = 32
    stability_noise_sigma: float = 2.0
    descriptor_window: int = 24
    saturation_level: int = 250
    shadow_level: int = 5
    border_px: int = 16
    border_exclusion_px: tuple[int, ...] = (0, 8, 16, 24)
    brightness_cells: int = 16
    contrast_ring_px: int = 8
    glare_share: float = 0.05
    edge_quantile: float = 0.9
    global_error_quantile: float = 0.9
    small_area_quantile: float = 0.25
    low_contrast_quantile: float = 0.25
    off_map_auroc: float = 0.9
    fp_cause_order: tuple[str, ...] = ("frame_border", "glare", "object_edge", "global")
    fp_cause_order_plan: tuple[str, ...] = ("global", "glare", "object_edge", "frame_border")
    significance_level: float = 0.05
    oracle_jitter: float = 1e-6
    mahalanobis_batch: int = 8192
    mean_image_batch: int = 64
    xai_panel_images: int = 4
    perturbation_example_steps: tuple[int, ...] = (0, 8, 32, 64, 128)
    error_panel_images: int = 2
    error_rate_unreliable: float = 0.5
    error_rate_supervised: float = 0.1
    fn_cause_order: tuple[str, ...] = ("small", "low_contrast", "off_map")
    min_group_size: int = 5
    drift_levels: tuple[DriftLevel, ...] = DRIFT_LEVELS
    drift_auroc_tolerance: float = 0.02
    drift_fpr_tolerance: float = 0.10
    tflite_variants: tuple[TfliteVariant, ...] = (
        TfliteVariant.FP32,
        TfliteVariant.FP16,
        TfliteVariant.DYNAMIC_INT8,
        TfliteVariant.FULL_INT8,
    )
    calibration_images: int = 64
    latency_warmup: int = 5
    latency_runs: int = 30
    latency_threads: tuple[int | None, ...] = (1, None)
    edge_input_sizes: tuple[int, ...] = (256, 224, 192)
    frame_budgets_ms: tuple[float, ...] = (33.0, 100.0)
    quantization_auroc_tolerance: float = 0.01
