"""Параметри лабораторної роботи №1.

Усі «магічні числа» (розмір тестової вибірки, пороги, гіперпараметри)
оголошуються тут, а не розкидаються по коду. Аналог у Java — клас
``@ConfigurationProperties`` або ``record`` із налаштуваннями.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mllabs.preprocessing import PreprocessingSettings


@dataclass(frozen=True)
class Lab1Config:
    """Незмінні параметри експерименту.

    Attributes:
        openml_data_id: Ідентифікатор датасету ``kick`` (Carvana «Don't Get
            Kicked!») на OpenML, версія 1.
        test_size: Частка даних, відкладена для тестування: 20 % — це ~14,6 тис.
            рядків і ~1,8 тис. позитивних, достатньо для стабільних метрик у ЛР2–3.
        max_missing_share: Колонки з більшою часткою пропусків видаляються:
            заповнювати понад половину значень вигаданими — безглуздо.
        high_cardinality_threshold: Від цієї кількості категорій ознака вважається
            високопотужною (One-Hot дав би десятки майже порожніх колонок).
            Пайплайн бере перелік таких ознак зі схеми (``categorical_high``);
            ЛР1 перевіряє на train, що він збігається з порогом
            (``EncodingBenchmark.check_schema``), і зупиняється з помилкою, якщо ні.
        rare_category_size: Категорія з меншою кількістю рядків у train вважається
            рідкісною (для таблиці кардинальності).
        correlation_threshold: Поріг |r| «сильної» кореляції для таблиці пар.
        vif_threshold: VIF вище 10 — загальноприйнята ознака мультиколінеарності.
        extra_collinear_candidates: Ознаки, вилучення яких додатково до переліку
            схеми лише перевіряється: роздрібні ціни MMR із VIF > 10. У FULL вони
            лишаються; ЛР1 вимірює альтернативу «вилучити ще», і підсумок
            показує висновок єдиного правила вибору.
        normality_alpha: Рівень значущості статистичних тестів.
        shapiro_sample_size: Розмір підвибірки для тесту Шапіро-Уїлка: у scipy
            його p-значення ненадійне для N > 5000. Стільки ж точок беруть Q-Q графіки.
        iqr_factor: Множник міжквартильного розмаху; 1.5 — класичне правило Тьюкі.
        zscore_threshold: Поріг |z|; 3 — «правило трьох сигм».
        outlier_contamination: Очікувана частка викидів — однакова для Isolation
            Forest і LOF, щоб порівняння було чесним (з ``"auto"`` вони позначають
            дуже різну кількість рядків).
        isolation_forest_estimators: Кількість дерев Isolation Forest; 200 дає
            стабільніші оцінки, ніж стандартні 100, а час лишається малим.
        lof_neighbors: Кількість сусідів LOF (стандартне значення sklearn).
        imputation_sample_size: Розмір підвибірки для експерименту з маскуванням:
            ``KNNImputer`` квадратичний за кількістю рядків.
        imputation_mask_share: Частка значень, які штучно ховаються в експерименті.
        preprocessing: Числові параметри цеглин препроцесингу (сусіди KNN,
            ітерації MICE, поріг рідкісних категорій One-Hot, поліном). Поля НЕ
            дублюються тут: єдине джерело — ``PreprocessingSettings`` спільного
            пакета, з тими самими значеннями за замовчуванням, з якими
            ЛР2–3 будуватимуть сценарій ``FULL``. Обґрунтування значень і
            пояснення щодо ``ConvergenceWarning`` MICE — у docstring
            ``PreprocessingSettings``.
        proxy_cv_folds: Кількість фолдів крос-валідації проксі-моделі. Правило вибору
            порівнює варіанти за парними різницями PR-AUC по фолдах і їхнім
            вибірковим σ (``ddof=1``); на трьох фолдах таке σ надто ненадійне,
            п'ять — компроміс між надійністю і часом (10 — удвічі довше).
        proxy_max_iter: Ліміт ітерацій логістичної регресії (для збіжності).
        smote_neighbors: Кількість сусідів SMOTE/ADASYN (стандартне значення).
        sampling_strategy: Після аугментації рідкісний клас становить 50 % від
            мажоритарного. Повне вирівнювання 1:1 створило б ~45 тис. синтетичних
            рядків і підвищило б ризик перенавчання.
        noise_scale: Стандартне відхилення гаусового шуму — 5 % від σ ознаки.
        mutual_information_sample_size: Розмір підвибірки для оцінки взаємної
            інформації (оцінювач на найближчих сусідах повільний на всьому train).
        pca_plot_sample_size: Кількість точок на діаграмах розсіювання — більше
            зливається в суцільну пляму.
        qq_plot_features: Скільки ознак показати на Q-Q графіках (найдальших від
            нормальності за |асиметрією|): рисунок половинної ширини, у звіті
            стоїть на ~55 % сторінки; із 13 панелями підписи стали б ~6 pt.
        distribution_plot_features: Скільки найбільш скошених ознак показувати
            на графіку «до/після» масштабування (3 рядки × 3 методи — більше на
            рисунку половинної ширини не читається).
    """

    openml_data_id: int = 41162
    test_size: float = 0.2
    max_missing_share: float = 0.5
    high_cardinality_threshold: int = 30
    rare_category_size: int = 10
    correlation_threshold: float = 0.9
    vif_threshold: float = 10.0
    extra_collinear_candidates: tuple[str, ...] = (
        "MMRAcquisitionRetailAveragePrice",
        "MMRCurrentRetailAveragePrice",
    )
    normality_alpha: float = 0.05
    shapiro_sample_size: int = 5000
    iqr_factor: float = 1.5
    zscore_threshold: float = 3.0
    outlier_contamination: float = 0.03
    isolation_forest_estimators: int = 200
    lof_neighbors: int = 20
    imputation_sample_size: int = 10_000
    imputation_mask_share: float = 0.1
    proxy_cv_folds: int = 5
    proxy_max_iter: int = 500
    smote_neighbors: int = 5
    sampling_strategy: float = 0.5
    noise_scale: float = 0.05
    mutual_information_sample_size: int = 20_000
    pca_plot_sample_size: int = 3000
    distribution_plot_features: int = 3
    qq_plot_features: int = 6
    # default_factory: кожен екземпляр отримує власний об'єкт налаштувань
    # (значення за замовчуванням для полів-об'єктів задається фабрикою).
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
