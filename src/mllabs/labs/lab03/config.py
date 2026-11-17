"""Параметри лабораторної роботи №3.

Усі «магічні числа» (обсяг вибірки, кількість фолдів, ітерацій, потоків)
оголошуються тут, а не розкидаються по коду. Аналог у Java — клас
``@ConfigurationProperties`` або ``record`` із налаштуваннями.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mllabs.labs.lab03.search_space import SearchSpace, hgb_search_space
from mllabs.modeling import ModelKind, ModelSettings
from mllabs.preprocessing import PreprocessingSettings


@dataclass(frozen=True)
class Lab3Config:
    """Незмінні параметри експерименту.

    Attributes:
        dataset_name: Каталог підготовленого в ЛР1 датасету в ``data/processed/``.
        model: Модель, гіперпараметри якої підбираються, — HistGradientBoosting
            (D1): найкраща модель ЛР2, має помітний розрив train − CV і багато
            реальних «важелів», а одне навчання вдвічі дешевше, ніж у лісу.
        model_settings: Параметри ЛР2 — вони ж baseline (``max_iter=200``,
            ``learning_rate=0.1``, раннє зупинення вимкнено). ``max_iter`` і
            ``early_stopping=False`` не змінюються й у пошуку: кількість дерев
            явна й однакова, час однієї оцінки передбачуваний.
        preprocessing: Параметри препроцесингу ЛР1 (звіряються з ``preprocessing.json``).
        scoring: Метрика оптимізації в усіх методах — PR-AUC
            (``average_precision``, D2): головна метрика ЛР1–2 при 12 % позитивних.
        dev_sample_size: Розмір «робочої» (dev) стратифікованої підвибірки train,
            на якій виконуються пошук, вкладена CV, baseline і фінальні refit (D4).
            Калібрування за правилом D4 («загальний час ≈ 14 × час самостійного
            Grid Search + ~3 хв»; > 25 хв → 15 000, < 12 хв → 30 000).
            Калібрувальний запуск 2026-11-16 (M1 Pro, 10 ядер; на початку
            запуску машину додатково вантажили сторонні процеси, load до ~140):
            самостійний Grid Search — 87,9 с, отже прогноз за правилом
            14 × 87,9 + 180 ≈ 1 411 с ≈ 23,5 хв; фактично весь запуск тривав
            1 149 с ≈ 19,1 хв (пошук 210 с, витік 62 с, вкладена CV 838 с,
            test 24 с). Обидва числа в межах 12–25 хв — розмір 20 000 лишається.
            Контрольний запуск з нуля 2026-11-17 на вільній машині (load ≈ 5):
            самостійний пошук 195 с (Grid 83 с), прогноз «вкладена CV ≈ 5 × пошук»
            ≈ 16 хв, разом ≈ 21 хв; фактично 1 115 с ≈ 18,6 хв (дані 9 с, пошук
            195 с, витік 57 с, вкладена CV 828 с, test 24 с) — 20 000 підтверджено.
            Фінальний запуск після рев'ю (2026-11-17, додано inner-оцінку baseline):
            ≈ 1 163 с ≈ 19,4 хв; точні часи кроків — у ``metrics.json``.
            Колишні запуски понад 1 год спричиняло переповнення ядер (процеси
            joblib × потоки OpenMP поверх сторонніх важких процесів); тепер потоки
            обмежено явно полями ``*_threads_per_worker`` і ``main_process_threads``.
            На 20 тис. у зовнішньому фолді ~490 позитивних, σ PR-AUC фолда ≈ 0,025.
        inner_folds: Внутрішні фолди (підбір параметрів) — за завданням K = 3.
        outer_folds: Зовнішні фолди вкладеної CV — за завданням K = 5.
        search_space: Простір гіперпараметрів (D3): 4 параметри, сітка 3⁴ = 81.
        random_iterations: Кількості ітерацій Random Search — за завданням N = {20, 50}.
        optuna_trials: Кількість спроб Optuna — за завданням N = 20.
        optuna_startup_trials: Скільки перших спроб TPE робить випадково (D6):
            5 із 20 — решта 15 керовані моделлю; типові 10 перетворили б
            «Bayesian» на половину випадкового пошуку.
        optuna_multivariate: TPE моделює параметри спільно (важлива пара
            ``learning_rate`` × ``max_leaf_nodes``).
        search_n_jobs: Процеси Grid/Random Search (−1 — усі ядра): кандидати ×
            фолди незалежні й рівномірно розкладаються на процеси.
        search_threads_per_worker: Потоки OpenMP у кожному процесі Grid/Random.
            Паралельним має бути ЛИШЕ один рівень: 10 процесів × 1 потік = 10 ядер.
        optuna_fold_n_jobs: Скільки фолдів однієї спроби Optuna рахується
            паралельно (= ``inner_folds``). Самі спроби — строго послідовно:
            TPE пропонує наступну точку за результатами попередніх, а з
            паралельними спробами історія стала б невідтворюваною.
        optuna_threads_per_worker: Потоки OpenMP у кожному з 3 процесів Optuna:
            3 × 3 = 9 ядер — режим, близький за завантаженням до Grid/Random.
        main_process_threads: Ліміт потоків OpenMP для навчань у головному
            процесі (refit у вкладеній CV і фінальні моделі). Заміряно
            2026-11-16 (одне навчання на 13,3 тис. рядків): 1 потік — 2,0 с,
            3 — 1,26 с, 8 — 1,30 с; більше 3–4 потоків бустинг не прискорюють,
            а на завантаженій машині зайві потоки його гальмують у рази. 4 — запас.
        leakage_folds: Фолди дослідження витоку — 5, як у ЛР1–2.
        cv_n_jobs: Паралельні фолди звичайної 5-кратної CV (baseline на
            зовнішніх фолдах, дослідження витоку) — як у ЛР2.
        cv_threads_per_worker: Потоки OpenMP у кожному з 5 процесів CV
            (5 × 2 = 10 ядер). Baseline на 3 внутрішніх фолдах виконується в
            режимі Optuna (``optuna_fold_n_jobs`` × ``optuna_threads_per_worker``).
        leakage_sample_sizes: Менші обсяги для розрізу «витік від обсягу даних»;
            повний train додається завжди. 20 000 — та сама dev-вибірка, що в пошуку.
        sampling_strategy: Частота SMOTE у варіантах витоку зі SMOTE — як у ЛР1–2.
        smote_neighbors: Кількість сусідів SMOTE — як у ЛР1–2.
        noise_scale: Параметр конструктора ``SamplerFactory`` (гаусів шум не
            використовується) — як у ЛР1–2.
        bootstrap_repeats: Повтори парного бутстрепу на test — як у ЛР2.
        confidence_level: Рівень довіри бутстреп-інтервалів.
    """

    dataset_name: str = "kick"
    model: ModelKind = ModelKind.HIST_GRADIENT_BOOSTING
    # default_factory: кожен екземпляр отримує власний об'єкт (як new у конструкторі Java).
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    scoring: str = "average_precision"
    dev_sample_size: int = 20_000
    inner_folds: int = 3
    outer_folds: int = 5
    search_space: SearchSpace = field(default_factory=hgb_search_space)
    random_iterations: tuple[int, ...] = (20, 50)
    optuna_trials: int = 20
    optuna_startup_trials: int = 5
    optuna_multivariate: bool = True
    search_n_jobs: int = -1
    search_threads_per_worker: int = 1
    optuna_fold_n_jobs: int = 3
    optuna_threads_per_worker: int = 3
    main_process_threads: int = 4
    leakage_folds: int = 5
    cv_n_jobs: int = 5
    cv_threads_per_worker: int = 2
    leakage_sample_sizes: tuple[int, ...] = (5_000, 20_000)
    sampling_strategy: float = 0.5
    smote_neighbors: int = 5
    noise_scale: float = 0.05
    bootstrap_repeats: int = 1000
    confidence_level: float = 0.95
