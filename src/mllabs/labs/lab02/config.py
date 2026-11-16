"""Параметри лабораторної роботи №2.

Усі «магічні числа» (кількість фолдів, частоти аугментації, розміри підвибірок)
оголошуються тут, а не розкидаються по коду. Аналог у Java — клас
``@ConfigurationProperties`` або ``record`` із налаштуваннями.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mllabs.modeling import ModelKind, ModelSettings
from mllabs.preprocessing import PreprocessingSettings


@dataclass(frozen=True)
class Lab2Config:
    """Незмінні параметри експерименту.

    Attributes:
        dataset_name: Каталог підготовленого в ЛР1 датасету в ``data/processed/``.
        models: Моделі, що порівнюються: три різні сімейства (лінійна, бегінг
            і бустинг дерев). SVM свідомо не взято: навчання квадратичне за
            кількістю рядків, а ймовірності — лише з дорогим ``probability=True``.
        model_settings: Базові гіперпараметри моделей (без підбору — це ЛР3).
        preprocessing: Числові параметри препроцесингу — ті самі, що в ЛР1;
            звіряються з ``preprocessing.json`` перед запуском.
        cv_folds: Кількість фолдів крос-валідації — як у ЛР1; п'ять пар
            значень потрібні для σ парних різниць.
        cv_n_jobs: Скільки фолдів крос-валідації виконувати паралельно (для всіх
            моделей). Заміряно 2026-11-16 на M1 Pro (10 ядер) під стороннім
            навантаженням машини, сценарій Б, 5 фолдів, «послідовно → паралельно»:
            LR 23,1 → 11,0 с; RF 19,5 → 13,7 с; HGB 271,6 → 8,3 с; із SMOTE:
            LR 33,8 → 6,3 с; RF 24,7 → 16,7 с; HGB 43,9 → 9,8 с. Паралельний
            режим швидший для всіх трьох: joblib обмежує кількість потоків
            OpenMP/BLAS у кожному процесі-працівнику, тож потоки бустингу й лісу
            не змагаються за ядра (послідовний HGB з усіма потоками OpenMP на
            завантаженій машині сповільнюється в десятки разів).
            Свідомий виняток із правила «паралелити лише один рівень»: ліс має
            ``forest_n_jobs=-1``, тож у кожному з 5 процесів фолдів він ще й
            запускає власні потоки (joblib виконує вкладений ``Parallel`` у
            потоках, і ліміт ``inner_max_num_threads`` на них не діє). Заміри
            вище показують, що для RF це все одно швидше (19,5 → 13,7 с), тому
            в ЛР2 виняток залишено; ЛР3 правило дотримує (у лісу там немає ролі,
            а в бустингу потоки OpenMP обмежуються явно).
        sampling_strategy: Цільове відношення «клас 1 / клас 0» після аугментації
            — як у ЛР1 (``Lab1Config.sampling_strategy``): клас 1 стає 33 %.
        smote_neighbors: Кількість сусідів SMOTE/ADASYN — як у ЛР1.
        noise_scale: Параметр гаусового шуму ``SamplerFactory`` — як у ЛР1. У ЛР2
            гаусів шум не використовується, але фабрика семплерів вимагає
            значення, тож передаємо те саме, що й ЛР1.
        ratio_sweep: Додаткові частоти аугментації SMOTE для аналізу
            перенавчання; точки «без аугментації» і 0.5 беруться з основної сітки.
        learning_curve_sizes: Розміри навчальної частини фолда для кривих
            навчання; повний розмір (~46,7 тис.) береться з основної сітки.
            2 500 рядків — це лише ~300 позитивних прикладів.
        learning_curve_models: Моделі на кривих навчання — усі три: логістична
            регресія є контролем із низькою місткістю.
        decomposition_models: Моделі для діагностичних варіантів А_missing і
            Б_MICE (лише крос-валідація; дешеві моделі — LR і HGB).
        bootstrap_repeats: Кількість повторів парного бутстрепу на test; 1000 —
            звичайний мінімум для стабільних 2,5-го і 97,5-го перцентилів.
        confidence_level: Рівень довіри бутстреп-інтервалів.
        curve_points: До скількох точок проріджуються криві PR і ROC для графіків.
        threshold_shift_model: Модель для графіка «зсув порога»: бустинг —
            найкраща модель проби плану; вибір зроблено ДО запуску, не за результатами.
    """

    dataset_name: str = "kick"
    models: tuple[ModelKind, ...] = (
        ModelKind.LOGREG,
        ModelKind.RANDOM_FOREST,
        ModelKind.HIST_GRADIENT_BOOSTING,
    )
    # default_factory: кожен екземпляр отримує власний об'єкт налаштувань.
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    cv_folds: int = 5
    cv_n_jobs: int = 5
    sampling_strategy: float = 0.5
    smote_neighbors: int = 5
    noise_scale: float = 0.05
    ratio_sweep: tuple[float, ...] = (0.25, 0.75, 1.0)
    learning_curve_sizes: tuple[int, ...] = (2500, 5000, 10_000, 20_000)
    learning_curve_models: tuple[ModelKind, ...] = (
        ModelKind.LOGREG,
        ModelKind.RANDOM_FOREST,
        ModelKind.HIST_GRADIENT_BOOSTING,
    )
    decomposition_models: tuple[ModelKind, ...] = (
        ModelKind.LOGREG,
        ModelKind.HIST_GRADIENT_BOOSTING,
    )
    bootstrap_repeats: int = 1000
    confidence_level: float = 0.95
    curve_points: int = 200
    threshold_shift_model: ModelKind = ModelKind.HIST_GRADIENT_BOOSTING
