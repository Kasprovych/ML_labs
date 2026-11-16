"""Фабрика нефітованих пайплайнів препроцесингу.

Одна фабрика збирає всі варіанти підготовки даних з однакових «цеглин»:
ЛР1 порівнює варіанти між собою, ЛР2 бере готові сценарії, ЛР3 будує з тих
самих цеглин «коректний» і «некоректний» пайплайни.

Фабрика повертає НЕФІТОВАНИЙ об'єкт: жодна статистика ще не порахована.
Навчання (``fit``) виконує той, хто знає, які рядки — навчальні: крос-валідація
або код лабораторної. Це і є захист від витоку даних.

Фабрика прив'язана до датасету Carvana ``kick``: ролі колонок вона бере зі
схеми, але інженерні ознаки (``RowFeatureEngineer``) і агрегації
(``KICK_AGGREGATIONS``) звертаються до конкретних колонок цього датасету
(``VehBCost``, ``WheelType``, ціни MMR тощо). Для іншого датасету знадобляться
власні трансформери ознак; решта цеглин (імп'ютація, кодування, масштабування)
від назв колонок не залежить.

Числові параметри цеглин мають ОДНЕ джерело — :class:`PreprocessingSettings`.
Їхні значення за замовчуванням обґрунтовано в ЛР1, і ЛР2–3 мають створювати
фабрику з тими самими налаштуваннями (ЛР1 записує їх у
``data/processed/kick/preprocessing.json``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

# IterativeImputer у sklearn досі «експериментальний»: щоб його імпортувати,
# треба спершу явно імпортувати модуль-перемикач enable_iterative_imputer.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from mllabs.data.schema import DatasetSchema
from mllabs.preprocessing.encoders import FrequencyEncoder
from mllabs.preprocessing.features import (
    KICK_AGGREGATIONS,
    GroupAggregateEncoder,
    RowFeatureEngineer,
)

#: Ключі ``manifest()``, які мають збігатися з записаними ЛР1.
_CHECKED_MANIFEST_KEYS = ("settings", "collinear_columns", "scenarios")


class PreprocessingMismatchError(ValueError):
    """Налаштування препроцесингу не збігаються з записаними в ЛР1."""


#: Окрема категорія для пропущених значень: сам факт пропуску буває інформативним.
MISSING_CATEGORY = "MISSING"

#: Назва кроку з ``ColumnTransformer`` у пайплайні, який повертає фабрика.
COLUMNS_STEP = "columns"


class PreprocessingScenario(StrEnum):
    """Готовий сценарій препроцесингу (для ЛР2: сценарії А та Б).

    ``StrEnum`` — enum, значення якого водночас є рядками.
    """

    MINIMAL = "minimal"  # мінімальне очищення: медіана/мода + Ordinal, без масштабування
    FULL = "full"  # повний препроцесинг, обраний за результатами ЛР1


class ImputerKind(StrEnum):
    """Спосіб заповнення пропусків у числових ознаках."""

    MEDIAN = "median"
    KNN = "knn"
    MICE = "mice"


class CategoricalImputation(StrEnum):
    """Спосіб заповнення пропусків у категоріальних ознаках."""

    MODE = "mode"  # найчастіша категорія
    MISSING_CATEGORY = "missing_category"  # окрема категорія «MISSING»


class LowCardinalityEncoding(StrEnum):
    """Кодування номінальних ознак із малою кількістю категорій."""

    ONEHOT = "onehot"
    ORDINAL = "ordinal"


class HighCardinalityEncoding(StrEnum):
    """Кодування ознак високої кардинальності (десятки й сотні категорій)."""

    TARGET = "target"
    FREQUENCY = "frequency"
    ONEHOT = "onehot"
    ORDINAL = "ordinal"


class ScalerKind(StrEnum):
    """Масштабування числових ознак."""

    NONE = "none"
    ROBUST = "robust"  # RobustScaler: (x − медіана) / IQR, стійкий до викидів
    POWER = "power"  # PowerTransformer (Yeo-Johnson): прибирає скошеність + стандартизує


@dataclass(frozen=True)
class PreprocessingOptions:
    """Які «цеглини» входять у пайплайн — один варіант препроцесингу.

    Attributes:
        imputer: Імп'ютація числових ознак.
        categorical_imputation: Імп'ютація категоріальних ознак.
        low_cardinality_encoding: Кодування номінальних ознак низької кардинальності.
        high_cardinality_encoding: Кодування ознак високої кардинальності.
        scaler: Масштабування числових ознак.
        drop_collinear: Чи прибирати ознаки-дублікати зі схеми
            (``DatasetSchema.collinear_columns``).
        add_row_features: Чи додавати відношення, логарифми, частини дати та
            індикатори пропусків (:class:`RowFeatureEngineer`).
        add_group_aggregates: Чи додавати агрегації за категоріями
            (:class:`GroupAggregateEncoder`). У ``FULL`` увімкнено за єдиним
            правилом вибору (``mllabs.preprocessing.selection``): парна різниця
            PR-AUC по фолдах Δ більша за її розкид s.
        add_polynomial_features: Чи додавати квадрати та попарні добутки
            вибраних ознак (``PolynomialFeatures``).
    """

    imputer: ImputerKind = ImputerKind.MEDIAN
    # Категорія MISSING — свідомий виняток із правила вибору (mllabs.preprocessing.selection):
    # «мода + індикатор» має менше ознак і рівну якість, але тоді інформація про пропуск
    # WheelType живе лише в індикаторі з іншої групи ознак; MISSING тримає її в самій ознаці.
    categorical_imputation: CategoricalImputation = CategoricalImputation.MISSING_CATEGORY
    low_cardinality_encoding: LowCardinalityEncoding = LowCardinalityEncoding.ONEHOT
    high_cardinality_encoding: HighCardinalityEncoding = HighCardinalityEncoding.TARGET
    scaler: ScalerKind = ScalerKind.POWER
    drop_collinear: bool = True
    add_row_features: bool = True
    add_group_aggregates: bool = True
    add_polynomial_features: bool = False


#: Сценарій А: лише те, без чого модель не запуститься.
_MINIMAL_OPTIONS = PreprocessingOptions(
    imputer=ImputerKind.MEDIAN,
    categorical_imputation=CategoricalImputation.MODE,
    low_cardinality_encoding=LowCardinalityEncoding.ORDINAL,
    high_cardinality_encoding=HighCardinalityEncoding.ORDINAL,
    scaler=ScalerKind.NONE,
    drop_collinear=False,
    add_row_features=False,
    add_group_aggregates=False,
    add_polynomial_features=False,
)

#: Сценарій Б: значення за замовчуванням ``PreprocessingOptions`` — вибір ЛР1 за
#: єдиним правилом із ``mllabs.preprocessing.selection`` (обґрунтування з числами —
#: у ``tables/preprocessing_summary.csv`` ЛР1).
_FULL_OPTIONS = PreprocessingOptions()


@dataclass(frozen=True)
class PreprocessingSettings:
    """Числові параметри цеглин препроцесингу — ЄДИНЕ їх джерело для ЛР1–3.

    ``Lab1Config`` не дублює ці поля, а вкладає готовий об'єкт налаштувань;
    ЛР2–3 створюють фабрику з налаштуваннями за замовчуванням і можуть звірити
    їх із записаними ЛР1 (:meth:`from_dict` + порівняння ``==``).

    Attributes:
        knn_neighbors: Кількість сусідів ``KNNImputer`` (стандартне значення sklearn).
        mice_max_iter: Максимум ітерацій ``IterativeImputer`` (MICE). Стандартних
            10 у пайплайні на train не вистачає (``ConvergenceWarning``): заміряно,
            що MICE сходиться за ~21 ітерацію, тому 25 — із запасом і без
            помітних витрат часу (~1 с). В експерименті з маскуванням ЛР1
            (стандартизовані ознаки, 10 % штучних пропусків у сильно
            корельованих цінах) MICE не сходиться навіть за 100 ітерацій:
            оцінки коливаються біля розв'язку на рівні допуску ``tol``, тож
            попередження там лишається і на результат практично не впливає.
        onehot_min_frequency: Категорії, що трапляються рідше, One-Hot об'єднує
            в одну колонку «рідкісні» (лише для ознак високої кардинальності):
            інакше ``Model`` дає понад 1000 колонок.
        target_encoder_folds: Кількість фолдів внутрішнього крос-фітингу
            ``TargetEncoder`` (стандартне значення sklearn — 5).
        polynomial_degree: Степінь ``PolynomialFeatures``: квадрати та попарні добутки.
        polynomial_features: Ознаки для полінома — лише три, щоб не роздувати матрицю.
    """

    knn_neighbors: int = 5
    mice_max_iter: int = 25
    onehot_min_frequency: int = 50
    target_encoder_folds: int = 5
    polynomial_degree: int = 2
    polynomial_features: tuple[str, ...] = ("VehicleAge", "VehOdo", "VehBCost")

    def to_dict(self) -> dict[str, Any]:
        """Словник зі звичайних типів (для запису в JSON)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PreprocessingSettings:
        """Відновлює налаштування зі словника, прочитаного з JSON (список → кортеж)."""
        values = {
            key: tuple(value) if isinstance(value, list) else value for key, value in raw.items()
        }
        return cls(**values)


class PreprocessorFactory:
    """Будує нефітовані пайплайни препроцесингу для датасету зі схемою.

    Результат ``build`` — ``sklearn.pipeline.Pipeline`` із кроками:

    1. ``row_features`` (необов'язковий) — ознаки в межах рядка;
    2. ``group_aggregates`` (необов'язковий) — агрегації за категоріями;
    3. ``columns`` — ``ColumnTransformer``: кожна група колонок проходить власний
       ланцюжок «імп'ютація → кодування/масштабування», результати склеюються.

    Args:
        schema: Схема очищеного датасету.
        random_state: Зерно для стохастичних цеглин (MICE, Target Encoding).
        settings: Числові параметри; за замовчуванням — стандартні.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        random_state: int,
        settings: PreprocessingSettings | None = None,
    ) -> None:
        self._schema = schema
        self._random_state = random_state
        self._settings = settings or PreprocessingSettings()

    @property
    def settings(self) -> PreprocessingSettings:
        """Числові параметри, з якими фабрика будує пайплайни."""
        return self._settings

    def manifest(self) -> dict[str, Any]:
        """Фактичні налаштування препроцесингу для запису поруч із даними.

        ЛР2–3 читають цей словник і перевіряють, що будують ТОЙ САМИЙ
        препроцесинг, який обґрунтовано в ЛР1.

        Returns:
            ``{"random_state": ..., "settings": {...}, "collinear_columns": [...],
            "scenarios": {назва: опції}}``.
        """
        return {
            "random_state": self._random_state,
            "settings": self._settings.to_dict(),
            "collinear_columns": list(self._schema.collinear_columns),
            # StrEnum серіалізується в JSON як звичайний рядок.
            "scenarios": {
                scenario.value: asdict(self.options_for(scenario))
                for scenario in PreprocessingScenario
            },
        }

    def check_manifest(self, recorded: Mapping[str, Any]) -> None:
        """Звіряє налаштування цієї фабрики із записаними ЛР1 (``preprocessing.json``).

        Порівнюються числові налаштування, перелік колінеарних ознак і опції
        обох сценаріїв. ``random_state`` не порівнюється: зерно задає контекст
        лабораторної, а не рішення препроцесингу.

        Args:
            recorded: Словник, прочитаний ``ProcessedDatasetRepository.load_preprocessing()``.

        Raises:
            PreprocessingMismatchError: Якщо хоч одне налаштування відрізняється;
                у повідомленні — перелік розбіжностей.
        """
        # Прогін через JSON зводить обидві сторони до однакових типів (кортеж → список).
        actual = json.loads(json.dumps(self.manifest()))
        differences = [
            f"{key}: ЛР1 {recorded.get(key)!r} ≠ зараз {actual[key]!r}"
            for key in _CHECKED_MANIFEST_KEYS
            if recorded.get(key) != actual[key]
        ]
        if differences:
            raise PreprocessingMismatchError(
                "Препроцесинг розходиться з обґрунтованим у ЛР1: " + "; ".join(differences)
            )

    @staticmethod
    def options_for(scenario: PreprocessingScenario) -> PreprocessingOptions:
        """Повертає набір опцій готового сценарію."""
        return _MINIMAL_OPTIONS if scenario is PreprocessingScenario.MINIMAL else _FULL_OPTIONS

    def build(self, scenario: PreprocessingScenario) -> Pipeline:
        """Будує нефітований пайплайн готового сценарію."""
        return self.build_from_options(self.options_for(scenario))

    def build_from_options(self, options: PreprocessingOptions) -> Pipeline:
        """Будує нефітований пайплайн із довільного набору опцій.

        Args:
            options: Які цеглини використати.

        Returns:
            Пайплайн, що приймає очищену таблицю (``DataFrame``) і після ``fit``
            повертає числову матрицю ознак.
        """
        steps: list[tuple[str, BaseEstimator]] = []
        if options.add_row_features:
            steps.append(("row_features", RowFeatureEngineer()))
        if options.add_group_aggregates:
            steps.append(("group_aggregates", GroupAggregateEncoder(KICK_AGGREGATIONS)))
        steps.append((COLUMNS_STEP, self._column_transformer(options)))
        return Pipeline(steps)

    def numeric_columns(self, options: PreprocessingOptions) -> list[str]:
        """Числові колонки, що йдуть у «звичайну» числову гілку пайплайна.

        Ознаки, для яких будується поліном, сюди не входять: їх (разом із
        квадратами й добутками) видає окрема поліноміальна гілка.
        """
        schema = self._schema
        columns = [
            column
            for column in schema.numeric
            if not (options.drop_collinear and column in schema.collinear_columns)
        ]
        if options.add_row_features:
            columns += RowFeatureEngineer.NUMERIC_FEATURES
        if options.add_group_aggregates:
            columns += GroupAggregateEncoder(KICK_AGGREGATIONS).output_features
        if options.add_polynomial_features:
            columns = [c for c in columns if c not in self._settings.polynomial_features]
        return columns

    def indicator_columns(self, options: PreprocessingOptions) -> list[str]:
        """Індикатори пропусків 0/1, які подаються в модель без змін.

        Індикатор пропуску категоріальної ознаки НЕ подається, якщо ця ознака
        кодується One-Hot з окремою категорією ``MISSING``: тоді колонка
        ``<ознака>_MISSING`` уже дослівно дорівнює індикатору, і друга копія
        дала б точну лінійну залежність (нескінченний VIF).
        """
        if not options.add_row_features:
            return []
        return [
            indicator
            for indicator in RowFeatureEngineer.INDICATOR_FEATURES
            if not self._has_missing_onehot_column(
                RowFeatureEngineer.CATEGORY_MISSING_INDICATORS.get(indicator), options
            )
        ]

    def _has_missing_onehot_column(self, column: str | None, options: PreprocessingOptions) -> bool:
        """Чи отримає категоріальна колонка окрему One-Hot колонку ``MISSING``."""
        if (
            column is None
            or options.categorical_imputation is not CategoricalImputation.MISSING_CATEGORY
        ):
            return False
        is_low_onehot = (
            column in self._schema.categorical_low
            and options.low_cardinality_encoding is LowCardinalityEncoding.ONEHOT
        )
        is_high_onehot = (
            column in self._schema.categorical_high
            and options.high_cardinality_encoding is HighCardinalityEncoding.ONEHOT
        )
        return is_low_onehot or is_high_onehot

    def _column_transformer(self, options: PreprocessingOptions) -> ColumnTransformer:
        """Збирає ``ColumnTransformer``: список трійок (назва, трансформер, колонки)."""
        schema = self._schema
        transformers: list[tuple[str, BaseEstimator | str, list[str]]] = [
            ("numeric", self._numeric_pipeline(options), self.numeric_columns(options)),
            (
                "low_cardinality",
                self._low_cardinality_pipeline(options),
                list(schema.categorical_low),
            ),
            (
                "high_cardinality",
                self._high_cardinality_pipeline(options),
                list(schema.categorical_high),
            ),
            ("binary", self._binary_pipeline(), list(schema.binary)),
        ]
        if options.add_polynomial_features:
            transformers.append(
                (
                    "polynomial",
                    self._polynomial_pipeline(options),
                    list(self._settings.polynomial_features),
                )
            )
        if options.add_row_features:
            transformers += [
                (
                    "date_parts",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    list(RowFeatureEngineer.DATE_FEATURES),
                ),
                # "passthrough" — колонки йдуть далі без змін (індикатори вже 0/1).
                ("indicators", "passthrough", self.indicator_columns(options)),
            ]
        # remainder="drop": усе, що не згадано явно (сама дата, прибрані дублікати),
        # у модель не потрапляє.
        return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

    def _numeric_pipeline(self, options: PreprocessingOptions) -> Pipeline:
        """Числова гілка: імп'ютація → масштабування."""
        steps = self._numeric_imputer_steps(options.imputer)
        scaler = self._scaler(options.scaler)
        if scaler is not None:
            steps.append(("scaler", scaler))
        return Pipeline(steps)

    def _polynomial_pipeline(self, options: PreprocessingOptions) -> Pipeline:
        """Поліноміальна гілка: імп'ютація → масштабування → поліном → стандартизація.

        Поліном будується ПІСЛЯ масштабування: квадрат пробігу в милях мав би
        порядок 10¹⁰ і ламав би оптимізатор, а квадрат стандартизованої ознаки —
        звичайне число. Центрування також зменшує кореляцію між ``x`` та ``x²``.
        """
        steps = self._numeric_imputer_steps(options.imputer)
        scaler = self._scaler(options.scaler)
        if scaler is not None:
            steps.append(("scaler", scaler))
        steps.append(
            (
                "polynomial",
                PolynomialFeatures(degree=self._settings.polynomial_degree, include_bias=False),
            )
        )
        if scaler is not None:
            steps.append(("rescale", StandardScaler()))
        return Pipeline(steps)

    def _numeric_imputer_steps(self, kind: ImputerKind) -> list[tuple[str, BaseEstimator]]:
        """Кроки імп'ютації числових ознак.

        * **Медіана** — пропуск замінюється медіаною колонки в train.
        * **KNN** — середнім значенням ознаки у ``k`` найсхожіших рядків. Схожість —
          евклідова відстань, тому ознаки спершу стандартизуються: інакше пробіг
          (десятки тисяч) повністю заглушив би вік авто (одиниці).
        * **MICE** (``IterativeImputer``) — кожна ознака з пропусками по черзі
          передбачається регресією від решти ознак, кілька кіл поспіль.
        """
        match kind:
            case ImputerKind.MEDIAN:
                return [("imputer", SimpleImputer(strategy="median"))]
            case ImputerKind.KNN:
                return [
                    ("standardize", StandardScaler()),
                    ("imputer", KNNImputer(n_neighbors=self._settings.knn_neighbors)),
                ]
            case ImputerKind.MICE:
                return [
                    (
                        "imputer",
                        IterativeImputer(
                            max_iter=self._settings.mice_max_iter, random_state=self._random_state
                        ),
                    )
                ]

    @staticmethod
    def _scaler(kind: ScalerKind) -> BaseEstimator | None:
        match kind:
            case ScalerKind.NONE:
                return None
            case ScalerKind.ROBUST:
                return RobustScaler()
            case ScalerKind.POWER:
                return PowerTransformer(method="yeo-johnson", standardize=True)

    @staticmethod
    def _categorical_imputer(kind: CategoricalImputation) -> SimpleImputer:
        if kind is CategoricalImputation.MODE:
            return SimpleImputer(strategy="most_frequent")
        return SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)

    def _low_cardinality_pipeline(self, options: PreprocessingOptions) -> Pipeline:
        """Гілка номінальних ознак низької кардинальності.

        One-Hot тут повний — по колонці на КОЖНУ категорію, без ``drop="first"``.
        Строго кажучи, колонки однієї ознаки в сумі дають 1 («пастка фіктивних
        змінних»), тобто між групами One-Hot є лінійна залежність. Для моделей
        з регуляризацією (L2 у логістичній регресії) і для дерев ЛР2 це не
        проблема, а відкидання категорії ускладнило б інтерпретацію та обробку
        невідомих у train категорій. Тому так зроблено свідомо.
        """
        if options.low_cardinality_encoding is LowCardinalityEncoding.ONEHOT:
            # handle_unknown="ignore": категорія, якої не було в train, дає рядок нулів.
            encoder: BaseEstimator = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        return Pipeline(
            [
                ("imputer", self._categorical_imputer(options.categorical_imputation)),
                ("encoder", encoder),
            ]
        )

    def _high_cardinality_pipeline(self, options: PreprocessingOptions) -> Pipeline:
        """Гілка ознак високої кардинальності.

        * **Target** — категорія замінюється згладженою часткою позитивного класу
          в ній. ВАЖЛИВО: у ``TargetEncoder`` sklearn ``fit_transform(X, y)`` і
          ``fit(X, y).transform(X)`` дають РІЗНИЙ результат на тих самих рядках:

          - ``fit_transform`` кодує навчальні рядки з крос-фітингом: кожен рядок
            отримує статистику, пораховану на інших фолдах, БЕЗ нього самого —
            інакше рідкісні категорії «запам'ятали» б власну відповідь;
          - ``fit(...).transform(...)`` кодує статистикою, порахованою на ВСІХ
            рядках ``fit`` (так і треба для нових, тестових даних).

          ``Pipeline.fit`` викликає саме ``fit_transform`` проміжних кроків, тож
          у коректному пайплайні (fit лише на train) train кодується чесно, а
          test — статистикою train. «Некоректний» пайплайн ЛР3 протікає саме
          тому, що енкодер навчається (``fit``) на ВСІХ даних, включно з тестом:
          тоді ``transform`` тестових рядків використовує їхні ж мітки. Увага:
          якщо «некоректний» пайплайн зробить ``fit_transform`` на ВСІХ даних,
          тестові рядки теж отримають крос-фітинг, і витік від ``TargetEncoder``
          майже зникне. Для демонстрації витоку потрібно саме ``fit(усі дані)``,
          а потім окремо ``transform(test)``.
        * **Frequency** — категорія замінюється своєю часткою в train.
        * **One-Hot** — по колонці на категорію; рідкісні об'єднуються.
        * **Ordinal** — довільний номер категорії (базовий варіант: порядок
          номерів змісту не має, для лінійної моделі це шум).
        """
        steps: list[tuple[str, BaseEstimator]] = [
            ("imputer", self._categorical_imputer(options.categorical_imputation))
        ]
        match options.high_cardinality_encoding:
            case HighCardinalityEncoding.TARGET:
                # cv задає внутрішній крос-фітинг: фолди перемішуються з фіксованим зерном.
                cross_fitting = StratifiedKFold(
                    n_splits=self._settings.target_encoder_folds,
                    shuffle=True,
                    random_state=self._random_state,
                )
                steps.append(("encoder", TargetEncoder(target_type="binary", cv=cross_fitting)))
            case HighCardinalityEncoding.FREQUENCY:
                steps.append(("encoder", FrequencyEncoder()))
            case HighCardinalityEncoding.ONEHOT:
                # Розріджена матриця: з ~тисячі колонок у рядку ненульових лише сім.
                steps.append(
                    (
                        "encoder",
                        OneHotEncoder(
                            min_frequency=self._settings.onehot_min_frequency,
                            handle_unknown="infrequent_if_exist",
                            sparse_output=True,
                        ),
                    )
                )
            case HighCardinalityEncoding.ORDINAL:
                steps.append(
                    (
                        "encoder",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    )
                )
        is_onehot = options.high_cardinality_encoding is HighCardinalityEncoding.ONEHOT
        if options.scaler is not ScalerKind.NONE and not is_onehot:
            # Закодована колонка — звичайне число; приводимо до масштабу решти ознак.
            steps.append(("scaler", StandardScaler()))
        return Pipeline(steps)

    @staticmethod
    def _binary_pipeline() -> Pipeline:
        """Гілка бінарних ознак: мода → 0/1."""
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
