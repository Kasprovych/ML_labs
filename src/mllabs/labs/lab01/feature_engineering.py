"""Інженерія ознак: перелік нових ознак, їх інформативність і вплив на модель."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from mllabs.labs.lab01.correlation import CorrelationAnalyzer
from mllabs.preprocessing import (
    KICK_AGGREGATIONS,
    CvScores,
    FeatureDescription,
    GroupAggregateEncoder,
    PreprocessingOptions,
    PreprocessorFactory,
    ProxyModelEvaluator,
    RowFeatureEngineer,
)
from mllabs.preprocessing.factory import COLUMNS_STEP

GROUP_ORIGINAL = "вихідна"
_GROUP_POLYNOMIAL = "поліноміальна"

SET_BASE = "Вихідні ознаки"
SET_ROW = "+ комбінації, дата, індикатори"
SET_AGGREGATES = "+ агрегації за категоріями"
SET_POLYNOMIAL = "+ поліноміальні"
SET_ROW_AND_AGGREGATES = "+ комбінації та агрегації"
SET_ALL = "Усі нові ознаки разом"

#: Групи нових ознак: «ключ у metrics.json → прапорець у PreprocessingOptions».
FEATURE_GROUPS = {
    "row_features": "add_row_features",
    "group_aggregates": "add_group_aggregates",
    "polynomial": "add_polynomial_features",
}


@dataclass(frozen=True)
class FeatureSetScores:
    """Результат порівняння наборів ознак.

    Attributes:
        sets: Набори R15 відносно препроцесингу без нових ознак: «назва → метрики».
        full: Остаточний сценарій FULL.
        toggled: FULL із перемкненою групою: «ключ групи → метрики».
        in_full: Чи входить група у FULL: «ключ групи → так/ні».
    """

    sets: dict[str, CvScores]
    full: CvScores
    toggled: dict[str, CvScores]
    in_full: dict[str, bool]


class FeatureEngineeringBenchmark:
    """Оцінює користь згенерованих ознак.

    Два погляди на користь:

    * **Взаємна інформація** (mutual information, MI) між ознакою та ціллю:
      скільки знання про клас дає одна окрема ознака. На відміну від кореляції,
      ловить і нелінійні залежності; 0 — ознака й ціль незалежні.
    * **Проксі-модель**: чи покращує набір ознак якість у крос-валідації.

    Args:
        numeric_features: Вихідні числові ознаки (для порівняння MI з новими).
        polynomial_features: Ознаки, з яких будуються поліноміальні комбінації.
        polynomial_degree: Степінь полінома.
        sample_size: Розмір підвибірки для оцінки MI.
        random_state: Зерно для підвибірки та оцінювача MI.
        label: Функція «назва колонки → підпис» (скорочення довгих назв цін MMR).
    """

    def __init__(
        self,
        numeric_features: Sequence[str],
        polynomial_features: Sequence[str],
        polynomial_degree: int,
        sample_size: int,
        random_state: int,
        label: Callable[[str], str] = str,
    ) -> None:
        self._numeric_features = list(numeric_features)
        self._polynomial_features = list(polynomial_features)
        self._polynomial_degree = polynomial_degree
        self._sample_size = sample_size
        self._random_state = random_state
        self._label = label

    def mutual_information(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Рахує MI з ціллю для вихідних числових і всіх нових ознак.

        Усі статистики (агрегати груп, медіани, параметри стандартизації)
        рахуються на переданій навчальній вибірці.

        Args:
            features: Очищені ознаки навчальної вибірки.
            target: Цільова змінна навчальної вибірки.

        Returns:
            Таблиця «Ознака» (підпис), «Група», «Формула», «MI», відсортована за
            спаданням MI.
        """
        sample = features.sample(
            n=min(self._sample_size, len(features)), random_state=self._random_state
        )
        engineered = GroupAggregateEncoder(KICK_AGGREGATIONS).fit_transform(
            RowFeatureEngineer().fit_transform(sample)
        )
        polynomial = self._polynomial_frame(sample)
        descriptions = self._descriptions(polynomial.columns)

        new_names = [d.name for d in descriptions if d.group != _GROUP_POLYNOMIAL]
        table = pd.concat(
            [engineered[self._numeric_features + new_names], polynomial], axis=1
        )  # axis=1 — склеювання колонками (пліч-о-пліч), а не рядками
        complete = SimpleImputer(strategy="median").fit_transform(table)
        # Дискретні ознаки (частини дати, індикатори) оцінюються іншою формулою MI.
        discrete = [
            name in RowFeatureEngineer.DATE_FEATURES + RowFeatureEngineer.INDICATOR_FEATURES
            for name in table.columns
        ]
        scores = mutual_info_classif(
            complete,
            target.loc[sample.index],
            discrete_features=discrete,
            random_state=self._random_state,
        )

        by_name = {description.name: description for description in descriptions}
        rows = []
        for name, score in zip(table.columns, scores, strict=True):
            description = by_name.get(name)  # None для вихідних ознак
            rows.append(
                {
                    "Ознака": self._label(name),
                    "Група": description.group if description else GROUP_ORIGINAL,
                    "Формула": description.formula if description else "—",
                    "MI": score,
                }
            )
        return pd.DataFrame(rows).sort_values("MI", ascending=False).reset_index(drop=True)

    def _polynomial_frame(self, sample: pd.DataFrame) -> pd.DataFrame:
        """Квадрати й попарні добутки стандартизованих ознак (без самих ознак)."""
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PolynomialFeatures(degree=self._polynomial_degree, include_bias=False),
        ).set_output(transform="pandas")
        expanded = pipeline.fit_transform(sample[self._polynomial_features])
        # sklearn називає добуток «a b»; «a×b» читається на графіку однозначно.
        expanded.columns = [name.replace(" ", "×") for name in expanded.columns]
        return expanded.drop(columns=self._polynomial_features)

    @staticmethod
    def _descriptions(polynomial_names: Sequence[str]) -> list[FeatureDescription]:
        """Описи всіх нових ознак: у межах рядка, агрегацій і поліноміальних."""
        polynomial = [
            FeatureDescription(
                name,
                _GROUP_POLYNOMIAL,
                # Добуток названо «a×b», квадрат — «a^2» (як у sklearn).
                name.replace("×", " × ") + " (стандартизовані)",
            )
            for name in polynomial_names
        ]
        return [
            *RowFeatureEngineer.describe(),
            *GroupAggregateEncoder(KICK_AGGREGATIONS).describe(),
            *polynomial,
        ]

    @staticmethod
    def new_features_table(mutual_information: pd.DataFrame) -> pd.DataFrame:
        """Лишає в таблиці MI тільки згенеровані ознаки."""
        is_new = mutual_information["Група"] != GROUP_ORIGINAL
        return mutual_information[is_new].reset_index(drop=True)

    @staticmethod
    def compare(
        factory: PreprocessorFactory,
        evaluator: ProxyModelEvaluator,
        base_options: PreprocessingOptions,
        full_options: PreprocessingOptions,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> FeatureSetScores:
        """Оцінює набори ознак (R15) і рішення про кожну групу відносно FULL.

        Кожен однаковий набір опцій оцінюється лише раз (результати кешуються),
        і всі варіанти — на тих самих фолдах оцінювача, тож їх можна
        порівнювати попарно.

        Args:
            factory: Фабрика пайплайнів препроцесингу.
            evaluator: Оцінювач проксі-моделі.
            base_options: Препроцесинг БЕЗ нових ознак; групи вмикаються по черзі.
            full_options: Остаточний сценарій FULL; у ньому кожна група
                перемикається («є» ↔ «немає»), щоб перевірити рішення про неї.
            features: Ознаки навчальної вибірки.
            target: Цільова змінна навчальної вибірки.
        """
        cache: dict[PreprocessingOptions, CvScores] = {}

        def evaluate(options: PreprocessingOptions) -> CvScores:
            # Вкладена функція бачить змінні зовнішньої (cache) — замикання, як лямбда в Java.
            if options not in cache:
                cache[options] = evaluator.evaluate(
                    factory.build_from_options(options), features, target
                )
            return cache[options]

        variants = FeatureEngineeringBenchmark.variants(base_options)
        toggles = FeatureEngineeringBenchmark.full_toggles(full_options)
        return FeatureSetScores(
            sets={name: evaluate(options) for name, options in variants.items()},
            full=evaluate(full_options),
            toggled={group: evaluate(options) for group, options in toggles.items()},
            in_full={
                group: bool(getattr(full_options, flag)) for group, flag in FEATURE_GROUPS.items()
            },
        )

    @staticmethod
    def full_toggles(full_options: PreprocessingOptions) -> dict[str, PreprocessingOptions]:
        """FULL, у якому одна група ознак перемкнена: «ключ групи → опції»."""
        return {
            group: replace(full_options, **{flag: not getattr(full_options, flag)})
            for group, flag in FEATURE_GROUPS.items()
        }

    @staticmethod
    def variants(base_options: PreprocessingOptions) -> dict[str, PreprocessingOptions]:
        """Набори ознак, що порівнюються: «назва набору → опції препроцесингу».

        Агрегації та поліном будуються й оцінюються тут незалежно від того, чи
        входять вони в сценарій FULL: бенчмарк показує внесок кожної групи.
        """
        return {
            SET_BASE: base_options,
            SET_ROW: replace(base_options, add_row_features=True),
            SET_AGGREGATES: replace(base_options, add_group_aggregates=True),
            SET_POLYNOMIAL: replace(base_options, add_polynomial_features=True),
            SET_ROW_AND_AGGREGATES: replace(
                base_options, add_row_features=True, add_group_aggregates=True
            ),
            SET_ALL: replace(
                base_options,
                add_row_features=True,
                add_group_aggregates=True,
                add_polynomial_features=True,
            ),
        }

    def full_pipeline_vif(
        self,
        factory: PreprocessorFactory,
        options: PreprocessingOptions,
        features: pd.DataFrame,
        target: pd.Series,
        analyzer: CorrelationAnalyzer,
    ) -> pd.DataFrame:
        """VIF числових ознак саме в тому вигляді, в якому їх отримує модель.

        На відміну від ``vif.csv`` (сирі вихідні ознаки), тут пайплайн
        сценарію навчається на train, і VIF рахується на виході числової гілки:
        після імп'ютації та степеневого перетворення, разом з інженерними
        ознаками. Точна лінійна залежність дала б тут VIF у тисячі й більше.

        Returns:
            Таблиця «Ознака», «VIF», відсортована за спаданням VIF.
        """
        pipeline = factory.build_from_options(options).fit(features, target)
        names = pipeline.named_steps[COLUMNS_STEP].get_feature_names_out()
        matrix = pd.DataFrame(pipeline.transform(features), columns=names)
        vif = analyzer.vif(matrix[factory.numeric_columns(options)])
        return (
            pd.DataFrame({"Ознака": vif.index.map(self._label), "VIF": vif.to_numpy()})
            .sort_values("VIF", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def comparison_table(scores: FeatureSetScores) -> pd.DataFrame:
        """Таблиця метрик проксі-моделі для кожного набору ознак (R15)."""
        return pd.DataFrame(
            [
                {
                    "Набір ознак": name,
                    "Ознак": result.n_features,
                    "PR-AUC": result.pr_auc,
                    "σ PR-AUC": result.pr_auc_std,
                    "ROC-AUC": result.roc_auc,
                    "F1": result.f1,
                }
                for name, result in scores.sets.items()
            ]
        )

    @staticmethod
    def to_metrics(
        new_features: pd.DataFrame,
        scores: FeatureSetScores,
        full_vif: pd.DataFrame,
        vif_threshold: float,
    ) -> dict[str, object]:
        """Словник для ``metrics.json``.

        Args:
            new_features: Таблиця нових ознак, відсортована за спаданням MI.
            scores: Результат :meth:`compare`.
            full_vif: Результат :meth:`full_pipeline_vif` для сценарію FULL.
            vif_threshold: Поріг VIF мультиколінеарності.
        """
        sets = scores.sets
        return {
            "n_features_full": scores.full.n_features,
            "pr_auc_full": scores.full.pr_auc,
            "pr_auc_std_full": scores.full.pr_auc_std,
            "full_numeric_n_features": len(full_vif),
            "full_numeric_max_vif": full_vif["VIF"].max(),
            "full_numeric_max_vif_feature": full_vif["Ознака"].iloc[0],
            "full_numeric_n_vif_above_threshold": int((full_vif["VIF"] > vif_threshold).sum()),
            "n_new_features": len(new_features),
            "pr_auc_before": sets[SET_BASE].pr_auc,
            "pr_auc_after": sets[SET_ALL].pr_auc,
            "pr_auc_row_features": sets[SET_ROW].pr_auc,
            "pr_auc_group_aggregates": sets[SET_AGGREGATES].pr_auc,
            "pr_auc_polynomial": sets[SET_POLYNOMIAL].pr_auc,
            "pr_auc_row_and_aggregates": sets[SET_ROW_AND_AGGREGATES].pr_auc,
            "pr_auc_std_before": sets[SET_BASE].pr_auc_std,
            # Рішення про кожну групу: FULL проти FULL із перемкненою групою, по фолдах.
            "decisions": {
                group: {
                    "in_full": scores.in_full[group],
                    "pr_auc_toggled": scores.toggled[group].pr_auc,
                    "n_features_toggled": scores.toggled[group].n_features,
                    "pr_auc_folds_full": list(scores.full.fold_pr_auc),
                    "pr_auc_folds_toggled": list(scores.toggled[group].fold_pr_auc),
                }
                for group in FEATURE_GROUPS
            },
            "top_feature_by_mi": new_features["Ознака"].iloc[0],
            "top_feature_mi": new_features["MI"].iloc[0],
        }
