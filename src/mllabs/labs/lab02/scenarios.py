"""Сценарії даних ЛР2: що відбувається з даними до того, як їх побачить модель.

Сценарії завдання:

* **А** — мінімальне очищення: медіана/мода, порядкові номери категорій, без
  масштабування (``PreprocessingScenario.MINIMAL`` з ЛР1);
* **Б** — повний препроцесинг, обґрунтований у ЛР1 (``PreprocessingScenario.FULL``);
* **В** — Б + аугментація рідкісного класу (SMOTE або ADASYN) лише в навчальній
  частині.

Діагностичні варіанти (лише крос-валідація, на test не оцінюються):

* **А_missing** — А, але пропуск у категоріальних ознаках стає окремою
  категорією MISSING (як у Б): показує, яку частку розриву «А → Б» дає саме
  втрата сигналу «пропущено ``WheelType``»;
* **Б_MICE** — Б, але числові пропуски заповнює MICE замість медіани: числом
  відповідає на питання, чи потрібна «просунута» числова імп'ютація.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import BaseEstimator

from mllabs.preprocessing import (
    CategoricalImputation,
    HighCardinalityEncoding,
    ImputerKind,
    LowCardinalityEncoding,
    PreprocessingOptions,
    PreprocessingScenario,
    PreprocessorFactory,
    SamplerFactory,
    SamplerKind,
    ScalerKind,
    build_model_pipeline,
)


class ScenarioId(StrEnum):
    """Ідентифікатор сценарію; значення — ключі таблиць і ``metrics.json``."""

    A = "A"
    A_MISSING = "A_missing"
    B = "B"
    B_MICE = "B_mice"
    C_SMOTE = "C_smote"
    C_ADASYN = "C_adasyn"


@dataclass(frozen=True)
class ScenarioSpec:
    """Опис одного сценарію даних.

    Attributes:
        scenario_id: Ідентифікатор.
        label: Короткий підпис для таблиць і графіків («А», «В-SMOTE»).
        options: Опції препроцесора (``PreprocessorFactory.build_from_options``).
        sampler_kind: Метод аугментації (``SamplerKind.NONE`` — без неї).
        on_test: Чи оцінюється сценарій на тестовій вибірці (діагностичні — ні).
    """

    scenario_id: ScenarioId
    label: str
    options: PreprocessingOptions
    sampler_kind: SamplerKind
    on_test: bool

    @property
    def is_augmented(self) -> bool:
        """Чи додає сценарій синтетичні рядки."""
        return self.sampler_kind is not SamplerKind.NONE


class ScenarioCatalog:
    """Перелік сценаріїв і збирання пайплайна «препроцесор → [семплер] → модель».

    Args:
        preprocessors: Фабрика препроцесорів ЛР1 (звірена з ``preprocessing.json``).
        sampling_strategy: Типове відношення «клас 1 / клас 0» після аугментації.
        k_neighbors: Сусіди SMOTE/ADASYN.
        noise_scale: Параметр фабрики семплерів (у ЛР2 не використовується).
        random_state: Зерно семплерів.
    """

    def __init__(
        self,
        preprocessors: PreprocessorFactory,
        sampling_strategy: float,
        k_neighbors: int,
        noise_scale: float,
        random_state: int,
    ) -> None:
        self._preprocessors = preprocessors
        self._sampling_strategy = sampling_strategy
        self._k_neighbors = k_neighbors
        self._noise_scale = noise_scale
        self._random_state = random_state
        minimal = preprocessors.options_for(PreprocessingScenario.MINIMAL)
        full = preprocessors.options_for(PreprocessingScenario.FULL)
        specs = (
            ScenarioSpec(ScenarioId.A, "А", minimal, SamplerKind.NONE, on_test=True),
            ScenarioSpec(
                ScenarioId.A_MISSING,
                "А_missing",
                # replace — копія frozen-об'єкта зі зміненим полем (як «wither» у Java record).
                replace(minimal, categorical_imputation=CategoricalImputation.MISSING_CATEGORY),
                SamplerKind.NONE,
                on_test=False,
            ),
            ScenarioSpec(ScenarioId.B, "Б", full, SamplerKind.NONE, on_test=True),
            ScenarioSpec(
                ScenarioId.B_MICE,
                "Б_MICE",
                replace(full, imputer=ImputerKind.MICE),
                SamplerKind.NONE,
                on_test=False,
            ),
            ScenarioSpec(ScenarioId.C_SMOTE, "В-SMOTE", full, SamplerKind.SMOTE, on_test=True),
            ScenarioSpec(ScenarioId.C_ADASYN, "В-ADASYN", full, SamplerKind.ADASYN, on_test=True),
        )
        self._specs = {spec.scenario_id: spec for spec in specs}

    def spec(self, scenario_id: ScenarioId) -> ScenarioSpec:
        """Опис сценарію за ідентифікатором."""
        return self._specs[scenario_id]

    def main(self) -> list[ScenarioSpec]:
        """Сценарії завдання (А, Б, В-SMOTE, В-ADASYN) — оцінюються і на test."""
        return [spec for spec in self._specs.values() if spec.on_test]

    def label(self, scenario_id: ScenarioId) -> str:
        """Короткий підпис сценарію."""
        return self._specs[scenario_id].label

    def preprocessor(self, spec: ScenarioSpec) -> BaseEstimator:
        """Нефітований препроцесор сценарію."""
        return self._preprocessors.build_from_options(spec.options)

    def sampler(
        self, spec: ScenarioSpec, sampling_strategy: float | None = None
    ) -> BaseEstimator | None:
        """Нефітований семплер сценарію (``None`` — без аугментації).

        Args:
            spec: Сценарій.
            sampling_strategy: Інша частота аугментації (для перебору частот);
                ``None`` — типова.
        """
        factory = SamplerFactory(
            sampling_strategy=(
                self._sampling_strategy if sampling_strategy is None else sampling_strategy
            ),
            k_neighbors=self._k_neighbors,
            noise_scale=self._noise_scale,
            random_state=self._random_state,
        )
        return factory.create(spec.sampler_kind)

    def pipeline(
        self,
        spec: ScenarioSpec,
        model: BaseEstimator,
        sampling_strategy: float | None = None,
        sampler_wrapper: type[BaseEstimator] | None = None,
    ) -> ImbalancedPipeline:
        """Плоский нефітований пайплайн «препроцесор → [семплер] → модель».

        Args:
            spec: Сценарій.
            model: Нефітована модель.
            sampling_strategy: Інша частота аугментації (``None`` — типова).
            sampler_wrapper: Клас-обгортка семплера (напр. такий, що запам'ятовує
                синтетичні рядки для аналізу); ``None`` — семплер як є.
        """
        sampler = self.sampler(spec, sampling_strategy)
        if sampler is not None and sampler_wrapper is not None:
            sampler = sampler_wrapper(sampler)
        return build_model_pipeline(self.preprocessor(spec), model, sampler)

    def transformations(self, spec: ScenarioSpec) -> str:
        """Перелік перетворень сценарію одним рядком (для таблиці стану даних)."""
        return "; ".join(describe_options(spec.options, spec.sampler_kind, self._sampling_strategy))


def describe_options(
    options: PreprocessingOptions, sampler_kind: SamplerKind, sampling_strategy: float
) -> Sequence[str]:
    """Короткий український опис кожної «цеглини» препроцесингу."""
    imputer = {
        ImputerKind.MEDIAN: "медіана",
        ImputerKind.KNN: "KNN",
        ImputerKind.MICE: "MICE",
    }[options.imputer]
    categorical = {
        CategoricalImputation.MODE: "мода",
        CategoricalImputation.MISSING_CATEGORY: "категорія MISSING",
    }[options.categorical_imputation]
    low = {
        LowCardinalityEncoding.ONEHOT: "One-Hot",
        LowCardinalityEncoding.ORDINAL: "Ordinal",
    }[options.low_cardinality_encoding]
    high = {
        HighCardinalityEncoding.TARGET: "Target",
        HighCardinalityEncoding.FREQUENCY: "Frequency",
        HighCardinalityEncoding.ONEHOT: "One-Hot",
        HighCardinalityEncoding.ORDINAL: "Ordinal",
    }[options.high_cardinality_encoding]
    scaler = {
        ScalerKind.NONE: "без масштабування",
        ScalerKind.ROBUST: "RobustScaler",
        ScalerKind.POWER: "Yeo-Johnson",
    }[options.scaler]
    parts = [
        f"числа: {imputer}",
        f"категорії: {categorical}",
        f"кодування: {low}/{high}" if low != high else f"кодування: {low}",
        scaler,
    ]
    if options.add_row_features or options.add_group_aggregates:
        parts.append("нові ознаки")
    if options.drop_collinear:
        parts.append("без колінеарних")
    if sampler_kind is not SamplerKind.NONE:
        parts.append(f"{sampler_kind.value.upper()} до {sampling_strategy:g}:1")
    return parts
