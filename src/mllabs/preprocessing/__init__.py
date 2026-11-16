"""Спільний препроцесинг: власні трансформери, семплери та фабрика пайплайнів.

Використовується лабораторними №1–3. Усе тут сумісне з інтерфейсом
scikit-learn (``fit`` / ``transform``), тому вбудовується в ``Pipeline`` і
навчається лише на тих даних, які передано у ``fit``.
"""

from mllabs.preprocessing.augmentation import GaussianNoiseSampler, SamplerFactory, SamplerKind
from mllabs.preprocessing.encoders import FrequencyEncoder
from mllabs.preprocessing.evaluation import CvScores, ProxyModelEvaluator
from mllabs.preprocessing.factory import (
    CategoricalImputation,
    HighCardinalityEncoding,
    ImputerKind,
    LowCardinalityEncoding,
    PreprocessingMismatchError,
    PreprocessingOptions,
    PreprocessingScenario,
    PreprocessingSettings,
    PreprocessorFactory,
    ScalerKind,
)
from mllabs.preprocessing.features import (
    KICK_AGGREGATIONS,
    FeatureDescription,
    GroupAggregateEncoder,
    GroupAggregation,
    RowFeatureEngineer,
)
from mllabs.preprocessing.pipelines import MODEL_STEP, SAMPLER_STEP, build_model_pipeline
from mllabs.preprocessing.selection import (
    SELECTION_RULE,
    FoldMismatchError,
    PairedDifference,
    paired_difference,
)

__all__ = [
    "KICK_AGGREGATIONS",
    "MODEL_STEP",
    "SAMPLER_STEP",
    "SELECTION_RULE",
    "CategoricalImputation",
    "CvScores",
    "FeatureDescription",
    "FoldMismatchError",
    "FrequencyEncoder",
    "GaussianNoiseSampler",
    "GroupAggregateEncoder",
    "GroupAggregation",
    "HighCardinalityEncoding",
    "ImputerKind",
    "LowCardinalityEncoding",
    "PairedDifference",
    "PreprocessingMismatchError",
    "PreprocessingOptions",
    "PreprocessingScenario",
    "PreprocessingSettings",
    "PreprocessorFactory",
    "ProxyModelEvaluator",
    "RowFeatureEngineer",
    "SamplerFactory",
    "SamplerKind",
    "ScalerKind",
    "build_model_pipeline",
    "paired_difference",
]
