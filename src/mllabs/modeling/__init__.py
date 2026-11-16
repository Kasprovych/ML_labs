"""Спільне моделювання: моделі, метрики, крос-валідація, поріг, бутстреп.

Використовується лабораторними №2–3: ЛР2 порівнює моделі на сценаріях даних,
ЛР3 бере ті самі моделі та протокол оцінки й підбирає гіперпараметри.
"""

from mllabs.modeling.bootstrap import BootstrapInterval, MetricFunction, PairedBootstrap
from mllabs.modeling.curves import CurveBuilder
from mllabs.modeling.fit_input import (
    FIT_INPUT_STEP,
    RESAMPLED_INPUT_STEP,
    FitInputMismatchError,
    FitInputRecorder,
    fit_input_proba,
    model_input_width,
    with_fit_input_recorder,
)
from mllabs.modeling.metrics import DEFAULT_THRESHOLD, ClassificationMetrics, MetricsCalculator
from mllabs.modeling.models import ModelFactory, ModelKind, ModelSettings
from mllabs.modeling.thresholds import F1ThresholdSelector
from mllabs.modeling.validation import (
    CrossValidationResult,
    CrossValidator,
    FoldInspector,
    FoldPrediction,
    mean_and_std,
    positive_proba,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "FIT_INPUT_STEP",
    "RESAMPLED_INPUT_STEP",
    "BootstrapInterval",
    "ClassificationMetrics",
    "CrossValidationResult",
    "CrossValidator",
    "CurveBuilder",
    "F1ThresholdSelector",
    "FitInputMismatchError",
    "FitInputRecorder",
    "FoldInspector",
    "FoldPrediction",
    "MetricFunction",
    "MetricsCalculator",
    "ModelFactory",
    "ModelKind",
    "ModelSettings",
    "PairedBootstrap",
    "fit_input_proba",
    "mean_and_std",
    "model_input_width",
    "positive_proba",
    "with_fit_input_recorder",
]
