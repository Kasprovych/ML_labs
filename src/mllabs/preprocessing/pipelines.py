"""Збирання повного навчального ланцюжка «препроцесор → [семплер] → модель»."""

from __future__ import annotations

from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

#: Назва кроку моделі в пайплайні, який будує :func:`build_model_pipeline`.
MODEL_STEP = "model"
#: Назва кроку семплера (якщо він є).
SAMPLER_STEP = "sampler"


def build_model_pipeline(
    preprocessor: BaseEstimator,
    model: BaseEstimator,
    sampler: BaseEstimator | None = None,
) -> ImbalancedPipeline:
    """Будує ПЛОСКИЙ пайплайн imblearn: кроки препроцесора, семплер, модель.

    Пайплайн imblearn (а не sklearn) — єдиний, що вміє застосовувати семплер
    (SMOTE, ADASYN, шум) лише під час ``fit`` і пропускати його під час
    ``predict``: синтетичні рядки потрапляють тільки в навчальні дані.
    imblearn не дозволяє вкладати ``Pipeline`` у ``Pipeline``, тому кроки
    препроцесора «розгортаються» в один ланцюжок (порядок і зміст ті самі).

    Усі складники клонуються: передані об'єкти не змінюються й не навчаються.

    Args:
        preprocessor: Нефітований препроцесор (``Pipeline`` з ``PreprocessorFactory``
            або будь-який трансформер).
        model: Нефітована модель (класифікатор sklearn).
        sampler: Нефітований семплер imblearn або ``None`` — без аугментації.

    Returns:
        Нефітований ``imblearn.pipeline.Pipeline``.
    """
    # clone робить «чисту» копію без навченого стану.
    prepared = clone(preprocessor)
    steps: list[tuple[str, BaseEstimator]] = (
        list(prepared.steps) if isinstance(prepared, Pipeline) else [("preprocess", prepared)]
    )
    if sampler is not None:
        steps.append((SAMPLER_STEP, clone(sampler)))
    steps.append((MODEL_STEP, clone(model)))
    return ImbalancedPipeline(steps)
