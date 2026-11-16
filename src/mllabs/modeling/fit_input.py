"""Прогноз на навчальних рядках у тих самих ознаках, які модель бачила під час ``fit``.

Чому не можна просто викликати ``pipeline.predict_proba(x_train)``: деякі
трансформери кодують навчальні рядки під час ``fit`` інакше, ніж під час
``transform``. Головний приклад — ``TargetEncoder``: ``Pipeline.fit``
викликає його ``fit_transform``, а той кодує кожен навчальний рядок
статистикою ІНШИХ фолдів (крос-фітинг, без власної мітки рядка). Натомість
``transform`` кодує той самий рядок статистикою ВСІХ навчальних рядків,
включно з його власною міткою. Тож ``predict_proba(x_train)`` дає ознаки, яких
модель не бачила, і «якість на train» (а з нею розрив train − валідація)
виходить спотвореною.

Рішення — прохідний крок :class:`FitInputRecorder`, що вставляється в пайплайн
одразу після препроцесингу (перед семплером або моделлю) і запам'ятовує
матрицю, яку отримав під час ``fit``. Якщо в пайплайні є семплер, одразу після
останнього семплера стає другий рекордер — він запам'ятовує вихід семплера.
Прогноз на train рахується моделлю на записаній матриці, проведеній через
кроки між рекордером і моделлю (наприклад, ``StandardScaler`` після SMOTE).

Обмеження. Семплер має ЗБЕРІГАТИ реальні рядки першими в початковому порядку
й лише дописувати синтетичні (так роблять SMOTE, ADASYN, гаусів шум проєкту).
Андерсемплери (``RandomUnderSampler``) і комбіновані методи (``SMOTEENN``,
``SMOTETomek``) видаляють або переставляють реальні рядки — тоді «прогноз на
власних навчальних рядках» не визначений, і :func:`fit_input_proba` кидає
:class:`FitInputMismatchError` замість мовчки повернути хибні числа.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone

#: Назва кроку-рекордера, що вставляється в пайплайн перед семплером або моделлю.
FIT_INPUT_STEP = "fit_input_recorder"
#: Назва другого рекордера — одразу після останнього семплера (якщо семплер є).
RESAMPLED_INPUT_STEP = "resampled_input_recorder"


class FitInputMismatchError(RuntimeError):
    """Не вдається відновити матрицю, на якій навчалася модель, для реальних рядків."""


class FitInputRecorder(TransformerMixin, BaseEstimator):
    """Прохідний крок: нічого не змінює, але запам'ятовує вхід ``fit_transform``.

    ``Pipeline.fit`` викликає у проміжних кроків саме ``fit_transform``, тож
    атрибут ``fit_input_`` після навчання — це матриця навчальних рядків у тих
    координатах, які пішли далі (у семплер або модель). На прогнозі
    (``transform``) крок просто пропускає дані.
    """

    def fit(self, X: np.ndarray, y: object = None) -> Self:  # noqa: N803
        """Запам'ятовує вхід (імена ``X``/``y`` вимушені API sklearn)."""
        self.fit_input_ = X
        self.n_features_in_ = X.shape[1]
        return self

    def fit_transform(self, X: np.ndarray, y: object = None) -> np.ndarray:  # noqa: N803
        """Запам'ятовує вхід і повертає його без змін."""
        self.fit(X, y)
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Повертає дані без змін."""
        return X

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Назви ознак не змінюються: крок прохідний.

        Потрібен, щоб ``pipeline[:-1].get_feature_names_out()`` працював і з рекордером.
        """
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray([f"x{index}" for index in range(self.n_features_in_)], dtype=object)


def _is_sampler(step: object) -> bool:
    """Семплер imblearn — крок із методом ``fit_resample`` (працює лише під час ``fit``)."""
    return hasattr(step, "fit_resample")


def with_fit_input_recorder(estimator: BaseEstimator) -> BaseEstimator:
    """Повертає КОПІЮ оцінювача з рекордером(-ами) матриці ``fit``.

    * Пайплайн (sklearn або imblearn): рекордер ``FIT_INPUT_STEP`` стає перед
      першим семплером (кроком із ``fit_resample``), а якщо семплера немає —
      перед моделлю. Якщо семплер є, рекордер ``RESAMPLED_INPUT_STEP`` стає
      одразу після останнього семплера: за ним перевіряється, що реальні рядки
      дійшли до моделі без змін (див. обмеження в описі модуля).
    * Обгортка пошуку (``GridSearchCV``, ``RandomizedSearchCV`` тощо, параметр
      ``estimator``): рекордер вставляється у вкладений пайплайн. Після
      ``refit`` у ``best_estimator_`` він містить матрицю фінального навчання.
    * Будь-що інше (окрема модель) повертається як є: модель бачить ``x_train`` без змін.

    Копія робиться через ``clone``, тож налаштування пайплайна (``memory``,
    ``verbose`` тощо) зберігаються, а переданий об'єкт не змінюється. Назви
    наявних кроків не змінюються, тож сітки параметрів на кшталт ``model__C``
    працюють як раніше.
    """
    if hasattr(estimator, "steps"):
        copied = clone(estimator)
        steps = list(copied.steps)
        sampler_positions = [index for index, (_, step) in enumerate(steps) if _is_sampler(step)]
        if sampler_positions:
            # Спершу вставка ПІСЛЯ останнього семплера — тоді позиція першого не зсувається.
            steps.insert(sampler_positions[-1] + 1, (RESAMPLED_INPUT_STEP, FitInputRecorder()))
            steps.insert(sampler_positions[0], (FIT_INPUT_STEP, FitInputRecorder()))
        else:
            steps.insert(len(steps) - 1, (FIT_INPUT_STEP, FitInputRecorder()))
        return copied.set_params(steps=steps)
    inner = estimator.get_params(deep=False).get("estimator")
    if inner is not None and hasattr(inner, "steps"):
        return clone(estimator).set_params(estimator=with_fit_input_recorder(inner))
    return estimator


def final_estimator(fitted: BaseEstimator) -> BaseEstimator:
    """Навчений пайплайн або модель усередині обгортки пошуку (``best_estimator_``).

    Raises:
        FitInputMismatchError: Якщо це обгортка пошуку з ``refit=False`` —
            фінальної моделі в неї немає.
    """
    if hasattr(fitted, "best_estimator_"):
        return fitted.best_estimator_
    if hasattr(fitted, "cv_results_"):
        raise FitInputMismatchError(
            "Обгортка пошуку навчена з refit=False: фінальної моделі (best_estimator_) немає, "
            "тож прогноз на навчальних рядках неможливий. Увімкніть refit=True."
        )
    return fitted


def model_input_width(fitted: BaseEstimator) -> int:
    """Кількість ознак на вході самої моделі (останнього кроку), з урахуванням обгорток пошуку."""
    inner = final_estimator(fitted)
    model = inner[-1] if hasattr(inner, "steps") else inner
    return int(model.n_features_in_)


def _model_matrix(pipeline: BaseEstimator) -> np.ndarray:
    """Матриця, яку модель бачила під час ``fit``, — лише для реальних навчальних рядків.

    Береться вихід рекордера, з нього (якщо був семплер) — перші ``n`` рядків
    після семплера, і проводиться через кроки між рекордером і моделлю
    (``transform``; семплери пропускаються — на прогнозі вони не діють).
    """
    names = [name for name, _ in pipeline.steps]
    recorded = pipeline.named_steps[FIT_INPUT_STEP].fit_input_
    start = names.index(FIT_INPUT_STEP)
    if RESAMPLED_INPUT_STEP in pipeline.named_steps:
        resampled = pipeline.named_steps[RESAMPLED_INPUT_STEP].fit_input_
        n_real = recorded.shape[0]
        head = resampled[:n_real]
        if resampled.shape[0] < n_real or not np.allclose(
            _dense(head), _dense(recorded), equal_nan=True
        ):
            raise FitInputMismatchError(
                "Семплер видалив або переставив реальні навчальні рядки (андерсемплінг, "
                "SMOTEENN, SMOTETomek?): прогноз на власних навчальних рядках не визначений."
            )
        recorded, start = head, names.index(RESAMPLED_INPUT_STEP)
    matrix = recorded
    for _, step in pipeline.steps[start + 1 : -1]:
        if not _is_sampler(step):
            matrix = step.transform(matrix)
    return matrix


def _dense(matrix: object) -> np.ndarray:
    """Щільна копія матриці (розріджену — через ``toarray``) для порівняння."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix, dtype=float)


def fit_input_proba(fitted: BaseEstimator, x_train: object, positive_label: int = 1) -> np.ndarray:
    """Ймовірності позитивного класу для навчальних рядків в ознаках часу ``fit``.

    Args:
        fitted: Навчений оцінювач, підготовлений :func:`with_fit_input_recorder`.
        x_train: Навчальні рядки — використовуються лише для оцінювача без
            пайплайна (тоді модель і так бачила їх без перетворень).
        positive_label: Мітка позитивного класу.

    Raises:
        FitInputMismatchError: Якщо семплер змінив реальні рядки або пошук без ``refit``.
    """
    inner = final_estimator(fitted)
    if hasattr(inner, "named_steps") and FIT_INPUT_STEP in inner.named_steps:
        matrix, model = _model_matrix(inner), inner[-1]
    else:
        matrix, model = x_train, inner
    column = list(model.classes_).index(positive_label)
    return model.predict_proba(matrix)[:, column]
