"""Масштабування та степеневі перетворення числових ознак."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler

METHOD_RAW = "Вихідна"
METHOD_ROBUST = "RobustScaler"
METHOD_YEO_JOHNSON = "Yeo-Johnson"
METHOD_BOX_COX = "Box-Cox"


@dataclass(frozen=True)
class ScalingResult:
    """Результат порівняння перетворень.

    Attributes:
        skewness: Асиметрія кожної ознаки (рядки) після кожного перетворення
            (колонки); Box-Cox порожній для ознак із недодатними значеннями.
        transformed: Словник «метод → перетворена таблиця» (для гістограм).
        mean_abs_skew: Середня |асиметрія| по ознаках для кожного методу.
        n_box_cox_features: Скільки ознак строго додатні й дали скінченний
            результат Box-Cox (середня асиметрія Box-Cox рахується лише по них,
            тому з іншими методами вона порівнянна лише наближено).
    """

    skewness: pd.DataFrame
    transformed: dict[str, pd.DataFrame]
    mean_abs_skew: dict[str, float]
    n_box_cox_features: int

    def to_metrics(self) -> dict[str, object]:
        """Словник для ``metrics.json``; ``chosen_method`` — метод із найменшою асиметрією."""
        candidates = {
            "robust": self.mean_abs_skew[METHOD_ROBUST],
            "power": self.mean_abs_skew[METHOD_YEO_JOHNSON],
        }
        return {
            "mean_abs_skew_raw": self.mean_abs_skew[METHOD_RAW],
            "mean_abs_skew_robust": candidates["robust"],
            "mean_abs_skew_power": candidates["power"],
            "mean_abs_skew_box_cox": self.mean_abs_skew.get(METHOD_BOX_COX),
            "n_box_cox_features": self.n_box_cox_features,
            "chosen_method": min(candidates, key=candidates.get),
        }


class ScalingBenchmark:
    """Порівнює RobustScaler і степеневі перетворення за асиметрією розподілів.

    * **RobustScaler**: ``(x − медіана) / IQR``. Це ЛІНІЙНЕ перетворення: змінює
      масштаб, але не форму розподілу — асиметрія лишається тією самою.
    * **PowerTransformer** — НЕЛІНІЙНЕ: підбирає степінь ``λ`` так, щоб розподіл
      став якомога ближчим до нормального, і стандартизує результат.
      **Box-Cox** працює лише для строго додатних значень, **Yeo-Johnson** —
      його узагальнення на нулі та від'ємні числа.

    Усі перетворення навчаються на переданих (навчальних) даних; пропуски
    попередньо заповнюються медіаною.
    """

    def run(self, numeric: pd.DataFrame) -> ScalingResult:
        """Застосовує перетворення і рахує асиметрію до та після.

        Args:
            numeric: Числові ознаки навчальної вибірки.
        """
        # set_output(transform="pandas") — трансформер повертає DataFrame з назвами
        # колонок замість «голого» масиву NumPy.
        imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
        complete = imputer.fit_transform(numeric)
        positive_columns = [column for column in complete.columns if complete[column].min() > 0]

        transformed = {
            METHOD_RAW: complete,
            METHOD_ROBUST: RobustScaler().set_output(transform="pandas").fit_transform(complete),
            METHOD_YEO_JOHNSON: PowerTransformer(method="yeo-johnson")
            .set_output(transform="pandas")
            .fit_transform(complete),
        }
        box_cox = self._box_cox(complete[positive_columns]) if positive_columns else None
        if box_cox is not None and not box_cox.empty:
            transformed[METHOD_BOX_COX] = box_cox
        # DataFrame зі словника Series вирівнює рядки за назвами ознак: для ознак,
        # до яких Box-Cox не застосовний, у його колонці лишається NaN.
        skewness = pd.DataFrame({method: table.skew() for method, table in transformed.items()})
        skewness = skewness.reindex(complete.columns)
        return ScalingResult(
            skewness=skewness,
            transformed=transformed,
            mean_abs_skew={method: float(skewness[method].abs().mean()) for method in skewness},
            n_box_cox_features=0 if box_cox is None else box_cox.shape[1],
        )

    @staticmethod
    def _box_cox(positive: pd.DataFrame) -> pd.DataFrame:
        """Box-Cox для строго додатних ознак; вироджені результати відкидаються.

        Box-Cox чутливий до зсуву: для ознаки на кшталт року випуску (2001–2010)
        відносний розкид мізерний, оптимальна степінь ``λ`` прямує до сотень, і
        ``x^λ`` переповнює float64. Такі ознаки (з нескінченними чи NaN
        значеннями після перетворення) не беруть участі в порівнянні.
        """
        # Попередження scipy про переповнення очікуване саме для таких ознак, і
        # вони обробляються нижче явно — тому тут їх тимчасово приглушуємо.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            result = (
                PowerTransformer(method="box-cox")
                .set_output(transform="pandas")
                .fit_transform(positive)
            )
        is_finite = np.isfinite(result).all() & result.std().gt(0)
        return result.loc[:, is_finite]

    @staticmethod
    def skewness_table(result: ScalingResult) -> pd.DataFrame:
        """Таблиця асиметрії для звіту (ознака — перша колонка)."""
        return result.skewness.rename_axis("Ознака").reset_index()

    @staticmethod
    def most_skewed(result: ScalingResult, count: int) -> list[str]:
        """Назви ``count`` ознак із найбільшою за модулем вихідною асиметрією."""
        ordered = result.skewness[METHOD_RAW].abs().sort_values(ascending=False)
        return list(ordered.index[:count])  # зріз [:count] — перші count елементів
