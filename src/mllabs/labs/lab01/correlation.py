"""Кореляційний аналіз і мультиколінеарність числових ознак."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from mllabs.data.schema import DatasetSchema
from mllabs.preprocessing import (
    CvScores,
    PreprocessingOptions,
    PreprocessingSettings,
    PreprocessorFactory,
    ProxyModelEvaluator,
)

VIF_BEFORE = "VIF до"
VIF_AFTER = "VIF після"
VIF_EXTRA = "VIF, якщо вилучити ще"


class CorrelationAnalyzer:
    """Рахує матриці кореляцій, шукає сильно пов'язані пари та VIF.

    * **Пірсон** вимірює ЛІНІЙНИЙ зв'язок і чутливий до викидів.
    * **Спірмен** — це Пірсон, порахований на рангах: ловить будь-який
      монотонний зв'язок і стійкий до викидів. Якщо Спірмен помітно більший
      за Пірсона — зв'язок є, але нелінійний.
    * **Мультиколінеарність** — коли ознака майже повністю виражається через
      інші. Лінійним моделям це шкодить: коефіцієнти стають нестійкими.

    Args:
        threshold: Поріг |r|, від якого пара вважається сильно корельованою.
    """

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def matrix(self, numeric: pd.DataFrame, method: str) -> pd.DataFrame:
        """Матриця кореляцій (пропуски ігноруються попарно).

        Args:
            numeric: Числові ознаки навчальної вибірки.
            method: ``"pearson"`` або ``"spearman"``.
        """
        return numeric.corr(method=method)

    def high_pairs(self, pearson: pd.DataFrame, spearman: pd.DataFrame) -> pd.DataFrame:
        """Пари ознак, у яких хоча б один коефіцієнт за модулем не нижчий за поріг.

        Returns:
            Таблиця «Ознака 1», «Ознака 2», «Пірсон», «Спірмен», відсортована
            за спаданням |Пірсон|.
        """
        names = pearson.columns
        # triu_indices(n, k=1) — номери клітинок вище головної діагоналі: кожна
        # пара один раз, без пар «ознака сама з собою».
        first, second = np.triu_indices(len(names), k=1)
        pairs = pd.DataFrame(
            {
                "Ознака 1": names[first],
                "Ознака 2": names[second],
                "Пірсон": pearson.to_numpy()[first, second],
                "Спірмен": spearman.to_numpy()[first, second],
            }
        )
        strongest = pairs[["Пірсон", "Спірмен"]].abs().max(axis=1)
        return (
            pairs[strongest >= self._threshold]
            .sort_values("Пірсон", key=np.abs, ascending=False)
            .reset_index(drop=True)
        )

    def vif(self, numeric: pd.DataFrame) -> pd.Series:
        """Коефіцієнт інфляції дисперсії (VIF) кожної ознаки.

        ``VIF_i = 1 / (1 − R²_i)``, де ``R²_i`` — наскільки добре ознаку ``i``
        пояснює лінійна регресія від решти ознак. VIF = 1 — ознака незалежна;
        VIF > 10 — майже повністю дублюється іншими.

        Окремі регресії не потрібні: VIF усіх ознак — це діагональ матриці,
        оберненої до кореляційної (відомий факт лінійної алгебри), тому
        обходимося без бібліотеки ``statsmodels``.

        Args:
            numeric: Числові ознаки; рядки з пропусками відкидаються.
        """
        correlation = numeric.dropna().corr().to_numpy()
        inverse = np.linalg.inv(correlation)
        return pd.Series(np.diag(inverse), index=numeric.columns, name="VIF")

    def vif_table(
        self,
        numeric: pd.DataFrame,
        dropped: Collection[str],
        extra_dropped: Collection[str] = (),
    ) -> pd.DataFrame:
        """Порівнює VIF до і після вилучення ознак-дублікатів.

        Args:
            numeric: Усі числові ознаки.
            dropped: Ознаки, які прибирає повний сценарій препроцесингу.
            extra_dropped: Ознаки, додаткове вилучення яких лише перевіряється
                (у сценарії вони лишаються); якщо задано — у таблиці є ще колонка
                VIF для варіанта, де їх вилучено.

        Returns:
            Таблиця «Ознака», «VIF до», «VIF після» (порожньо для вилучених) і,
            за потреби, «VIF, якщо вилучити ще».
        """
        before = self.vif(numeric)
        after = self.vif(numeric.drop(columns=list(dropped)))
        table = pd.DataFrame(
            {
                "Ознака": before.index,
                VIF_BEFORE: before.to_numpy(),
                # reindex вирівнює за назвами: для вилучених ознак значення немає → NaN.
                VIF_AFTER: after.reindex(before.index).to_numpy(),
            }
        )
        if extra_dropped:
            extra = self.vif(numeric.drop(columns=[*dropped, *extra_dropped]))
            table[VIF_EXTRA] = extra.reindex(before.index).to_numpy()
        return table

    @staticmethod
    def to_metrics(
        pairs: pd.DataFrame, vif_table: pd.DataFrame, vif_threshold: float
    ) -> dict[str, object]:
        """Словник для ``metrics.json``: сильні пари та VIF до/після вилучення.

        Args:
            pairs: Результат :meth:`high_pairs`.
            vif_table: Результат :meth:`vif_table`.
            vif_threshold: Поріг VIF, вище якого ознака вважається мультиколінеарною.
        """
        metrics: dict[str, object] = {
            "vif_threshold": vif_threshold,
            "n_pairs_above_threshold": len(pairs),
            "max_abs_pearson": pairs["Пірсон"].abs().max(),
            "max_abs_spearman": pairs["Спірмен"].abs().max(),
            "max_vif_before": vif_table[VIF_BEFORE].max(),
            "max_vif_after": vif_table[VIF_AFTER].max(),
            "n_vif_above_threshold_before": int((vif_table[VIF_BEFORE] > vif_threshold).sum()),
            "n_vif_above_threshold_after": int((vif_table[VIF_AFTER] > vif_threshold).sum()),
        }
        if VIF_EXTRA in vif_table:
            metrics["max_vif_extra_drop"] = vif_table[VIF_EXTRA].max()
            metrics["n_vif_above_threshold_extra_drop"] = int(
                (vif_table[VIF_EXTRA] > vif_threshold).sum()
            )
        return metrics


@dataclass(frozen=True)
class ExtraDropScores:
    """Проксі-метрики повного сценарію з додатковим вилученням ознак і без нього.

    Attributes:
        columns: Ознаки-кандидати на додаткове вилучення.
        dropped: Той самий сценарій, але без цих ознак.
        kept: Повний сценарій як є (ознаки залишено).
    """

    columns: tuple[str, ...]
    dropped: CvScores
    kept: CvScores


class ExtraCollinearDropCheck:
    """Перевіряє, чи варто вилучити ще ознаки, що лишаються з VIF > порога.

    Високий VIF сам по собі не шкодить якості прогнозу — він робить нестійкими
    окремі коефіцієнти лінійної моделі. Тому рішення ухвалюється за ЄДИНИМ
    правилом вибору препроцесингу (``mllabs.preprocessing.selection``): обидва
    варіанти оцінюються на тих самих фолдах, d_k = PR-AUC «залишити» − PR-AUC
    «вилучити», Δ = mean(d), s = std(d, ddof=1). Залишити ознаки (складніший
    варіант: більше ознак, вищий VIF) варто, лише якщо Δ > s; інакше — вилучити.

    Кандидати НЕ входять у ``schema.collinear_columns`` (сценарій їх залишає);
    клас вимірює альтернативу «вилучити ще», щоб підсумкова таблиця показала,
    що застосоване рішення відповідає правилу.

    Args:
        schema: Схема очищеного датасету.
        extra_columns: Ознаки-кандидати на додаткове вилучення.
        settings: Числові параметри препроцесингу (ті самі, що й в основній фабриці).
        random_state: Зерно стохастичних цеглин.

    Raises:
        ValueError: Якщо кандидат уже є в переліку колінеарних ознак схеми або
            не є числовою ознакою схеми.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        extra_columns: Sequence[str],
        settings: PreprocessingSettings,
        random_state: int,
    ) -> None:
        invalid = [
            column
            for column in extra_columns
            if column in schema.collinear_columns or column not in schema.numeric
        ]
        if invalid:
            raise ValueError(
                f"Ознаки {invalid} не можна перевірити: їх уже вилучено (collinear_columns) "
                "або це не числові ознаки схеми"
            )
        self._schema = schema
        self._extra_columns = tuple(extra_columns)
        self._settings = settings
        self._random_state = random_state

    def compare(
        self,
        evaluator: ProxyModelEvaluator,
        options: PreprocessingOptions,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> ExtraDropScores:
        """Оцінює проксі-модель зі сценарієм ``options`` з вилученням і без нього."""
        reduced_schema = replace(
            self._schema,
            collinear_columns=self._schema.collinear_columns + self._extra_columns,
        )
        kept, dropped = (
            evaluator.evaluate(
                PreprocessorFactory(schema, self._random_state, self._settings).build_from_options(
                    options
                ),
                features,
                target,
            )
            for schema in (self._schema, reduced_schema)
        )
        return ExtraDropScores(self._extra_columns, dropped=dropped, kept=kept)

    def to_metrics(self, scores: ExtraDropScores) -> dict[str, object]:
        """Словник для ``metrics.json`` (розділ ``correlation``)."""
        return {
            "extra_drop_columns": [self._schema.label(column) for column in scores.columns],
            "pr_auc_full": scores.kept.pr_auc,
            "pr_auc_std_full": scores.kept.pr_auc_std,
            "pr_auc_full_extra_drop": scores.dropped.pr_auc,
            # PR-AUC по фолдах обох варіантів — для парного порівняння за правилом вибору.
            "pr_auc_folds_full": list(scores.kept.fold_pr_auc),
            "pr_auc_folds_extra_drop": list(scores.dropped.fold_pr_auc),
            "n_features_full": scores.kept.n_features,
            "n_features_extra_drop": scores.dropped.n_features,
        }
