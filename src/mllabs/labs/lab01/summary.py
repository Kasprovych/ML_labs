"""Підсумкова таблиця обраного препроцесингу з числовим обґрунтуванням.

Кожне рішення пояснюється одним і тим самим правилом вибору
(``mllabs.preprocessing.selection.SELECTION_RULE``): на тих самих фолдах
d_k = PR-AUC складнішого − PR-AUC простішого; Δ = mean(d), s = std(d, ddof=1);
складніший варіант обирається, лише якщо Δ > s. Висновок «обрано простіший»
чи «обрано складніший» генерується з чисел ``metrics.json``. Якщо застосований
у коді варіант розходиться з тим, що радить правило, рядок це прямо називає —
окремо позначено два свідомі винятки, оголошені в ``_DECLARED_EXCEPTIONS``
(кодування високої кардинальності та категоріальна імп'ютація).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pandas as pd

from mllabs.data.schema import DatasetSchema
from mllabs.preprocessing import (
    CategoricalImputation,
    HighCardinalityEncoding,
    ImputerKind,
    LowCardinalityEncoding,
    PreprocessingOptions,
    ScalerKind,
    paired_difference,
)

_IMPUTERS = {
    ImputerKind.MEDIAN: "медіана",
    ImputerKind.KNN: "KNNImputer",
    ImputerKind.MICE: "IterativeImputer (MICE)",
}
_CATEGORICAL = {
    CategoricalImputation.MODE: "мода",
    CategoricalImputation.MISSING_CATEGORY: "окрема категорія MISSING",
}
_ENCODINGS = {
    HighCardinalityEncoding.TARGET: "Target Encoding",
    HighCardinalityEncoding.FREQUENCY: "Frequency Encoding",
    HighCardinalityEncoding.ONEHOT: "One-Hot з об'єднанням рідкісних",
    HighCardinalityEncoding.ORDINAL: "Ordinal",
}
#: Порядок складності (від простішого): впливає лише на те, що вважати «простішим».
_IMPUTER_COMPLEXITY = [ImputerKind.MEDIAN, ImputerKind.KNN, ImputerKind.MICE]
_ENCODING_COMPLEXITY = [
    HighCardinalityEncoding.ORDINAL,
    HighCardinalityEncoding.FREQUENCY,
    HighCardinalityEncoding.TARGET,
    HighCardinalityEncoding.ONEHOT,
]
_LOW_CARDINALITY = {
    LowCardinalityEncoding.ONEHOT: "One-Hot",
    LowCardinalityEncoding.ORDINAL: "Ordinal",
}
#: Як кожне масштабування поводиться з викидами (рядки не видаляються ніколи).
_OUTLIER_HANDLING = {
    ScalerKind.NONE: "рядки не видаляються; масштабування немає, викиди лишаються як є",
    ScalerKind.ROBUST: "рядки не видаляються; RobustScaler (медіана та IQR) стійкий до них",
    ScalerKind.POWER: "рядки не видаляються; хвости стискає степеневе перетворення",
}
_SCALERS = {
    ScalerKind.NONE: "без масштабування",
    ScalerKind.ROBUST: "RobustScaler",
    ScalerKind.POWER: "PowerTransformer (Yeo-Johnson)",
}


class PreprocessingSummaryBuilder:
    """Зводить рішення сценарію повного препроцесингу та числа, на які вони спираються.

    Колонка «Обрано» формується з фактичних опцій сценарію (тобто з коду), а
    «Підстава» — з ``metrics.json`` цього запуску за єдиним правилом вибору:
    на тих самих фолдах d_k = PR-AUC складнішого − PR-AUC простішого;
    Δ = mean(d), s = std(d, ddof=1); складніший варіант (більше ознак, вищий
    VIF, повільніший метод) обирається, лише якщо Δ > s, інакше — простіший.
    Кожне рішення перевіряється відносно остаточного сценарію FULL. Тому
    таблиця не може розійтися ні з кодом, ні з результатами, а розбіжність із
    правилом стає видимою (позначка «УВАГА»).

    Args:
        schema: Схема очищеного датасету.
        options: Опції сценарію повного препроцесингу.
        indicators: Індикатори пропусків, які сценарій фактично подає в модель
            (``PreprocessorFactory.indicator_columns``).
    """

    def __init__(
        self, schema: DatasetSchema, options: PreprocessingOptions, indicators: Sequence[str]
    ) -> None:
        self._schema = schema
        self._options = options
        self._indicators = list(indicators)

    def build(self, metrics: Mapping[str, Any]) -> pd.DataFrame:
        """Будує таблицю «Етап», «Обрано», «Підстава».

        Args:
            metrics: Вміст ``metrics.json`` після виконання попередніх кроків.
        """
        options = self._options
        dataset = metrics["dataset"]
        outliers = metrics["outliers"]
        scaling = metrics["scaling"]
        augmentation = metrics["augmentation"]
        rows = [
            (
                "Приховані пропуски",
                "ціни ≤ 1 і NOT AVAIL замінено на NaN",
                f"{dataset['hidden_missing_cells']} клітинок",
            ),
            self._collinearity_row(metrics["correlation"]),
            (
                "Викиди",
                _OUTLIER_HANDLING[options.scaler],
                f"IQR позначає {outliers['iqr_share']:.1%} рядків, "
                f"Z-score {outliers['zscore_share']:.1%}; "
                f"усі чотири методи разом {outliers['all_methods_share']:.2%}",
            ),
            self._numeric_imputation_row(metrics["imputation"]),
            self._categorical_imputation_row(metrics["imputation"]),
            self._high_cardinality_row(metrics["encoding"]),
            (
                "Кодування інших категорій",
                f"{_LOW_CARDINALITY[options.low_cardinality_encoding]} (номінальні), "
                "Ordinal 0/1 (бінарні)",
                f"не більше {metrics['encoding']['max_onehot_cardinality']} категорій "
                "на номінальну ознаку",
            ),
            (
                "Масштабування",
                _SCALERS[options.scaler],
                f"середня |асиметрія|: вихідна {scaling['mean_abs_skew_raw']:.2f}, "
                f"RobustScaler {scaling['mean_abs_skew_robust']:.2f}, "
                f"Yeo-Johnson {scaling['mean_abs_skew_power']:.2f}",
            ),
            (
                "Аугментація",
                "у базовий сценарій не входить; вмикається окремо семплером у Pipeline",
                "PR-AUC / recall: без {none}, SMOTE {smote}, ADASYN {adasyn}, шум {noise}".format(
                    none=_quality(augmentation["none"]),
                    smote=_quality(augmentation["smote"]),
                    adasyn=_quality(augmentation["adasyn"]),
                    noise=_quality(augmentation["gaussian_noise"]),
                ),
            ),
            self._feature_engineering_row(metrics["feature_engineering"]),
        ]
        return pd.DataFrame(rows, columns=["Етап", "Обрано", "Підстава"])

    def decisions_to_metrics(self, metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Δ, s і висновок кожного рішення таблиці — для ``metrics.json`` (розділ ``selection``).

        Звіт підставляє ці числа через ``{{metrics.selection...}}``, а не
        переписує їх із CSV. Числа ті самі, що й у тексті ``preprocessing_summary.csv``.

        Args:
            metrics: Вміст ``metrics.json`` після виконання попередніх кроків.

        Returns:
            Словник «ключ рішення → поля :class:`Decision`» (без тексту).
        """
        return {decision.key: decision.to_metrics() for decision in self.decisions(metrics)}

    def decisions(self, metrics: Mapping[str, Any]) -> list[Decision]:
        """Усі рішення сценарію FULL, перевірені за єдиним правилом вибору."""
        return [
            self._collinearity_decision(metrics["correlation"]),
            *self._imputer_decisions(metrics["imputation"]),
            self._categorical_decision(metrics["imputation"]),
            *self._encoding_decisions(metrics["encoding"]),
            *self._feature_group_decisions(metrics["feature_engineering"]),
        ]

    def _collinearity_row(self, correlation: Mapping[str, Any]) -> tuple[str, str, str]:
        """Вилучення колінеарних ознак: дублікати та додаткове вилучення за правилом."""
        dropped = [self._schema.label(c) for c in self._schema.collinear_columns]
        extra = ", ".join(correlation["extra_drop_columns"])
        threshold = correlation["vif_threshold"]
        basis = (
            f"найбільший VIF {correlation['max_vif_before']:.0f} → "
            f"{correlation['max_vif_after']:.0f}; VIF > {threshold:.0f}: "
            f"{correlation['n_vif_above_threshold_before']} → "
            f"{correlation['n_vif_above_threshold_after']} ознак (якщо вилучити ще {extra}, "
            f"було б {correlation['n_vif_above_threshold_extra_drop']}, найбільший VIF "
            f"{correlation['max_vif_extra_drop']:.0f}); "
            + self._collinearity_decision(correlation).text
        )
        chosen = (
            f"вилучено: {', '.join(dropped)}" if self._options.drop_collinear else "не вилучаються"
        )
        return ("Мультиколінеарність", chosen, basis)

    def _collinearity_decision(self, correlation: Mapping[str, Any]) -> Decision:
        extra_columns = correlation["extra_drop_columns"]
        dropped = {self._schema.label(c) for c in self._schema.collinear_columns}
        extra_applied = self._options.drop_collinear and set(extra_columns) <= dropped
        extra = ", ".join(extra_columns)
        return _decide(
            key="extra_collinear_drop",
            simpler=f"без {extra}",
            more_complex=f"з {extra}",
            simpler_folds=correlation["pr_auc_folds_extra_drop"],
            complex_folds=correlation["pr_auc_folds_full"],
            n_features_simple=correlation["n_features_extra_drop"],
            n_features_complex=correlation["n_features_full"],
            complex_applied=not extra_applied,
        )

    def _numeric_imputation_row(self, imputation: Mapping[str, Any]) -> tuple[str, str, str]:
        """Імп'ютація числових ознак: обраний метод проти кожного іншого за правилом."""
        chosen = self._options.imputer
        basis = (
            f"RMSE відновлення: медіана {imputation['rmse_median']:.2f}, "
            f"KNN {imputation['rmse_knn']:.2f}, MICE {imputation['rmse_mice']:.2f}; "
            f"PR-AUC у FULL: {_IMPUTERS[chosen]} {imputation['pr_auc'][chosen.value]:.4f}; "
            + "; ".join(decision.text for decision in self._imputer_decisions(imputation))
        )
        return ("Імп'ютація числових", _IMPUTERS[chosen], basis)

    def _imputer_decisions(self, imputation: Mapping[str, Any]) -> list[Decision]:
        chosen = self._options.imputer
        folds = imputation["pr_auc_folds"]
        n_features = imputation["n_features"]
        decisions = []
        for other in ImputerKind:
            if other is chosen:
                continue
            # Простіший — той, що раніше в _IMPUTER_COMPLEXITY (медіана < KNN < MICE).
            simpler, more_complex = sorted((chosen, other), key=_IMPUTER_COMPLEXITY.index)
            decisions.append(
                _decide(
                    key=f"imputer_{other.value}",
                    simpler=_IMPUTERS[simpler],
                    more_complex=_IMPUTERS[more_complex],
                    simpler_folds=folds[simpler.value],
                    complex_folds=folds[more_complex.value],
                    n_features_simple=n_features[simpler.value],
                    n_features_complex=n_features[more_complex.value],
                    complex_applied=more_complex is chosen,
                )
            )
        return decisions

    def _categorical_imputation_row(self, imputation: Mapping[str, Any]) -> tuple[str, str, str]:
        """Імп'ютація категоріальних ознак: «мода + індикатор» проти категорії MISSING."""
        chosen = self._options.categorical_imputation
        pr_auc = imputation["pr_auc_categorical"]
        mode, missing = CategoricalImputation.MODE, CategoricalImputation.MISSING_CATEGORY
        basis = (
            f"PR-AUC у FULL: {_CATEGORICAL[mode]} + індикатор {pr_auc[mode.value]:.4f}, "
            f"{_CATEGORICAL[missing]} {pr_auc[missing.value]:.4f}; "
            + self._categorical_decision(imputation).text
            + f"; частка класу 1 за пропуску WheelType "
            f"{imputation['wheel_type_missing_positive_rate']:.1%} проти "
            f"{imputation['wheel_type_present_positive_rate']:.1%}"
        )
        return ("Імп'ютація категоріальних", _CATEGORICAL[chosen], basis)

    def _categorical_decision(self, imputation: Mapping[str, Any]) -> Decision:
        """MISSING складніший: окрема колонка «_MISSING» у кожній ознаці з пропусками."""
        folds = imputation["pr_auc_folds_categorical"]
        n_features = imputation["n_features_categorical"]
        mode, missing = CategoricalImputation.MODE, CategoricalImputation.MISSING_CATEGORY
        return _decide(
            key="categorical_imputation",
            simpler=f"{_CATEGORICAL[mode]} + індикатор",
            more_complex=_CATEGORICAL[missing],
            simpler_folds=folds[mode.value],
            complex_folds=folds[missing.value],
            n_features_simple=n_features[mode.value],
            n_features_complex=n_features[missing.value],
            complex_applied=self._options.categorical_imputation is missing,
        )

    def _high_cardinality_row(self, encoding: Mapping[str, Any]) -> tuple[str, str, str]:
        """Кодування високої кардинальності: Target проти решти (One-Hot — свідомий виняток)."""
        chosen = self._options.high_cardinality_encoding
        pr_auc = encoding["pr_auc"]
        n_features = encoding["n_output_features"]
        listing = ", ".join(
            f"{_ENCODINGS[kind]} {pr_auc[kind.value]:.3f} ({n_features[kind.value]})"
            for kind in HighCardinalityEncoding
        )
        basis = f"PR-AUC у FULL (ознак на виході): {listing}; " + "; ".join(
            decision.text for decision in self._encoding_decisions(encoding)
        )
        return ("Кодування високої кардинальності", _ENCODINGS[chosen], basis)

    def _encoding_decisions(self, encoding: Mapping[str, Any]) -> list[Decision]:
        """Складність: Frequency і Ordinal простіші за Target, One-Hot — складніший."""
        chosen = self._options.high_cardinality_encoding
        folds = encoding["pr_auc_folds"]
        n_features = encoding["n_output_features"]
        decisions = []
        for other in HighCardinalityEncoding:
            if other is chosen:
                continue
            simpler, more_complex = sorted((chosen, other), key=_ENCODING_COMPLEXITY.index)
            decisions.append(
                _decide(
                    key=f"encoding_{other.value}",
                    simpler=_ENCODINGS[simpler],
                    more_complex=_ENCODINGS[more_complex],
                    simpler_folds=folds[simpler.value],
                    complex_folds=folds[more_complex.value],
                    n_features_simple=n_features[simpler.value],
                    n_features_complex=n_features[more_complex.value],
                    complex_applied=more_complex is chosen,
                )
            )
        return decisions

    def _feature_engineering_row(self, engineering: Mapping[str, Any]) -> tuple[str, str, str]:
        """Які групи нових ознак увійшли в FULL — кожна за правилом відносно FULL."""
        decisions = self._feature_group_decisions(engineering)
        names = self._feature_group_names()
        chosen = [names[group] for group in names if engineering["decisions"][group]["in_full"]]
        basis = (
            "; ".join(
                f"{names[group]}: {decision.text}"
                for group, decision in zip(names, decisions, strict=True)
            )
            + f". Підсумок FULL: {engineering['n_features_full']} ознак, PR-AUC "
            f"{engineering['pr_auc_full']:.4f} (σ ≈ {engineering['pr_auc_std_full']:.4f}); "
            f"у числовій частині {engineering['full_numeric_n_vif_above_threshold']} із "
            f"{engineering['full_numeric_n_features']} ознак мають VIF > 10, найбільший "
            f"{engineering['full_numeric_max_vif']:.0f} "
            f"({engineering['full_numeric_max_vif_feature']})"
        )
        return (
            "Інженерія ознак",
            "; ".join(chosen) if chosen else "нові ознаки не додаються",
            basis,
        )

    def _feature_group_names(self) -> dict[str, str]:
        row_name = (
            "комбінації, частини дати, індикатори ("
            + (", ".join(self._indicators) or "немає")
            + ")"
        )
        return {
            "row_features": row_name,
            "group_aggregates": "агрегації за категоріями",
            "polynomial": "поліноміальні",
        }

    def _feature_group_decisions(self, engineering: Mapping[str, Any]) -> list[Decision]:
        decisions = []
        for group in self._feature_group_names():
            decision = engineering["decisions"][group]
            in_full = decision["in_full"]
            full = (decision["pr_auc_folds_full"], engineering["n_features_full"])
            toggled = (decision["pr_auc_folds_toggled"], decision["n_features_toggled"])
            # Складніший варіант — той, де група Є; у FULL вона або є, або перемкнена.
            (with_folds, with_n), (without_folds, without_n) = (
                (full, toggled) if in_full else (toggled, full)
            )
            decisions.append(
                _decide(
                    key=f"feature_group_{group}",
                    simpler="без групи",
                    more_complex="з групою",
                    simpler_folds=without_folds,
                    complex_folds=with_folds,
                    n_features_simple=without_n,
                    n_features_complex=with_n,
                    complex_applied=in_full,
                )
            )
        return decisions


@dataclass(frozen=True)
class Decision:
    """Одне рішення сценарію FULL, перевірене за єдиним правилом вибору.

    Attributes:
        key: Ключ рішення в ``metrics.json`` (розділ ``selection``).
        simpler: Назва простішого варіанта.
        more_complex: Назва складнішого варіанта.
        delta: Δ = mean(d), d_k = PR-AUC складнішого − PR-AUC простішого по фолдах.
        spread: s = std(d, ddof=1).
        n_features_simple: Ознак на вході моделі в простішому варіанті.
        n_features_complex: Ознак у складнішому варіанті.
        chosen: Назва варіанта, застосованого в коді.
        rule_verdict: Назва варіанта, який радить правило.
        exception: Розбіжність із правилом — явно оголошений свідомий виняток.
        exception_reason: Обґрунтування винятку (порожньо, якщо винятку немає).
    """

    key: str
    simpler: str
    more_complex: str
    delta: float
    spread: float
    n_features_simple: int
    n_features_complex: int
    chosen: str
    rule_verdict: str
    exception: bool
    exception_reason: str

    @property
    def text(self) -> str:
        """Фраза для таблиці: «Δ, s, ознак → що радить правило [виняток/УВАГА]»."""
        relation = "Δ > s" if self.delta > self.spread else "Δ ≤ s"
        text = (
            f"{self.more_complex} ({self.n_features_complex} ознак) проти {self.simpler} "
            f"({self.n_features_simple} ознак): Δ = {self.delta:+.4f}, s = {self.spread:.4f} "
            f"({relation}) → за правилом {self.rule_verdict}"
        )
        if self.exception:
            text += f"; обрано {self.chosen} — свідомий ВИНЯТОК із правила: {self.exception_reason}"
        elif self.chosen != self.rule_verdict:
            text += f" (УВАГА: у коді застосовано {self.chosen})"
        return text

    def to_metrics(self) -> dict[str, Any]:
        """Поля рішення для ``metrics.json`` (без тексту таблиці)."""
        return {
            "simpler": self.simpler,
            "more_complex": self.more_complex,
            "delta": self.delta,
            "spread": self.spread,
            "n_features_simple": self.n_features_simple,
            "n_features_complex": self.n_features_complex,
            "chosen": self.chosen,
            "rule_verdict": self.rule_verdict,
            "exception": self.exception,
        }


#: Свідомі винятки з правила вибору: «ключ рішення → обґрунтування». Лише для
#: цих рішень розбіжність із правилом позначається як виняток, а не «УВАГА».
_DECLARED_EXCEPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "encoding_onehot": (
            "завдання вимагає для ознак високої кардинальності Target/Frequency Encoding, "
            "а ознак на виході в кілька разів менше"
        ),
        "categorical_imputation": (
            "з модою інформація про невипадковий пропуск WheelType (найсильніший предиктор) "
            "існує лише в індикаторі з іншої групи ознак, і будь-яка конфігурація без цієї "
            "групи (база R14/R15, сценарії ЛР2–3) її втрачає; "
            "MISSING тримає інформацію в самій ознаці, і рішення не залежать одне від одного; "
            "ціна — кілька додаткових ознак при різниці в межах шуму"
        ),
    }
)


def _decide(
    key: str,
    simpler: str,
    more_complex: str,
    simpler_folds: Sequence[float],
    complex_folds: Sequence[float],
    n_features_simple: int,
    n_features_complex: int,
    complex_applied: bool,
) -> Decision:
    """Застосовує єдине правило вибору до пари варіантів.

    Правило (``mllabs.preprocessing.selection``): на тих самих фолдах
    d_k = PR-AUC складнішого − PR-AUC простішого; Δ = mean(d), s = std(d, ddof=1);
    складніший обирається, лише якщо Δ > s, інакше — простіший. Розбіжність
    коду з правилом — свідомий виняток, якщо ключ є в ``_DECLARED_EXCEPTIONS``.
    """
    difference = paired_difference(simpler_folds, complex_folds)
    rule_verdict = more_complex if difference.complex_wins else simpler
    chosen = more_complex if complex_applied else simpler
    is_exception = chosen != rule_verdict and key in _DECLARED_EXCEPTIONS
    return Decision(
        key=key,
        simpler=simpler,
        more_complex=more_complex,
        delta=difference.delta,
        spread=difference.spread,
        n_features_simple=n_features_simple,
        n_features_complex=n_features_complex,
        chosen=chosen,
        rule_verdict=rule_verdict,
        exception=is_exception,
        exception_reason=_DECLARED_EXCEPTIONS[key] if is_exception else "",
    )


def _quality(method_metrics: Mapping[str, float]) -> str:
    """Форматує пару «PR-AUC / recall» одного методу аугментації."""
    return f"{method_metrics['pr_auc']:.3f} / {method_metrics['recall']:.2f}"
