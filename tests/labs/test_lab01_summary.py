"""Тести таблиці рішень ЛР1: правило вибору, свідомі винятки та позначка «УВАГА»."""

from mllabs.labs.lab01.summary import _decide

_SIMPLE = [0.40, 0.41, 0.42, 0.43, 0.44]


def _equal_quality() -> list[float]:
    # Δ = 0, s > 0: якість однакова, правило радить простіший.
    return [0.401, 0.409, 0.421, 0.429, 0.44]


def test_rule_picks_simpler_variant_and_complex_choice_is_flagged_when_not_declared():
    decision = _decide(
        key="feature_group_polynomial",
        simpler="без групи",
        more_complex="з групою",
        simpler_folds=_SIMPLE,
        complex_folds=_equal_quality(),
        n_features_simple=10,
        n_features_complex=12,
        complex_applied=True,
    )

    assert decision.rule_verdict == "без групи"
    assert not decision.exception
    assert "УВАГА" in decision.text


def test_declared_exception_is_marked_as_such_without_warning():
    decision = _decide(
        key="categorical_imputation",
        simpler="мода + індикатор",
        more_complex="окрема категорія MISSING",
        simpler_folds=_SIMPLE,
        complex_folds=_equal_quality(),
        n_features_simple=88,
        n_features_complex=92,
        complex_applied=True,
    )

    assert decision.exception
    assert "свідомий ВИНЯТОК" in decision.text
    assert "УВАГА" not in decision.text
    assert decision.to_metrics()["n_features_complex"] == 92


def test_declared_key_is_not_an_exception_when_code_follows_the_rule():
    decision = _decide(
        key="categorical_imputation",
        simpler="мода + індикатор",
        more_complex="окрема категорія MISSING",
        simpler_folds=_SIMPLE,
        complex_folds=_equal_quality(),
        n_features_simple=88,
        n_features_complex=92,
        complex_applied=False,
    )

    assert not decision.exception
    assert decision.chosen == decision.rule_verdict
