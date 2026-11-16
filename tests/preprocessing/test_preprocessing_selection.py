"""Тести єдиного правила вибору варіантів препроцесингу (парні різниці по фолдах)."""

import numpy as np
import pytest

from mllabs.preprocessing import FoldMismatchError, PairedDifference, paired_difference


def test_paired_difference_uses_fold_by_fold_differences_and_sample_std():
    simpler = [0.40, 0.42, 0.44, 0.41, 0.43]
    more_complex = [0.41, 0.44, 0.45, 0.43, 0.44]

    result = paired_difference(simpler, more_complex)

    differences = np.array([0.01, 0.02, 0.01, 0.02, 0.01])
    assert result.delta == pytest.approx(differences.mean())
    assert result.spread == pytest.approx(differences.std(ddof=1))  # вибіркове, не ddof=0
    assert result.spread > differences.std(ddof=0)
    assert result.n_folds == 5


def test_common_fold_difficulty_cancels_out_in_paired_difference():
    # Фолди дуже різні за складністю, але складніший варіант стабільно кращий на 0.01:
    # σ кожного варіанта велике, а розкид різниць — нульовий, тож правило його обирає.
    simpler = [0.30, 0.50, 0.40, 0.60, 0.35]
    more_complex = [value + 0.01 for value in simpler]

    result = paired_difference(simpler, more_complex)

    assert result.spread == pytest.approx(0.0, abs=1e-12)
    assert result.complex_wins


def test_simpler_variant_is_kept_when_delta_equals_spread():
    assert not PairedDifference(delta=0.004, spread=0.004, n_folds=5).complex_wins
    assert PairedDifference(delta=0.0041, spread=0.004, n_folds=5).complex_wins
    assert not PairedDifference(delta=-0.01, spread=0.001, n_folds=5).complex_wins


def test_paired_difference_rejects_different_fold_counts():
    with pytest.raises(FoldMismatchError):
        paired_difference([0.4, 0.5, 0.6], [0.4, 0.5])
    with pytest.raises(FoldMismatchError):
        paired_difference([0.4], [0.5])
