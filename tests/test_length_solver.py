"""Tests for length_solver module."""

from random import Random

from twoway_lib.length_solver import (
    compute_helix_budget,
    is_combo_feasible,
    random_helix_assignment,
)


class TestComputeHelixBudget:
    """Tests for compute_helix_budget function."""

    def test_basic_computation(self):
        # target=50, motifs=[5,5], p5=5, p3=5, hairpin=4
        # fixed = 5 + 5 + 5 + 5 + 4 = 24
        # remaining = 50 - 24 = 26, budget = 26 / 2 = 13
        budget = compute_helix_budget(50, [5, 5], 5, 5, 4)
        assert budget == 13

    def test_with_spacers(self):
        # fixed = 5 + 5 + 10 + 4 + 3 + 2 = 29
        # remaining = 50 - 29 = 21, budget = 21 / 2 is not int => None
        budget = compute_helix_budget(50, [5, 5], 5, 5, 4, 3, 2)
        assert budget is None

        # fixed = 5 + 5 + 10 + 4 + 2 + 2 = 28
        # remaining = 50 - 28 = 22, budget = 11
        budget = compute_helix_budget(50, [5, 5], 5, 5, 4, 2, 2)
        assert budget == 11

    def test_impossible_negative(self):
        # target too small
        budget = compute_helix_budget(10, [5, 5], 5, 5, 4)
        assert budget is None

    def test_odd_remainder(self):
        # fixed = 5 + 5 + 5 + 4 = 19, remaining = 30 - 19 = 11 (odd)
        budget = compute_helix_budget(30, [5], 5, 5, 4)
        assert budget is None

    def test_zero_budget(self):
        # fixed = 5 + 5 + 10 + 4 = 24, target = 24, budget = 0
        budget = compute_helix_budget(24, [5, 5], 5, 5, 4)
        assert budget == 0

    def test_single_motif(self):
        budget = compute_helix_budget(40, [6], 5, 5, 4)
        # fixed = 5 + 5 + 6 + 4 = 20, remaining = 20, budget = 10
        assert budget == 10


class TestRandomHelixAssignment:
    """Tests for random_helix_assignment function."""

    def test_uniform_assignment(self):
        rng = Random(42)
        result = random_helix_assignment(9, 3, 3, 3, rng)
        assert result == (3, 3, 3)

    def test_variable_assignment(self):
        rng = Random(42)
        result = random_helix_assignment(10, 3, 2, 5, rng)
        assert result is not None
        assert sum(result) == 10
        assert len(result) == 3
        assert all(2 <= h <= 5 for h in result)

    def test_infeasible_too_small(self):
        rng = Random(42)
        # 3 helices, min 3 each = 9, but only 6 budget
        result = random_helix_assignment(6, 3, 3, 5, rng)
        assert result is None

    def test_infeasible_too_large(self):
        rng = Random(42)
        # 3 helices, max 3 each = 9, but 12 budget
        result = random_helix_assignment(12, 3, 2, 3, rng)
        assert result is None

    def test_single_helix(self):
        rng = Random(42)
        result = random_helix_assignment(4, 1, 2, 5, rng)
        assert result == (4,)

    def test_zero_helices(self):
        rng = Random(42)
        result = random_helix_assignment(0, 0, 2, 5, rng)
        assert result is None

    def test_reproducible_with_seed(self):
        result1 = random_helix_assignment(15, 4, 2, 5, Random(42))
        result2 = random_helix_assignment(15, 4, 2, 5, Random(42))
        assert result1 == result2

    def test_all_at_max(self):
        rng = Random(42)
        result = random_helix_assignment(15, 3, 5, 5, rng)
        assert result == (5, 5, 5)


class TestIsComboFeasible:
    """Tests for is_combo_feasible function."""

    def test_feasible(self):
        assert is_combo_feasible(
            motif_lengths=[5, 5],
            target_length=50,
            min_helix=2,
            max_helix=5,
            p5_len=5,
            p3_len=5,
            hairpin_len=4,
        )

    def test_infeasible_too_long(self):
        assert not is_combo_feasible(
            motif_lengths=[5, 5],
            target_length=200,
            min_helix=2,
            max_helix=3,
            p5_len=5,
            p3_len=5,
            hairpin_len=4,
        )

    def test_infeasible_too_short(self):
        assert not is_combo_feasible(
            motif_lengths=[5, 5],
            target_length=20,
            min_helix=2,
            max_helix=5,
            p5_len=5,
            p3_len=5,
            hairpin_len=4,
        )

    def test_with_spacers(self):
        assert is_combo_feasible(
            motif_lengths=[5, 5],
            target_length=54,
            min_helix=2,
            max_helix=5,
            p5_len=5,
            p3_len=5,
            hairpin_len=4,
            spacer_5p_len=2,
            spacer_3p_len=2,
        )

    def test_odd_remainder_infeasible(self):
        # fixed=19, target=30, remaining=11 (odd) -> not feasible
        assert not is_combo_feasible(
            motif_lengths=[5],
            target_length=30,
            min_helix=2,
            max_helix=10,
            p5_len=5,
            p3_len=5,
            hairpin_len=4,
        )
