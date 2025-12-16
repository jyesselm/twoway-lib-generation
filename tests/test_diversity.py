"""Tests for diversity module."""

import numpy as np

from twoway_lib.diversity import (
    calculate_diversity_score,
    compute_distance_matrix,
    diversity_from_matrix,
    edit_distance,
    min_distance_to_set,
    parallel_distance_matrix,
)


class TestEditDistance:
    """Tests for edit_distance function."""

    def test_identical_sequences(self):
        assert edit_distance("GGGAAACCC", "GGGAAACCC") == 0

    def test_single_substitution(self):
        assert edit_distance("GGGAAACCC", "GGGAAACUC") == 1

    def test_single_insertion(self):
        assert edit_distance("GGGAAACCC", "GGGAAAACCC") == 1

    def test_single_deletion(self):
        assert edit_distance("GGGAAACCC", "GGGAACCC") == 1

    def test_completely_different(self):
        assert edit_distance("AAAA", "CCCC") == 4

    def test_empty_sequences(self):
        assert edit_distance("", "") == 0
        assert edit_distance("AAA", "") == 3
        assert edit_distance("", "AAA") == 3


class TestMinDistanceToSet:
    """Tests for min_distance_to_set function."""

    def test_empty_set(self):
        assert min_distance_to_set("AAAA", []) == 0

    def test_single_sequence(self):
        dist = min_distance_to_set("AAAA", ["AAAC"])
        assert dist == 1

    def test_multiple_sequences(self):
        dist = min_distance_to_set("AAAA", ["CCCC", "AAAC", "GGGG"])
        assert dist == 1

    def test_identical_in_set(self):
        dist = min_distance_to_set("AAAA", ["CCCC", "AAAA", "GGGG"])
        assert dist == 0


class TestCalculateDiversityScore:
    """Tests for calculate_diversity_score function."""

    def test_empty_list(self):
        assert calculate_diversity_score([]) == 0.0

    def test_single_sequence(self):
        assert calculate_diversity_score(["AAAA"]) == 0.0

    def test_identical_sequences(self):
        score = calculate_diversity_score(["AAAA", "AAAA", "AAAA"])
        assert score == 0.0

    def test_different_sequences(self):
        score = calculate_diversity_score(["AAAA", "CCCC", "GGGG"])
        assert score > 0

    def test_higher_diversity_higher_score(self):
        low_div = calculate_diversity_score(["AAAA", "AAAC", "AACC"])
        high_div = calculate_diversity_score(["AAAA", "CCCC", "GGGG"])
        assert high_div > low_div


class TestComputeDistanceMatrix:
    """Tests for compute_distance_matrix function."""

    def test_empty_list(self):
        matrix = compute_distance_matrix([])
        assert matrix.shape == (0, 0)

    def test_single_sequence(self):
        matrix = compute_distance_matrix(["AAAA"])
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 0

    def test_symmetric(self):
        matrix = compute_distance_matrix(["AAAA", "CCCC", "GGGG"])
        assert np.array_equal(matrix, matrix.T)

    def test_diagonal_zeros(self):
        matrix = compute_distance_matrix(["AAAA", "CCCC", "GGGG"])
        assert np.all(np.diag(matrix) == 0)

    def test_correct_distances(self):
        matrix = compute_distance_matrix(["AAAA", "AAAC"])
        assert matrix[0, 1] == 1
        assert matrix[1, 0] == 1


class TestParallelDistanceMatrix:
    """Tests for parallel_distance_matrix function."""

    def test_small_list_uses_serial(self):
        sequences = ["AAAA", "CCCC", "GGGG"]
        matrix = parallel_distance_matrix(sequences)
        serial_matrix = compute_distance_matrix(sequences)
        assert np.array_equal(matrix, serial_matrix)

    def test_returns_correct_shape(self):
        sequences = ["AAAA", "CCCC", "GGGG", "UUUU"]
        matrix = parallel_distance_matrix(sequences, n_workers=2)
        assert matrix.shape == (4, 4)


class TestDiversityFromMatrix:
    """Tests for diversity_from_matrix function."""

    def test_all_indices(self):
        sequences = ["AAAA", "CCCC", "GGGG"]
        matrix = compute_distance_matrix(sequences)
        score = diversity_from_matrix(matrix)
        expected = calculate_diversity_score(sequences)
        assert abs(score - expected) < 0.01

    def test_subset_indices(self):
        sequences = ["AAAA", "AAAC", "CCCC", "GGGG"]
        matrix = compute_distance_matrix(sequences)
        score = diversity_from_matrix(matrix, indices=[0, 2, 3])
        subset_score = calculate_diversity_score(["AAAA", "CCCC", "GGGG"])
        assert abs(score - subset_score) < 0.01

    def test_single_index(self):
        matrix = np.zeros((3, 3))
        score = diversity_from_matrix(matrix, indices=[0])
        assert score == 0.0
