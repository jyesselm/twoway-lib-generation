"""Tests for validation module."""

import pytest

from twoway_lib.construct import Construct
from twoway_lib.motif import Motif
from twoway_lib.validation import (
    FoldResult,
    ValidationResult,
    compare_structures,
    count_structure_differences,
    fold_sequence,
    validate_construct,
    validate_structure_format,
)


class TestFoldSequence:
    """Tests for fold_sequence function."""

    def test_returns_fold_result(self):
        result = fold_sequence("GCGCGCGCGC")
        assert isinstance(result, FoldResult)
        assert result.sequence == "GCGCGCGCGC"

    def test_predicted_structure_length(self):
        seq = "GGGAAACCC"
        result = fold_sequence(seq)
        assert len(result.predicted_structure) == len(seq)

    def test_returns_mfe(self):
        result = fold_sequence("GCGCGCGCGC")
        assert isinstance(result.mfe, float)

    def test_returns_ensemble_defect(self):
        result = fold_sequence("GCGCGCGCGC")
        assert isinstance(result.ensemble_defect, float)
        assert result.ensemble_defect >= 0


class TestCompareStructures:
    """Tests for compare_structures function."""

    def test_identical_structures(self):
        assert compare_structures("(((...)))", "(((...))).") == 0.0
        assert compare_structures("(((...)))", "(((...)))") == 1.0

    def test_partial_match(self):
        match = compare_structures("(((...)))", "((.....))")
        assert 0 < match < 1

    def test_no_match(self):
        match = compare_structures("((((((((((", "))))))))))")
        assert match == 0.0

    def test_different_lengths(self):
        assert compare_structures("(((", "((((") == 0.0

    def test_empty_structures(self):
        assert compare_structures("", "") == 1.0


class TestCountStructureDifferences:
    """Tests for count_structure_differences function."""

    def test_identical(self):
        counts = count_structure_differences("(((...)))", "(((...)))")
        assert counts["total_mismatches"] == 0

    def test_length_mismatch(self):
        counts = count_structure_differences("(((", "((((")
        assert "length_mismatch" in counts

    def test_unpaired_to_paired(self):
        counts = count_structure_differences("...", "(()")
        assert counts["unpaired_to_paired"] == 3

    def test_paired_to_unpaired(self):
        counts = count_structure_differences("(()", "...")
        assert counts["paired_to_unpaired"] == 3

    def test_bracket_swap(self):
        counts = count_structure_differences("(((", ")))")
        assert counts["bracket_swap"] == 3


class TestValidateConstruct:
    """Tests for validate_construct function."""

    @pytest.fixture
    def simple_construct(self):
        """A simple construct for testing."""
        motif = Motif.from_string("GAC&GC", "(.(&))")
        return Construct(
            sequence="GGGAAACCC",
            structure="(((...)))",
            motifs=[motif],
        )

    def test_returns_validation_result(self, simple_construct):
        result = validate_construct(
            simple_construct,
            max_ensemble_defect=10.0,
            allow_differences=True,
        )
        assert isinstance(result, ValidationResult)

    def test_ensemble_defect_threshold(self, simple_construct):
        result = validate_construct(
            simple_construct,
            max_ensemble_defect=0.001,
            allow_differences=True,
        )
        assert result.is_valid is False
        assert "Ensemble defect" in result.reason

    def test_strict_structure_match(self):
        construct = Construct(
            sequence="AAAAAAAAAA",
            structure="((((()))))",
            motifs=[],
        )
        result = validate_construct(
            construct,
            max_ensemble_defect=100.0,
            allow_differences=False,
        )
        assert result.is_valid is False or result.structure_match == 1.0

    def test_allows_differences(self):
        construct = Construct(
            sequence="GGGAAACCC",
            structure="(((...)))",
            motifs=[],
        )
        result = validate_construct(
            construct,
            max_ensemble_defect=100.0,
            allow_differences=True,
            min_structure_match=0.5,
        )
        assert result.fold_result is not None

    def test_returns_fold_result(self, simple_construct):
        result = validate_construct(
            simple_construct,
            max_ensemble_defect=100.0,
            allow_differences=True,
        )
        assert result.fold_result is not None
        assert result.fold_result.sequence == simple_construct.sequence


class TestValidateStructureFormat:
    """Tests for validate_structure_format function."""

    def test_valid_structure(self):
        is_valid, error = validate_structure_format("GGGAAACCC", "(((...)))")
        assert is_valid is True
        assert error == ""

    def test_length_mismatch(self):
        is_valid, error = validate_structure_format("GGGAAA", "(((...)))")
        assert is_valid is False
        assert "length" in error.lower()

    def test_unbalanced_brackets(self):
        is_valid, error = validate_structure_format("GGGAAACCC", "(((...))")
        assert is_valid is False

    def test_valid_multi_strand(self):
        is_valid, error = validate_structure_format("GAC&GC", "(.(&))")
        assert is_valid is True
        assert error == ""

    def test_strand_separator_mismatch(self):
        is_valid, error = validate_structure_format("GAC&GC", "(.(.))")
        assert is_valid is False
        assert "strand" in error.lower()
