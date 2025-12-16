"""Tests for motif module."""

from pathlib import Path

import pytest

from twoway_lib.motif import Motif, load_motifs, validate_motif


class TestMotif:
    """Tests for Motif dataclass."""

    def test_from_string_basic(self):
        motif = Motif.from_string("GAC&GC", "(.(&))")
        assert motif.sequence == "GAC&GC"
        assert motif.structure == "(.(&))"
        assert motif.strand1_seq == "GAC"
        assert motif.strand2_seq == "GC"
        assert motif.strand1_ss == "(.("
        assert motif.strand2_ss == "))"

    def test_from_string_converts_t_to_u(self):
        motif = Motif.from_string("GAT&GC", "(.(&))")
        assert motif.strand1_seq == "GAU"

    def test_from_string_uppercase(self):
        motif = Motif.from_string("gac&gc", "(.(&))")
        assert motif.strand1_seq == "GAC"

    def test_total_length(self):
        motif = Motif.from_string("GAC&GC", "(.(&))")
        assert motif.total_length() == 5

    def test_strand1_length(self):
        motif = Motif.from_string("GAC&GC", "(.(&))")
        assert motif.strand1_length() == 3

    def test_strand2_length(self):
        motif = Motif.from_string("GAC&GC", "(.(&))")
        assert motif.strand2_length() == 2

    def test_from_string_no_separator(self):
        with pytest.raises(ValueError, match="must contain '&' separator"):
            Motif.from_string("GACGC", "(.(&))")

    def test_from_string_no_structure_separator(self):
        with pytest.raises(ValueError, match="must contain '&' separator"):
            Motif.from_string("GAC&GC", "(.(.))")

    def test_from_string_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            Motif.from_string("GACAA&GC", "(.(&))")

    def test_from_string_invalid_nucleotide(self):
        with pytest.raises(ValueError, match="invalid nucleotides"):
            Motif.from_string("GAX&GC", "(.(&))")

    def test_from_string_invalid_structure(self):
        with pytest.raises(ValueError, match="invalid characters"):
            Motif.from_string("GAC&GC", "([(&))")

    def test_frozen_dataclass(self):
        motif = Motif.from_string("GAC&GC", "(.(&))")
        with pytest.raises(AttributeError):
            motif.sequence = "AAA&GC"


class TestLoadMotifs:
    """Tests for load_motifs function."""

    def test_load_valid_csv(self, temp_motifs_file: Path):
        motifs = load_motifs(temp_motifs_file)
        assert len(motifs) == 3
        assert motifs[0].sequence == "GAC&GC"

    def test_load_with_string_path(self, temp_motifs_file: Path):
        motifs = load_motifs(str(temp_motifs_file))
        assert len(motifs) == 3

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_motifs(Path("/nonexistent/motifs.csv"))

    def test_load_missing_columns(self, temp_dir: Path):
        csv_path = temp_dir / "bad.csv"
        csv_path.write_text("name,seq\ntest,AAA")
        with pytest.raises(ValueError, match="missing required columns"):
            load_motifs(csv_path)


class TestValidateMotif:
    """Tests for validate_motif function."""

    def test_valid_motif(self, sample_motif: Motif):
        validate_motif(sample_motif)

    def test_unbalanced_parentheses(self):
        # 3 opens in strand1, only 2 closes in strand2
        motif = Motif.from_string("GAC&GC", "(((&))")
        with pytest.raises(ValueError, match="Unbalanced parentheses"):
            validate_motif(motif)

    def test_closing_bracket_in_strand1(self):
        # strand1 = ".)(" has ')' which is not allowed for two-way junctions
        # but overall structure is balanced (1 open, 1 close)
        motif = Motif.from_string("GAC&GC", ".)(&.)")
        with pytest.raises(ValueError, match="Strand 1 should only have"):
            validate_motif(motif)

    def test_opening_bracket_in_strand2(self):
        # strand2 has '(' which is not allowed for two-way junctions
        # Character check now happens before balance check
        motif = Motif.from_string("GAC&GCC", "(.(&(.)")
        with pytest.raises(ValueError, match="Strand 2 should only have"):
            validate_motif(motif)


class TestMotifFlip:
    """Tests for Motif.flip method."""

    def test_flip_swaps_strands(self):
        # GGA&CCU with ((.&.)) -> strand1=GGA/((., strand2=CCU/.))
        motif = Motif.from_string("GGA&CCU", "((.&.))")
        flipped = motif.flip()
        assert flipped.strand1_seq == "CCU"
        assert flipped.strand2_seq == "GGA"

    def test_flip_swaps_structures(self):
        # After flip: strand1 gets old strand2's structure, strand2 gets old strand1's
        motif = Motif.from_string("GGA&CCU", "((.&.))")
        flipped = motif.flip()
        assert flipped.strand1_ss == ".))"  # was strand2_ss
        assert flipped.strand2_ss == "((."  # was strand1_ss

    def test_flip_updates_sequence(self):
        motif = Motif.from_string("GGA&CCU", "((.&.))")
        flipped = motif.flip()
        assert flipped.sequence == "CCU&GGA"

    def test_flip_updates_structure(self):
        motif = Motif.from_string("GGA&CCU", "((.&.))")
        flipped = motif.flip()
        assert flipped.structure == ".))&((."

    def test_flip_is_reversible(self):
        motif = Motif.from_string("GGA&CCU", "((.&.))")
        double_flipped = motif.flip().flip()
        assert double_flipped.sequence == motif.sequence
        assert double_flipped.structure == motif.structure

    def test_flip_with_unpaired(self):
        # (.(&.)) -> strand1=(.( strand2=.))
        motif = Motif.from_string("GAC&GUC", "(.(&.))")
        flipped = motif.flip()
        assert flipped.sequence == "GUC&GAC"
        assert flipped.structure == ".))&(.("
