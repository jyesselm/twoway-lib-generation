"""Tests for construct module."""

import pytest

from twoway_lib.construct import (
    Construct,
    MotifPosition,
    assemble_construct,
    calculate_construct_length,
    estimate_construct_length,
)
from twoway_lib.hairpin import Hairpin
from twoway_lib.helix import Helix
from twoway_lib.motif import Motif


class TestConstruct:
    """Tests for Construct dataclass."""

    def test_length(self, sample_construct: Construct):
        assert sample_construct.length() == len(sample_construct.sequence)

    def test_motifs_list(self, sample_construct: Construct):
        assert len(sample_construct.motifs) == 1

    def test_is_valid_true(self):
        construct = Construct(
            sequence="GGGAAACCC",
            structure="(((...)))",
            motifs=[],
        )
        assert construct.is_valid() is True

    def test_is_valid_false_length_mismatch(self):
        construct = Construct(
            sequence="GGGAAA",
            structure="(((...)))",
            motifs=[],
        )
        assert construct.is_valid() is False

    def test_validate_raises_on_invalid(self):
        construct = Construct(
            sequence="GGGAAA",
            structure="(((...)))",
            motifs=[],
        )
        with pytest.raises(ValueError):
            construct.validate()

    def test_to_secstruct(self):
        from rna_secstruct import SecStruct
        construct = Construct(
            sequence="GGGAAACCC",
            structure="(((...)))",
            motifs=[],
        )
        ss = construct.to_secstruct()
        assert isinstance(ss, SecStruct)
        assert ss.sequence == "GGGAAACCC"
        assert ss.structure == "(((...)))"


class TestAssembleConstruct:
    """Tests for assemble_construct function."""

    @pytest.fixture
    def assembly_components(self):
        """Components for construct assembly."""
        motifs = [
            Motif.from_string("GAC&GC", "(.(&))"),
            Motif.from_string("AAG&CUU", "(.(&.))"),
        ]
        helices = [
            Helix(strand1="AGC", strand2="GCU", structure1="(((", structure2=")))"),
            Helix(strand1="UAG", strand2="CUA", structure1="(((", structure2=")))"),
            Helix(strand1="GGC", strand2="GCC", structure1="(((", structure2=")))"),
        ]
        hairpin = Hairpin(sequence="GAAA", structure="....")
        return motifs, helices, hairpin

    def test_assemble_basic(self, assembly_components):
        motifs, helices, hairpin = assembly_components
        construct = assemble_construct(
            motifs=motifs,
            helices=helices,
            hairpin=hairpin,
            p5_seq="GG",
            p5_ss="((",
            p3_seq="CC",
            p3_ss="))",
        )
        assert isinstance(construct, Construct)
        assert len(construct.motifs) == 2

    def test_sequence_contains_components(self, assembly_components):
        motifs, helices, hairpin = assembly_components
        construct = assemble_construct(
            motifs=motifs,
            helices=helices,
            hairpin=hairpin,
            p5_seq="GG",
            p5_ss="((",
            p3_seq="CC",
            p3_ss="))",
        )
        assert "GG" in construct.sequence
        assert "CC" in construct.sequence
        assert "GAC" in construct.sequence
        assert "GAAA" in construct.sequence

    def test_structure_contains_components(self, assembly_components):
        motifs, helices, hairpin = assembly_components
        construct = assemble_construct(
            motifs=motifs,
            helices=helices,
            hairpin=hairpin,
            p5_seq="GG",
            p5_ss="((",
            p3_seq="CC",
            p3_ss="))",
        )
        assert "((" in construct.structure
        assert "))" in construct.structure
        assert "...." in construct.structure

    def test_wrong_number_of_helices(self, assembly_components):
        motifs, helices, hairpin = assembly_components
        with pytest.raises(ValueError, match="Need 3 helices"):
            assemble_construct(
                motifs=motifs,
                helices=helices[:2],
                hairpin=hairpin,
                p5_seq="GG",
                p5_ss="((",
                p3_seq="CC",
                p3_ss="))",
            )

    def test_sequence_structure_same_length(self, assembly_components):
        motifs, helices, hairpin = assembly_components
        construct = assemble_construct(
            motifs=motifs,
            helices=helices,
            hairpin=hairpin,
            p5_seq="GG",
            p5_ss="((",
            p3_seq="CC",
            p3_ss="))",
        )
        assert len(construct.sequence) == len(construct.structure)


class TestCalculateConstructLength:
    """Tests for calculate_construct_length function."""

    def test_basic_calculation(self):
        length = calculate_construct_length(
            num_motifs=2,
            motif_lengths=[5, 6],
            helix_length=3,
            hairpin_length=4,
            p5_length=5,
            p3_length=5,
        )
        expected = 5 + 5 + 5 + 6 + 3 * 3 * 2 + 4
        assert length == expected

    def test_single_motif(self):
        length = calculate_construct_length(
            num_motifs=1,
            motif_lengths=[4],
            helix_length=2,
            hairpin_length=3,
            p5_length=3,
            p3_length=3,
        )
        expected = 3 + 3 + 4 + 2 * 2 * 2 + 3
        assert length == expected


class TestEstimateConstructLength:
    """Tests for estimate_construct_length function."""

    def test_estimate_matches_calculate(self):
        avg_len = 5
        num = 3
        estimate = estimate_construct_length(
            num_motifs=num,
            avg_motif_length=avg_len,
            helix_length=3,
            hairpin_length=4,
            p5_length=5,
            p3_length=5,
        )
        actual = calculate_construct_length(
            num_motifs=num,
            motif_lengths=[avg_len] * num,
            helix_length=3,
            hairpin_length=4,
            p5_length=5,
            p3_length=5,
        )
        assert estimate == actual


class TestMotifPosition:
    """Tests for MotifPosition dataclass."""

    def test_all_positions(self):
        motif = Motif.from_string("GAC&GC", "(.(&))")
        mp = MotifPosition(motif=motif, strand1_positions=[5, 6, 7], strand2_positions=[20, 21])
        assert mp.all_positions() == [5, 6, 7, 20, 21]

    def test_to_string(self):
        motif = Motif.from_string("GAC&GC", "(.(&))")
        mp = MotifPosition(motif=motif, strand1_positions=[5, 6, 7], strand2_positions=[20, 21])
        assert mp.to_string() == "[5,6,7,20,21]"


class TestAssembleConstructPositions:
    """Tests for position tracking in assemble_construct."""

    @pytest.fixture
    def simple_assembly(self):
        """Simple assembly for position testing."""
        motifs = [Motif.from_string("GAC&GC", "(.(&))")]
        helices = [
            Helix(strand1="AA", strand2="UU", structure1="((", structure2="))"),
            Helix(strand1="CC", strand2="GG", structure1="((", structure2="))"),
        ]
        hairpin = Hairpin(sequence="GAAA", structure="....")
        return motifs, helices, hairpin

    def test_has_motif_positions(self, simple_assembly):
        motifs, helices, hairpin = simple_assembly
        construct = assemble_construct(
            motifs=motifs,
            helices=helices,
            hairpin=hairpin,
            p5_seq="G",
            p5_ss="(",
            p3_seq="C",
            p3_ss=")",
        )
        assert len(construct.motif_positions) == 1

    def test_strand1_positions_correct(self, simple_assembly):
        motifs, helices, hairpin = simple_assembly
        construct = assemble_construct(
            motifs=motifs,
            helices=helices,
            hairpin=hairpin,
            p5_seq="G",
            p5_ss="(",
            p3_seq="C",
            p3_ss=")",
        )
        # Layout: G + AA + GAC + CC + GAAA + GG + GC + UU + C
        # Positions:0   1-2   3-5   6-7  8-11  12-13 14-15 16-17 18
        # Motif strand1 (GAC) should be at positions 3, 4, 5
        mp = construct.motif_positions[0]
        assert mp.strand1_positions == [3, 4, 5]

    def test_strand2_positions_correct(self, simple_assembly):
        motifs, helices, hairpin = simple_assembly
        construct = assemble_construct(
            motifs=motifs,
            helices=helices,
            hairpin=hairpin,
            p5_seq="G",
            p5_ss="(",
            p3_seq="C",
            p3_ss=")",
        )
        # Layout: G + AA + GAC + CC + GAAA + GG + GC + UU + C
        # Motif strand2 (GC) should be at positions 14, 15
        mp = construct.motif_positions[0]
        assert mp.strand2_positions == [14, 15]

    def test_positions_match_sequence(self, simple_assembly):
        motifs, helices, hairpin = simple_assembly
        construct = assemble_construct(
            motifs=motifs,
            helices=helices,
            hairpin=hairpin,
            p5_seq="G",
            p5_ss="(",
            p3_seq="C",
            p3_ss=")",
        )
        mp = construct.motif_positions[0]
        # Verify strand1 positions contain correct nucleotides
        strand1_from_pos = "".join(construct.sequence[i] for i in mp.strand1_positions)
        assert strand1_from_pos == motifs[0].strand1_seq
        # Verify strand2 positions contain correct nucleotides
        strand2_from_pos = "".join(construct.sequence[i] for i in mp.strand2_positions)
        assert strand2_from_pos == motifs[0].strand2_seq

    def test_get_positions_string(self, simple_assembly):
        motifs, helices, hairpin = simple_assembly
        construct = assemble_construct(
            motifs=motifs,
            helices=helices,
            hairpin=hairpin,
            p5_seq="G",
            p5_ss="(",
            p3_seq="C",
            p3_ss=")",
        )
        pos_str = construct.get_positions_string()
        assert "M1:" in pos_str
        assert "[3,4,5,14,15]" in pos_str
