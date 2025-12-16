"""Tests for helix module."""

from random import Random

import pytest

from twoway_lib.helix import (
    Helix,
    generate_all_helices,
    helix_from_strand1,
    random_helix,
)


class TestHelix:
    """Tests for Helix dataclass."""

    def test_helix_length(self, sample_helix: Helix):
        assert sample_helix.length() == 3

    def test_helix_attributes(self, sample_helix: Helix):
        assert sample_helix.strand1 == "AGC"
        assert sample_helix.strand2 == "GCU"
        assert sample_helix.structure1 == "((("
        assert sample_helix.structure2 == ")))"

    def test_frozen_dataclass(self, sample_helix: Helix):
        with pytest.raises(AttributeError):
            sample_helix.strand1 = "AAA"


class TestGenerateAllHelices:
    """Tests for generate_all_helices function."""

    def test_length_1(self):
        helices = generate_all_helices(1)
        assert len(helices) == 4
        strands = {h.strand1 for h in helices}
        assert strands == {"A", "U", "G", "C"}

    def test_length_2(self):
        helices = generate_all_helices(2)
        assert len(helices) == 16

    def test_length_3(self):
        helices = generate_all_helices(3)
        assert len(helices) == 64

    def test_structure_correct(self):
        helices = generate_all_helices(2)
        for h in helices:
            assert h.structure1 == "(("
            assert h.structure2 == "))"

    def test_complement_correct(self):
        helices = generate_all_helices(1)
        for h in helices:
            if h.strand1 == "A":
                assert h.strand2 == "U"
            elif h.strand1 == "U":
                assert h.strand2 == "A"
            elif h.strand1 == "G":
                assert h.strand2 == "C"
            elif h.strand1 == "C":
                assert h.strand2 == "G"

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="at least 1"):
            generate_all_helices(0)


class TestRandomHelix:
    """Tests for random_helix function."""

    def test_correct_length(self):
        helix = random_helix(5)
        assert helix.length() == 5
        assert len(helix.strand1) == 5
        assert len(helix.strand2) == 5

    def test_correct_structure(self):
        helix = random_helix(4)
        assert helix.structure1 == "(((("
        assert helix.structure2 == "))))"

    def test_with_seed(self):
        rng1 = Random(42)
        rng2 = Random(42)
        h1 = random_helix(3, rng1)
        h2 = random_helix(3, rng2)
        assert h1.strand1 == h2.strand1

    def test_different_with_different_seeds(self):
        rng1 = Random(42)
        rng2 = Random(99)
        h1 = random_helix(5, rng1)
        h2 = random_helix(5, rng2)
        assert h1.strand1 != h2.strand1

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="at least 1"):
            random_helix(0)


class TestHelixFromStrand1:
    """Tests for helix_from_strand1 function."""

    def test_basic_complement(self):
        helix = helix_from_strand1("AGC")
        assert helix.strand1 == "AGC"
        assert helix.strand2 == "GCU"

    def test_all_nucleotides(self):
        helix = helix_from_strand1("AUGC")
        assert helix.strand2 == "GCAU"

    def test_structure(self):
        helix = helix_from_strand1("AAA")
        assert helix.structure1 == "((("
        assert helix.structure2 == ")))"

    def test_invalid_nucleotide(self):
        with pytest.raises(ValueError, match="Invalid nucleotide"):
            helix_from_strand1("AXG")
