"""Tests for helix module."""

from random import Random

import pytest

from twoway_lib.helix import (
    ALL_PAIRS,
    WOBBLE_PAIRS,
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


class TestWobblePairs:
    """Tests for G-U wobble pair support."""

    def test_wobble_pairs_constant(self):
        assert WOBBLE_PAIRS == [("G", "U"), ("U", "G")]

    def test_all_pairs_includes_wobble(self):
        assert len(ALL_PAIRS) == 6
        assert ("G", "U") in ALL_PAIRS
        assert ("U", "G") in ALL_PAIRS

    def test_generate_all_helices_with_wobble(self):
        # Without wobble: 4^1 = 4
        helices_wc = generate_all_helices(1, allow_wobble=False)
        assert len(helices_wc) == 4

        # With wobble: 6^1 = 6
        helices_wobble = generate_all_helices(1, allow_wobble=True)
        assert len(helices_wobble) == 6

    def test_generate_all_helices_length_2_with_wobble(self):
        # With wobble: 6^2 = 36
        helices = generate_all_helices(2, allow_wobble=True)
        assert len(helices) == 36

    def test_random_helix_with_wobble(self):
        rng = Random(42)
        # Generate many helices with wobble to check GU pairs can appear
        strands1 = set()
        strands2 = set()
        for _ in range(100):
            helix = random_helix(3, rng, allow_wobble=True)
            strands1.add(helix.strand1)
            strands2.add(helix.strand2)

        # With wobble pairs, we should see G-U combinations
        # Strand1 can have G pairing with U in strand2
        all_chars1 = "".join(strands1)
        all_chars2 = "".join(strands2)
        assert "G" in all_chars1 and "U" in all_chars2

    def test_random_helix_without_wobble_no_gu_pairs(self):
        rng = Random(42)
        # Generate helices without wobble - no G-U base pairs
        for _ in range(50):
            helix = random_helix(3, rng, allow_wobble=False)
            # Each position should have WC pairs only
            for s1, s2 in zip(helix.strand1, reversed(helix.strand2), strict=True):
                pair = (s1, s2)
                assert pair not in WOBBLE_PAIRS

    def test_wobble_helix_structure(self):
        # G-U pairs should still use ( and ) structure
        helices = generate_all_helices(2, allow_wobble=True)
        for h in helices:
            assert h.structure1 == "(("
            assert h.structure2 == "))"


class TestHasWobblePair:
    """Tests for has_wobble_pair function."""

    def test_wc_helix_no_wobble(self):
        from twoway_lib.helix import has_wobble_pair, helix_from_strand1

        helix = helix_from_strand1("AGC")
        assert has_wobble_pair(helix) is False

    def test_wobble_helix_detected(self):
        from twoway_lib.helix import has_wobble_pair

        # G-U pair: strand1='G', strand2 reversed -> need strand2 reversed has 'U'
        helix = Helix.from_sequences("G", "U", "(", ")")
        assert has_wobble_pair(helix) is True

    def test_mixed_helix(self):
        from twoway_lib.helix import has_wobble_pair

        # AGG strand1, strand2 reversed = 'U' at pos 2 paired with 'G' at pos 0
        helix = Helix.from_sequences("AGG", "CCU", "(((", ")))")
        # pos 0: A paired with reversed strand2[2]=U -> (A,U) WC
        # pos 1: G paired with reversed strand2[1]=C -> (G,C) WC
        # pos 2: G paired with reversed strand2[0]=C -> (G,C) WC
        assert has_wobble_pair(helix) is False


class TestRandomHelixWithGuRequirement:
    """Tests for random_helix_with_gu_requirement function."""

    def test_has_gu_pair(self):
        from twoway_lib.helix import has_wobble_pair, random_helix_with_gu_requirement

        rng = Random(42)
        for _ in range(20):
            helix = random_helix_with_gu_requirement(3, rng, require_gu=True)
            assert has_wobble_pair(helix) is True

    def test_correct_length(self):
        from twoway_lib.helix import random_helix_with_gu_requirement

        rng = Random(42)
        helix = random_helix_with_gu_requirement(5, rng)
        assert helix.length() == 5

    def test_no_gu_required(self):
        from twoway_lib.helix import random_helix_with_gu_requirement

        rng = Random(42)
        helix = random_helix_with_gu_requirement(3, rng, require_gu=False)
        assert helix.length() == 3

    def test_invalid_length(self):
        from twoway_lib.helix import random_helix_with_gu_requirement

        with pytest.raises(ValueError):
            random_helix_with_gu_requirement(0, Random(42))

    def test_length_1_has_gu(self):
        from twoway_lib.helix import has_wobble_pair, random_helix_with_gu_requirement

        rng = Random(42)
        helix = random_helix_with_gu_requirement(1, rng, require_gu=True)
        assert has_wobble_pair(helix) is True
