"""Tests for hairpin module."""

from random import Random

import pytest

from twoway_lib.hairpin import (
    Hairpin,
    generate_all_hairpins,
    get_stable_tetraloops,
    random_hairpin,
    random_stable_tetraloop,
)


class TestHairpin:
    """Tests for Hairpin dataclass."""

    def test_hairpin_length(self, sample_hairpin: Hairpin):
        assert sample_hairpin.length() == 4

    def test_hairpin_attributes(self, sample_hairpin: Hairpin):
        assert sample_hairpin.sequence == "GAAA"
        assert sample_hairpin.structure == "...."

    def test_frozen_dataclass(self, sample_hairpin: Hairpin):
        with pytest.raises(AttributeError):
            sample_hairpin.sequence = "AAAA"


class TestGenerateAllHairpins:
    """Tests for generate_all_hairpins function."""

    def test_length_3(self):
        hairpins = generate_all_hairpins(3)
        assert len(hairpins) == 64

    def test_length_4(self):
        hairpins = generate_all_hairpins(4)
        assert len(hairpins) == 256

    def test_structure_all_unpaired(self):
        hairpins = generate_all_hairpins(4)
        for h in hairpins:
            assert h.structure == "...."

    def test_sequences_correct_length(self):
        hairpins = generate_all_hairpins(5)
        for h in hairpins:
            assert len(h.sequence) == 5

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="at least 3"):
            generate_all_hairpins(2)


class TestRandomHairpin:
    """Tests for random_hairpin function."""

    def test_correct_length(self):
        hairpin = random_hairpin(5)
        assert hairpin.length() == 5

    def test_structure_all_unpaired(self):
        hairpin = random_hairpin(6)
        assert hairpin.structure == "......"

    def test_with_seed(self):
        rng1 = Random(42)
        rng2 = Random(42)
        h1 = random_hairpin(4, rng1)
        h2 = random_hairpin(4, rng2)
        assert h1.sequence == h2.sequence

    def test_different_with_different_seeds(self):
        rng1 = Random(42)
        rng2 = Random(99)
        h1 = random_hairpin(6, rng1)
        h2 = random_hairpin(6, rng2)
        assert h1.sequence != h2.sequence

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="at least 3"):
            random_hairpin(2)


class TestGetStableTetraloops:
    """Tests for get_stable_tetraloops function."""

    def test_returns_list(self):
        tetraloops = get_stable_tetraloops()
        assert isinstance(tetraloops, list)
        assert len(tetraloops) > 0

    def test_all_length_4(self):
        tetraloops = get_stable_tetraloops()
        for hp in tetraloops:
            assert hp.length() == 4

    def test_contains_common_tetraloops(self):
        tetraloops = get_stable_tetraloops()
        sequences = {hp.sequence for hp in tetraloops}
        assert "GAAA" in sequences
        assert "UUCG" in sequences


class TestRandomStableTetraloop:
    """Tests for random_stable_tetraloop function."""

    def test_returns_tetraloop(self):
        hp = random_stable_tetraloop()
        assert hp.length() == 4

    def test_is_from_stable_set(self):
        stable = get_stable_tetraloops()
        stable_seqs = {h.sequence for h in stable}
        hp = random_stable_tetraloop()
        assert hp.sequence in stable_seqs

    def test_with_seed(self):
        rng1 = Random(42)
        rng2 = Random(42)
        h1 = random_stable_tetraloop(rng1)
        h2 = random_stable_tetraloop(rng2)
        assert h1.sequence == h2.sequence
