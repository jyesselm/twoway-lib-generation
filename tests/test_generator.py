"""Tests for generator module."""

import pytest

from twoway_lib.config import LibraryConfig, OptimizationConfig, ValidationConfig
from twoway_lib.generator import (
    LibraryGenerator,
    estimate_feasible_lengths,
    generate_library,
)
from twoway_lib.motif import Motif


@pytest.fixture
def generation_config() -> LibraryConfig:
    """Config for generation testing."""
    return LibraryConfig(
        target_length_min=40,
        target_length_max=100,
        motifs_per_construct_min=2,
        motifs_per_construct_max=3,
        p5_sequence="GGAAC",
        p5_structure="(((..",
        p3_sequence="GUUCC",
        p3_structure="..)))",
        helix_length=3,
        hairpin_loop_length=4,
        validation=ValidationConfig(
            enabled=False,
        ),
        optimization=OptimizationConfig(
            iterations=100,
            target_library_size=10,
        ),
    )


@pytest.fixture
def test_motifs() -> list[Motif]:
    """Motifs for testing."""
    return [
        Motif.from_string("GAC&GC", "(.(&))"),
        Motif.from_string("AAG&CUU", "(.(&.))"),
        Motif.from_string("UUG&CAA", "(.(&.))"),
        Motif.from_string("GGA&UCC", "(.(&.))"),
    ]


class TestLibraryGenerator:
    """Tests for LibraryGenerator class."""

    def test_init(self, generation_config, test_motifs):
        generator = LibraryGenerator(generation_config, test_motifs)
        assert generator.config == generation_config
        assert len(generator.motifs) == 4

    def test_generate_produces_constructs(self, generation_config, test_motifs):
        generator = LibraryGenerator(generation_config, test_motifs, seed=42)
        constructs = generator.generate(num_candidates=50)
        assert len(constructs) > 0

    def test_generate_respects_length_constraints(self, generation_config, test_motifs):
        generator = LibraryGenerator(generation_config, test_motifs, seed=42)
        constructs = generator.generate(num_candidates=50)
        for c in constructs:
            assert generation_config.target_length_min <= c.length()
            assert c.length() <= generation_config.target_length_max

    def test_generate_with_seed_reproducible(self, generation_config, test_motifs):
        gen1 = LibraryGenerator(generation_config, test_motifs, seed=42)
        gen2 = LibraryGenerator(generation_config, test_motifs, seed=42)
        c1 = gen1.generate(num_candidates=20)
        c2 = gen2.generate(num_candidates=20)
        assert [c.sequence for c in c1] == [c.sequence for c in c2]

    def test_stats_tracked(self, generation_config, test_motifs):
        generator = LibraryGenerator(generation_config, test_motifs, seed=42)
        generator.generate(num_candidates=50)
        assert generator.stats.candidates_generated > 0
        assert generator.stats.candidates_valid >= 0

    def test_constructs_have_motifs(self, generation_config, test_motifs):
        generator = LibraryGenerator(generation_config, test_motifs, seed=42)
        constructs = generator.generate(num_candidates=50)
        for c in constructs:
            assert len(c.motifs) >= generation_config.motifs_per_construct_min
            assert len(c.motifs) <= generation_config.motifs_per_construct_max


class TestGenerateLibrary:
    """Tests for generate_library function."""

    def test_convenience_function(self, generation_config, test_motifs):
        constructs = generate_library(
            generation_config,
            test_motifs,
            num_candidates=50,
            seed=42,
        )
        assert len(constructs) > 0


class TestEstimateFeasibleLengths:
    """Tests for estimate_feasible_lengths function."""

    def test_returns_tuple(self, generation_config, test_motifs):
        min_len, max_len = estimate_feasible_lengths(generation_config, test_motifs)
        assert isinstance(min_len, int)
        assert isinstance(max_len, int)

    def test_min_less_than_max(self, generation_config, test_motifs):
        min_len, max_len = estimate_feasible_lengths(generation_config, test_motifs)
        assert min_len <= max_len

    def test_varies_with_motif_counts(self, test_motifs):
        config1 = LibraryConfig(
            target_length_min=60,
            target_length_max=80,
            motifs_per_construct_min=2,
            motifs_per_construct_max=2,
            p5_sequence="GG",
            p5_structure="((",
            p3_sequence="CC",
            p3_structure="))",
            helix_length=3,
            hairpin_loop_length=4,
        )
        config2 = LibraryConfig(
            target_length_min=60,
            target_length_max=80,
            motifs_per_construct_min=4,
            motifs_per_construct_max=4,
            p5_sequence="GG",
            p5_structure="((",
            p3_sequence="CC",
            p3_structure="))",
            helix_length=3,
            hairpin_loop_length=4,
        )
        min1, max1 = estimate_feasible_lengths(config1, test_motifs)
        min2, max2 = estimate_feasible_lengths(config2, test_motifs)
        assert min2 > min1
        assert max2 > max1
