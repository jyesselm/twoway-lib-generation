"""Tests for optimizer module."""

import pytest

from twoway_lib.config import OptimizationConfig
from twoway_lib.construct import Construct
from twoway_lib.motif import Motif
from twoway_lib.optimizer import LibraryOptimizer, select_diverse_subset


@pytest.fixture
def sample_constructs() -> list[Construct]:
    """Create sample constructs for testing."""
    motif = Motif.from_string("GAC&GC", "(.(&))")
    sequences = [
        "GGGAAACCC",
        "GGGAAACUC",
        "GGGAAAUCC",
        "CCCAAAGGG",
        "UUUAAACCC",
        "GGGAAAGGG",
        "CCCAAAUUU",
        "AAAUUUGGG",
        "GGGCCCAAA",
        "UUUGGGUUU",
    ]
    structures = ["." * len(s) for s in sequences]
    return [
        Construct(s, ss, [motif]) for s, ss in zip(sequences, structures, strict=True)
    ]


class TestLibraryOptimizer:
    """Tests for LibraryOptimizer class."""

    def test_init(self, sample_constructs):
        config = OptimizationConfig(
            iterations=100,
            target_library_size=5,
        )
        optimizer = LibraryOptimizer(sample_constructs, config)
        assert optimizer.config == config
        assert len(optimizer.constructs) == 10

    def test_optimize_returns_indices(self, sample_constructs):
        config = OptimizationConfig(
            iterations=100,
            target_library_size=5,
        )
        optimizer = LibraryOptimizer(sample_constructs, config, seed=42)
        indices = optimizer.optimize()
        assert len(indices) == 5
        assert all(0 <= i < len(sample_constructs) for i in indices)

    def test_optimize_with_seed_reproducible(self, sample_constructs):
        config = OptimizationConfig(iterations=100, target_library_size=5)
        opt1 = LibraryOptimizer(sample_constructs, config, seed=42)
        opt2 = LibraryOptimizer(sample_constructs, config, seed=42)
        indices1 = opt1.optimize()
        indices2 = opt2.optimize()
        assert indices1 == indices2

    def test_optimize_target_larger_than_pool(self, sample_constructs):
        config = OptimizationConfig(iterations=100, target_library_size=20)
        optimizer = LibraryOptimizer(sample_constructs, config, seed=42)
        indices = optimizer.optimize()
        assert len(indices) == 10

    def test_progress_callback(self, sample_constructs):
        config = OptimizationConfig(iterations=2000, target_library_size=5)
        optimizer = LibraryOptimizer(sample_constructs, config, seed=42)
        progress_calls = []
        optimizer.set_progress_callback(lambda p: progress_calls.append(p))
        optimizer.optimize()
        assert len(progress_calls) > 0


class TestSelectDiverseSubset:
    """Tests for select_diverse_subset function."""

    def test_select_subset(self, sample_constructs):
        indices = select_diverse_subset(
            sample_constructs,
            target_size=5,
            iterations=100,
            seed=42,
        )
        assert len(indices) == 5
        assert len(set(indices)) == 5

    def test_select_all_when_target_exceeds(self, sample_constructs):
        indices = select_diverse_subset(
            sample_constructs,
            target_size=20,
            iterations=100,
            seed=42,
        )
        assert len(indices) == 10
