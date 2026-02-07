"""Tests for preprocessing module."""

import json
from pathlib import Path

import pytest

from twoway_lib.motif import Motif
from twoway_lib.preprocessing import (
    MotifTestResult,
    load_motif_results,
    preprocess_motifs,
    save_motif_results,
)


@pytest.fixture
def test_motifs() -> list[Motif]:
    """Motifs for preprocessing testing."""
    return [
        Motif.from_string("GAC&GC", "(.(&))"),
        Motif.from_string("AAG&CUU", "(.(&.))"),
        Motif.from_string("UUG&CAA", "(.(&.))"),
    ]


class TestMotifTestResult:
    """Tests for MotifTestResult dataclass."""

    def test_create_from_fields(self):
        result = MotifTestResult(
            motif_sequence="GAC&GC",
            motif_structure="(.(&))",
            passes=True,
            match_fraction=1.0,
            instances_failed=0,
            total_instances=10,
            designed_ss="(.(&))",
            predicted_ss="(.(&))",
            avg_ensemble_defect=2.5,
        )
        assert result.passes is True
        assert result.avg_ensemble_defect == 2.5


class TestPreprocessMotifs:
    """Tests for preprocess_motifs function."""

    def test_returns_three_lists(self, test_motifs):
        passing, failing, results = preprocess_motifs(
            test_motifs,
            helix_length=3,
            num_contexts=3,
            seed=42,
        )
        assert isinstance(passing, list)
        assert isinstance(failing, list)
        assert isinstance(results, list)
        assert len(passing) + len(failing) == len(test_motifs)
        assert len(results) == len(test_motifs)

    def test_results_have_ensemble_defect(self, test_motifs):
        _, _, results = preprocess_motifs(
            test_motifs,
            helix_length=3,
            num_contexts=3,
            seed=42,
        )
        for r in results:
            assert isinstance(r, MotifTestResult)
            assert r.avg_ensemble_defect >= 0

    def test_reproducible(self, test_motifs):
        p1, f1, r1 = preprocess_motifs(test_motifs, seed=42)
        p2, f2, r2 = preprocess_motifs(test_motifs, seed=42)
        assert [m.sequence for m in p1] == [m.sequence for m in p2]
        assert [m.sequence for m in f1] == [m.sequence for m in f2]


class TestSaveLoadMotifResults:
    """Tests for save/load motif results."""

    def test_round_trip(self, test_motifs, temp_dir: Path):
        _, _, results = preprocess_motifs(
            test_motifs,
            helix_length=3,
            num_contexts=3,
            seed=42,
        )
        path = temp_dir / "motif_results.json"
        save_motif_results(results, path)
        assert path.exists()

        loaded = load_motif_results(path)
        assert len(loaded) == len(results)
        for orig, load in zip(results, loaded, strict=True):
            assert orig.motif_sequence == load.motif_sequence
            assert orig.passes == load.passes
            assert abs(orig.avg_ensemble_defect - load.avg_ensemble_defect) < 1e-6

    def test_save_creates_valid_json(self, test_motifs, temp_dir: Path):
        _, _, results = preprocess_motifs(
            test_motifs,
            helix_length=3,
            num_contexts=3,
            seed=42,
        )
        path = temp_dir / "motif_results.json"
        save_motif_results(results, path)

        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == len(results)
        assert "avg_ensemble_defect" in data[0]

    def test_load_nonexistent_raises(self, temp_dir: Path):
        with pytest.raises(FileNotFoundError):
            load_motif_results(temp_dir / "nonexistent.json")
