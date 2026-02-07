"""Tests for io module."""

import json
from pathlib import Path

import pytest

from twoway_lib.construct import Construct
from twoway_lib.io import (
    format_construct_dict,
    get_library_summary,
    load_library_json,
    save_fasta,
    save_library_json,
    save_sequences_txt,
)
from twoway_lib.motif import Motif


@pytest.fixture
def test_constructs() -> list[Construct]:
    """Create test constructs."""
    motif1 = Motif.from_string("GAC&GC", "(.(&))")
    motif2 = Motif.from_string("AAG&CUU", "(.(&.))")
    return [
        Construct("GGGAAACCC", "(((...)))", [motif1]),
        Construct("CCCAAAGGG", "(((...)))", [motif2]),
        Construct("UUUAAAUUU", "(((...)))", [motif1, motif2]),
    ]


class TestFormatConstructDict:
    """Tests for format_construct_dict function."""

    def test_basic_formatting(self, test_constructs):
        data = format_construct_dict(test_constructs[0], 0)
        assert data["index"] == 0
        assert data["sequence"] == "GGGAAACCC"
        assert data["structure"] == "(((...)))"
        assert data["length"] == 9

    def test_motifs_list(self, test_constructs):
        data = format_construct_dict(test_constructs[2], 2)
        assert len(data["motifs"]) == 2
        assert data["motifs"][0]["sequence"] == "GAC&GC"
        assert data["motifs"][1]["sequence"] == "AAG&CUU"

    def test_motif_structure_included(self, test_constructs):
        data = format_construct_dict(test_constructs[0], 0)
        assert data["motifs"][0]["structure"] == "(.(&))"


class TestSaveLibraryJson:
    """Tests for save_library_json function."""

    def test_creates_file(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.json"
        save_library_json(test_constructs, output)
        assert output.exists()

    def test_valid_json(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.json"
        save_library_json(test_constructs, output)
        with open(output) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_json_structure(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.json"
        save_library_json(test_constructs, output)
        with open(output) as f:
            data = json.load(f)
        assert "index" in data[0]
        assert "sequence" in data[0]
        assert "structure" in data[0]
        assert "motifs" in data[0]

    def test_accepts_string_path(self, test_constructs, temp_dir: Path):
        output = str(temp_dir / "library.json")
        save_library_json(test_constructs, output)
        assert Path(output).exists()


class TestLoadLibraryJson:
    """Tests for load_library_json function."""

    def test_loads_saved_library(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.json"
        save_library_json(test_constructs, output)
        rows = load_library_json(output)
        assert len(rows) == 3

    def test_row_structure(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.json"
        save_library_json(test_constructs, output)
        rows = load_library_json(output)
        assert "sequence" in rows[0]
        assert "structure" in rows[0]
        assert "motifs" in rows[0]

    def test_motifs_data(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.json"
        save_library_json(test_constructs, output)
        rows = load_library_json(output)
        assert isinstance(rows[0]["motifs"], list)
        assert rows[0]["motifs"][0]["sequence"] == "GAC&GC"

    def test_nonexistent_file(self, temp_dir: Path):
        with pytest.raises(FileNotFoundError):
            load_library_json(temp_dir / "nonexistent.json")


class TestSaveSequencesTxt:
    """Tests for save_sequences_txt function."""

    def test_creates_file(self, test_constructs, temp_dir: Path):
        output = temp_dir / "sequences.txt"
        save_sequences_txt(test_constructs, output)
        assert output.exists()

    def test_one_sequence_per_line(self, test_constructs, temp_dir: Path):
        output = temp_dir / "sequences.txt"
        save_sequences_txt(test_constructs, output)
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_sequences_correct(self, test_constructs, temp_dir: Path):
        output = temp_dir / "sequences.txt"
        save_sequences_txt(test_constructs, output)
        lines = output.read_text().strip().split("\n")
        assert lines[0] == "GGGAAACCC"


class TestSaveFasta:
    """Tests for save_fasta function."""

    def test_creates_file(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.fasta"
        save_fasta(test_constructs, output)
        assert output.exists()

    def test_fasta_format(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.fasta"
        save_fasta(test_constructs, output)
        content = output.read_text()
        assert ">construct_0" in content
        assert "GGGAAACCC" in content

    def test_custom_prefix(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.fasta"
        save_fasta(test_constructs, output, prefix="seq")
        content = output.read_text()
        assert ">seq_0" in content


class TestGetLibrarySummary:
    """Tests for get_library_summary function."""

    def test_empty_library(self):
        summary = get_library_summary([])
        assert summary["count"] == 0

    def test_count(self, test_constructs):
        summary = get_library_summary(test_constructs)
        assert summary["count"] == 3

    def test_length_stats(self, test_constructs):
        summary = get_library_summary(test_constructs)
        assert "length_min" in summary
        assert "length_max" in summary
        assert "length_mean" in summary

    def test_motif_stats(self, test_constructs):
        summary = get_library_summary(test_constructs)
        assert "unique_motifs_used" in summary
        assert "total_motif_usages" in summary
        assert summary["total_motif_usages"] == 4

    def test_edit_distance_stats(self, test_constructs):
        summary = get_library_summary(test_constructs)
        assert "avg_edit_distance" in summary
        assert summary["avg_edit_distance"] >= 0


class TestGetLibrarySummaryEnhanced:
    """Tests for enhanced get_library_summary."""

    def test_per_motif_usage(self, test_constructs):
        summary = get_library_summary(test_constructs)
        assert "per_motif_usage" in summary
        assert isinstance(summary["per_motif_usage"], dict)
        assert "GAC&GC" in summary["per_motif_usage"]


class TestSaveDetailedSummary:
    """Tests for save_detailed_summary function."""

    def test_creates_file(self, test_constructs, temp_dir):
        from twoway_lib.io import save_detailed_summary

        path = temp_dir / "detailed.json"
        save_detailed_summary(test_constructs, path)
        assert path.exists()

    def test_valid_json(self, test_constructs, temp_dir):
        from twoway_lib.io import save_detailed_summary

        path = temp_dir / "detailed.json"
        save_detailed_summary(test_constructs, path)
        with open(path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "constructs" in data

    def test_with_ensemble_defects(self, test_constructs, temp_dir):
        from twoway_lib.io import save_detailed_summary

        path = temp_dir / "detailed.json"
        eds = [1.0, 2.0, 3.0]
        save_detailed_summary(test_constructs, path, ensemble_defects=eds)
        with open(path) as f:
            data = json.load(f)
        assert "ensemble_defect_stats" in data
        assert data["ensemble_defect_stats"]["min"] == 1.0

    def test_with_motif_results(self, test_constructs, temp_dir):
        from twoway_lib.io import save_detailed_summary

        path = temp_dir / "detailed.json"
        motif_results = [{"motif": "GAC&GC", "passes": True}]
        save_detailed_summary(
            test_constructs,
            path,
            motif_test_results=motif_results,
        )
        with open(path) as f:
            data = json.load(f)
        assert "motif_test_results" in data
