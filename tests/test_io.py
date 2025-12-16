"""Tests for io module."""

from pathlib import Path

import pytest

from twoway_lib.construct import Construct
from twoway_lib.io import (
    format_construct_row,
    get_library_summary,
    load_library_csv,
    save_fasta,
    save_library_csv,
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


class TestFormatConstructRow:
    """Tests for format_construct_row function."""

    def test_basic_formatting(self, test_constructs):
        row = format_construct_row(test_constructs[0], 0)
        assert row["index"] == 0
        assert row["sequence"] == "GGGAAACCC"
        assert row["structure"] == "(((...)))"
        assert row["length"] == 9

    def test_motifs_joined(self, test_constructs):
        row = format_construct_row(test_constructs[2], 2)
        assert "GAC&GC" in row["motifs"]
        assert "AAG&CUU" in row["motifs"]
        assert ";" in row["motifs"]

    def test_num_motifs(self, test_constructs):
        row = format_construct_row(test_constructs[2], 2)
        assert row["num_motifs"] == 2


class TestSaveLibraryCsv:
    """Tests for save_library_csv function."""

    def test_creates_file(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.csv"
        save_library_csv(test_constructs, output)
        assert output.exists()

    def test_file_has_header(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.csv"
        save_library_csv(test_constructs, output)
        content = output.read_text()
        assert "index" in content
        assert "sequence" in content
        assert "structure" in content

    def test_file_has_correct_rows(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.csv"
        save_library_csv(test_constructs, output)
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 4

    def test_accepts_string_path(self, test_constructs, temp_dir: Path):
        output = str(temp_dir / "library.csv")
        save_library_csv(test_constructs, output)
        assert Path(output).exists()


class TestLoadLibraryCsv:
    """Tests for load_library_csv function."""

    def test_loads_saved_library(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.csv"
        save_library_csv(test_constructs, output)
        rows = load_library_csv(output)
        assert len(rows) == 3

    def test_row_structure(self, test_constructs, temp_dir: Path):
        output = temp_dir / "library.csv"
        save_library_csv(test_constructs, output)
        rows = load_library_csv(output)
        assert "sequence" in rows[0]
        assert "structure" in rows[0]

    def test_nonexistent_file(self, temp_dir: Path):
        with pytest.raises(FileNotFoundError):
            load_library_csv(temp_dir / "nonexistent.csv")


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
