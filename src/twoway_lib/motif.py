"""Motif representation and parsing for two-way junctions."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Motif:
    """
    Represents a two-way junction motif with sequence and structure.

    A motif consists of two strands separated by '&' in both sequence
    and secondary structure notation. For example:
    - sequence: "GAC&GC"
    - structure: "(.(&))"

    Attributes:
        sequence: Full motif sequence with & separator.
        structure: Full dot-bracket structure with & separator.
        strand1_seq: First strand sequence (5' to 3').
        strand2_seq: Second strand sequence (5' to 3').
        strand1_ss: First strand secondary structure.
        strand2_ss: Second strand secondary structure.
    """

    sequence: str
    structure: str
    strand1_seq: str
    strand2_seq: str
    strand1_ss: str
    strand2_ss: str

    @classmethod
    def from_string(cls, sequence: str, structure: str) -> "Motif":
        """
        Create a Motif from sequence and structure strings.

        Args:
            sequence: Motif sequence with & separator (e.g., "GAC&GC").
            structure: Dot-bracket structure with & separator (e.g., "(.(&))").

        Returns:
            Motif object with parsed strands.

        Raises:
            ValueError: If format is invalid or strands don't match.
        """
        sequence = sequence.upper().replace("T", "U")
        _validate_motif_format(sequence, structure)

        strand1_seq, strand2_seq = sequence.split("&")
        strand1_ss, strand2_ss = structure.split("&")

        return cls(
            sequence=sequence,
            structure=structure,
            strand1_seq=strand1_seq,
            strand2_seq=strand2_seq,
            strand1_ss=strand1_ss,
            strand2_ss=strand2_ss,
        )

    def total_length(self) -> int:
        """Return total length of both strands."""
        return len(self.strand1_seq) + len(self.strand2_seq)

    def strand1_length(self) -> int:
        """Return length of first strand."""
        return len(self.strand1_seq)

    def strand2_length(self) -> int:
        """Return length of second strand."""
        return len(self.strand2_seq)

    def flip(self) -> "Motif":
        """
        Return a new motif with strands swapped.

        The flip operation swaps strand1 and strand2, creating a new motif
        where what was the 5' side becomes the 3' side and vice versa.

        Example:
            GGA&CCU with ((&)) becomes CCU&GGA with )(&((

        Returns:
            New Motif with swapped strands.
        """
        new_sequence = f"{self.strand2_seq}&{self.strand1_seq}"
        new_structure = f"{self.strand2_ss}&{self.strand1_ss}"
        return Motif(
            sequence=new_sequence,
            structure=new_structure,
            strand1_seq=self.strand2_seq,
            strand2_seq=self.strand1_seq,
            strand1_ss=self.strand2_ss,
            strand2_ss=self.strand1_ss,
        )


def _validate_motif_format(sequence: str, structure: str) -> None:
    """
    Validate motif sequence and structure format.

    Args:
        sequence: Motif sequence string.
        structure: Motif structure string.

    Raises:
        ValueError: If format is invalid.
    """
    _validate_separators(sequence, structure)
    seq_parts = sequence.split("&")
    ss_parts = structure.split("&")
    _validate_strand_counts(seq_parts, ss_parts, sequence, structure)
    _validate_strand_lengths(seq_parts, ss_parts, sequence, structure)
    _validate_strand_characters(seq_parts, ss_parts)


def _validate_separators(sequence: str, structure: str) -> None:
    """Validate that separators are present."""
    if "&" not in sequence:
        raise ValueError(f"Motif sequence must contain '&' separator: {sequence}")
    if "&" not in structure:
        raise ValueError(f"Motif structure must contain '&' separator: {structure}")


def _validate_strand_counts(
    seq_parts: list[str], ss_parts: list[str], sequence: str, structure: str
) -> None:
    """Validate that there are exactly 2 strands."""
    if len(seq_parts) != 2:
        raise ValueError(f"Motif must have exactly 2 strands: {sequence}")
    if len(ss_parts) != 2:
        raise ValueError(f"Structure must have exactly 2 parts: {structure}")


def _validate_strand_lengths(
    seq_parts: list[str], ss_parts: list[str], sequence: str, structure: str
) -> None:
    """Validate that sequence and structure lengths match."""
    if len(seq_parts[0]) != len(ss_parts[0]):
        raise ValueError(
            f"Strand 1 sequence/structure length mismatch: {sequence}, {structure}"
        )
    if len(seq_parts[1]) != len(ss_parts[1]):
        raise ValueError(
            f"Strand 2 sequence/structure length mismatch: {sequence}, {structure}"
        )


def _validate_strand_characters(seq_parts: list[str], ss_parts: list[str]) -> None:
    """Validate characters in sequences and structures."""
    valid_nts = set("AUGC")
    valid_ss = set("().")

    for i, seq in enumerate(seq_parts):
        invalid = set(seq) - valid_nts
        if invalid:
            raise ValueError(f"Strand {i + 1} contains invalid nucleotides: {invalid}")

    for i, ss in enumerate(ss_parts):
        invalid = set(ss) - valid_ss
        if invalid:
            raise ValueError(
                f"Structure {i + 1} contains invalid characters: {invalid}"
            )


def load_motifs(path: Path | str) -> list[Motif]:
    """
    Load motifs from a CSV file.

    Expected CSV format:
        sequence,structure
        GAC&GC,(.(&))
        UUCG&CGAA,(..((&))...)

    Args:
        path: Path to CSV file with motif data.

    Returns:
        List of parsed Motif objects.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If CSV format is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Motif file not found: {path}")

    df = pd.read_csv(path)
    _validate_motif_csv(df)

    motifs = []
    for _, row in df.iterrows():
        motif = Motif.from_string(row["sequence"], row["structure"])
        motifs.append(motif)

    return motifs


def _validate_motif_csv(df: pd.DataFrame) -> None:
    """
    Validate motif CSV DataFrame structure.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"sequence", "structure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Motif CSV missing required columns: {missing}")


def validate_motif(motif: Motif) -> None:
    """
    Perform additional validation on a motif.

    Checks that the structure is balanced and makes sense for a two-way junction.

    Args:
        motif: Motif to validate.

    Raises:
        ValueError: If motif structure is invalid.
    """
    open_count = motif.strand1_ss.count("(")
    close_count = motif.strand2_ss.count(")")

    if open_count != close_count:
        raise ValueError(
            f"Unbalanced parentheses in motif: {open_count} opens, {close_count} closes"
        )

    if ")" in motif.strand1_ss:
        raise ValueError("Strand 1 should only have '(' or '.' characters")
    if "(" in motif.strand2_ss:
        raise ValueError("Strand 2 should only have ')' or '.' characters")
