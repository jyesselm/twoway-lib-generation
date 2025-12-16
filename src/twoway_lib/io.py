"""CSV I/O operations for library data."""

from pathlib import Path

import pandas as pd

from twoway_lib.construct import Construct


def save_library_csv(constructs: list[Construct], path: Path | str) -> None:
    """
    Save library to CSV file.

    Output columns: index, sequence, structure, length, motifs

    Args:
        constructs: List of constructs to save.
        path: Output file path.
    """
    path = Path(path)
    rows = [format_construct_row(c, i) for i, c in enumerate(constructs)]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def format_construct_row(construct: Construct, index: int) -> dict:
    """
    Format a construct as a CSV row dictionary.

    Args:
        construct: Construct to format.
        index: Index in the library.

    Returns:
        Dictionary with row data.
    """
    motif_seqs = [m.sequence for m in construct.motifs]
    row = {
        "index": index,
        "sequence": construct.sequence,
        "structure": construct.structure,
        "length": construct.length(),
        "num_motifs": len(construct.motifs),
        "motifs": ";".join(motif_seqs),
    }
    if construct.motif_positions:
        row["motif_positions"] = construct.get_positions_string()
    return row


def load_library_csv(path: Path | str) -> list[dict]:
    """
    Load library from CSV file.

    Args:
        path: Path to CSV file.

    Returns:
        List of row dictionaries.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Library file not found: {path}")

    df = pd.read_csv(path)
    return df.to_dict("records")


def save_sequences_txt(constructs: list[Construct], path: Path | str) -> None:
    """
    Save sequences to a plain text file (one per line).

    Args:
        constructs: List of constructs.
        path: Output file path.
    """
    path = Path(path)
    with open(path, "w") as f:
        for construct in constructs:
            f.write(f"{construct.sequence}\n")


def save_fasta(
    constructs: list[Construct],
    path: Path | str,
    prefix: str = "construct",
) -> None:
    """
    Save constructs to FASTA format.

    Args:
        constructs: List of constructs.
        path: Output file path.
        prefix: Prefix for sequence names.
    """
    path = Path(path)
    with open(path, "w") as f:
        for i, construct in enumerate(constructs):
            f.write(f">{prefix}_{i}\n")
            f.write(f"{construct.sequence}\n")


def get_library_summary(constructs: list[Construct]) -> dict:
    """
    Generate summary statistics for a library.

    Args:
        constructs: List of constructs.

    Returns:
        Dictionary with summary statistics.
    """
    if not constructs:
        return {"count": 0}

    lengths = [c.length() for c in constructs]
    num_motifs = [len(c.motifs) for c in constructs]

    all_motifs = []
    for c in constructs:
        all_motifs.extend(m.sequence for m in c.motifs)
    unique_motifs = len(set(all_motifs))

    return {
        "count": len(constructs),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_mean": sum(lengths) / len(lengths),
        "motifs_per_construct_min": min(num_motifs),
        "motifs_per_construct_max": max(num_motifs),
        "unique_motifs_used": unique_motifs,
        "total_motif_usages": len(all_motifs),
    }
