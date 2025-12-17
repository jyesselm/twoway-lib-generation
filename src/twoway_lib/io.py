"""I/O operations for library data."""

import json
from pathlib import Path

from twoway_lib.construct import Construct
from twoway_lib.diversity import calculate_diversity_score


def save_library_json(constructs: list[Construct], path: Path | str) -> None:
    """
    Save library to JSON file.

    Args:
        constructs: List of constructs to save.
        path: Output file path.
    """
    path = Path(path)
    data = [format_construct_dict(c, i) for i, c in enumerate(constructs)]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_library_json(path: Path | str) -> list[dict]:
    """
    Load library from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        List of construct dictionaries.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Library file not found: {path}")

    with open(path) as f:
        return json.load(f)


def format_construct_dict(construct: Construct, index: int) -> dict:
    """
    Format a construct as a dictionary for JSON output.

    Args:
        construct: Construct to format.
        index: Index in the library.

    Returns:
        Dictionary with construct data.
    """
    motifs_data = []
    for i, motif in enumerate(construct.motifs):
        motif_dict = {
            "sequence": motif.sequence,
            "structure": motif.structure,
        }
        if construct.motif_positions and i < len(construct.motif_positions):
            mp = construct.motif_positions[i]
            motif_dict["positions"] = {
                "strand1": mp.strand1_positions,
                "strand2": mp.strand2_positions,
            }
        motifs_data.append(motif_dict)

    return {
        "index": index,
        "sequence": construct.sequence,
        "structure": construct.structure,
        "length": construct.length(),
        "motifs": motifs_data,
    }


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
    sequences = [c.sequence for c in constructs]

    all_motifs = []
    motif_counts: dict[str, int] = {}
    for c in constructs:
        for m in c.motifs:
            all_motifs.append(m.sequence)
            motif_counts[m.sequence] = motif_counts.get(m.sequence, 0) + 1

    unique_motifs = len(set(all_motifs))
    used_counts = [v for v in motif_counts.values() if v > 0]

    # Calculate average edit distance diversity
    avg_edit_distance = calculate_diversity_score(sequences)

    return {
        "count": len(constructs),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_mean": sum(lengths) / len(lengths),
        "motifs_per_construct_min": min(num_motifs),
        "motifs_per_construct_max": max(num_motifs),
        "unique_motifs_used": unique_motifs,
        "total_motif_usages": len(all_motifs),
        "motif_usage_min": min(used_counts) if used_counts else 0,
        "motif_usage_max": max(used_counts) if used_counts else 0,
        "motif_usage_mean": sum(used_counts) / len(used_counts) if used_counts else 0,
        "avg_edit_distance": avg_edit_distance,
    }
