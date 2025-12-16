"""Edit distance calculations for sequence diversity."""

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import editdistance
import numpy as np


def edit_distance(seq1: str, seq2: str) -> int:
    """
    Calculate Levenshtein edit distance between two sequences.

    Args:
        seq1: First sequence.
        seq2: Second sequence.

    Returns:
        Edit distance (minimum edits to transform seq1 to seq2).
    """
    return editdistance.eval(seq1, seq2)


def min_distance_to_set(sequence: str, existing: list[str]) -> int:
    """
    Find minimum edit distance from sequence to any in a set.

    Args:
        sequence: Query sequence.
        existing: List of existing sequences to compare against.

    Returns:
        Minimum edit distance, or 0 if existing is empty.
    """
    if not existing:
        return 0

    return min(edit_distance(sequence, other) for other in existing)


def calculate_diversity_score(sequences: list[str]) -> float:
    """
    Calculate average minimum pairwise edit distance.

    For each sequence, finds its minimum distance to any other sequence,
    then returns the average of these minimum distances.

    Args:
        sequences: List of sequences.

    Returns:
        Average minimum pairwise distance, or 0.0 for < 2 sequences.
    """
    n = len(sequences)
    if n < 2:
        return 0.0

    min_distances = []
    for i, seq in enumerate(sequences):
        others = sequences[:i] + sequences[i + 1 :]
        min_dist = min_distance_to_set(seq, others)
        min_distances.append(min_dist)

    return sum(min_distances) / n


def compute_distance_matrix(sequences: list[str]) -> np.ndarray:
    """
    Compute full pairwise distance matrix.

    Args:
        sequences: List of sequences.

    Returns:
        NxN numpy array of pairwise distances.
    """
    n = len(sequences)
    matrix = np.zeros((n, n), dtype=np.int32)

    for i, j in combinations(range(n), 2):
        dist = edit_distance(sequences[i], sequences[j])
        matrix[i, j] = dist
        matrix[j, i] = dist

    return matrix


def _compute_distances_chunk(
    args: tuple[list[str], list[tuple[int, int]]],
) -> list[tuple[int, int, int]]:
    """
    Compute distances for a chunk of index pairs.

    Args:
        args: Tuple of (sequences, list of (i, j) pairs).

    Returns:
        List of (i, j, distance) tuples.
    """
    sequences, pairs = args
    results = []
    for i, j in pairs:
        dist = edit_distance(sequences[i], sequences[j])
        results.append((i, j, dist))
    return results


def parallel_distance_matrix(sequences: list[str], n_workers: int = 4) -> np.ndarray:
    """
    Compute pairwise distance matrix in parallel.

    Args:
        sequences: List of sequences.
        n_workers: Number of parallel workers.

    Returns:
        NxN numpy array of pairwise distances.
    """
    n = len(sequences)
    if n < 100:
        return compute_distance_matrix(sequences)

    pairs = list(combinations(range(n), 2))
    chunk_size = max(1, len(pairs) // n_workers)
    chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    matrix = np.zeros((n, n), dtype=np.int32)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        args_list = [(sequences, chunk) for chunk in chunks]
        results = executor.map(_compute_distances_chunk, args_list)

        for result_chunk in results:
            for i, j, dist in result_chunk:
                matrix[i, j] = dist
                matrix[j, i] = dist

    return matrix


def diversity_from_matrix(
    matrix: np.ndarray, indices: list[int] | None = None
) -> float:
    """
    Calculate diversity score from a precomputed distance matrix.

    Args:
        matrix: Pairwise distance matrix.
        indices: Subset of indices to consider (all if None).

    Returns:
        Average minimum pairwise distance.
    """
    if indices is None:
        indices = list(range(matrix.shape[0]))

    n = len(indices)
    if n < 2:
        return 0.0

    submatrix = matrix[np.ix_(indices, indices)].astype(np.float64).copy()
    np.fill_diagonal(submatrix, np.finfo(np.float64).max)

    min_distances = np.min(submatrix, axis=1)
    return float(np.mean(min_distances))
