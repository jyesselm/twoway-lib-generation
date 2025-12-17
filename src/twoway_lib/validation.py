"""Structure validation using Vienna RNA fold."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rna_secstruct import SecStruct

if TYPE_CHECKING:
    from twoway_lib.construct import Construct
    from twoway_lib.motif import Motif


@dataclass
class FoldResult:
    """Result of RNA folding prediction."""

    sequence: str
    predicted_structure: str
    mfe: float
    ensemble_defect: float


@dataclass
class ValidationResult:
    """Result of construct validation."""

    is_valid: bool
    reason: str
    fold_result: FoldResult | None = None
    structure_match: float = 0.0


def fold_sequence(sequence: str) -> FoldResult:
    """
    Fold an RNA sequence using ViennaRNA.

    Args:
        sequence: RNA sequence to fold.

    Returns:
        FoldResult with predicted structure, MFE, and ensemble defect.
    """
    import RNA

    fc = RNA.fold_compound(sequence)
    structure, mfe = fc.mfe()
    fc.pf()
    ensemble_defect = fc.ensemble_defect(structure)

    return FoldResult(
        sequence=sequence,
        predicted_structure=structure,
        mfe=mfe,
        ensemble_defect=ensemble_defect,
    )


def validate_construct(
    construct: "Construct",
    max_ensemble_defect: float,
    allow_differences: bool,
    min_structure_match: float = 0.9,
) -> ValidationResult:
    """
    Validate a construct using Vienna fold prediction.

    Args:
        construct: Construct to validate.
        max_ensemble_defect: Maximum allowed ensemble defect.
        allow_differences: Allow structure prediction differences.
        min_structure_match: Minimum fraction of matching positions.

    Returns:
        ValidationResult indicating if construct is valid.
    """
    fold_result = fold_sequence(construct.sequence)

    if fold_result.ensemble_defect > max_ensemble_defect:
        return ValidationResult(
            is_valid=False,
            reason=f"Ensemble defect {fold_result.ensemble_defect:.2f} > {max_ensemble_defect}",
            fold_result=fold_result,
        )

    structure_match = compare_structures(
        construct.structure, fold_result.predicted_structure
    )

    if not allow_differences and structure_match < 1.0:
        return ValidationResult(
            is_valid=False,
            reason=f"Structure mismatch: {structure_match:.1%} match",
            fold_result=fold_result,
            structure_match=structure_match,
        )

    if allow_differences and structure_match < min_structure_match:
        return ValidationResult(
            is_valid=False,
            reason=f"Structure match {structure_match:.1%} < {min_structure_match:.1%}",
            fold_result=fold_result,
            structure_match=structure_match,
        )

    return ValidationResult(
        is_valid=True,
        reason="Valid",
        fold_result=fold_result,
        structure_match=structure_match,
    )


def compare_structures(designed: str, predicted: str) -> float:
    """
    Calculate fraction of matching positions between structures.

    Uses rna_secstruct.SecStruct.structural_similarity for comparison.

    Args:
        designed: Designed secondary structure.
        predicted: Predicted secondary structure.

    Returns:
        Fraction of positions that match (0.0 to 1.0).
    """
    if len(designed) != len(predicted):
        return 0.0

    if len(designed) == 0:
        return 1.0

    # Use dummy sequences since we only compare structure
    dummy_seq = "N" * len(designed)
    designed_ss = SecStruct(dummy_seq, designed)
    predicted_ss = SecStruct(dummy_seq, predicted)
    return designed_ss.structural_similarity(predicted_ss)


def count_structure_differences(designed: str, predicted: str) -> dict[str, int]:
    """
    Count different types of structure mismatches.

    Args:
        designed: Designed secondary structure.
        predicted: Predicted secondary structure.

    Returns:
        Dictionary with counts of different mismatch types.
    """
    if len(designed) != len(predicted):
        return {"length_mismatch": abs(len(designed) - len(predicted))}

    counts = {
        "total_mismatches": 0,
        "unpaired_to_paired": 0,
        "paired_to_unpaired": 0,
        "bracket_swap": 0,
    }

    for d, p in zip(designed, predicted, strict=True):
        if d == p:
            continue
        counts["total_mismatches"] += 1
        if d == "." and p in "()":
            counts["unpaired_to_paired"] += 1
        elif d in "()" and p == ".":
            counts["paired_to_unpaired"] += 1
        elif d in "()" and p in "()":
            counts["bracket_swap"] += 1

    return counts


def validate_structure_format(sequence: str, structure: str) -> tuple[bool, str]:
    """
    Validate that a sequence and structure are well-formed using SecStruct.

    Checks for:
    - Matching lengths
    - Balanced brackets
    - Valid characters

    Args:
        sequence: RNA sequence.
        structure: Dot-bracket secondary structure.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    try:
        ss = SecStruct(sequence, structure)
        ss.validate()
        return True, ""
    except ValueError as e:
        return False, str(e)


def has_consecutive_nucleotides(sequence: str, max_count: int = 4) -> bool:
    """
    Check if sequence has too many consecutive identical nucleotides.

    Args:
        sequence: RNA sequence to check.
        max_count: Maximum allowed consecutive identical nucleotides.

    Returns:
        True if sequence has max_count or more consecutive identical nucleotides.
    """
    pattern = rf"(.)\1{{{max_count - 1},}}"
    return bool(re.search(pattern, sequence))


def find_consecutive_nucleotides(sequence: str, max_count: int = 4) -> list[tuple[int, str]]:
    """
    Find all runs of consecutive identical nucleotides that exceed max_count.

    Args:
        sequence: RNA sequence to check.
        max_count: Maximum allowed consecutive identical nucleotides.

    Returns:
        List of (position, nucleotide) tuples for each violation.
    """
    pattern = rf"(.)\1{{{max_count - 1},}}"
    violations = []
    for match in re.finditer(pattern, sequence):
        violations.append((match.start(), match.group(0)[0]))
    return violations


def has_consecutive_gc_pairs(
    sequence: str, structure: str, max_count: int = 3
) -> bool:
    """
    Check if structure has too many consecutive GC/CG base pairs.

    Args:
        sequence: RNA sequence.
        structure: Dot-bracket secondary structure.
        max_count: Maximum allowed consecutive GC/CG pairs.

    Returns:
        True if structure has max_count or more consecutive GC/CG pairs.
    """
    gc_pairs = _find_gc_pair_runs(sequence, structure)
    return any(run_length >= max_count for run_length in gc_pairs)


def _find_gc_pair_runs(sequence: str, structure: str) -> list[int]:
    """
    Find lengths of consecutive GC/CG pair runs in the structure.

    Args:
        sequence: RNA sequence.
        structure: Dot-bracket secondary structure.

    Returns:
        List of run lengths for consecutive GC/CG pairs.
    """
    if len(sequence) != len(structure):
        return []

    # Build base pair mapping using SecStruct
    ss = SecStruct(sequence, structure)

    # Build pairs list: pairs[i] = partner of position i, or -1 if unpaired
    pairs = []
    for i in range(len(ss)):
        if ss.is_paired(i):
            _, partner = ss.get_basepair(i)
            pairs.append(partner)
        else:
            pairs.append(-1)

    # Find consecutive GC pairs by walking through helices
    gc_runs = []
    visited = set()

    for i, j in enumerate(pairs):
        if j == -1 or i in visited or i > j:
            continue

        # Start of a potential helix - walk through consecutive pairs
        run_length = 0
        pos = i
        while pos < len(pairs) and pairs[pos] != -1:
            partner = pairs[pos]
            if partner < pos:
                break

            nt1 = sequence[pos].upper()
            nt2 = sequence[partner].upper()

            if (nt1 == "G" and nt2 == "C") or (nt1 == "C" and nt2 == "G"):
                run_length += 1
                visited.add(pos)
                visited.add(partner)
            else:
                if run_length > 0:
                    gc_runs.append(run_length)
                run_length = 0

            # Check if next position continues the helix
            if pos + 1 < len(pairs) and pairs[pos + 1] == partner - 1:
                pos += 1
            else:
                break

        if run_length > 0:
            gc_runs.append(run_length)

    return gc_runs


def test_motif_folding(
    motif: "Motif",
    helix_length: int = 3,
    repeats: int = 5,
    seed: int = 42,
) -> tuple[bool, float]:
    """
    Test if a motif folds correctly when embedded in a helix context.

    Builds a test construct with the motif repeated multiple times
    between random helices, folds it, and checks if the motif regions
    match the designed structure.

    Args:
        motif: Motif to test.
        helix_length: Length of flanking helices.
        repeats: Number of times to repeat motif in test construct.
        seed: Random seed for helix generation.

    Returns:
        Tuple of (passes, match_fraction) where passes is True if motif
        regions match 100%, and match_fraction is the fraction of matching
        positions in motif regions.
    """
    from random import Random

    from twoway_lib.helix import random_helix

    rng = Random(seed)

    # Build test construct
    seq_parts = []
    ss_parts = []
    helices = [random_helix(helix_length, rng) for _ in range(repeats + 1)]

    # 5' arm
    for i in range(repeats):
        seq_parts.append(helices[i].strand1)
        ss_parts.append(helices[i].structure1)
        seq_parts.append(motif.strand1_seq)
        ss_parts.append(motif.strand1_ss)

    seq_parts.append(helices[repeats].strand1)
    ss_parts.append(helices[repeats].structure1)

    # Hairpin loop
    seq_parts.append("GAAA")
    ss_parts.append("....")

    # 3' arm
    seq_parts.append(helices[repeats].strand2)
    ss_parts.append(helices[repeats].structure2)

    for i in range(repeats - 1, -1, -1):
        seq_parts.append(motif.strand2_seq)
        ss_parts.append(motif.strand2_ss)
        seq_parts.append(helices[i].strand2)
        ss_parts.append(helices[i].structure2)

    sequence = "".join(seq_parts)
    designed_ss = "".join(ss_parts)

    # Fold
    fold_result = fold_sequence(sequence)
    predicted_ss = fold_result.predicted_structure

    # Find motif positions and check match
    motif_s1_len = len(motif.strand1_seq)
    motif_s2_len = len(motif.strand2_seq)

    motif_matches = 0
    motif_total = 0

    # 5' arm motif positions
    pos = 0
    for i in range(repeats):
        pos += helix_length
        for j in range(motif_s1_len):
            motif_total += 1
            if designed_ss[pos + j] == predicted_ss[pos + j]:
                motif_matches += 1
        pos += motif_s1_len

    # Skip final helix + loop + final helix
    pos += helix_length + 4 + helix_length

    # 3' arm motif positions
    for i in range(repeats):
        for j in range(motif_s2_len):
            motif_total += 1
            if designed_ss[pos + j] == predicted_ss[pos + j]:
                motif_matches += 1
        pos += motif_s2_len + helix_length

    match_fraction = motif_matches / motif_total if motif_total > 0 else 1.0
    passes = match_fraction == 1.0

    return passes, match_fraction


@dataclass
class MotifFoldResult:
    """Result of motif fold test."""

    motif: "Motif"
    passes: bool
    match_fraction: float
    instances_failed: int
    total_instances: int
    designed_ss: str
    predicted_ss: str


def filter_foldable_motifs(
    motifs: list["Motif"],
    helix_length: int = 3,
    repeats: int = 5,
    seed: int = 42,
) -> tuple[list["Motif"], list["Motif"], list[MotifFoldResult]]:
    """
    Filter motifs to only those that fold correctly.

    Args:
        motifs: List of motifs to test.
        helix_length: Length of flanking helices for test.
        repeats: Number of repeats in test construct.
        seed: Random seed for helix generation.

    Returns:
        Tuple of (passing_motifs, failing_motifs, failed_results).
    """
    passing = []
    failing = []
    failed_results = []

    for motif in motifs:
        result = test_motif_folding_detailed(motif, helix_length, repeats, seed)
        if result.passes:
            passing.append(motif)
        else:
            failing.append(motif)
            failed_results.append(result)

    return passing, failing, failed_results


def test_motif_folding_detailed(
    motif: "Motif",
    helix_length: int = 3,
    repeats: int = 5,
    seed: int = 42,
) -> MotifFoldResult:
    """
    Test if a motif folds correctly and return detailed results.

    Args:
        motif: Motif to test.
        helix_length: Length of flanking helices.
        repeats: Number of times to repeat motif in test construct.
        seed: Random seed for helix generation.

    Returns:
        MotifFoldResult with detailed fold test information.
    """
    from random import Random

    from twoway_lib.helix import random_helix

    rng = Random(seed)

    # Build test construct
    seq_parts = []
    ss_parts = []
    helices = [random_helix(helix_length, rng) for _ in range(repeats + 1)]

    # 5' arm
    for i in range(repeats):
        seq_parts.append(helices[i].strand1)
        ss_parts.append(helices[i].structure1)
        seq_parts.append(motif.strand1_seq)
        ss_parts.append(motif.strand1_ss)

    seq_parts.append(helices[repeats].strand1)
    ss_parts.append(helices[repeats].structure1)

    # Hairpin loop
    seq_parts.append("GAAA")
    ss_parts.append("....")

    # 3' arm
    seq_parts.append(helices[repeats].strand2)
    ss_parts.append(helices[repeats].structure2)

    for i in range(repeats - 1, -1, -1):
        seq_parts.append(motif.strand2_seq)
        ss_parts.append(motif.strand2_ss)
        seq_parts.append(helices[i].strand2)
        ss_parts.append(helices[i].structure2)

    sequence = "".join(seq_parts)
    designed_ss = "".join(ss_parts)

    # Fold
    fold_result = fold_sequence(sequence)
    predicted_ss = fold_result.predicted_structure

    # Find motif positions and check match
    motif_s1_len = len(motif.strand1_seq)
    motif_s2_len = len(motif.strand2_seq)

    instances_failed = 0
    motif_matches = 0
    motif_total = 0

    # Store positions for first instance of each strand
    first_s1_pos = None
    first_s2_pos = None

    # 5' arm motif positions
    pos = 0
    for i in range(repeats):
        pos += helix_length
        if first_s1_pos is None:
            first_s1_pos = pos
        instance_match = True
        for j in range(motif_s1_len):
            motif_total += 1
            if designed_ss[pos + j] == predicted_ss[pos + j]:
                motif_matches += 1
            else:
                instance_match = False
        if not instance_match:
            instances_failed += 1
        pos += motif_s1_len

    # Skip final helix + loop + final helix
    pos += helix_length + 4 + helix_length

    # 3' arm motif positions
    for i in range(repeats):
        if first_s2_pos is None:
            first_s2_pos = pos
        instance_match = True
        for j in range(motif_s2_len):
            motif_total += 1
            if designed_ss[pos + j] == predicted_ss[pos + j]:
                motif_matches += 1
            else:
                instance_match = False
        if not instance_match:
            instances_failed += 1
        pos += motif_s2_len + helix_length

    match_fraction = motif_matches / motif_total if motif_total > 0 else 1.0
    passes = match_fraction == 1.0

    # Build predicted structure string matching motif format: strand1&strand2
    pred_s1 = predicted_ss[first_s1_pos : first_s1_pos + motif_s1_len]
    pred_s2 = predicted_ss[first_s2_pos : first_s2_pos + motif_s2_len]
    predicted_motif_ss = f"{pred_s1}&{pred_s2}"

    return MotifFoldResult(
        motif=motif,
        passes=passes,
        match_fraction=match_fraction,
        instances_failed=instances_failed,
        total_instances=repeats * 2,  # repeats on each arm
        designed_ss=motif.structure,
        predicted_ss=predicted_motif_ss,
    )


def check_sequence_constraints(
    sequence: str,
    structure: str,
    motif_sequences: list[str] | None = None,
    max_consecutive_nt: int = 4,
    max_consecutive_gc: int = 3,
) -> tuple[bool, str]:
    """
    Check sequence constraints, excluding patterns present in motifs.

    Args:
        sequence: Full construct sequence.
        structure: Full construct structure.
        motif_sequences: List of motif sequences (violations in motifs are allowed).
        max_consecutive_nt: Max consecutive identical nucleotides allowed.
        max_consecutive_gc: Max consecutive GC/CG pairs allowed.

    Returns:
        Tuple of (is_valid, reason).
    """
    # Check consecutive nucleotides
    violations = find_consecutive_nucleotides(sequence, max_consecutive_nt)

    # Filter out violations that exist in the motifs
    if violations and motif_sequences:
        filtered = []
        for pos, nt in violations:
            run = nt * max_consecutive_nt
            if not any(run in motif for motif in motif_sequences):
                filtered.append((pos, nt))
        violations = filtered

    if violations:
        nt = violations[0][1]
        return False, f"Has {max_consecutive_nt}+ consecutive {nt} nucleotides"

    # Check consecutive GC pairs
    if has_consecutive_gc_pairs(sequence, structure, max_consecutive_gc):
        # Check if this pattern exists in any motif
        if motif_sequences:
            # For GC pairs, we need to check if the motif itself has this pattern
            # This is harder to check, so for now we just flag it
            pass
        return False, f"Has {max_consecutive_gc}+ consecutive GC/CG pairs"

    return True, ""
