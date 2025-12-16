"""Structure validation using Vienna RNA fold."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rna_secstruct import SecStruct

if TYPE_CHECKING:
    from twoway_lib.construct import Construct


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
