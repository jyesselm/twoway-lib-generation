"""Motif preprocessing pipeline for testing and filtering motifs."""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

import pandas as pd
import structlog

from twoway_lib.validation import (
    MotifFoldResult,
    fold_sequence,
)

if True:  # TYPE_CHECKING workaround for runtime imports
    from twoway_lib.motif import Motif

logger = structlog.get_logger(__name__)


@dataclass
class MotifTestResult:
    """Extended motif fold test result with ensemble defect info."""

    motif_sequence: str
    motif_structure: str
    passes: bool
    match_fraction: float
    instances_failed: int
    total_instances: int
    designed_ss: str
    predicted_ss: str
    avg_ensemble_defect: float

    @classmethod
    def from_fold_result(
        cls,
        result: MotifFoldResult,
        avg_ensemble_defect: float,
    ) -> "MotifTestResult":
        """Create from a MotifFoldResult with added ensemble defect.

        Args:
            result: Base fold result.
            avg_ensemble_defect: Average ensemble defect across contexts.

        Returns:
            MotifTestResult with all fields populated.
        """
        return cls(
            motif_sequence=result.motif.sequence,
            motif_structure=result.motif.structure,
            passes=result.passes,
            match_fraction=result.match_fraction,
            instances_failed=result.instances_failed,
            total_instances=result.total_instances,
            designed_ss=result.designed_ss,
            predicted_ss=result.predicted_ss,
            avg_ensemble_defect=avg_ensemble_defect,
        )


def preprocess_motifs(
    motifs: list["Motif"],
    helix_length: int = 3,
    num_contexts: int = 5,
    seed: int = 42,
) -> tuple[list["Motif"], list["Motif"], list[MotifTestResult]]:
    """
    Test each motif in multiple random helix contexts.

    Computes fold pass/fail and average ensemble defect for each motif.

    Args:
        motifs: List of motifs to test.
        helix_length: Length of flanking helices for test constructs.
        num_contexts: Number of random helix contexts to test each motif.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (passing_motifs, failing_motifs, all_results).
    """

    rng = Random(seed)
    passing: list[Motif] = []
    failing: list[Motif] = []
    all_results: list[MotifTestResult] = []

    for motif in motifs:
        fold_result, avg_ed = _test_motif_in_contexts(
            motif,
            helix_length,
            num_contexts,
            rng,
        )
        test_result = MotifTestResult.from_fold_result(fold_result, avg_ed)
        all_results.append(test_result)

        if test_result.passes:
            passing.append(motif)
        else:
            failing.append(motif)
            logger.debug(
                "Motif failed preprocessing",
                motif=motif.sequence,
                match=f"{test_result.match_fraction:.1%}",
                avg_ed=f"{avg_ed:.2f}",
            )

    logger.info(
        "Preprocessing complete",
        passing=len(passing),
        failing=len(failing),
        total=len(motifs),
    )
    return passing, failing, all_results


def _test_motif_in_contexts(
    motif: "Motif",
    helix_length: int,
    num_contexts: int,
    rng: Random,
) -> tuple[MotifFoldResult, float]:
    """Test a motif across multiple helix contexts, return result and avg ED."""
    from twoway_lib.helix import random_helix
    from twoway_lib.validation import test_motif_folding_detailed

    seed = rng.randint(0, 2**31)
    fold_result = test_motif_folding_detailed(
        motif,
        helix_length,
        num_contexts,
        seed,
    )

    # Compute average ensemble defect across contexts
    context_rng = Random(seed)
    helices = [random_helix(helix_length, context_rng) for _ in range(num_contexts + 1)]

    seq_parts: list[str] = []
    ss_parts: list[str] = []

    for i in range(num_contexts):
        seq_parts.append(helices[i].strand1)
        ss_parts.append(helices[i].structure1)
        seq_parts.append(motif.strand1_seq)
        ss_parts.append(motif.strand1_ss)

    seq_parts.append(helices[num_contexts].strand1)
    ss_parts.append(helices[num_contexts].structure1)
    seq_parts.append("GAAA")
    ss_parts.append("....")
    seq_parts.append(helices[num_contexts].strand2)
    ss_parts.append(helices[num_contexts].structure2)

    for i in range(num_contexts - 1, -1, -1):
        seq_parts.append(motif.strand2_seq)
        ss_parts.append(motif.strand2_ss)
        seq_parts.append(helices[i].strand2)
        ss_parts.append(helices[i].structure2)

    sequence = "".join(seq_parts)
    fr = fold_sequence(sequence)
    avg_ed = fr.ensemble_defect

    return fold_result, avg_ed


def save_motif_results(results: list[MotifTestResult], path: Path | str) -> None:
    """
    Serialize motif test results to JSON.

    Args:
        results: List of MotifTestResult to save.
        path: Output file path.
    """
    path = Path(path)
    data = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved motif results", path=str(path), count=len(results))


def load_motif_results(path: Path | str) -> list[MotifTestResult]:
    """
    Deserialize motif test results from JSON.

    Args:
        path: Path to JSON file.

    Returns:
        List of MotifTestResult.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Motif results file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    results = [MotifTestResult(**entry) for entry in data]
    logger.info("Loaded motif results", path=str(path), count=len(results))
    return results


@dataclass
class PredictedMotifResult:
    """Result of predicting a motif's ViennaRNA structure.

    Attributes:
        motif_id: Unique motif identifier.
        pdb_id: Source PDB structure ID.
        sequence: Motif sequence with & separator (e.g., "CGCGG&CCCG").
        predicted_structure: ViennaRNA consensus structure.
        pdb_structure: Original crystal structure.
        match_fraction: Fraction of positions where PDB matches prediction.
        consensus_agreement: Fraction of contexts agreeing with consensus.
    """

    motif_id: str
    pdb_id: str
    sequence: str
    predicted_structure: str
    pdb_structure: str
    match_fraction: float
    consensus_agreement: float


def predict_motif_structures(
    entries: list,
    num_contexts: int = 50,
    helix_length: int = 3,
    seed: int = 42,
) -> list[PredictedMotifResult]:
    """Predict ViennaRNA structures for curated PDB motifs.

    Embeds each motif in multiple random helix contexts, folds with
    ViennaRNA, and computes a per-position consensus prediction.

    Args:
        entries: List of CuratedMotifEntry objects.
        num_contexts: Number of independent helix contexts to test.
        helix_length: Length of flanking helices.
        seed: Random seed for reproducibility.

    Returns:
        List of PredictedMotifResult with predicted structures.
    """
    from twoway_lib.motif import Motif

    rng = Random(seed)
    results: list[PredictedMotifResult] = []

    for idx, entry in enumerate(entries):
        if idx % 50 == 0:
            logger.info("Predicting motif structures", index=idx, total=len(entries))

        motif = Motif.from_string(entry.sequence, entry.structure)
        predictions = [
            _predict_single_context(motif, helix_length, Random(rng.randint(0, 2**31)))
            for _ in range(num_contexts)
        ]
        consensus, agreement = _compute_consensus(predictions)
        match = _compute_match_fraction(consensus, entry.structure)

        results.append(
            PredictedMotifResult(
                motif_id=entry.motif_id,
                pdb_id=entry.pdb_id,
                sequence=entry.sequence,
                predicted_structure=consensus,
                pdb_structure=entry.structure,
                match_fraction=match,
                consensus_agreement=agreement,
            )
        )

    # Filter out invalid predicted structures
    kept = []
    helix_only = []
    unpaired_ends = []
    for r in results:
        if _is_pure_helix(r.predicted_structure):
            helix_only.append(r)
        elif _has_unpaired_ends(r.predicted_structure):
            unpaired_ends.append(r)
        else:
            kept.append(r)

    if helix_only:
        logger.warning(
            "Removed motifs predicted as pure helix (no internal loops/bulges)",
            removed=len(helix_only),
        )
        for r in helix_only:
            logger.warning(
                "Pure helix motif removed",
                motif_id=r.motif_id,
                sequence=r.sequence,
                predicted=r.predicted_structure,
            )

    if unpaired_ends:
        logger.warning(
            "Removed motifs with unpaired flanking positions",
            removed=len(unpaired_ends),
        )
        for r in unpaired_ends:
            logger.warning(
                "Unpaired ends motif removed",
                motif_id=r.motif_id,
                sequence=r.sequence,
                predicted=r.predicted_structure,
            )

    total_removed = len(helix_only) + len(unpaired_ends)
    logger.info(
        "Structure prediction complete",
        total=len(kept),
        removed_helix=len(helix_only),
        removed_unpaired_ends=len(unpaired_ends),
    )
    return kept


def _is_pure_helix(structure: str) -> bool:
    """Check if a predicted structure is all paired with no dots.

    Args:
        structure: Structure string with & separator.

    Returns:
        True if structure contains only '(' , ')' and '&' (no dots).
    """
    return "." not in structure


def _has_unpaired_ends(structure: str) -> bool:
    """Check if either strand has unpaired dots at its ends.

    The flanking positions must be paired to connect the motif to
    adjacent helices. Dots on the ends mean ViennaRNA doesn't predict
    the closing base pair.

    Args:
        structure: Structure string with & separator.

    Returns:
        True if any strand starts or ends with '.'.
    """
    parts = structure.split("&")
    s1, s2 = parts[0], parts[1]
    return s1[0] == "." or s1[-1] == "." or s2[0] == "." or s2[-1] == "."


def _build_pair_table(structure: str) -> list[int]:
    """Build a pair table from dot-bracket structure.

    Args:
        structure: Dot-bracket secondary structure string.

    Returns:
        List where pair_table[i] = j if i pairs with j, else -1.
    """
    pair_table = [-1] * len(structure)
    stack: list[int] = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_table[i] = j
                pair_table[j] = i
    return pair_table


def _clean_cross_boundary_pairs(
    structure: str,
    motif_positions: set[int],
    pair_table: list[int],
) -> str:
    """Replace brackets that pair across motif/helix boundary with dots.

    If a motif position pairs with a non-motif position, both are
    artifacts of the extraction and should be treated as unpaired.

    Args:
        structure: Full predicted structure.
        motif_positions: Set of positions belonging to the motif.
        pair_table: Pair table from _build_pair_table.

    Returns:
        Cleaned structure string with cross-boundary pairs replaced by '.'.
    """
    chars = list(structure)
    for pos in motif_positions:
        partner = pair_table[pos]
        if partner != -1 and partner not in motif_positions:
            chars[pos] = "."
    return "".join(chars)


def _remove_invalid_chars(strand_ss: str, invalid: str) -> str:
    """Replace invalid bracket characters with dots and clean orphaned partners.

    For a two-way junction, strand1 should only have '(' and '.',
    strand2 should only have ')' and '.'. This removes any violations
    and also removes the now-orphaned partner bracket.

    Args:
        strand_ss: Structure string for one strand.
        invalid: Character to remove ('(' or ')').

    Returns:
        Cleaned structure string.
    """
    if invalid not in strand_ss:
        return strand_ss

    chars = list(strand_ss)
    pair_table = _build_pair_table(strand_ss)
    for i, char in enumerate(chars):
        if char == invalid:
            chars[i] = "."
            partner = pair_table[i]
            if 0 <= partner < len(chars):
                chars[partner] = "."
    return "".join(chars)


def _predict_single_context(
    motif: "Motif",
    helix_length: int,
    rng: Random,
) -> str:
    """Embed motif in one helix context and return predicted structure.

    Builds: helix + motif_strand1 + helix + GAAA + helix + motif_strand2 + helix
    Folds with ViennaRNA and extracts the predicted structure for the motif region.

    Args:
        motif: Motif to test.
        helix_length: Length of flanking helices.
        rng: Random number generator.

    Returns:
        Predicted structure string for the motif (e.g., "((.(&)))").
    """
    from twoway_lib.helix import random_helix

    h1 = random_helix(helix_length, rng)
    h2 = random_helix(helix_length, rng)

    sequence = (
        h1.strand1
        + motif.strand1_seq
        + h2.strand1
        + "GAAA"
        + h2.strand2
        + motif.strand2_seq
        + h1.strand2
    )

    fr = fold_sequence(sequence)
    predicted = fr.predicted_structure

    s1_start = helix_length
    s1_end = s1_start + len(motif.strand1_seq)
    s2_start = helix_length + len(motif.strand1_seq) + helix_length + 4 + helix_length
    s2_end = s2_start + len(motif.strand2_seq)

    motif_positions = set(range(s1_start, s1_end)) | set(range(s2_start, s2_end))
    pair_table = _build_pair_table(predicted)
    cleaned = _clean_cross_boundary_pairs(predicted, motif_positions, pair_table)

    pred_s1 = cleaned[s1_start:s1_end]
    pred_s2 = cleaned[s2_start:s2_end]

    # Remove strand-local hairpins: ) in strand1 or ( in strand2
    pred_s1 = _remove_invalid_chars(pred_s1, invalid=")")
    pred_s2 = _remove_invalid_chars(pred_s2, invalid="(")
    return pred_s1 + "&" + pred_s2


def _compute_consensus(predictions: list[str]) -> tuple[str, float]:
    """Select the most common full structure from predictions.

    Uses whole-structure voting rather than per-position voting to
    guarantee the result is a valid, balanced structure.

    Args:
        predictions: List of predicted structure strings.

    Returns:
        Tuple of (consensus_structure, agreement_fraction).
    """
    if not predictions:
        return "", 0.0

    counts: Counter[str] = Counter(predictions)
    best_structure, best_count = counts.most_common(1)[0]
    agreement = best_count / len(predictions)
    return best_structure, agreement


def _compute_match_fraction(predicted: str, reference: str) -> float:
    """Compute fraction of positions where predicted matches reference.

    Args:
        predicted: Predicted structure string.
        reference: Reference (PDB) structure string.

    Returns:
        Fraction of matching positions, ignoring the '&' separator.
    """
    pred_chars = [c for c in predicted if c != "&"]
    ref_chars = [c for c in reference if c != "&"]

    if not ref_chars:
        return 0.0

    matches = sum(p == r for p, r in zip(pred_chars, ref_chars, strict=False))
    return matches / len(ref_chars)


def save_predicted_motifs_csv(
    results: list[PredictedMotifResult],
    path: Path | str,
) -> None:
    """Save predicted structures as CSV compatible with load_motifs().

    Output CSV has sequence and structure columns (for load_motifs compatibility)
    plus extra metadata columns.

    Args:
        results: Prediction results to save.
        path: Output file path.
    """
    rows = [
        {
            "sequence": r.sequence,
            "structure": r.predicted_structure,
            "pdb_structure": r.pdb_structure,
            "motif_id": r.motif_id,
            "pdb_id": r.pdb_id,
            "match_fraction": r.match_fraction,
            "consensus_agreement": r.consensus_agreement,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Saved predicted motifs CSV", path=str(path), count=len(results))
