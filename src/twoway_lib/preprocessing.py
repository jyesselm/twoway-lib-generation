"""Motif preprocessing pipeline for testing and filtering motifs."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

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
