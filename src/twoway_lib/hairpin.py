"""Hairpin loop generation for RNA constructs."""

from dataclasses import dataclass
from itertools import product
from random import Random

# RNA nucleotides for loop generation
RNA_NUCLEOTIDES: list[str] = ["A", "U", "G", "C"]

# Common stable tetraloop sequences
STABLE_TETRALOOPS: list[str] = ["GAAA", "GCAA", "GAGA", "UUCG", "CUUG", "GUAA"]


@dataclass(frozen=True)
class Hairpin:
    """
    A hairpin loop sequence.

    The hairpin loop sits at the top of the construct, connecting
    the 5' and 3' arms of the hairpin stem.

    Attributes:
        sequence: Loop sequence (e.g., "GAAA").
        structure: Loop structure (all unpaired, e.g., "....").
    """

    sequence: str
    structure: str

    def length(self) -> int:
        """Return the loop length."""
        return len(self.sequence)


def generate_all_hairpins(length: int) -> list[Hairpin]:
    """
    Generate all possible hairpin loops of given length.

    For length n, generates 4^n hairpins (all nucleotide combinations).

    Args:
        length: Number of nucleotides in the loop.

    Returns:
        List of all possible Hairpin objects.

    Raises:
        ValueError: If length is less than 3.
    """
    if length < 3:
        raise ValueError("Hairpin loop length must be at least 3")

    hairpins = []
    structure = "." * length

    for nts in product(RNA_NUCLEOTIDES, repeat=length):
        sequence = "".join(nts)
        hairpins.append(Hairpin(sequence=sequence, structure=structure))

    return hairpins


def random_hairpin(length: int, rng: Random | None = None) -> Hairpin:
    """
    Generate a random hairpin loop.

    Args:
        length: Number of nucleotides in the loop.
        rng: Random number generator (uses default if None).

    Returns:
        Randomly generated Hairpin object.

    Raises:
        ValueError: If length is less than 3.
    """
    if length < 3:
        raise ValueError("Hairpin loop length must be at least 3")

    if rng is None:
        rng = Random()

    sequence = "".join(rng.choice(RNA_NUCLEOTIDES) for _ in range(length))
    structure = "." * length

    return Hairpin(sequence=sequence, structure=structure)


def get_stable_tetraloops() -> list[Hairpin]:
    """
    Get list of known stable tetraloop sequences.

    These tetraloops are thermodynamically stable and commonly
    used in RNA structure design.

    Returns:
        List of Hairpin objects with stable tetraloop sequences.
    """
    return [Hairpin(sequence=seq, structure="....") for seq in STABLE_TETRALOOPS]


def random_stable_tetraloop(rng: Random | None = None) -> Hairpin:
    """
    Get a random stable tetraloop.

    Args:
        rng: Random number generator (uses default if None).

    Returns:
        Randomly selected stable tetraloop Hairpin.
    """
    if rng is None:
        rng = Random()

    seq = rng.choice(STABLE_TETRALOOPS)
    return Hairpin(sequence=seq, structure="....")


def hairpin_from_sequence(sequence: str) -> Hairpin:
    """
    Create a hairpin from a specific sequence.

    Args:
        sequence: RNA sequence for the hairpin loop.

    Returns:
        Hairpin object with the specified sequence.

    Raises:
        ValueError: If sequence is too short or contains invalid nucleotides.
    """
    sequence = sequence.upper().replace("T", "U")
    if len(sequence) < 3:
        raise ValueError("Hairpin loop length must be at least 3")
    valid_nts = set("AUGC")
    invalid = set(sequence) - valid_nts
    if invalid:
        raise ValueError(f"Hairpin sequence contains invalid nucleotides: {invalid}")
    structure = "." * len(sequence)
    return Hairpin(sequence=sequence, structure=structure)
