"""Watson-Crick helix generation for RNA constructs."""

from dataclasses import dataclass
from itertools import product
from random import Random

from rna_secstruct import SecStruct

# Standard Watson-Crick base pairs
WC_PAIRS: list[tuple[str, str]] = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")]


@dataclass(frozen=True)
class Helix:
    """
    A Watson-Crick base-paired helix segment.

    Strand 1 goes 5' to 3' on the left arm of the hairpin.
    Strand 2 goes 5' to 3' on the right arm (reverse complement direction).

    Internally uses rna_secstruct.SecStruct for each strand.
    """

    _strand1_ss: SecStruct
    _strand2_ss: SecStruct

    @classmethod
    def from_sequences(
        cls, strand1: str, strand2: str, structure1: str, structure2: str
    ) -> "Helix":
        """Create a Helix from strand sequences and structures."""
        return cls(
            _strand1_ss=SecStruct(strand1, structure1),
            _strand2_ss=SecStruct(strand2, structure2),
        )

    @property
    def strand1(self) -> str:
        """First strand sequence (e.g., 'AGC')."""
        return self._strand1_ss.sequence

    @property
    def strand2(self) -> str:
        """Second strand sequence, reverse complement (e.g., 'GCU')."""
        return self._strand2_ss.sequence

    @property
    def structure1(self) -> str:
        """First strand structure (e.g., '(((')."""
        return self._strand1_ss.structure

    @property
    def structure2(self) -> str:
        """Second strand structure (e.g., ')))')."""
        return self._strand2_ss.structure

    def length(self) -> int:
        """Return the number of base pairs."""
        return len(self._strand1_ss)

    def strand1_secstruct(self) -> SecStruct:
        """Return SecStruct for strand1."""
        return self._strand1_ss

    def strand2_secstruct(self) -> SecStruct:
        """Return SecStruct for strand2."""
        return self._strand2_ss


def generate_all_helices(length: int) -> list[Helix]:
    """
    Generate all possible Watson-Crick helices of given length.

    For length n, generates 4^n helices (all combinations of 4 base pairs).

    Args:
        length: Number of base pairs in the helix.

    Returns:
        List of all possible Helix objects.

    Raises:
        ValueError: If length is less than 1.
    """
    if length < 1:
        raise ValueError("Helix length must be at least 1")

    helices = []
    for bp_combo in product(WC_PAIRS, repeat=length):
        helix = _create_helix_from_pairs(bp_combo)
        helices.append(helix)

    return helices


def _create_helix_from_pairs(bp_combo: tuple[tuple[str, str], ...]) -> Helix:
    """
    Create a Helix from a tuple of base pair tuples.

    Args:
        bp_combo: Tuple of (nt1, nt2) base pairs.

    Returns:
        Helix object with assembled strands.
    """
    strand1 = "".join(bp[0] for bp in bp_combo)
    strand2 = "".join(bp[1] for bp in reversed(bp_combo))
    structure1 = "(" * len(bp_combo)
    structure2 = ")" * len(bp_combo)

    return Helix.from_sequences(strand1, strand2, structure1, structure2)


def random_helix(length: int, rng: Random | None = None) -> Helix:
    """
    Generate a random Watson-Crick helix.

    Args:
        length: Number of base pairs in the helix.
        rng: Random number generator (uses default if None).

    Returns:
        Randomly generated Helix object.

    Raises:
        ValueError: If length is less than 1.
    """
    if length < 1:
        raise ValueError("Helix length must be at least 1")

    if rng is None:
        rng = Random()

    bp_combo = tuple(rng.choice(WC_PAIRS) for _ in range(length))
    return _create_helix_from_pairs(bp_combo)


def helix_from_strand1(strand1: str) -> Helix:
    """
    Create a helix from the first strand sequence.

    Automatically generates the complementary second strand.

    Args:
        strand1: First strand sequence (only A, U, G, C).

    Returns:
        Helix with computed complement.

    Raises:
        ValueError: If strand contains invalid nucleotides.
    """
    complement_map = {"A": "U", "U": "A", "G": "C", "C": "G"}

    strand2_chars = []
    for nt in reversed(strand1):
        if nt not in complement_map:
            raise ValueError(f"Invalid nucleotide in strand1: {nt}")
        strand2_chars.append(complement_map[nt])

    strand2 = "".join(strand2_chars)
    structure1 = "(" * len(strand1)
    structure2 = ")" * len(strand1)

    return Helix.from_sequences(strand1, strand2, structure1, structure2)
