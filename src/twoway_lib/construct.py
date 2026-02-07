"""Construct assembly for RNA hairpin libraries using rna_secstruct."""

from dataclasses import dataclass, field

from rna_secstruct import SecStruct

from twoway_lib.hairpin import Hairpin
from twoway_lib.helix import Helix
from twoway_lib.motif import Motif


@dataclass
class MotifPosition:
    """
    Tracks the position of a motif within a construct.

    Attributes:
        motif: The motif object.
        strand1_positions: List of 0-indexed positions for strand1 nucleotides.
        strand2_positions: List of 0-indexed positions for strand2 nucleotides.
    """

    motif: Motif
    strand1_positions: list[int]
    strand2_positions: list[int]

    def all_positions(self) -> list[int]:
        """Return all positions for this motif (strand1 + strand2)."""
        return self.strand1_positions + self.strand2_positions

    def to_string(self) -> str:
        """Format positions as a readable string."""
        all_pos = self.all_positions()
        return f"[{','.join(str(p) for p in all_pos)}]"


@dataclass
class Construct:
    """
    A complete RNA hairpin construct with motifs.

    The construct is a single-stranded RNA that folds into a hairpin
    structure with multiple two-way junction motifs embedded in the stem.

    Attributes:
        sequence: Full RNA sequence.
        structure: Full dot-bracket secondary structure.
        motifs: List of motifs included in this construct.
        motif_positions: List of MotifPosition objects tracking nucleotide locations.
    """

    sequence: str
    structure: str
    motifs: list[Motif]
    motif_positions: list[MotifPosition] = field(default_factory=list)

    def length(self) -> int:
        """Return the sequence length."""
        return len(self.sequence)

    def get_positions_string(self) -> str:
        """Return a formatted string of all motif positions."""
        parts = []
        for i, mp in enumerate(self.motif_positions):
            parts.append(f"M{i + 1}:{mp.to_string()}")
        return ";".join(parts)

    def to_secstruct(self) -> SecStruct:
        """Convert to rna_secstruct.SecStruct object."""
        return SecStruct(self.sequence, self.structure)

    def is_valid(self) -> bool:
        """Check if the construct has valid sequence/structure using SecStruct."""
        try:
            return self.to_secstruct().is_valid()
        except ValueError:
            return False

    def validate(self) -> None:
        """
        Validate that the construct has well-formed sequence and structure.

        Raises:
            ValueError: If the construct is invalid.
        """
        self.to_secstruct().validate()


def assemble_construct(
    motifs: list[Motif],
    helices: list[Helix],
    hairpin: Hairpin,
    p5_seq: str,
    p5_ss: str,
    p3_seq: str,
    p3_ss: str,
    spacer_5p_seq: str | None = None,
    spacer_5p_ss: str | None = None,
    spacer_3p_seq: str | None = None,
    spacer_3p_ss: str | None = None,
) -> Construct:
    """
    Assemble a complete construct from components using rna_secstruct.

    Layout (5' to 3'):
    P5 -- Spacer5p -- H -- M1_s1 -- H -- ... -- Mn_s1 -- H -- hairpin
    -- H -- Mn_s2 -- ... -- M1_s2 -- H -- Spacer3p -- P3

    Args:
        motifs: List of motifs to include.
        helices: List of helices (need num_motifs + 1 helices).
        hairpin: Hairpin loop at the top.
        p5_seq: 5' common sequence.
        p5_ss: 5' common structure.
        p3_seq: 3' common sequence.
        p3_ss: 3' common structure.
        spacer_5p_seq: Optional 5' spacer sequence.
        spacer_5p_ss: Optional 5' spacer structure.
        spacer_3p_seq: Optional 3' spacer sequence.
        spacer_3p_ss: Optional 3' spacer structure.

    Returns:
        Assembled Construct object with tracked motif positions.

    Raises:
        ValueError: If number of helices doesn't match requirements.
    """
    expected_helices = len(motifs) + 1
    if len(helices) != expected_helices:
        raise ValueError(f"Need {expected_helices} helices, got {len(helices)}")

    # Build left arm using SecStruct operations
    left_ss, strand1_positions = _build_left_arm_secstruct(
        motifs,
        helices,
        p5_seq,
        p5_ss,
        spacer_5p_seq,
        spacer_5p_ss,
    )

    # Add hairpin
    hairpin_ss = SecStruct(hairpin.sequence, hairpin.structure)
    left_length = len(left_ss)

    # Build right arm
    right_ss, strand2_positions = _build_right_arm_secstruct(
        motifs,
        helices,
        p3_seq,
        p3_ss,
        left_length + len(hairpin_ss),
        spacer_3p_seq,
        spacer_3p_ss,
    )

    # Combine: left + hairpin + right
    full_ss = left_ss + hairpin_ss + right_ss

    # Create motif positions
    motif_positions = _create_motif_positions(
        motifs, strand1_positions, strand2_positions
    )

    return Construct(
        sequence=full_ss.sequence,
        structure=full_ss.structure,
        motifs=list(motifs),
        motif_positions=motif_positions,
    )


def _build_left_arm_secstruct(
    motifs: list[Motif],
    helices: list[Helix],
    p5_seq: str,
    p5_ss: str,
    spacer_5p_seq: str | None = None,
    spacer_5p_ss: str | None = None,
) -> tuple[SecStruct, list[list[int]]]:
    """
    Build the left (5') arm using SecStruct operations.

    Assembles: P5 -- Spacer5p -- H -- M1_s1 -- H -- ... -- Mn_s1 -- H

    Returns:
        Tuple of (SecStruct, strand1_positions) for the left arm.
    """
    result = SecStruct(p5_seq, p5_ss)
    current_pos = len(p5_seq)
    strand1_positions: list[list[int]] = []

    # Add 5' spacer if provided
    if spacer_5p_seq is not None and spacer_5p_ss is not None:
        spacer_ss = SecStruct(spacer_5p_seq, spacer_5p_ss)
        result = result + spacer_ss
        current_pos += len(spacer_ss)

    for i, motif in enumerate(motifs):
        # Add helix
        helix_ss = SecStruct(helices[i].strand1, helices[i].structure1)
        result = result + helix_ss
        current_pos += len(helix_ss)

        # Track motif strand1 position
        motif_start = current_pos
        motif_ss = SecStruct(motif.strand1_seq, motif.strand1_ss)
        result = result + motif_ss
        current_pos += len(motif_ss)
        strand1_positions.append(list(range(motif_start, current_pos)))

    # Add final helix
    final_helix_ss = SecStruct(helices[-1].strand1, helices[-1].structure1)
    result = result + final_helix_ss

    return result, strand1_positions


def _build_right_arm_secstruct(
    motifs: list[Motif],
    helices: list[Helix],
    p3_seq: str,
    p3_ss: str,
    start_offset: int,
    spacer_3p_seq: str | None = None,
    spacer_3p_ss: str | None = None,
) -> tuple[SecStruct, list[list[int]]]:
    """
    Build the right (3') arm using SecStruct operations.

    Assembles: H -- Mn_s2 -- ... -- M1_s2 -- H -- Spacer3p -- P3

    Returns:
        Tuple of (SecStruct, strand2_positions) for the right arm.
        strand2_positions is in original motif order.
    """
    result = SecStruct(helices[-1].strand2, helices[-1].structure2)
    current_pos = start_offset + len(helices[-1].strand2)
    strand2_positions: list[list[int]] = [[] for _ in motifs]

    for i in range(len(motifs) - 1, -1, -1):
        # Track motif strand2 position
        motif_start = current_pos
        motif_ss = SecStruct(motifs[i].strand2_seq, motifs[i].strand2_ss)
        result = result + motif_ss
        current_pos += len(motif_ss)
        strand2_positions[i] = list(range(motif_start, current_pos))

        # Add helix
        helix_ss = SecStruct(helices[i].strand2, helices[i].structure2)
        result = result + helix_ss
        current_pos += len(helix_ss)

    # Add 3' spacer if provided
    if spacer_3p_seq is not None and spacer_3p_ss is not None:
        spacer_ss = SecStruct(spacer_3p_seq, spacer_3p_ss)
        result = result + spacer_ss
        current_pos += len(spacer_ss)

    # Add P3
    p3_ss_struct = SecStruct(p3_seq, p3_ss)
    result = result + p3_ss_struct

    return result, strand2_positions


def _create_motif_positions(
    motifs: list[Motif],
    strand1_positions: list[list[int]],
    strand2_positions: list[list[int]],
) -> list[MotifPosition]:
    """Create MotifPosition objects from position lists."""
    return [
        MotifPosition(
            motif=motif,
            strand1_positions=s1_pos,
            strand2_positions=s2_pos,
        )
        for motif, s1_pos, s2_pos in zip(
            motifs, strand1_positions, strand2_positions, strict=True
        )
    ]


def calculate_construct_length(
    num_motifs: int,
    motif_lengths: list[int],
    helix_length: int | list[int],
    hairpin_length: int,
    p5_length: int,
    p3_length: int,
    spacer_5p_length: int = 0,
    spacer_3p_length: int = 0,
) -> int:
    """
    Calculate total length of an assembled construct.

    Args:
        num_motifs: Number of motifs in the construct.
        motif_lengths: Total length of each motif (strand1 + strand2).
        helix_length: Length of each helix (int for uniform, list for variable).
        hairpin_length: Length of the hairpin loop.
        p5_length: Length of the 5' common sequence.
        p3_length: Length of the 3' common sequence.
        spacer_5p_length: Length of 5' spacer sequence.
        spacer_3p_length: Length of 3' spacer sequence.

    Returns:
        Total construct length in nucleotides.
    """
    num_helices = num_motifs + 1
    if isinstance(helix_length, list):
        total_helix_length = sum(helix_length) * 2
    else:
        total_helix_length = num_helices * helix_length * 2
    total_motif_length = sum(motif_lengths)

    return (
        p5_length
        + p3_length
        + total_motif_length
        + total_helix_length
        + hairpin_length
        + spacer_5p_length
        + spacer_3p_length
    )


def estimate_construct_length(
    num_motifs: int,
    avg_motif_length: int,
    helix_length: int,
    hairpin_length: int,
    p5_length: int,
    p3_length: int,
) -> int:
    """
    Estimate construct length using average motif length.

    Useful for planning when exact motifs aren't yet selected.

    Args:
        num_motifs: Number of motifs.
        avg_motif_length: Average total length per motif.
        helix_length: Helix length in base pairs.
        hairpin_length: Hairpin loop length.
        p5_length: 5' sequence length.
        p3_length: 3' sequence length.

    Returns:
        Estimated total length.
    """
    motif_lengths = [avg_motif_length] * num_motifs
    return calculate_construct_length(
        num_motifs, motif_lengths, helix_length, hairpin_length, p5_length, p3_length
    )
