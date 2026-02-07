"""Length solver for computing exact helix lengths to hit target construct length."""

from random import Random


def compute_helix_budget(
    target_length: int,
    motif_lengths: list[int],
    p5_len: int,
    p3_len: int,
    hairpin_len: int,
    spacer_5p_len: int = 0,
    spacer_3p_len: int = 0,
) -> int | None:
    """
    Compute total helix base pairs needed to reach target construct length.

    Each helix contributes 2 * bp to the total length (strand1 + strand2).
    The number of helices is len(motif_lengths) + 1.

    Args:
        target_length: Desired total construct length.
        motif_lengths: Total length of each motif (strand1 + strand2).
        p5_len: Length of 5' common sequence.
        p3_len: Length of 3' common sequence.
        hairpin_len: Length of the hairpin loop.
        spacer_5p_len: Length of 5' spacer.
        spacer_3p_len: Length of 3' spacer.

    Returns:
        Total helix base pairs needed, or None if impossible (negative).
    """
    fixed_length = (
        p5_len
        + p3_len
        + sum(motif_lengths)
        + hairpin_len
        + spacer_5p_len
        + spacer_3p_len
    )
    remaining = target_length - fixed_length
    # Each helix bp contributes 2 nucleotides (one on each arm)
    if remaining < 0 or remaining % 2 != 0:
        return None
    return remaining // 2


def random_helix_assignment(
    total_bp: int,
    num_helices: int,
    min_length: int,
    max_length: int,
    rng: Random,
) -> tuple[int, ...] | None:
    """
    Distribute helix base pairs across helices randomly.

    Algorithm: start all helices at min_length, then distribute remaining
    bp one at a time to random helices, capped at max_length.

    Args:
        total_bp: Total base pairs to distribute.
        num_helices: Number of helices to assign lengths to.
        min_length: Minimum length per helix.
        max_length: Maximum length per helix.
        rng: Random number generator.

    Returns:
        Tuple of helix lengths, or None if infeasible.
    """
    if num_helices < 1:
        return None
    if total_bp < num_helices * min_length:
        return None
    if total_bp > num_helices * max_length:
        return None

    lengths = [min_length] * num_helices
    remaining = total_bp - num_helices * min_length

    # Distribute remaining bp randomly
    available = list(range(num_helices))
    while remaining > 0:
        # Filter to helices that can still accept more
        can_grow = [i for i in available if lengths[i] < max_length]
        if not can_grow:
            return None
        idx = rng.choice(can_grow)
        lengths[idx] += 1
        remaining -= 1

    return tuple(lengths)


def is_combo_feasible(
    motif_lengths: list[int],
    target_length: int,
    min_helix: int,
    max_helix: int,
    p5_len: int,
    p3_len: int,
    hairpin_len: int,
    spacer_5p_len: int = 0,
    spacer_3p_len: int = 0,
) -> bool:
    """
    Quick feasibility check for a motif combination at a target length.

    Args:
        motif_lengths: Total length of each motif (strand1 + strand2).
        target_length: Desired total construct length.
        min_helix: Minimum helix length in base pairs.
        max_helix: Maximum helix length in base pairs.
        p5_len: Length of 5' common sequence.
        p3_len: Length of 3' common sequence.
        hairpin_len: Length of the hairpin loop.
        spacer_5p_len: Length of 5' spacer.
        spacer_3p_len: Length of 3' spacer.

    Returns:
        True if it's possible to hit target_length with given constraints.
    """
    budget = compute_helix_budget(
        target_length,
        motif_lengths,
        p5_len,
        p3_len,
        hairpin_len,
        spacer_5p_len,
        spacer_3p_len,
    )
    if budget is None:
        return False

    num_helices = len(motif_lengths) + 1
    min_total = num_helices * min_helix
    max_total = num_helices * max_helix
    return min_total <= budget <= max_total
