"""Main library generation pipeline."""

import logging
from dataclasses import dataclass
from random import Random

from twoway_lib.config import LibraryConfig
from twoway_lib.construct import (
    Construct,
    assemble_construct,
    calculate_construct_length,
)
from twoway_lib.hairpin import hairpin_from_sequence, random_hairpin
from twoway_lib.helix import Helix, random_helix
from twoway_lib.motif import Motif
from twoway_lib.optimizer import LibraryOptimizer
from twoway_lib.validation import validate_construct

logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """Statistics from library generation."""

    candidates_generated: int
    candidates_valid: int
    candidates_rejected_length: int
    candidates_rejected_validation: int
    final_library_size: int


class LibraryGenerator:
    """
    Main class for generating two-way junction libraries.

    Generates candidate constructs, filters by length and validation,
    then selects a diverse subset using simulated annealing.
    """

    def __init__(
        self,
        config: LibraryConfig,
        motifs: list[Motif],
        seed: int | None = None,
    ):
        """
        Initialize the generator.

        Args:
            config: Library configuration.
            motifs: Available motifs to use.
            seed: Random seed for reproducibility.
        """
        self.config = config
        self.motifs = motifs
        self.rng = Random(seed)
        self.stats = GenerationStats(0, 0, 0, 0, 0)

    def generate(self, num_candidates: int) -> list[Construct]:
        """
        Generate a library of diverse constructs.

        Args:
            num_candidates: Number of candidates to generate before selection.

        Returns:
            List of selected diverse constructs.
        """
        self.stats = GenerationStats(0, 0, 0, 0, 0)

        logger.info(f"Generating {num_candidates} candidate constructs...")
        candidates = self._generate_candidates(num_candidates)
        logger.info(f"Generated {len(candidates)} valid candidates")

        if len(candidates) <= self.config.optimization.target_library_size:
            self.stats.final_library_size = len(candidates)
            return candidates

        logger.info("Selecting diverse subset with simulated annealing...")
        selected = self._select_diverse_library(candidates)
        logger.info(f"Selected {len(selected)} constructs")

        self.stats.final_library_size = len(selected)
        return selected

    def _generate_candidates(self, count: int) -> list[Construct]:
        """Generate valid candidate constructs."""
        candidates = []
        attempts = 0
        max_attempts = count * 10

        while len(candidates) < count and attempts < max_attempts:
            attempts += 1
            construct = self._try_generate_construct()
            if construct is not None:
                candidates.append(construct)

        self.stats.candidates_generated = attempts
        self.stats.candidates_valid = len(candidates)
        return candidates

    def _try_generate_construct(self) -> Construct | None:
        """Attempt to generate a single valid construct."""
        num_motifs = self.rng.randint(*self.config.motifs_per_construct)
        selected_motifs = self._select_motifs(num_motifs)

        construct = self._assemble_with_length_constraint(selected_motifs)
        if construct is None:
            self.stats.candidates_rejected_length += 1
            return None

        if not self._validate(construct):
            self.stats.candidates_rejected_validation += 1
            return None

        return construct

    def _select_motifs(self, count: int) -> list[Motif]:
        """Select random motifs for a construct, optionally flipping."""
        selected = []
        for _ in range(count):
            motif = self.rng.choice(self.motifs)
            if self.config.allow_motif_flip and self.rng.random() < 0.5:
                motif = motif.flip()
            selected.append(motif)
        return selected

    def _assemble_with_length_constraint(
        self,
        motifs: list[Motif],
    ) -> Construct | None:
        """Assemble construct and check length constraint."""
        helices = self._generate_helices(len(motifs) + 1)
        if self.config.hairpin_sequence:
            hairpin = hairpin_from_sequence(self.config.hairpin_sequence)
        else:
            hairpin = random_hairpin(self.config.hairpin_loop_length, self.rng)

        construct = assemble_construct(
            motifs=motifs,
            helices=helices,
            hairpin=hairpin,
            p5_seq=self.config.p5_sequence,
            p5_ss=self.config.p5_structure,
            p3_seq=self.config.p3_sequence,
            p3_ss=self.config.p3_structure,
        )

        if not self._check_length(construct):
            return None
        return construct

    def _generate_helices(self, count: int) -> list[Helix]:
        """Generate helices for a construct."""
        return [random_helix(self.config.helix_length, self.rng) for _ in range(count)]

    def _check_length(self, construct: Construct) -> bool:
        """Check if construct length is within target range."""
        length = construct.length()
        return self.config.target_length_min <= length <= self.config.target_length_max

    def _validate(self, construct: Construct) -> bool:
        """Validate construct with Vienna fold."""
        if not self.config.validation.enabled:
            return True

        result = validate_construct(
            construct,
            max_ensemble_defect=self.config.validation.max_ensemble_defect,
            allow_differences=self.config.validation.allow_structure_differences,
            min_structure_match=self.config.validation.min_structure_match,
        )
        return result.is_valid

    def _select_diverse_library(
        self,
        candidates: list[Construct],
    ) -> list[Construct]:
        """Select diverse subset using simulated annealing."""
        optimizer = LibraryOptimizer(
            constructs=candidates,
            config=self.config.optimization,
            seed=self.rng.randint(0, 2**32 - 1),
        )
        selected_indices = optimizer.optimize()
        return [candidates[i] for i in selected_indices]


def generate_library(
    config: LibraryConfig,
    motifs: list[Motif],
    num_candidates: int = 50000,
    seed: int | None = None,
) -> list[Construct]:
    """
    Generate a two-way junction library.

    Convenience function for the full generation pipeline.

    Args:
        config: Library configuration.
        motifs: Available motifs.
        num_candidates: Number of candidates to generate.
        seed: Random seed.

    Returns:
        List of diverse constructs.
    """
    generator = LibraryGenerator(config, motifs, seed=seed)
    return generator.generate(num_candidates)


def estimate_feasible_lengths(
    config: LibraryConfig,
    motifs: list[Motif],
) -> tuple[int, int]:
    """
    Estimate feasible construct length range given config and motifs.

    Args:
        config: Library configuration.
        motifs: Available motifs.

    Returns:
        Tuple of (min_length, max_length) that can be achieved.
    """
    motif_lengths = [m.total_length() for m in motifs]
    min_motif = min(motif_lengths)
    max_motif = max(motif_lengths)

    min_n, max_n = config.motifs_per_construct
    helix_len = config.helix_length
    hairpin_len = config.hairpin_loop_length
    p5_len = config.p5_length
    p3_len = config.p3_length

    min_total = calculate_construct_length(
        min_n, [min_motif] * min_n, helix_len, hairpin_len, p5_len, p3_len
    )
    max_total = calculate_construct_length(
        max_n, [max_motif] * max_n, helix_len, hairpin_len, p5_len, p3_len
    )

    return min_total, max_total
