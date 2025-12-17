"""Main library generation pipeline."""

from dataclasses import dataclass
from random import Random

import structlog

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
from twoway_lib.validation import check_sequence_constraints, validate_construct

logger = structlog.get_logger(__name__)


@dataclass
class GenerationStats:
    """Statistics from library generation."""

    candidates_generated: int
    candidates_valid: int
    candidates_rejected_length: int
    candidates_rejected_validation: int
    candidates_rejected_sequence: int = 0
    final_library_size: int = 0
    motif_usage: dict[str, int] | None = None


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
        self._motif_usage: dict[str, int] = {m.sequence: 0 for m in motifs}

    def generate(self, num_candidates: int) -> list[Construct]:
        """
        Generate a library of diverse constructs.

        Args:
            num_candidates: Number of candidates to generate before selection.

        Returns:
            List of selected diverse constructs.
        """
        self.stats = GenerationStats(0, 0, 0, 0, 0)
        self._motif_usage = {m.sequence: 0 for m in self.motifs}

        logger.info("Generating candidate constructs", count=num_candidates)
        candidates = self._generate_candidates(num_candidates)
        logger.info("Generated valid candidates", count=len(candidates))

        if len(candidates) <= self.config.optimization.target_library_size:
            self.stats.final_library_size = len(candidates)
            self.stats.motif_usage = dict(self._motif_usage)
            return candidates

        logger.info("Selecting diverse subset with simulated annealing")
        selected = self._select_diverse_library(candidates)
        logger.info("Selected constructs", count=len(selected))

        self.stats.final_library_size = len(selected)
        self._update_usage_for_selected(selected)
        return selected

    def _update_usage_for_selected(self, selected: list[Construct]) -> None:
        """Update motif usage stats for selected constructs."""
        usage: dict[str, int] = {m.sequence: 0 for m in self.motifs}
        for construct in selected:
            for motif in construct.motifs:
                key = motif.sequence
                if key in usage:
                    usage[key] += 1
                else:
                    flipped_key = motif.flip().sequence
                    if flipped_key in usage:
                        usage[flipped_key] += 1
        self.stats.motif_usage = usage

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

        if not self._check_sequence_constraints(construct):
            self.stats.candidates_rejected_sequence += 1
            return None

        if not self._validate(construct):
            self.stats.candidates_rejected_validation += 1
            return None

        return construct

    def _select_motifs(self, count: int) -> list[Motif]:
        """Select motifs with preference for less-used ones, optionally flipping."""
        selected = []
        for _ in range(count):
            motif = self._select_weighted_motif()
            if self.config.allow_motif_flip and self.rng.random() < 0.5:
                motif = motif.flip()
            selected.append(motif)
            self._motif_usage[motif.sequence] = self._motif_usage.get(motif.sequence, 0) + 1
        return selected

    def _select_weighted_motif(self) -> Motif:
        """Select a motif with higher probability for less-used ones."""
        if not self._motif_usage or all(v == 0 for v in self._motif_usage.values()):
            return self.rng.choice(self.motifs)

        max_usage = max(self._motif_usage.values()) + 1
        weights = [max_usage - self._motif_usage.get(m.sequence, 0) for m in self.motifs]
        total = sum(weights)
        if total == 0:
            return self.rng.choice(self.motifs)

        r = self.rng.random() * total
        cumulative = 0
        for motif, weight in zip(self.motifs, weights):
            cumulative += weight
            if r <= cumulative:
                return motif
        return self.motifs[-1]

    def _assemble_with_length_constraint(
        self,
        motifs: list[Motif],
    ) -> Construct | None:
        """Assemble construct and check length constraint."""
        helices = self._generate_helices(len(motifs) + 1)
        if self.config.hairpin_sequence:
            hairpin = hairpin_from_sequence(
                self.config.hairpin_sequence, self.config.hairpin_structure
            )
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

    def _check_sequence_constraints(self, construct: Construct) -> bool:
        """Check sequence constraints (consecutive nucleotides, GC pairs)."""
        val_config = self.config.validation

        # Skip if validation is disabled or neither constraint is enabled
        if not val_config.enabled:
            return True
        if not val_config.avoid_consecutive_nucleotides and not val_config.avoid_consecutive_gc_pairs:
            return True

        # Get motif sequences for exclusion (patterns in motifs are allowed)
        motif_sequences = [m.sequence.replace("&", "") for m in construct.motifs]

        # Check consecutive nucleotides
        if val_config.avoid_consecutive_nucleotides:
            from twoway_lib.validation import find_consecutive_nucleotides

            violations = find_consecutive_nucleotides(
                construct.sequence, val_config.max_consecutive_nucleotides
            )
            # Filter out violations that exist in motifs
            for pos, nt in violations:
                run = nt * val_config.max_consecutive_nucleotides
                if not any(run in motif for motif in motif_sequences):
                    return False

        # Check consecutive GC pairs
        if val_config.avoid_consecutive_gc_pairs:
            from twoway_lib.validation import has_consecutive_gc_pairs

            if has_consecutive_gc_pairs(
                construct.sequence,
                construct.structure,
                val_config.max_consecutive_gc_pairs,
            ):
                return False

        return True

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
