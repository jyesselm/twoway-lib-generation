"""Main library generation pipeline."""

from concurrent.futures import ProcessPoolExecutor
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
from twoway_lib.helix import Helix, random_helix, random_helix_with_gu_requirement
from twoway_lib.length_solver import compute_helix_budget, random_helix_assignment
from twoway_lib.motif import Motif
from twoway_lib.optimizer import LibraryOptimizer
from twoway_lib.validation import (
    filter_foldable_motifs,
    validate_construct,
)

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
        filter_motifs: bool = True,
    ):
        """
        Initialize the generator.

        Args:
            config: Library configuration.
            motifs: Available motifs to use.
            seed: Random seed for reproducibility.
            filter_motifs: If True, filter out motifs that don't fold correctly.
        """
        self.config = config
        self.rng = Random(seed)
        self.excluded_motifs: list[Motif] = []

        # Filter motifs if requested
        if filter_motifs:
            passing, failing, failed_results = filter_foldable_motifs(
                motifs, helix_length=config.helix_length
            )
            self.motifs = passing
            self.excluded_motifs = failing
            if failing:
                logger.info(
                    "Filtered motifs by fold test",
                    passing=len(passing),
                    excluded=len(failing),
                )
                for result in failed_results:
                    logger.debug(
                        "Excluded motif",
                        motif=result.motif.sequence,
                        failed=f"{result.instances_failed}/{result.total_instances}",
                        designed=result.designed_ss,
                        predicted=result.predicted_ss,
                    )
            if len(passing) < config.motifs_per_construct_min:
                logger.warning(
                    "Few motifs pass fold test",
                    passing=len(passing),
                    required=config.motifs_per_construct_min,
                    hint="Use --no-filter-motifs to include all motifs",
                )
        else:
            self.motifs = motifs

        self.stats = GenerationStats(0, 0, 0, 0, 0)
        self._motif_usage: dict[str, int] = {m.sequence: 0 for m in self.motifs}

    def generate(
        self,
        num_candidates: int,
        max_attempts: int | None = None,
        auto_tune: bool = False,
        parallel: bool = False,
        n_workers: int = 4,
    ) -> list[Construct]:
        """
        Generate a library of diverse constructs.

        Args:
            num_candidates: Number of candidates to generate before selection.
            max_attempts: Maximum attempts to generate candidates.
            auto_tune: If True, auto-tune annealing parameters.
            parallel: If True, use parallel generation.
            n_workers: Number of workers for parallel generation.

        Returns:
            List of selected diverse constructs.
        """
        self.stats = GenerationStats(0, 0, 0, 0, 0)
        self._motif_usage = {m.sequence: 0 for m in self.motifs}

        logger.info("Generating candidate constructs", count=num_candidates)

        if parallel:
            candidates = self._generate_candidates_parallel(
                num_candidates,
                max_attempts,
                n_workers,
            )
        else:
            candidates = self._generate_candidates(num_candidates, max_attempts)

        logger.info("Generated valid candidates", count=len(candidates))

        if len(candidates) <= self.config.optimization.target_library_size:
            self.stats.final_library_size = len(candidates)
            self.stats.motif_usage = dict(self._motif_usage)
            return candidates

        logger.info("Selecting diverse subset with simulated annealing")
        selected = self._select_diverse_library(candidates, auto_tune=auto_tune)
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

    def _generate_candidates(
        self, count: int, max_attempts: int | None = None
    ) -> list[Construct]:
        """Generate valid candidate constructs."""
        candidates: list[Construct] = []
        attempts = 0
        if max_attempts is None:
            max_attempts = count * 100
        log_interval = max(1, count // 10)  # Log every 10%

        logger.debug("Starting generation", target=count, max_attempts=max_attempts)

        while len(candidates) < count and attempts < max_attempts:
            attempts += 1
            construct = self._try_generate_construct()
            if construct is not None:
                candidates.append(construct)
                if len(candidates) % log_interval == 0:
                    logger.debug(
                        "Generation progress",
                        valid=len(candidates),
                        attempts=attempts,
                        target=count,
                    )

        self.stats.candidates_generated = attempts
        self.stats.candidates_valid = len(candidates)

        if len(candidates) < count:
            logger.warning(
                "Could not generate all requested candidates",
                requested=count,
                generated=len(candidates),
                attempts=attempts,
                hint="Try --max-attempts to increase attempts or relax constraints",
            )

        return candidates

    def _generate_candidates_parallel(
        self,
        count: int,
        max_attempts: int | None = None,
        n_workers: int = 4,
    ) -> list[Construct]:
        """
        Generate candidates using multiple processes.

        Args:
            count: Total number of candidates to generate.
            max_attempts: Maximum attempts per batch.
            n_workers: Number of parallel workers.

        Returns:
            List of valid candidate constructs.
        """
        batch_size = count // n_workers
        if max_attempts is None:
            batch_max = batch_size * 100
        else:
            batch_max = max_attempts // n_workers

        # Generate seeds for each worker
        seeds = [self.rng.randint(0, 2**31) for _ in range(n_workers)]

        logger.info(
            "Starting parallel generation",
            workers=n_workers,
            batch_size=batch_size,
        )

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _generate_candidate_batch,
                    self.config,
                    self.motifs,
                    batch_size,
                    batch_max,
                    seed,
                )
                for seed in seeds
            ]
            all_candidates: list[Construct] = []
            for future in futures:
                all_candidates.extend(future.result())

        self.stats.candidates_generated = count
        self.stats.candidates_valid = len(all_candidates)
        return all_candidates

    def _try_generate_construct(self) -> Construct | None:
        """Attempt to generate a single valid construct."""
        num_motifs = self.rng.randint(*self.config.motifs_per_construct)
        selected_motifs = self._select_motifs(num_motifs)

        logger.debug(
            "Attempting construct",
            num_motifs=num_motifs,
            motifs=[m.sequence for m in selected_motifs],
        )

        construct = self._assemble_with_length_solver(selected_motifs)
        if construct is None:
            logger.debug("Rejected: length out of range")
            self.stats.candidates_rejected_length += 1
            return None

        passed, reason = self._check_sequence_constraints(construct)
        if not passed:
            logger.debug(
                "Rejected: sequence constraint",
                length=construct.length(),
                reason=reason,
                sequence=construct.sequence,
                structure=construct.structure,
            )
            self.stats.candidates_rejected_sequence += 1
            return None

        if not self._validate(construct):
            logger.debug(
                "Rejected: validation failed",
                length=construct.length(),
                sequence=construct.sequence,
                structure=construct.structure,
            )
            self.stats.candidates_rejected_validation += 1
            return None

        logger.debug(
            "Accepted construct",
            length=construct.length(),
            sequence=construct.sequence,
            structure=construct.structure,
            motifs=[m.sequence for m in construct.motifs],
        )
        return construct

    def _select_motifs(self, count: int) -> list[Motif]:
        """Select motifs with preference for less-used ones, optionally flipping."""
        selected = []
        for _ in range(count):
            motif = self._select_weighted_motif()
            if self.config.allow_motif_flip and self.rng.random() < 0.5:
                motif = motif.flip()
            selected.append(motif)
            self._motif_usage[motif.sequence] = (
                self._motif_usage.get(motif.sequence, 0) + 1
            )
        return selected

    def _select_weighted_motif(self) -> Motif:
        """Select a motif with higher probability for less-used ones."""
        if not self._motif_usage or all(v == 0 for v in self._motif_usage.values()):
            return self.rng.choice(self.motifs)

        max_usage = max(self._motif_usage.values()) + 1
        weights = [
            max_usage - self._motif_usage.get(m.sequence, 0) for m in self.motifs
        ]
        total = sum(weights)
        if total == 0:
            return self.rng.choice(self.motifs)

        r = self.rng.random() * total
        cumulative = 0
        for motif, weight in zip(self.motifs, weights, strict=True):
            cumulative += weight
            if r <= cumulative:
                return motif
        return self.motifs[-1]

    def _assemble_with_length_solver(
        self,
        motifs: list[Motif],
    ) -> Construct | None:
        """
        Assemble construct using length solver for exact helix lengths.

        Picks a random target length, computes helix budget, assigns
        lengths to individual helices, then assembles.
        """
        target = self.rng.randint(
            self.config.target_length_min,
            self.config.target_length_max,
        )
        motif_lengths = [m.total_length() for m in motifs]
        num_helices = len(motifs) + 1
        min_h, max_h = self.config.effective_helix_length_range

        assert self.config.hairpin_loop_length is not None
        budget = compute_helix_budget(
            target_length=target,
            motif_lengths=motif_lengths,
            p5_len=self.config.p5_length,
            p3_len=self.config.p3_length,
            hairpin_len=self.config.hairpin_loop_length,
            spacer_5p_len=self.config.spacer_5p_length,
            spacer_3p_len=self.config.spacer_3p_length,
        )
        if budget is None:
            return None

        assignment = random_helix_assignment(
            budget,
            num_helices,
            min_h,
            max_h,
            self.rng,
        )
        if assignment is None:
            return None

        helices = self._generate_variable_helices(assignment)

        if self.config.hairpin_sequence:
            hairpin = hairpin_from_sequence(
                self.config.hairpin_sequence,
                self.config.hairpin_structure,
            )
        else:
            hairpin = random_hairpin(self.config.hairpin_loop_length, self.rng)

        return assemble_construct(
            motifs=motifs,
            helices=helices,
            hairpin=hairpin,
            p5_seq=self.config.p5_sequence,
            p5_ss=self.config.p5_structure,
            p3_seq=self.config.p3_sequence,
            p3_ss=self.config.p3_structure,
            spacer_5p_seq=self.config.spacer_5p_sequence,
            spacer_5p_ss=self.config.spacer_5p_structure,
            spacer_3p_seq=self.config.spacer_3p_sequence,
            spacer_3p_ss=self.config.spacer_3p_structure,
        )

    def _generate_variable_helices(
        self,
        assignment: tuple[int, ...],
    ) -> list[Helix]:
        """
        Generate helices with variable lengths, enforcing GU requirements.

        Args:
            assignment: Tuple of helix lengths (one per helix).

        Returns:
            List of Helix objects with assigned lengths.
        """
        helices = []
        gu_threshold = self.config.gu_required_above_length
        for length in assignment:
            if gu_threshold is not None and length >= gu_threshold:
                helix = random_helix_with_gu_requirement(
                    length,
                    self.rng,
                    require_gu=True,
                )
            else:
                helix = random_helix(
                    length,
                    self.rng,
                    self.config.allow_wobble_pairs,
                )
            helices.append(helix)
        return helices

    def _check_sequence_constraints(
        self,
        construct: Construct,
    ) -> tuple[bool, str]:
        """Check sequence constraints (consecutive nucleotides, GC pairs).

        Only checks the core region, excluding p5 and p3 sequences which are fixed.
        """
        val_config = self.config.validation

        # Skip if validation is disabled or neither constraint is enabled
        if not val_config.enabled:
            return True, ""
        if (
            not val_config.avoid_consecutive_nucleotides
            and not val_config.avoid_consecutive_gc_pairs
        ):
            return True, ""

        # Extract core sequence/structure (exclude p5 and p3)
        p5_len = self.config.p5_length
        p3_len = self.config.p3_length
        core_seq = construct.sequence[p5_len : len(construct.sequence) - p3_len]
        core_ss = construct.structure[p5_len : len(construct.structure) - p3_len]

        # Get motif sequences for exclusion (patterns in motifs are allowed)
        motif_sequences = [m.sequence.replace("&", "") for m in construct.motifs]

        # Check consecutive nucleotides
        if val_config.avoid_consecutive_nucleotides:
            from twoway_lib.validation import find_consecutive_nucleotides

            violations = find_consecutive_nucleotides(
                core_seq,
                val_config.max_consecutive_nucleotides,
            )
            # Filter out violations that exist in motifs
            for pos, nt in violations:
                run = nt * val_config.max_consecutive_nucleotides
                if not any(run in motif for motif in motif_sequences):
                    return False, f"{run} at position {pos + p5_len}"

        # Check consecutive GC pairs
        if val_config.avoid_consecutive_gc_pairs:
            from twoway_lib.validation import has_consecutive_gc_pairs

            if has_consecutive_gc_pairs(
                core_seq,
                core_ss,
                val_config.max_consecutive_gc_pairs,
            ):
                return (
                    False,
                    f"{val_config.max_consecutive_gc_pairs}+ consecutive GC pairs",
                )

        return True, ""

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
        auto_tune: bool = False,
    ) -> list[Construct]:
        """Select diverse subset using simulated annealing."""
        optimizer = LibraryOptimizer(
            constructs=candidates,
            config=self.config.optimization,
            seed=self.rng.randint(0, 2**32 - 1),
        )
        selected_indices = optimizer.optimize(auto_tune=auto_tune)
        return [candidates[i] for i in selected_indices]


def _generate_candidate_batch(
    config: LibraryConfig,
    motifs: list[Motif],
    batch_size: int,
    max_attempts: int,
    seed: int,
) -> list[Construct]:
    """
    Generate a batch of candidates (module-level for pickling).

    Args:
        config: Library configuration.
        motifs: Available motifs.
        batch_size: Number of candidates to generate.
        max_attempts: Maximum attempts.
        seed: Random seed for this batch.

    Returns:
        List of valid candidate constructs.
    """
    generator = LibraryGenerator(config, motifs, seed=seed, filter_motifs=False)
    return generator._generate_candidates(batch_size, max_attempts)


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
    min_h, max_h = config.effective_helix_length_range
    assert config.hairpin_loop_length is not None
    hairpin_len = config.hairpin_loop_length
    p5_len = config.p5_length
    p3_len = config.p3_length
    spacer_5p = config.spacer_5p_length
    spacer_3p = config.spacer_3p_length

    min_total = calculate_construct_length(
        min_n,
        [min_motif] * min_n,
        min_h,
        hairpin_len,
        p5_len,
        p3_len,
        spacer_5p,
        spacer_3p,
    )
    max_total = calculate_construct_length(
        max_n,
        [max_motif] * max_n,
        max_h,
        hairpin_len,
        p5_len,
        p3_len,
        spacer_5p,
        spacer_3p,
    )

    return min_total, max_total
