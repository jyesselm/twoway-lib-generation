"""Simulated annealing optimizer for library selection."""

import math
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

import numpy as np
import structlog

from twoway_lib.config import OptimizationConfig
from twoway_lib.diversity import diversity_from_matrix, parallel_distance_matrix

if TYPE_CHECKING:
    from twoway_lib.construct import Construct

logger = structlog.get_logger(__name__)


@dataclass
class OptimizationProgress:
    """Progress information during optimization."""

    iteration: int
    temperature: float
    current_score: float
    best_score: float
    acceptance_rate: float


@dataclass
class TunedParams:
    """Auto-tuned annealing parameters."""

    initial_temperature: float
    cooling_rate: float
    avg_energy_delta: float
    sample_size: int


class LibraryOptimizer:
    """
    Simulated annealing optimizer for selecting diverse library subsets.

    Uses edit distance as the diversity metric and simulated annealing
    to find a subset that maximizes average minimum pairwise distance.
    Can also optimize for balanced motif usage.
    """

    def __init__(
        self,
        constructs: list["Construct"],
        config: OptimizationConfig,
        seed: int | None = None,
    ):
        """
        Initialize the optimizer.

        Args:
            constructs: List of candidate constructs.
            config: Optimization configuration.
            seed: Random seed for reproducibility.
        """
        self.constructs = constructs
        self.config = config
        self.rng = Random(seed)
        self.sequences = [c.sequence for c in constructs]
        self.distance_matrix: np.ndarray | None = None
        self._progress_callback: callable | None = None

        # Build motif usage mapping: construct index -> set of motif sequences
        self._construct_motifs: list[set[str]] = []
        self._all_motifs: set[str] = set()
        for c in constructs:
            motif_seqs = {m.sequence for m in c.motifs}
            self._construct_motifs.append(motif_seqs)
            self._all_motifs.update(motif_seqs)

    def set_progress_callback(self, callback: callable) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def optimize(self, auto_tune: bool = False) -> list[int]:
        """
        Run simulated annealing optimization.

        Args:
            auto_tune: If True, run brief optimization to find good parameters first.

        Returns:
            List of selected construct indices.
        """
        self._precompute_distances()
        initial_indices = self._random_initial_selection()

        if auto_tune:
            tuned = self.auto_tune(initial_indices)
            logger.info(
                "Auto-tuned parameters",
                initial_temp=round(tuned.initial_temperature, 4),
                cooling_rate=round(tuned.cooling_rate, 8),
                avg_delta=round(tuned.avg_energy_delta, 4),
            )
            return self._run_annealing(
                initial_indices,
                initial_temp=tuned.initial_temperature,
                cooling_rate=tuned.cooling_rate,
            )

        return self._run_annealing(initial_indices)

    def auto_tune(
        self,
        initial_indices: set[int] | None = None,
        sample_steps: int = 500,
        target_accept_rate: float = 0.5,
    ) -> TunedParams:
        """
        Run brief optimization to find good annealing parameters.

        Samples energy differences and calculates initial temperature
        to achieve target acceptance rate, and cooling rate to reach
        near-zero acceptance by the end.

        Args:
            initial_indices: Starting selection (random if None).
            sample_steps: Number of steps to sample energy differences.
            target_accept_rate: Target acceptance rate for initial temperature.

        Returns:
            TunedParams with recommended initial_temperature and cooling_rate.
        """
        if self.distance_matrix is None:
            self._precompute_distances()

        if initial_indices is None:
            initial_indices = self._random_initial_selection()

        # Sample energy differences by doing random moves
        current_indices = initial_indices.copy()
        current_energy = self._compute_energy(current_indices)
        energy_deltas: list[float] = []

        logger.debug("Auto-tuning: sampling energy differences", steps=sample_steps)

        for _ in range(sample_steps):
            new_indices = self._propose_move(current_indices)
            new_energy = self._compute_energy(new_indices)
            delta = new_energy - current_energy

            if delta > 0:
                energy_deltas.append(delta)

            # Always accept to explore the space
            current_indices = new_indices
            current_energy = new_energy

        if not energy_deltas:
            # No uphill moves found, use config defaults
            logger.warning("Auto-tune: no uphill moves found, using defaults")
            return TunedParams(
                initial_temperature=self.config.initial_temperature,
                cooling_rate=self.config.cooling_rate,
                avg_energy_delta=0.0,
                sample_size=sample_steps,
            )

        # Calculate initial temperature for target acceptance rate
        # P(accept) = exp(-delta/T) = target_accept_rate
        # T = -delta / ln(target_accept_rate)
        avg_delta = sum(energy_deltas) / len(energy_deltas)
        initial_temp = -avg_delta / math.log(target_accept_rate)

        # Calculate cooling rate to reach low acceptance by final iteration
        # At iteration N, T = T0 * exp(-rate * N)
        # We want T_final ≈ avg_delta / 10 (very low acceptance)
        # T_final = T0 * exp(-rate * iterations)
        # rate = -ln(T_final / T0) / iterations
        final_temp = avg_delta / 10  # ~10% of delta means ~0.01% acceptance
        iterations = self.config.iterations
        cooling_rate = -math.log(final_temp / initial_temp) / iterations

        logger.debug(
            "Auto-tune results",
            samples=len(energy_deltas),
            avg_delta=round(avg_delta, 4),
            initial_temp=round(initial_temp, 4),
            cooling_rate=round(cooling_rate, 8),
        )

        return TunedParams(
            initial_temperature=initial_temp,
            cooling_rate=cooling_rate,
            avg_energy_delta=avg_delta,
            sample_size=len(energy_deltas),
        )

    def _precompute_distances(self) -> None:
        """Precompute pairwise distance matrix for efficiency."""
        self.distance_matrix = parallel_distance_matrix(self.sequences)

    def _random_initial_selection(self) -> set[int]:
        """Select random initial subset of constructs."""
        n_select = min(self.config.target_library_size, len(self.constructs))
        all_indices = list(range(len(self.constructs)))
        selected = self.rng.sample(all_indices, n_select)
        return set(selected)

    def _run_annealing(
        self,
        initial_indices: set[int],
        initial_temp: float | None = None,
        cooling_rate: float | None = None,
    ) -> list[int]:
        """
        Execute the simulated annealing algorithm.

        Args:
            initial_indices: Initial selection of construct indices.
            initial_temp: Override initial temperature (uses config if None).
            cooling_rate: Override cooling rate (uses config if None).

        Returns:
            Optimized list of selected indices.
        """
        current_indices = initial_indices.copy()
        current_energy = self._compute_energy(current_indices)
        best_indices = current_indices.copy()
        best_energy = current_energy

        # Use provided params or fall back to config
        temp_start = initial_temp if initial_temp is not None else self.config.initial_temperature
        rate = cooling_rate if cooling_rate is not None else self.config.cooling_rate

        temperature = temp_start
        accepts = 0
        window_size = 1000
        log_interval = max(1, self.config.iterations // 10)

        logger.info(
            "Starting annealing",
            iterations=self.config.iterations,
            initial_temp=round(temp_start, 4),
            cooling_rate=round(rate, 8),
        )

        for iteration in range(self.config.iterations):
            new_indices = self._propose_move(current_indices)
            new_energy = self._compute_energy(new_indices)

            if self._accept_move(current_energy, new_energy, temperature):
                current_indices = new_indices
                current_energy = new_energy
                accepts += 1

                if new_energy < best_energy:
                    best_indices = new_indices.copy()
                    best_energy = new_energy

            temperature = temp_start * math.exp(-rate * iteration)

            # Log progress at info level (10 updates total)
            if iteration > 0 and iteration % log_interval == 0:
                pct = round(100 * iteration / self.config.iterations)
                logger.info(
                    "Optimization progress",
                    progress=f"{pct}%",
                    iteration=iteration,
                    best_score=round(-best_energy, 2),
                    accept_rate=round(accepts / iteration, 3),
                )

            self._report_progress(
                iteration,
                temperature,
                current_energy,
                best_energy,
                accepts,
                window_size,
            )

        logger.info(
            "Annealing complete",
            final_score=round(-best_energy, 2),
            accept_rate=round(accepts / self.config.iterations, 3),
        )

        return sorted(best_indices)

    def _compute_energy(self, indices: set[int]) -> float:
        """
        Compute energy (negative diversity + motif usage penalty).

        Lower energy = higher diversity + better motif balance = better.

        Args:
            indices: Set of selected construct indices.

        Returns:
            Energy value combining diversity and motif usage.
        """
        if self.distance_matrix is None:
            raise RuntimeError("Distance matrix not computed")

        # Diversity component (negative = better)
        diversity = diversity_from_matrix(self.distance_matrix, list(indices))
        diversity_energy = -diversity

        # Motif usage component
        motif_penalty = self._compute_motif_usage_penalty(indices)

        return diversity_energy + motif_penalty

    def _compute_motif_usage_penalty(self, indices: set[int]) -> float:
        """
        Compute penalty for motif usage outside min/max bounds.

        Args:
            indices: Set of selected construct indices.

        Returns:
            Penalty value (0 if all motifs within bounds).
        """
        min_usage = self.config.min_motif_usage
        max_usage = self.config.max_motif_usage

        if min_usage is None and max_usage is None:
            return 0.0

        # Count motif usage in selected constructs
        usage_counts: dict[str, int] = {m: 0 for m in self._all_motifs}
        for idx in indices:
            for motif in self._construct_motifs[idx]:
                usage_counts[motif] = usage_counts.get(motif, 0) + 1

        # Calculate penalty for violations
        penalty = 0.0
        for motif, count in usage_counts.items():
            if min_usage is not None and count < min_usage:
                penalty += (min_usage - count) ** 2
            if max_usage is not None and count > max_usage:
                penalty += (count - max_usage) ** 2

        return penalty * self.config.motif_usage_weight

    def _propose_move(self, current: set[int]) -> set[int]:
        """
        Propose a swap move: remove one, add another.

        Args:
            current: Current selection set.

        Returns:
            New selection set with one swap.
        """
        new_indices = current.copy()
        available = set(range(len(self.constructs))) - current
        if not available:
            return new_indices

        remove_idx = self.rng.choice(list(current))
        add_idx = self.rng.choice(list(available))

        new_indices.remove(remove_idx)
        new_indices.add(add_idx)
        return new_indices

    def _accept_move(
        self,
        current_energy: float,
        new_energy: float,
        temperature: float,
    ) -> bool:
        """
        Decide whether to accept proposed move (Metropolis criterion).

        Args:
            current_energy: Energy of current state.
            new_energy: Energy of proposed state.
            temperature: Current temperature.

        Returns:
            True if move should be accepted.
        """
        if new_energy < current_energy:
            return True
        if temperature <= 0:
            return False
        delta = new_energy - current_energy
        probability = math.exp(-delta / temperature)
        return self.rng.random() < probability

    def _cool_temperature(self, temperature: float, iteration: int) -> float:
        """
        Apply cooling schedule.

        Uses exponential cooling: T = T0 * exp(-rate * iteration)

        Args:
            temperature: Current temperature.
            iteration: Current iteration number.

        Returns:
            New temperature.
        """
        return self.config.initial_temperature * math.exp(
            -self.config.cooling_rate * iteration
        )

    def _report_progress(
        self,
        iteration: int,
        temperature: float,
        current_energy: float,
        best_energy: float,
        accepts: int,
        window_size: int,
    ) -> None:
        """Report progress via callback if set."""
        if self._progress_callback is None:
            return
        if iteration % window_size != 0:
            return

        rate = accepts / max(1, iteration) if iteration > 0 else 1.0
        progress = OptimizationProgress(
            iteration=iteration,
            temperature=temperature,
            current_score=-current_energy,
            best_score=-best_energy,
            acceptance_rate=rate,
        )
        self._progress_callback(progress)


def select_diverse_subset(
    constructs: list["Construct"],
    target_size: int,
    iterations: int = 100000,
    seed: int | None = None,
) -> list[int]:
    """
    Select a diverse subset of constructs.

    Convenience function that creates an optimizer with default settings.

    Args:
        constructs: List of candidate constructs.
        target_size: Number of constructs to select.
        iterations: Number of annealing iterations.
        seed: Random seed.

    Returns:
        List of selected construct indices.
    """
    config = OptimizationConfig(
        iterations=iterations,
        target_library_size=target_size,
    )
    optimizer = LibraryOptimizer(constructs, config, seed=seed)
    return optimizer.optimize()
