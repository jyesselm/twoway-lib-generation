"""Simulated annealing optimizer for library selection."""

import math
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

import numpy as np

from twoway_lib.config import OptimizationConfig
from twoway_lib.diversity import diversity_from_matrix, parallel_distance_matrix

if TYPE_CHECKING:
    from twoway_lib.construct import Construct


@dataclass
class OptimizationProgress:
    """Progress information during optimization."""

    iteration: int
    temperature: float
    current_score: float
    best_score: float
    acceptance_rate: float


class LibraryOptimizer:
    """
    Simulated annealing optimizer for selecting diverse library subsets.

    Uses edit distance as the diversity metric and simulated annealing
    to find a subset that maximizes average minimum pairwise distance.
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

    def set_progress_callback(self, callback: callable) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def optimize(self) -> list[int]:
        """
        Run simulated annealing optimization.

        Returns:
            List of selected construct indices.
        """
        self._precompute_distances()
        initial_indices = self._random_initial_selection()
        return self._run_annealing(initial_indices)

    def _precompute_distances(self) -> None:
        """Precompute pairwise distance matrix for efficiency."""
        self.distance_matrix = parallel_distance_matrix(self.sequences)

    def _random_initial_selection(self) -> set[int]:
        """Select random initial subset of constructs."""
        n_select = min(self.config.target_library_size, len(self.constructs))
        all_indices = list(range(len(self.constructs)))
        selected = self.rng.sample(all_indices, n_select)
        return set(selected)

    def _run_annealing(self, initial_indices: set[int]) -> list[int]:
        """
        Execute the simulated annealing algorithm.

        Args:
            initial_indices: Initial selection of construct indices.

        Returns:
            Optimized list of selected indices.
        """
        current_indices = initial_indices.copy()
        current_energy = self._compute_energy(current_indices)
        best_indices = current_indices.copy()
        best_energy = current_energy
        temperature = self.config.initial_temperature
        accepts = 0
        window_size = 1000

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

            temperature = self._cool_temperature(temperature, iteration)
            self._report_progress(
                iteration,
                temperature,
                current_energy,
                best_energy,
                accepts,
                window_size,
            )

        return sorted(best_indices)

    def _compute_energy(self, indices: set[int]) -> float:
        """
        Compute energy (negative diversity score).

        Lower energy = higher diversity = better.

        Args:
            indices: Set of selected construct indices.

        Returns:
            Energy value (negative average minimum distance).
        """
        if self.distance_matrix is None:
            raise RuntimeError("Distance matrix not computed")
        diversity = diversity_from_matrix(self.distance_matrix, list(indices))
        return -diversity

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
