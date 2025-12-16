"""Configuration management for two-way junction library generation."""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml
from rna_secstruct import SecStruct


@dataclass
class ValidationConfig:
    """Configuration for construct validation with Vienna RNA fold."""

    enabled: bool = True
    max_ensemble_defect: float = 3.0
    allow_structure_differences: bool = False
    min_structure_match: float = 0.9


@dataclass
class OptimizationConfig:
    """Configuration for simulated annealing optimization."""

    iterations: int = 100000
    initial_temperature: float = 10.0
    cooling_rate: float = 0.00001
    target_library_size: int = 3000


@dataclass
class LibraryConfig:
    """Main configuration for library generation."""

    target_length_min: int
    target_length_max: int
    motifs_per_construct_min: int
    motifs_per_construct_max: int
    p5_sequence: str
    p5_structure: str
    p3_sequence: str
    p3_structure: str
    helix_length: int = 3
    hairpin_loop_length: int = 4
    hairpin_sequence: str | None = None
    allow_motif_flip: bool = False
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    def __post_init__(self) -> None:
        """Validate hairpin_sequence if provided."""
        if self.hairpin_sequence is not None:
            self.hairpin_sequence = self.hairpin_sequence.upper().replace("T", "U")

    @property
    def target_length(self) -> tuple[int, int]:
        """Return target length as tuple."""
        return (self.target_length_min, self.target_length_max)

    @property
    def motifs_per_construct(self) -> tuple[int, int]:
        """Return motifs per construct as tuple."""
        return (self.motifs_per_construct_min, self.motifs_per_construct_max)

    @property
    def p5_length(self) -> int:
        """Length of 5' common sequence."""
        return len(self.p5_sequence)

    @property
    def p3_length(self) -> int:
        """Length of 3' common sequence."""
        return len(self.p3_sequence)

    @cached_property
    def p5_secstruct(self) -> SecStruct:
        """Return SecStruct for 5' common region."""
        return SecStruct(self.p5_sequence, self.p5_structure)

    @cached_property
    def p3_secstruct(self) -> SecStruct:
        """Return SecStruct for 3' common region."""
        return SecStruct(self.p3_sequence, self.p3_structure)


def load_config(path: Path | str) -> LibraryConfig:
    """
    Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed and validated LibraryConfig object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If configuration is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    config = _parse_config_dict(data)
    validate_config(config)
    return config


def _parse_config_dict(data: dict[str, Any]) -> LibraryConfig:
    """
    Parse configuration dictionary into LibraryConfig object.

    Args:
        data: Raw dictionary from YAML parsing.

    Returns:
        LibraryConfig object with all settings.
    """
    validation = _parse_validation_config(data.get("validation", {}))
    optimization = _parse_optimization_config(data.get("optimization", {}))

    target_length = data.get("target_length", {})
    motifs_per = data.get("motifs_per_construct", {})

    return LibraryConfig(
        target_length_min=target_length.get("min", 145),
        target_length_max=target_length.get("max", 150),
        motifs_per_construct_min=motifs_per.get("min", 6),
        motifs_per_construct_max=motifs_per.get("max", 7),
        p5_sequence=data.get("p5_sequence", ""),
        p5_structure=data.get("p5_structure", ""),
        p3_sequence=data.get("p3_sequence", ""),
        p3_structure=data.get("p3_structure", ""),
        helix_length=data.get("helix_length", 3),
        hairpin_loop_length=data.get("hairpin_loop_length", 4),
        hairpin_sequence=data.get("hairpin_sequence"),
        allow_motif_flip=data.get("allow_motif_flip", False),
        validation=validation,
        optimization=optimization,
    )


def _parse_validation_config(data: dict[str, Any]) -> ValidationConfig:
    """Parse validation configuration section."""
    return ValidationConfig(
        enabled=data.get("enabled", True),
        max_ensemble_defect=data.get("max_ensemble_defect", 3.0),
        allow_structure_differences=data.get("allow_structure_differences", False),
        min_structure_match=data.get("min_structure_match", 0.9),
    )


def _parse_optimization_config(data: dict[str, Any]) -> OptimizationConfig:
    """Parse optimization configuration section."""
    return OptimizationConfig(
        iterations=data.get("iterations", 100000),
        initial_temperature=data.get("initial_temperature", 10.0),
        cooling_rate=data.get("cooling_rate", 0.00001),
        target_library_size=data.get("target_library_size", 3000),
    )


def validate_config(config: LibraryConfig) -> None:
    """
    Validate configuration values and structure compatibility.

    Args:
        config: LibraryConfig to validate.

    Raises:
        ValueError: If any configuration values are invalid.
    """
    _validate_lengths(config)
    _validate_sequences(config)
    _validate_hairpin_sequence(config)
    _validate_validation_config(config.validation)
    _validate_optimization_config(config.optimization)


def _validate_lengths(config: LibraryConfig) -> None:
    """Validate length-related configuration."""
    if config.target_length_min > config.target_length_max:
        raise ValueError("target_length min must be <= max")
    if config.target_length_min < 50:
        raise ValueError("target_length min must be >= 50")
    if config.motifs_per_construct_min > config.motifs_per_construct_max:
        raise ValueError("motifs_per_construct min must be <= max")
    if config.motifs_per_construct_min < 1:
        raise ValueError("motifs_per_construct min must be >= 1")
    if config.helix_length < 1:
        raise ValueError("helix_length must be >= 1")
    if config.hairpin_loop_length < 3:
        raise ValueError("hairpin_loop_length must be >= 3")


def _validate_sequences(config: LibraryConfig) -> None:
    """Validate sequence and structure configuration."""
    valid_nts = set("AUGC")
    valid_ss = set("().")

    # Basic character validation first
    for name, seq in [
        ("p5_sequence", config.p5_sequence),
        ("p3_sequence", config.p3_sequence),
    ]:
        if not seq:
            raise ValueError(f"{name} cannot be empty")
        invalid = set(seq.upper()) - valid_nts
        if invalid:
            raise ValueError(f"{name} contains invalid nucleotides: {invalid}")

    for name, ss in [
        ("p5_structure", config.p5_structure),
        ("p3_structure", config.p3_structure),
    ]:
        if not ss:
            raise ValueError(f"{name} cannot be empty")
        invalid = set(ss) - valid_ss
        if invalid:
            raise ValueError(f"{name} contains invalid characters: {invalid}")

    # Length validation - p5/p3 are fragments so brackets won't be balanced
    if len(config.p5_sequence) != len(config.p5_structure):
        raise ValueError("p5_sequence and p5_structure must have same length")
    if len(config.p3_sequence) != len(config.p3_structure):
        raise ValueError("p3_sequence and p3_structure must have same length")


def _validate_validation_config(config: ValidationConfig) -> None:
    """Validate validation configuration."""
    if config.max_ensemble_defect < 0:
        raise ValueError("max_ensemble_defect must be >= 0")
    if not 0 <= config.min_structure_match <= 1:
        raise ValueError("min_structure_match must be between 0 and 1")


def _validate_optimization_config(config: OptimizationConfig) -> None:
    """Validate optimization configuration."""
    if config.iterations < 1:
        raise ValueError("iterations must be >= 1")
    if config.initial_temperature <= 0:
        raise ValueError("initial_temperature must be > 0")
    if config.cooling_rate <= 0:
        raise ValueError("cooling_rate must be > 0")
    if config.target_library_size < 1:
        raise ValueError("target_library_size must be >= 1")


def _validate_hairpin_sequence(config: LibraryConfig) -> None:
    """Validate hairpin_sequence if provided."""
    if config.hairpin_sequence is None:
        return
    valid_nts = set("AUGC")
    invalid = set(config.hairpin_sequence.upper()) - valid_nts
    if invalid:
        raise ValueError(f"hairpin_sequence contains invalid nucleotides: {invalid}")
    if len(config.hairpin_sequence) != config.hairpin_loop_length:
        raise ValueError(
            f"hairpin_sequence length ({len(config.hairpin_sequence)}) must match "
            f"hairpin_loop_length ({config.hairpin_loop_length})"
        )


def generate_default_config() -> LibraryConfig:
    """
    Generate a default configuration with example values.

    Returns:
        LibraryConfig with sensible default values.
    """
    return LibraryConfig(
        target_length_min=115,
        target_length_max=140,
        motifs_per_construct_min=6,
        motifs_per_construct_max=7,
        p5_sequence="GGAACAGCACUUCGGUGCC",
        p5_structure="((((.(((((....)))))",
        p3_sequence="GAAGUGGCUGACCCC",
        p3_structure=")))))....))))))",
        helix_length=3,
        hairpin_loop_length=4,
        hairpin_sequence=None,
        allow_motif_flip=False,
        validation=ValidationConfig(),
        optimization=OptimizationConfig(),
    )


def save_config(config: LibraryConfig, path: Path | str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: LibraryConfig to save.
        path: Output file path.
    """
    path = Path(path)
    data = _config_to_dict(config)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _config_to_dict(config: LibraryConfig) -> dict[str, Any]:
    """Convert LibraryConfig to dictionary for YAML serialization."""
    data: dict[str, Any] = {
        "target_length": {"min": config.target_length_min, "max": config.target_length_max},
        "motifs_per_construct": {
            "min": config.motifs_per_construct_min,
            "max": config.motifs_per_construct_max,
        },
        "p5_sequence": config.p5_sequence,
        "p5_structure": config.p5_structure,
        "p3_sequence": config.p3_sequence,
        "p3_structure": config.p3_structure,
        "helix_length": config.helix_length,
        "hairpin_loop_length": config.hairpin_loop_length,
    }
    if config.hairpin_sequence is not None:
        data["hairpin_sequence"] = config.hairpin_sequence
    if config.allow_motif_flip:
        data["allow_motif_flip"] = config.allow_motif_flip
    data["validation"] = {
        "enabled": config.validation.enabled,
        "max_ensemble_defect": config.validation.max_ensemble_defect,
        "allow_structure_differences": config.validation.allow_structure_differences,
        "min_structure_match": config.validation.min_structure_match,
    }
    data["optimization"] = {
        "iterations": config.optimization.iterations,
        "initial_temperature": config.optimization.initial_temperature,
        "cooling_rate": config.optimization.cooling_rate,
        "target_library_size": config.optimization.target_library_size,
    }
    return data
