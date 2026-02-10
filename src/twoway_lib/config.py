"""Configuration management for two-way junction library generation."""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rna_secstruct import SecStruct


@dataclass
class ValidationConfig:
    """Configuration for construct validation with Vienna RNA fold."""

    enabled: bool = True
    max_ensemble_defect: float = 3.0
    allow_structure_differences: bool = False
    min_structure_match: float = 0.9
    avoid_consecutive_nucleotides: bool = True
    max_consecutive_nucleotides: int = 4
    avoid_consecutive_gc_pairs: bool = True
    max_consecutive_gc_pairs: int = 3


@dataclass
class OptimizationConfig:
    """Configuration for simulated annealing optimization."""

    iterations: int = 100000
    initial_temperature: float = 10.0
    cooling_rate: float = 0.00001
    target_library_size: int = 3000
    min_motif_usage: int | None = None
    max_motif_usage: int | None = None
    motif_usage_weight: float = 1.0  # Weight of motif balance vs diversity
    target_motif_usage: int | None = None  # Desired usage count per motif


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
    helix_length_min: int | None = None  # Variable helix length range minimum
    helix_length_max: int | None = None  # Variable helix length range maximum
    gu_required_above_length: int | None = None  # Helices >= this length require GU
    hairpin_loop_length: int | None = None  # Derived from hairpin_sequence if provided
    hairpin_sequence: str | None = None
    hairpin_structure: str | None = None
    allow_motif_flip: bool = False
    allow_wobble_pairs: bool = False  # Allow G-U/U-G pairs in helices
    spacer_5p_sequence: str | None = None  # Linker between p5 and first helix
    spacer_5p_structure: str | None = None
    spacer_3p_sequence: str | None = None  # Linker between last helix and p3
    spacer_3p_structure: str | None = None
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    def __post_init__(self) -> None:
        """Validate and derive hairpin_loop_length from hairpin_sequence."""
        if self.hairpin_sequence is not None:
            self.hairpin_sequence = self.hairpin_sequence.upper().replace("T", "U")
            # Derive length from sequence
            self.hairpin_loop_length = len(self.hairpin_sequence)
        elif self.hairpin_loop_length is None:
            # Default to 4 if neither specified
            self.hairpin_loop_length = 4

        # Derive helix_length_min/max from helix_length if not set
        if self.helix_length_min is None:
            self.helix_length_min = self.helix_length
        if self.helix_length_max is None:
            self.helix_length_max = self.helix_length

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

    @property
    def spacer_5p_length(self) -> int:
        """Length of 5' spacer sequence (0 if not set)."""
        return len(self.spacer_5p_sequence) if self.spacer_5p_sequence else 0

    @property
    def spacer_3p_length(self) -> int:
        """Length of 3' spacer sequence (0 if not set)."""
        return len(self.spacer_3p_sequence) if self.spacer_3p_sequence else 0

    @property
    def effective_helix_length_range(self) -> tuple[int, int]:
        """Return the effective helix length range (min, max)."""
        # Always set after __post_init__
        assert self.helix_length_min is not None
        assert self.helix_length_max is not None
        return (self.helix_length_min, self.helix_length_max)

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

    # Handle p5/p3 by name or direct sequence
    p5_seq, p5_struct = _resolve_primer_config(data, "p5")
    p3_seq, p3_struct = _resolve_primer_config(data, "p3")

    return LibraryConfig(
        target_length_min=target_length.get("min", 145),
        target_length_max=target_length.get("max", 150),
        motifs_per_construct_min=motifs_per.get("min", 6),
        motifs_per_construct_max=motifs_per.get("max", 7),
        p5_sequence=p5_seq,
        p5_structure=p5_struct,
        p3_sequence=p3_seq,
        p3_structure=p3_struct,
        helix_length=data.get("helix_length", 3),
        helix_length_min=data.get("helix_length_min"),
        helix_length_max=data.get("helix_length_max"),
        gu_required_above_length=data.get("gu_required_above_length"),
        hairpin_loop_length=data.get("hairpin_loop_length", 4),
        hairpin_sequence=data.get("hairpin_sequence"),
        hairpin_structure=data.get("hairpin_structure"),
        allow_motif_flip=data.get("allow_motif_flip", False),
        allow_wobble_pairs=data.get("allow_wobble_pairs", False),
        spacer_5p_sequence=data.get("spacer_5p_sequence"),
        spacer_5p_structure=data.get("spacer_5p_structure"),
        spacer_3p_sequence=data.get("spacer_3p_sequence"),
        spacer_3p_structure=data.get("spacer_3p_structure"),
        validation=validation,
        optimization=optimization,
    )


def _resolve_primer_config(data: dict[str, Any], prefix: str) -> tuple[str, str]:
    """Resolve p5 or p3 config from name or direct sequence."""
    name_key = f"{prefix}_name"
    seq_key = f"{prefix}_sequence"
    struct_key = f"{prefix}_structure"

    if name_key in data:
        name = data[name_key]
        if prefix == "p5":
            return get_p5_by_name(name)
        else:
            return get_p3_by_name(name)
    else:
        seq = data.get(seq_key, "")
        struct = data.get(struct_key, "")
        return seq, struct


def _parse_validation_config(data: dict[str, Any]) -> ValidationConfig:
    """Parse validation configuration section."""
    return ValidationConfig(
        enabled=data.get("enabled", True),
        max_ensemble_defect=data.get("max_ensemble_defect", 3.0),
        allow_structure_differences=data.get("allow_structure_differences", False),
        min_structure_match=data.get("min_structure_match", 0.9),
        avoid_consecutive_nucleotides=data.get("avoid_consecutive_nucleotides", True),
        max_consecutive_nucleotides=data.get("max_consecutive_nucleotides", 4),
        avoid_consecutive_gc_pairs=data.get("avoid_consecutive_gc_pairs", True),
        max_consecutive_gc_pairs=data.get("max_consecutive_gc_pairs", 3),
    )


def _parse_optimization_config(data: dict[str, Any]) -> OptimizationConfig:
    """Parse optimization configuration section."""
    return OptimizationConfig(
        iterations=data.get("iterations", 100000),
        initial_temperature=data.get("initial_temperature", 10.0),
        cooling_rate=data.get("cooling_rate", 0.00001),
        target_library_size=data.get("target_library_size", 3000),
        min_motif_usage=data.get("min_motif_usage"),
        max_motif_usage=data.get("max_motif_usage"),
        motif_usage_weight=data.get("motif_usage_weight", 1.0),
        target_motif_usage=data.get("target_motif_usage"),
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
    _validate_spacer_sequences(config)
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
    _validate_helix_range(config)
    if config.hairpin_loop_length is not None and config.hairpin_loop_length < 3:
        raise ValueError("hairpin_loop_length must be >= 3")


def _validate_helix_range(config: LibraryConfig) -> None:
    """Validate variable helix length and GU threshold settings."""
    if config.helix_length_min is not None and config.helix_length_min < 1:
        raise ValueError("helix_length_min must be >= 1")
    if config.helix_length_max is not None and config.helix_length_max < 1:
        raise ValueError("helix_length_max must be >= 1")
    if (
        config.helix_length_min is not None
        and config.helix_length_max is not None
        and config.helix_length_min > config.helix_length_max
    ):
        raise ValueError("helix_length_min must be <= helix_length_max")
    if (
        config.gu_required_above_length is not None
        and config.gu_required_above_length < 1
    ):
        raise ValueError("gu_required_above_length must be >= 1")


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
    """Validate hairpin_sequence and hairpin_structure if provided."""
    if config.hairpin_sequence is None and config.hairpin_structure is None:
        return

    valid_nts = set("AUGC")
    valid_ss = set("().")

    if config.hairpin_sequence is not None:
        invalid = set(config.hairpin_sequence.upper()) - valid_nts
        if invalid:
            raise ValueError(
                f"hairpin_sequence contains invalid nucleotides: {invalid}"
            )

    if config.hairpin_structure is not None:
        invalid = set(config.hairpin_structure) - valid_ss
        if invalid:
            raise ValueError(
                f"hairpin_structure contains invalid characters: {invalid}"
            )

    # If both provided, they must have same length
    if (
        config.hairpin_sequence
        and config.hairpin_structure
        and len(config.hairpin_sequence) != len(config.hairpin_structure)
    ):
        raise ValueError("hairpin_sequence and hairpin_structure must have same length")


def _validate_spacer_sequences(config: LibraryConfig) -> None:
    """Validate spacer sequences and structures if provided."""
    valid_nts = set("AUGC")
    valid_ss = set("().")

    for name, seq, ss in [
        ("spacer_5p", config.spacer_5p_sequence, config.spacer_5p_structure),
        ("spacer_3p", config.spacer_3p_sequence, config.spacer_3p_structure),
    ]:
        if seq is None and ss is None:
            continue
        if seq is not None:
            invalid = set(seq.upper()) - valid_nts
            if invalid:
                raise ValueError(
                    f"{name}_sequence contains invalid nucleotides: {invalid}"
                )
        if ss is not None:
            invalid = set(ss) - valid_ss
            if invalid:
                raise ValueError(
                    f"{name}_structure contains invalid characters: {invalid}"
                )
        if seq is not None and ss is not None and len(seq) != len(ss):
            raise ValueError(
                f"{name}_sequence and {name}_structure must have same length"
            )


def create_example_config(output_path: Path | str) -> None:
    """
    Create an example configuration file with inline documentation.

    Writes a YAML file with section headers and per-field comments
    explaining every parameter. The output is directly parseable by
    load_config().

    Args:
        output_path: Where to write the example config file.
    """
    example = """\
# Two-Way Junction Library Configuration
# Edit values below to customize your library generation.

# =============================================================================
# TARGET LENGTH - Total construct length in nucleotides
# =============================================================================
target_length:
  # Minimum construct length (must be >= 50)
  min: 100
  # Maximum construct length (must be >= min)
  max: 130

# =============================================================================
# MOTIFS PER CONSTRUCT - How many motifs to include in each construct
# =============================================================================
motifs_per_construct:
  # Minimum motifs per construct (must be >= 1)
  min: 6
  # Maximum motifs per construct (must be >= min)
  max: 7

# =============================================================================
# PRIMER SEQUENCES - 5' and 3' common regions flanking the variable region
# You can specify sequences directly or use named primers (p5_name / p3_name).
# =============================================================================
# Direct sequence specification:
p5_sequence: "GGGCGAAAGCCC"
p5_structure: "((((....))))"

p3_sequence: "AAAGAAACAACAACAACAAC"
p3_structure: "...................."

# Or use named primers from seq_tools (uncomment to use):
# p5_name: "T7_P5"
# p3_name: "tail_P3"

# =============================================================================
# HAIRPIN LOOP - The loop at the end of the hairpin construct
# =============================================================================
# Length of random hairpin loop (used when hairpin_sequence is not set)
hairpin_loop_length: 4

# Optional: specify an exact hairpin loop sequence and structure.
# When set, hairpin_loop_length is derived from the sequence length.
# hairpin_sequence: "GAAA"
# hairpin_structure: "...."

# =============================================================================
# HELIX SETTINGS - Random helices separating motifs
# =============================================================================
# Default helix length in base pairs (must be >= 1)
helix_length: 3

# Optional: allow variable helix lengths (range).
# When set, each helix is randomly assigned a length in [min, max].
# helix_length_min: 3
# helix_length_max: 5

# Optional: require at least one G-U wobble pair in helices at or above
# this length. Helps avoid overly stable helices.
# gu_required_above_length: 5

# Allow G-U / U-G wobble pairs in random helices (default: false)
allow_wobble_pairs: false

# Allow motifs to be inserted in both orientations (default: false)
allow_motif_flip: false

# =============================================================================
# SPACER SEQUENCES - Optional linkers between primers and the variable region
# =============================================================================
# 5' spacer: inserted between p5 and the first helix
# spacer_5p_sequence: "AA"
# spacer_5p_structure: ".."

# 3' spacer: inserted between the last helix and p3
# spacer_3p_sequence: "AA"
# spacer_3p_structure: ".."

# =============================================================================
# VALIDATION - Vienna RNA fold validation of generated constructs
# =============================================================================
validation:
  # Enable/disable fold validation (default: true)
  enabled: true

  # Maximum ensemble defect allowed (lower = stricter, must be >= 0)
  max_ensemble_defect: 3.0

  # Allow minor differences between designed and predicted structure
  allow_structure_differences: false

  # Minimum fraction of positions matching predicted structure (0.0-1.0)
  min_structure_match: 0.9

  # Reject constructs with long runs of the same nucleotide
  avoid_consecutive_nucleotides: true
  # Maximum allowed consecutive identical nucleotides
  max_consecutive_nucleotides: 4

  # Reject constructs with long runs of G-C pairs
  avoid_consecutive_gc_pairs: true
  # Maximum allowed consecutive G-C base pairs
  max_consecutive_gc_pairs: 3

# =============================================================================
# OPTIMIZATION - Simulated annealing for library selection
# =============================================================================
optimization:
  # Number of simulated annealing iterations
  iterations: 100000

  # Starting temperature for simulated annealing (must be > 0)
  initial_temperature: 10.0

  # Temperature decrease per iteration (must be > 0)
  cooling_rate: 0.00001

  # Desired number of constructs in the final library (must be >= 1)
  target_library_size: 3000

  # Optional: constrain per-motif usage counts
  # min_motif_usage: 10
  # max_motif_usage: 100

  # Weight balancing motif usage uniformity vs sequence diversity (default: 1.0)
  # motif_usage_weight: 1.0

  # Optional: target usage count per motif (overrides min/max if set)
  # target_motif_usage: 50
"""
    output_path = Path(output_path)
    output_path.write_text(example)


def generate_default_config() -> LibraryConfig:
    """
    Generate a default configuration with example values.

    Returns:
        LibraryConfig with sensible default values.
    """
    return LibraryConfig(
        target_length_min=100,
        target_length_max=130,
        motifs_per_construct_min=6,
        motifs_per_construct_max=7,
        p5_sequence="GGGCGAAAGCCC",
        p5_structure="((((....))))",
        p3_sequence="AAAGAAACAACAACAACAAC",
        p3_structure="....................",
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
        "target_length": {
            "min": config.target_length_min,
            "max": config.target_length_max,
        },
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
    _add_optional_fields(data, config)
    data["validation"] = _validation_config_to_dict(config.validation)
    data["optimization"] = _optimization_config_to_dict(config.optimization)
    return data


def _add_optional_fields(data: dict[str, Any], config: LibraryConfig) -> None:
    """Add optional config fields to the serialization dict."""
    if (
        config.helix_length_min is not None
        and config.helix_length_min != config.helix_length
    ):
        data["helix_length_min"] = config.helix_length_min
    if (
        config.helix_length_max is not None
        and config.helix_length_max != config.helix_length
    ):
        data["helix_length_max"] = config.helix_length_max
    if config.gu_required_above_length is not None:
        data["gu_required_above_length"] = config.gu_required_above_length
    if config.hairpin_sequence is not None:
        data["hairpin_sequence"] = config.hairpin_sequence
    if config.hairpin_structure is not None:
        data["hairpin_structure"] = config.hairpin_structure
    if config.allow_motif_flip:
        data["allow_motif_flip"] = config.allow_motif_flip
    if config.allow_wobble_pairs:
        data["allow_wobble_pairs"] = config.allow_wobble_pairs
    for attr in (
        "spacer_5p_sequence",
        "spacer_5p_structure",
        "spacer_3p_sequence",
        "spacer_3p_structure",
    ):
        val = getattr(config, attr)
        if val is not None:
            data[attr] = val


def _validation_config_to_dict(config: ValidationConfig) -> dict[str, Any]:
    """Convert ValidationConfig to dictionary."""
    return {
        "enabled": config.enabled,
        "max_ensemble_defect": config.max_ensemble_defect,
        "allow_structure_differences": config.allow_structure_differences,
        "min_structure_match": config.min_structure_match,
        "avoid_consecutive_nucleotides": config.avoid_consecutive_nucleotides,
        "max_consecutive_nucleotides": config.max_consecutive_nucleotides,
        "avoid_consecutive_gc_pairs": config.avoid_consecutive_gc_pairs,
        "max_consecutive_gc_pairs": config.max_consecutive_gc_pairs,
    }


def _optimization_config_to_dict(config: OptimizationConfig) -> dict[str, Any]:
    """Convert OptimizationConfig to dictionary."""
    opt_data: dict[str, Any] = {
        "iterations": config.iterations,
        "initial_temperature": config.initial_temperature,
        "cooling_rate": config.cooling_rate,
        "target_library_size": config.target_library_size,
    }
    if config.min_motif_usage is not None:
        opt_data["min_motif_usage"] = config.min_motif_usage
    if config.max_motif_usage is not None:
        opt_data["max_motif_usage"] = config.max_motif_usage
    if config.motif_usage_weight != 1.0:
        opt_data["motif_usage_weight"] = config.motif_usage_weight
    if config.target_motif_usage is not None:
        opt_data["target_motif_usage"] = config.target_motif_usage
    return opt_data


def get_p5_sequences() -> dict[str, tuple[str, str]]:
    """
    Get available p5 sequences from seq_tools resources.

    Returns:
        Dictionary mapping name to (sequence, structure) tuple.
    """
    return _load_sequence_resource("p5_sequences.csv")


def get_p3_sequences() -> dict[str, tuple[str, str]]:
    """
    Get available p3 sequences from seq_tools resources.

    Returns:
        Dictionary mapping name to (sequence, structure) tuple.
    """
    return _load_sequence_resource("p3_sequences.csv")


def _load_sequence_resource(filename: str) -> dict[str, tuple[str, str]]:
    """Load sequence resource from seq_tools."""
    try:
        from seq_tools.utils import get_resources_path

        resource_path = get_resources_path() / filename
        if not resource_path.exists():
            return {}
        df = pd.read_csv(resource_path)
        result = {}
        for _, row in df.iterrows():
            name = row["name"]
            seq = row["sequence"]
            struct = row.get("structure", "." * len(seq))
            result[name] = (seq, struct)
        return result
    except (ImportError, FileNotFoundError):
        return {}


def get_p5_by_name(name: str) -> tuple[str, str]:
    """
    Get a p5 sequence by name.

    Args:
        name: Name of the p5 sequence.

    Returns:
        Tuple of (sequence, structure).

    Raises:
        ValueError: If name not found.
    """
    sequences = get_p5_sequences()
    if name not in sequences:
        available = list(sequences.keys())
        raise ValueError(f"Unknown p5 sequence '{name}'. Available: {available}")
    return sequences[name]


def get_p3_by_name(name: str) -> tuple[str, str]:
    """
    Get a p3 sequence by name.

    Args:
        name: Name of the p3 sequence.

    Returns:
        Tuple of (sequence, structure).

    Raises:
        ValueError: If name not found.
    """
    sequences = get_p3_sequences()
    if name not in sequences:
        available = list(sequences.keys())
        raise ValueError(f"Unknown p3 sequence '{name}'. Available: {available}")
    return sequences[name]


def list_available_primers() -> dict[str, list[str]]:
    """
    List available p5 and p3 primer names.

    Returns:
        Dictionary with 'p5' and 'p3' keys containing lists of available names.
    """
    return {
        "p5": list(get_p5_sequences().keys()),
        "p3": list(get_p3_sequences().keys()),
    }
