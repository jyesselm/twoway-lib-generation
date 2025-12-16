"""Tests for config module."""

from pathlib import Path

import pytest

from twoway_lib.config import (
    LibraryConfig,
    OptimizationConfig,
    ValidationConfig,
    generate_default_config,
    load_config,
    save_config,
    validate_config,
)


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_defaults(self):
        config = ValidationConfig()
        assert config.enabled is True
        assert config.max_ensemble_defect == 3.0
        assert config.allow_structure_differences is False

    def test_custom_values(self):
        config = ValidationConfig(
            enabled=False,
            max_ensemble_defect=5.0,
            allow_structure_differences=True,
        )
        assert config.enabled is False
        assert config.max_ensemble_defect == 5.0


class TestOptimizationConfig:
    """Tests for OptimizationConfig dataclass."""

    def test_defaults(self):
        config = OptimizationConfig()
        assert config.iterations == 100000
        assert config.initial_temperature == 10.0
        assert config.target_library_size == 3000

    def test_custom_values(self):
        config = OptimizationConfig(iterations=5000, target_library_size=500)
        assert config.iterations == 5000
        assert config.target_library_size == 500


class TestLibraryConfig:
    """Tests for LibraryConfig dataclass."""

    def test_target_length_property(self, sample_config: LibraryConfig):
        assert sample_config.target_length == (100, 120)

    def test_motifs_per_construct_property(self, sample_config: LibraryConfig):
        assert sample_config.motifs_per_construct == (3, 4)

    def test_p5_length(self, sample_config: LibraryConfig):
        assert sample_config.p5_length == 5

    def test_p3_length(self, sample_config: LibraryConfig):
        assert sample_config.p3_length == 5


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, temp_config_file: Path):
        config = load_config(temp_config_file)
        assert config.target_length_min == 100
        assert config.target_length_max == 120
        assert config.p5_sequence == "GGAAC"

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_load_with_string_path(self, temp_config_file: Path):
        config = load_config(str(temp_config_file))
        assert config.target_length_min == 100


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config(self, sample_config: LibraryConfig):
        validate_config(sample_config)

    def test_invalid_target_length_range(self, sample_config: LibraryConfig):
        sample_config.target_length_min = 150
        sample_config.target_length_max = 100
        with pytest.raises(ValueError, match="target_length min must be <= max"):
            validate_config(sample_config)

    def test_target_length_too_small(self, sample_config: LibraryConfig):
        sample_config.target_length_min = 30
        with pytest.raises(ValueError, match="target_length min must be >= 50"):
            validate_config(sample_config)

    def test_invalid_motifs_per_construct(self, sample_config: LibraryConfig):
        sample_config.motifs_per_construct_min = 5
        sample_config.motifs_per_construct_max = 3
        with pytest.raises(ValueError, match="motifs_per_construct min must be <= max"):
            validate_config(sample_config)

    def test_invalid_helix_length(self, sample_config: LibraryConfig):
        sample_config.helix_length = 0
        with pytest.raises(ValueError, match="helix_length must be >= 1"):
            validate_config(sample_config)

    def test_invalid_hairpin_length(self, sample_config: LibraryConfig):
        sample_config.hairpin_loop_length = 2
        with pytest.raises(ValueError, match="hairpin_loop_length must be >= 3"):
            validate_config(sample_config)

    def test_empty_p5_sequence(self, sample_config: LibraryConfig):
        sample_config.p5_sequence = ""
        with pytest.raises(ValueError, match="p5_sequence cannot be empty"):
            validate_config(sample_config)

    def test_invalid_p5_nucleotides(self, sample_config: LibraryConfig):
        sample_config.p5_sequence = "GGXAC"
        with pytest.raises(ValueError, match="invalid nucleotides"):
            validate_config(sample_config)

    def test_invalid_p5_structure(self, sample_config: LibraryConfig):
        sample_config.p5_structure = "(([.."
        with pytest.raises(ValueError, match="invalid characters"):
            validate_config(sample_config)

    def test_mismatched_p5_lengths(self, sample_config: LibraryConfig):
        sample_config.p5_structure = "(((.."
        sample_config.p5_sequence = "GGAA"
        with pytest.raises(ValueError, match="same length"):
            validate_config(sample_config)

    def test_negative_ensemble_defect(self, sample_config: LibraryConfig):
        sample_config.validation.max_ensemble_defect = -1.0
        with pytest.raises(ValueError, match="max_ensemble_defect must be >= 0"):
            validate_config(sample_config)

    def test_invalid_structure_match(self, sample_config: LibraryConfig):
        sample_config.validation.min_structure_match = 1.5
        with pytest.raises(ValueError, match="min_structure_match must be between"):
            validate_config(sample_config)

    def test_invalid_iterations(self, sample_config: LibraryConfig):
        sample_config.optimization.iterations = 0
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            validate_config(sample_config)

    def test_hairpin_sequence_valid(self, sample_config: LibraryConfig):
        sample_config.hairpin_sequence = "GAAA"
        validate_config(sample_config)

    def test_hairpin_sequence_invalid_nucleotides(self, sample_config: LibraryConfig):
        sample_config.hairpin_sequence = "GXAA"
        with pytest.raises(ValueError, match="invalid nucleotides"):
            validate_config(sample_config)

    def test_hairpin_sequence_length_mismatch(self, sample_config: LibraryConfig):
        sample_config.hairpin_sequence = "GAA"  # 3 nt, but hairpin_loop_length is 4
        with pytest.raises(ValueError, match="length.*must match"):
            validate_config(sample_config)


class TestGenerateDefaultConfig:
    """Tests for generate_default_config function."""

    def test_returns_library_config(self):
        config = generate_default_config()
        assert isinstance(config, LibraryConfig)

    def test_config_is_valid(self):
        config = generate_default_config()
        validate_config(config)

    def test_has_p5_and_p3_sequences(self):
        config = generate_default_config()
        assert config.p5_sequence
        assert config.p3_sequence


class TestSaveConfig:
    """Tests for save_config function."""

    def test_saves_to_file(self, sample_config: LibraryConfig, temp_dir: Path):
        output_path = temp_dir / "config.yaml"
        save_config(sample_config, output_path)
        assert output_path.exists()

    def test_loadable_after_save(self, sample_config: LibraryConfig, temp_dir: Path):
        output_path = temp_dir / "config.yaml"
        save_config(sample_config, output_path)
        loaded = load_config(output_path)
        assert loaded.target_length_min == sample_config.target_length_min

    def test_saves_hairpin_sequence(self, sample_config: LibraryConfig, temp_dir: Path):
        sample_config.hairpin_sequence = "GAAA"
        output_path = temp_dir / "config.yaml"
        save_config(sample_config, output_path)
        loaded = load_config(output_path)
        assert loaded.hairpin_sequence == "GAAA"

    def test_saves_allow_motif_flip(self, sample_config: LibraryConfig, temp_dir: Path):
        sample_config.allow_motif_flip = True
        output_path = temp_dir / "config.yaml"
        save_config(sample_config, output_path)
        loaded = load_config(output_path)
        assert loaded.allow_motif_flip is True
