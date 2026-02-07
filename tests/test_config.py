"""Tests for config module."""

from pathlib import Path

import pytest

from twoway_lib.config import (
    LibraryConfig,
    OptimizationConfig,
    ValidationConfig,
    generate_default_config,
    get_p3_sequences,
    get_p5_sequences,
    list_available_primers,
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

    def test_hairpin_sequence_derives_length(self, sample_config: LibraryConfig):
        # Length is derived from sequence, so setting sequence updates length
        sample_config.hairpin_sequence = "GAAA"
        sample_config.__post_init__()  # Re-run to derive length
        assert sample_config.hairpin_loop_length == 4
        validate_config(sample_config)  # Should not raise


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


class TestPrimerLookup:
    """Tests for primer lookup functions."""

    def test_get_p5_sequences_returns_dict(self):
        result = get_p5_sequences()
        assert isinstance(result, dict)

    def test_get_p3_sequences_returns_dict(self):
        result = get_p3_sequences()
        assert isinstance(result, dict)

    def test_list_available_primers(self):
        result = list_available_primers()
        assert "p5" in result
        assert "p3" in result
        assert isinstance(result["p5"], list)
        assert isinstance(result["p3"], list)

    def test_p5_sequences_have_sequence_structure(self):
        sequences = get_p5_sequences()
        for _name, (seq, struct) in sequences.items():
            assert isinstance(seq, str)
            assert isinstance(struct, str)
            assert len(seq) == len(struct)

    def test_p3_sequences_have_sequence_structure(self):
        sequences = get_p3_sequences()
        for _name, (seq, struct) in sequences.items():
            assert isinstance(seq, str)
            assert isinstance(struct, str)
            assert len(seq) == len(struct)


class TestConfigWithPrimerName:
    """Tests for loading config with p5_name/p3_name."""

    def test_load_config_with_p5_name(self, temp_dir: Path):
        p5_seqs = get_p5_sequences()
        if not p5_seqs:
            pytest.skip("No p5 sequences available")

        p5_name = list(p5_seqs.keys())[0]
        config_content = f"""
target_length:
  min: 100
  max: 120
motifs_per_construct:
  min: 3
  max: 4
p5_name: "{p5_name}"
p3_sequence: "AAAGAAAC"
p3_structure: "........"
helix_length: 3
hairpin_loop_length: 4
"""
        config_file = temp_dir / "config.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)
        expected_seq, expected_struct = p5_seqs[p5_name]
        assert config.p5_sequence == expected_seq
        assert config.p5_structure == expected_struct

    def test_load_config_with_p3_name(self, temp_dir: Path):
        p3_seqs = get_p3_sequences()
        if not p3_seqs:
            pytest.skip("No p3 sequences available")

        p3_name = list(p3_seqs.keys())[0]
        config_content = f"""
target_length:
  min: 100
  max: 120
motifs_per_construct:
  min: 3
  max: 4
p5_sequence: "GGGCGAAAGCCC"
p5_structure: "((((....))))"
p3_name: "{p3_name}"
helix_length: 3
hairpin_loop_length: 4
"""
        config_file = temp_dir / "config.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)
        expected_seq, expected_struct = p3_seqs[p3_name]
        assert config.p3_sequence == expected_seq
        assert config.p3_structure == expected_struct

    def test_unknown_p5_name_raises(self, temp_dir: Path):
        config_content = """
target_length:
  min: 100
  max: 120
motifs_per_construct:
  min: 3
  max: 4
p5_name: "nonexistent_primer_xyz"
p3_sequence: "AAAGAAAC"
p3_structure: "........"
helix_length: 3
hairpin_loop_length: 4
"""
        config_file = temp_dir / "config.yaml"
        config_file.write_text(config_content)

        with pytest.raises(ValueError, match="Unknown p5 sequence"):
            load_config(config_file)


class TestVariableHelixConfig:
    """Tests for variable helix length configuration."""

    def test_defaults_from_helix_length(self, sample_config):
        assert sample_config.helix_length_min == 3
        assert sample_config.helix_length_max == 3

    def test_effective_helix_length_range(self, sample_config):
        assert sample_config.effective_helix_length_range == (3, 3)

    def test_custom_range(self):
        config = LibraryConfig(
            target_length_min=100,
            target_length_max=120,
            motifs_per_construct_min=3,
            motifs_per_construct_max=4,
            p5_sequence="GGAAC",
            p5_structure="(((..",
            p3_sequence="GUUCC",
            p3_structure="..)))",
            helix_length=3,
            helix_length_min=2,
            helix_length_max=5,
        )
        assert config.effective_helix_length_range == (2, 5)

    def test_invalid_range(self):
        config = LibraryConfig(
            target_length_min=100,
            target_length_max=120,
            motifs_per_construct_min=3,
            motifs_per_construct_max=4,
            p5_sequence="GGAAC",
            p5_structure="(((..",
            p3_sequence="GUUCC",
            p3_structure="..)))",
            helix_length=3,
            helix_length_min=5,
            helix_length_max=2,
        )
        with pytest.raises(
            ValueError, match="helix_length_min must be <= helix_length_max"
        ):
            validate_config(config)

    def test_gu_required_above_length(self):
        config = LibraryConfig(
            target_length_min=100,
            target_length_max=120,
            motifs_per_construct_min=3,
            motifs_per_construct_max=4,
            p5_sequence="GGAAC",
            p5_structure="(((..",
            p3_sequence="GUUCC",
            p3_structure="..)))",
            helix_length=3,
            gu_required_above_length=4,
        )
        validate_config(config)
        assert config.gu_required_above_length == 4

    def test_gu_required_invalid(self):
        config = LibraryConfig(
            target_length_min=100,
            target_length_max=120,
            motifs_per_construct_min=3,
            motifs_per_construct_max=4,
            p5_sequence="GGAAC",
            p5_structure="(((..",
            p3_sequence="GUUCC",
            p3_structure="..)))",
            helix_length=3,
            gu_required_above_length=0,
        )
        with pytest.raises(ValueError, match="gu_required_above_length must be >= 1"):
            validate_config(config)


class TestSpacerConfig:
    """Tests for spacer configuration."""

    def test_spacer_lengths(self):
        config = LibraryConfig(
            target_length_min=100,
            target_length_max=120,
            motifs_per_construct_min=3,
            motifs_per_construct_max=4,
            p5_sequence="GGAAC",
            p5_structure="(((..",
            p3_sequence="GUUCC",
            p3_structure="..)))",
            helix_length=3,
            spacer_5p_sequence="AA",
            spacer_5p_structure="..",
            spacer_3p_sequence="UUU",
            spacer_3p_structure="...",
        )
        assert config.spacer_5p_length == 2
        assert config.spacer_3p_length == 3

    def test_no_spacer_default(self, sample_config):
        assert sample_config.spacer_5p_length == 0
        assert sample_config.spacer_3p_length == 0

    def test_spacer_validation_invalid_nucleotides(self):
        config = LibraryConfig(
            target_length_min=100,
            target_length_max=120,
            motifs_per_construct_min=3,
            motifs_per_construct_max=4,
            p5_sequence="GGAAC",
            p5_structure="(((..",
            p3_sequence="GUUCC",
            p3_structure="..)))",
            helix_length=3,
            spacer_5p_sequence="AXA",
            spacer_5p_structure="...",
        )
        with pytest.raises(ValueError, match="invalid nucleotides"):
            validate_config(config)

    def test_spacer_validation_length_mismatch(self):
        config = LibraryConfig(
            target_length_min=100,
            target_length_max=120,
            motifs_per_construct_min=3,
            motifs_per_construct_max=4,
            p5_sequence="GGAAC",
            p5_structure="(((..",
            p3_sequence="GUUCC",
            p3_structure="..)))",
            helix_length=3,
            spacer_5p_sequence="AA",
            spacer_5p_structure="...",
        )
        with pytest.raises(ValueError, match="same length"):
            validate_config(config)

    def test_round_trip_with_spacers(self, temp_dir):
        config = LibraryConfig(
            target_length_min=100,
            target_length_max=120,
            motifs_per_construct_min=3,
            motifs_per_construct_max=4,
            p5_sequence="GGAAC",
            p5_structure="(((..",
            p3_sequence="GUUCC",
            p3_structure="..)))",
            helix_length=3,
            spacer_5p_sequence="AA",
            spacer_5p_structure="..",
            spacer_3p_sequence="UU",
            spacer_3p_structure="..",
        )
        path = temp_dir / "config.yaml"
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.spacer_5p_sequence == "AA"
        assert loaded.spacer_3p_sequence == "UU"


class TestTargetMotifUsageConfig:
    """Tests for target_motif_usage configuration."""

    def test_default_none(self):
        config = OptimizationConfig()
        assert config.target_motif_usage is None

    def test_set_value(self):
        config = OptimizationConfig(target_motif_usage=50)
        assert config.target_motif_usage == 50

    def test_round_trip(self, temp_dir):
        config = LibraryConfig(
            target_length_min=100,
            target_length_max=120,
            motifs_per_construct_min=3,
            motifs_per_construct_max=4,
            p5_sequence="GGAAC",
            p5_structure="(((..",
            p3_sequence="GUUCC",
            p3_structure="..)))",
            helix_length=3,
            optimization=OptimizationConfig(target_motif_usage=50),
        )
        path = temp_dir / "config.yaml"
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.optimization.target_motif_usage == 50
