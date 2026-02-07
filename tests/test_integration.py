"""Integration tests for the full library generation pipeline."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from twoway_lib.cli import cli
from twoway_lib.config import (
    LibraryConfig,
    OptimizationConfig,
    ValidationConfig,
    load_config,
)
from twoway_lib.generator import LibraryGenerator, generate_library
from twoway_lib.io import load_library_json, save_library_json
from twoway_lib.motif import Motif, load_motifs
from twoway_lib.validation import compare_structures, fold_sequence

# Path to example files
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


class TestFullPipelineWithExamples:
    """Integration tests using the example config and motifs files."""

    @pytest.fixture
    def example_config_path(self) -> Path:
        return EXAMPLES_DIR / "config.yaml"

    @pytest.fixture
    def example_motifs_path(self) -> Path:
        return EXAMPLES_DIR / "motifs.csv"

    def test_example_files_exist(self, example_config_path, example_motifs_path):
        """Verify example files are present."""
        assert example_config_path.exists(), f"Config not found: {example_config_path}"
        assert example_motifs_path.exists(), f"Motifs not found: {example_motifs_path}"

    def test_load_example_config(self, example_config_path):
        """Load and validate the example config."""
        config = load_config(example_config_path)
        assert isinstance(config, LibraryConfig)
        assert config.target_length_min < config.target_length_max
        assert config.p5_sequence
        assert config.p3_sequence

    def test_load_example_motifs(self, example_motifs_path):
        """Load the example motifs."""
        motifs = load_motifs(example_motifs_path)
        assert len(motifs) > 0
        assert all(isinstance(m, Motif) for m in motifs)

    def test_generate_constructs_from_examples(
        self, example_config_path, example_motifs_path
    ):
        """Generate constructs using example config and motifs."""
        config = load_config(example_config_path)
        motifs = load_motifs(example_motifs_path)

        # Use fewer candidates for faster testing
        config = LibraryConfig(
            target_length_min=config.target_length_min,
            target_length_max=config.target_length_max,
            motifs_per_construct_min=config.motifs_per_construct_min,
            motifs_per_construct_max=config.motifs_per_construct_max,
            p5_sequence=config.p5_sequence,
            p5_structure=config.p5_structure,
            p3_sequence=config.p3_sequence,
            p3_structure=config.p3_structure,
            helix_length=config.helix_length,
            hairpin_loop_length=config.hairpin_loop_length,
            validation=ValidationConfig(enabled=False),
            optimization=OptimizationConfig(
                iterations=100,
                target_library_size=20,
            ),
        )

        constructs = generate_library(config, motifs, num_candidates=100, seed=42)

        assert len(constructs) > 0, "Should generate at least one construct"

        # Verify all constructs meet length constraints
        for c in constructs:
            assert config.target_length_min <= c.length() <= config.target_length_max
            assert len(c.sequence) == len(c.structure)

    def test_full_pipeline_with_validation(
        self, example_config_path, example_motifs_path
    ):
        """Test full pipeline with Vienna RNA validation enabled."""
        config = load_config(example_config_path)
        motifs = load_motifs(example_motifs_path)

        # Create config with validation enabled
        config = LibraryConfig(
            target_length_min=config.target_length_min,
            target_length_max=config.target_length_max,
            motifs_per_construct_min=config.motifs_per_construct_min,
            motifs_per_construct_max=config.motifs_per_construct_max,
            p5_sequence=config.p5_sequence,
            p5_structure=config.p5_structure,
            p3_sequence=config.p3_sequence,
            p3_structure=config.p3_structure,
            helix_length=config.helix_length,
            hairpin_loop_length=config.hairpin_loop_length,
            validation=ValidationConfig(
                enabled=True,
                max_ensemble_defect=10.0,
                allow_structure_differences=True,
                min_structure_match=0.5,
            ),
            optimization=OptimizationConfig(
                iterations=100,
                target_library_size=10,
            ),
        )

        generator = LibraryGenerator(config, motifs, seed=42)
        generator.generate(num_candidates=100)

        # May get few or no constructs due to validation
        # Just verify the pipeline completes without error
        assert generator.stats.candidates_generated > 0


class TestPipelineWithIO:
    """Test the full pipeline including file I/O."""

    def test_generate_and_save_library(self):
        """Test generating a library and saving to JSON."""
        config = LibraryConfig(
            target_length_min=50,
            target_length_max=80,
            motifs_per_construct_min=2,
            motifs_per_construct_max=3,
            p5_sequence="GGGCGAAAGCCC",
            p5_structure="((((....))))",
            p3_sequence="AAAGAAAC",
            p3_structure="........",
            helix_length=3,
            hairpin_loop_length=4,
            validation=ValidationConfig(enabled=False),
            optimization=OptimizationConfig(
                iterations=100,
                target_library_size=10,
            ),
        )

        motifs = [
            Motif.from_string("GAC&GC", "(.(&))"),
            Motif.from_string("AAG&CUU", "(.(&.))"),
            Motif.from_string("UUG&CAA", "(.(&.))"),
        ]

        constructs = generate_library(config, motifs, num_candidates=50, seed=42)
        assert len(constructs) > 0

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "library.json"
            save_library_json(constructs, output_path)

            assert output_path.exists()

            # Load and verify
            loaded = load_library_json(output_path)
            assert len(loaded) == len(constructs)

            # Verify data integrity
            assert "sequence" in loaded[0]
            assert "structure" in loaded[0]
            assert "length" in loaded[0]
            assert "motifs" in loaded[0]

    def test_saved_library_has_correct_data(self):
        """Verify saved library contains correct construct data."""
        config = LibraryConfig(
            target_length_min=50,
            target_length_max=100,
            motifs_per_construct_min=2,
            motifs_per_construct_max=3,
            p5_sequence="GGGGG",
            p5_structure=".....",
            p3_sequence="CCCCC",
            p3_structure=".....",
            helix_length=3,
            hairpin_loop_length=4,
            validation=ValidationConfig(enabled=False),
            optimization=OptimizationConfig(
                iterations=50,
                target_library_size=5,
            ),
        )

        motifs = [
            Motif.from_string("GAC&GC", "(.(&))"),
            Motif.from_string("AAG&CUU", "(.(&.))"),
        ]

        constructs = generate_library(config, motifs, num_candidates=100, seed=42)
        assert len(constructs) > 0, "No constructs generated"

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "library.json"
            save_library_json(constructs, output_path)

            data = load_library_json(output_path)

            # Check each row matches original construct
            for i, c in enumerate(constructs):
                row = data[i]
                assert row["sequence"] == c.sequence
                assert row["structure"] == c.structure
                assert row["length"] == c.length()


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    @pytest.fixture
    def runner(self):
        from click.testing import CliRunner

        return CliRunner()

    def test_cli_generate_command(self, runner):
        """Test the generate CLI command with a working config."""
        # Create a simple config that will definitely work
        config_content = """
target_length:
  min: 50
  max: 100
motifs_per_construct:
  min: 2
  max: 3
p5_sequence: "GGGGG"
p5_structure: "....."
p3_sequence: "CCCCC"
p3_structure: "....."
helix_length: 3
hairpin_loop_length: 4
validation:
  enabled: false
optimization:
  iterations: 100
  target_library_size: 10
"""
        motifs_content = """sequence,structure
GAC&GC,(.(&))
AAG&CUU,(.(&.))
UUG&CAA,(.(&.))
"""

        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            motifs_path = Path(tmpdir) / "motifs.csv"
            output = Path(tmpdir) / "output.json"

            config_path.write_text(config_content)
            motifs_path.write_text(motifs_content)

            result = runner.invoke(
                cli,
                [
                    "generate",
                    str(config_path),
                    str(motifs_path),
                    "-o",
                    str(output),
                    "-n",
                    "50",
                    "-s",
                    "42",
                ],
            )

            # Check command completed
            if result.exit_code != 0:
                print(f"CLI output: {result.output}")
                print(f"Exception: {result.exception}")

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            assert output.exists(), "Output file not created"

            # Verify output has constructs
            with open(output) as f:
                data = json.load(f)
            assert len(data) > 0, "No constructs generated"

    def test_cli_check_command(self, runner):
        """Test the check CLI command."""
        example_config = EXAMPLES_DIR / "config.yaml"
        example_motifs = EXAMPLES_DIR / "motifs.csv"

        if not example_config.exists() or not example_motifs.exists():
            pytest.skip("Example files not found")

        result = runner.invoke(
            cli,
            ["check", str(example_config), str(example_motifs)],
        )

        assert result.exit_code == 0
        assert "Configuration loaded" in result.output
        assert "feasible" in result.output.lower()

    def test_cli_config_generation(self, runner):
        """Test default config generation."""
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "default_config.yaml"
            result = runner.invoke(
                cli,
                ["config", "-o", str(output)],
            )

            assert result.exit_code == 0
            assert output.exists()

            # Verify generated config is valid
            config = load_config(output)
            assert isinstance(config, LibraryConfig)


class TestMotifPositionTracking:
    """Test that motif positions are tracked correctly through the pipeline."""

    def test_constructs_have_motif_positions(self):
        """Verify motif positions are tracked in generated constructs."""
        config = LibraryConfig(
            target_length_min=50,
            target_length_max=100,
            motifs_per_construct_min=2,
            motifs_per_construct_max=3,
            p5_sequence="GGGGG",
            p5_structure=".....",
            p3_sequence="CCCCC",
            p3_structure=".....",
            helix_length=3,
            hairpin_loop_length=4,
            validation=ValidationConfig(enabled=False),
            optimization=OptimizationConfig(
                iterations=50,
                target_library_size=5,
            ),
        )

        motifs = [
            Motif.from_string("GAC&GC", "(.(&))"),
            Motif.from_string("AAG&CUU", "(.(&.))"),
        ]

        constructs = generate_library(config, motifs, num_candidates=100, seed=42)
        assert len(constructs) > 0, "No constructs generated"

        for c in constructs:
            # Each construct should have motif positions
            assert hasattr(c, "motif_positions")
            assert len(c.motif_positions) == len(c.motifs)

            # Positions should match sequence
            for mp in c.motif_positions:
                strand1_seq = "".join(c.sequence[i] for i in mp.strand1_positions)
                assert strand1_seq == mp.motif.strand1_seq

                strand2_seq = "".join(c.sequence[i] for i in mp.strand2_positions)
                assert strand2_seq == mp.motif.strand2_seq

    def test_positions_saved_to_json(self):
        """Verify motif positions are included in saved JSON."""
        config = LibraryConfig(
            target_length_min=50,
            target_length_max=100,
            motifs_per_construct_min=2,
            motifs_per_construct_max=3,
            p5_sequence="GGGGG",
            p5_structure=".....",
            p3_sequence="CCCCC",
            p3_structure=".....",
            helix_length=3,
            hairpin_loop_length=4,
            validation=ValidationConfig(enabled=False),
            optimization=OptimizationConfig(
                iterations=50,
                target_library_size=5,
            ),
        )

        motifs = [
            Motif.from_string("GAC&GC", "(.(&))"),
            Motif.from_string("AAG&CUU", "(.(&.))"),
        ]

        constructs = generate_library(config, motifs, num_candidates=100, seed=42)
        assert len(constructs) > 0, "No constructs generated"

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "library.json"
            save_library_json(constructs, output_path)

            data = load_library_json(output_path)
            for row in data:
                assert "motifs" in row
                for motif in row["motifs"]:
                    assert "positions" in motif
                    assert "strand1" in motif["positions"]
                    assert "strand2" in motif["positions"]


class TestRefoldVerification:
    """Test that generated constructs refold to expected structure."""

    def test_constructs_refold_correctly(self):
        """Generate a small library and verify each construct refolds correctly."""
        config = LibraryConfig(
            target_length_min=50,
            target_length_max=100,
            motifs_per_construct_min=2,
            motifs_per_construct_max=3,
            p5_sequence="GGGCGAAAGCCC",
            p5_structure="((((....))))",
            p3_sequence="AAAGAAAC",
            p3_structure="........",
            helix_length=3,
            hairpin_loop_length=4,
            validation=ValidationConfig(enabled=False),
            optimization=OptimizationConfig(
                iterations=100,
                target_library_size=10,
            ),
        )

        motifs = [
            Motif.from_string("GAC&GC", "(.(&))"),
            Motif.from_string("AAG&CUU", "(.(&.))"),
            Motif.from_string("UUG&CAA", "(.(&.))"),
        ]

        constructs = generate_library(config, motifs, num_candidates=50, seed=42)
        assert len(constructs) > 0, "No constructs generated"

        # Refold each construct and verify structure matches
        for construct in constructs:
            fold_result = fold_sequence(construct.sequence)
            match_fraction = compare_structures(
                construct.structure, fold_result.predicted_structure
            )
            # Allow some flexibility - at least 70% match
            assert match_fraction >= 0.7, (
                f"Structure mismatch for construct:\n"
                f"  Sequence:  {construct.sequence}\n"
                f"  Designed:  {construct.structure}\n"
                f"  Predicted: {fold_result.predicted_structure}\n"
                f"  Match: {match_fraction:.1%}"
            )

    def test_validated_constructs_refold_accurately(self):
        """Constructs that passed validation should refold with high accuracy."""
        config = LibraryConfig(
            target_length_min=50,
            target_length_max=100,
            motifs_per_construct_min=2,
            motifs_per_construct_max=3,
            p5_sequence="GGGCGAAAGCCC",
            p5_structure="((((....))))",
            p3_sequence="AAAGAAAC",
            p3_structure="........",
            helix_length=3,
            hairpin_loop_length=4,
            validation=ValidationConfig(
                enabled=True,
                max_ensemble_defect=10.0,
                allow_structure_differences=True,
                min_structure_match=0.8,
            ),
            optimization=OptimizationConfig(
                iterations=100,
                target_library_size=5,
            ),
        )

        motifs = [
            Motif.from_string("GAC&GC", "(.(&))"),
            Motif.from_string("AAG&CUU", "(.(&.))"),
            Motif.from_string("UUG&CAA", "(.(&.))"),
        ]

        constructs = generate_library(config, motifs, num_candidates=100, seed=42)

        # All validated constructs should refold with at least 80% accuracy
        for construct in constructs:
            fold_result = fold_sequence(construct.sequence)
            match_fraction = compare_structures(
                construct.structure, fold_result.predicted_structure
            )
            assert match_fraction >= 0.8, (
                f"Validated construct has poor refold:\n"
                f"  Sequence:  {construct.sequence}\n"
                f"  Designed:  {construct.structure}\n"
                f"  Predicted: {fold_result.predicted_structure}\n"
                f"  Match: {match_fraction:.1%}"
            )
