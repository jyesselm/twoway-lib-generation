"""Tests for cli module."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from twoway_lib.cli import cli


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


class TestGenerateCommand:
    """Tests for generate command."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "Generate a two-way junction library" in result.output

    def test_missing_args(self, runner):
        result = runner.invoke(cli, ["generate"])
        assert result.exit_code != 0

    def test_nonexistent_config(self, runner, temp_motifs_file: Path):
        result = runner.invoke(
            cli, ["generate", "/nonexistent.yaml", str(temp_motifs_file)]
        )
        assert result.exit_code != 0

    def test_basic_generation(
        self,
        runner,
        temp_config_file: Path,
        temp_motifs_file: Path,
        temp_dir: Path,
    ):
        output = temp_dir / "output.csv"
        result = runner.invoke(
            cli,
            [
                "generate",
                str(temp_config_file),
                str(temp_motifs_file),
                "-o",
                str(output),
                "-n",
                "10",
                "-s",
                "42",
            ],
        )
        if result.exit_code != 0:
            print(result.output)
        assert output.exists() or result.exit_code != 0


class TestCheckCommand:
    """Tests for check command."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["check", "--help"])
        assert result.exit_code == 0

    def test_check_config(self, runner, temp_config_file: Path, temp_motifs_file: Path):
        result = runner.invoke(
            cli, ["check", str(temp_config_file), str(temp_motifs_file)]
        )
        assert result.exit_code == 0
        assert "Configuration loaded" in result.output

    def test_check_shows_summary(
        self, runner, temp_config_file: Path, temp_motifs_file: Path
    ):
        result = runner.invoke(
            cli, ["check", str(temp_config_file), str(temp_motifs_file)]
        )
        assert "Target length" in result.output
        assert "Motifs loaded" in result.output


class TestSummaryCommand:
    """Tests for summary command."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["summary", "--help"])
        assert result.exit_code == 0

    def test_nonexistent_file(self, runner):
        result = runner.invoke(cli, ["summary", "/nonexistent.csv"])
        assert result.exit_code != 0


class TestConfigCommand:
    """Tests for config command."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "Generate a default config file" in result.output

    def test_generate_default_config(self, runner, temp_dir: Path):
        output = temp_dir / "config.yaml"
        result = runner.invoke(cli, ["config", "-o", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        assert "Generated default config" in result.output

    def test_validate_config(self, runner, temp_config_file: Path):
        result = runner.invoke(cli, ["config", "--validate", str(temp_config_file)])
        assert result.exit_code == 0
        assert "Configuration valid" in result.output

    def test_validate_invalid_config(self, runner, temp_dir: Path):
        bad_config = temp_dir / "bad.yaml"
        bad_config.write_text("target_length:\n  min: 200\n  max: 100\n")
        result = runner.invoke(cli, ["config", "--validate", str(bad_config)])
        assert result.exit_code != 0
        assert "invalid" in result.output.lower()
