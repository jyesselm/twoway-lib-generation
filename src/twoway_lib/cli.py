"""Command line interface for two-way junction library generation."""

import sys

import click
import structlog

from twoway_lib.config import (
    generate_default_config,
    list_available_primers,
    load_config,
    save_config,
)
from twoway_lib.generator import LibraryGenerator, estimate_feasible_lengths
from twoway_lib.io import get_library_summary, save_library_json
from twoway_lib.motif import load_motifs


@click.group()
@click.version_option()
def cli() -> None:
    """Two-way junction RNA library generator."""
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("motifs_path", type=click.Path(exists=True))
@click.option("-o", "--output", default="library.json", help="Output JSON file path")
@click.option(
    "-n", "--num-candidates", default=50000, help="Number of candidates to generate"
)
@click.option(
    "-s", "--seed", type=int, default=None, help="Random seed for reproducibility"
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def generate(
    config_path: str,
    motifs_path: str,
    output: str,
    num_candidates: int,
    seed: int | None,
    verbose: bool,
) -> None:
    """
    Generate a two-way junction library.

    CONFIG_PATH: Path to YAML configuration file
    MOTIFS_PATH: Path to CSV file with motifs
    """
    _setup_logging(verbose)
    log = structlog.get_logger()

    try:
        config = load_config(config_path)
        motifs = load_motifs(motifs_path)
        log.info("Loaded config", path=config_path)
        log.info("Loaded motifs", count=len(motifs), path=motifs_path)

        generator = LibraryGenerator(config, motifs, seed=seed)
        constructs = generator.generate(num_candidates)

        save_library_json(constructs, output)
        log.info("Saved constructs", count=len(constructs), path=output)

        _print_summary(constructs)

    except Exception as e:
        log.error("Error during generation", error=str(e))
        sys.exit(1)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("motifs_path", type=click.Path(exists=True))
def check(config_path: str, motifs_path: str) -> None:
    """
    Check configuration and estimate feasible lengths.

    CONFIG_PATH: Path to YAML configuration file
    MOTIFS_PATH: Path to CSV file with motifs
    """
    try:
        config = load_config(config_path)
        motifs = load_motifs(motifs_path)

        click.echo(f"Configuration loaded: {config_path}")
        click.echo(f"Motifs loaded: {len(motifs)} motifs from {motifs_path}")
        click.echo()

        _print_config_summary(config)
        click.echo()

        min_len, max_len = estimate_feasible_lengths(config, motifs)
        click.echo("Estimated feasible length range:")
        click.echo(f"  Minimum: {min_len} nt")
        click.echo(f"  Maximum: {max_len} nt")
        click.echo(
            f"  Target:  {config.target_length_min}-{config.target_length_max} nt"
        )
        click.echo()

        if min_len > config.target_length_max:
            click.echo("WARNING: Minimum feasible length exceeds target maximum!")
        elif max_len < config.target_length_min:
            click.echo("WARNING: Maximum feasible length is below target minimum!")
        else:
            click.echo("Configuration appears feasible.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("library_path", type=click.Path(exists=True))
def summary(library_path: str) -> None:
    """
    Display summary statistics for a generated library.

    LIBRARY_PATH: Path to library JSON file
    """
    from twoway_lib.io import load_library_json

    try:
        rows = load_library_json(library_path)
        click.echo(f"Library: {library_path}")
        click.echo(f"Constructs: {len(rows)}")

        if rows:
            lengths = [r["length"] for r in rows]
            click.echo(f"Length range: {min(lengths)}-{max(lengths)} nt")
            click.echo(f"Average length: {sum(lengths) / len(lengths):.1f} nt")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("-o", "--output", default="config.yaml", help="Output config file path")
@click.option(
    "--validate", "validate_path", type=click.Path(exists=True),
    help="Validate an existing config file instead of generating"
)
def config(output: str, validate_path: str | None) -> None:
    """
    Generate a default config file or validate an existing one.

    Use --validate to check an existing configuration file.
    """
    if validate_path:
        try:
            cfg = load_config(validate_path)
            click.echo(f"Configuration valid: {validate_path}")
            _print_config_summary(cfg)
        except Exception as e:
            click.echo(f"Configuration invalid: {e}", err=True)
            sys.exit(1)
    else:
        try:
            cfg = generate_default_config()
            save_config(cfg, output)
            click.echo(f"Generated default config: {output}")
            click.echo("Edit the file to customize your library generation settings.")
        except Exception as e:
            click.echo(f"Error generating config: {e}", err=True)
            sys.exit(1)


@cli.command()
def primers() -> None:
    """List available p5 and p3 primer sequences."""
    available = list_available_primers()

    click.echo("Available p5 sequences:")
    if available["p5"]:
        for name in available["p5"]:
            click.echo(f"  - {name}")
    else:
        click.echo("  (none available - seq_tools package may not be installed)")

    click.echo()
    click.echo("Available p3 sequences:")
    if available["p3"]:
        for name in available["p3"]:
            click.echo(f"  - {name}")
    else:
        click.echo("  (none available - seq_tools package may not be installed)")

    click.echo()
    click.echo("Use p5_name or p3_name in your config file to reference these by name.")


def _setup_logging(verbose: bool) -> None:
    """Configure structlog based on verbosity."""
    import logging

    level = logging.DEBUG if verbose else logging.INFO

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=level,
    )


def _print_config_summary(config) -> None:
    """Print configuration summary."""
    click.echo("Configuration summary:")
    click.echo(
        f"  Target length: {config.target_length_min}-{config.target_length_max} nt"
    )
    click.echo(
        f"  Motifs per construct: {config.motifs_per_construct_min}-{config.motifs_per_construct_max}"
    )
    click.echo(f"  Helix length: {config.helix_length} bp")
    click.echo(f"  Hairpin loop length: {config.hairpin_loop_length} nt")
    click.echo(f"  5' sequence length: {config.p5_length} nt")
    click.echo(f"  3' sequence length: {config.p3_length} nt")
    click.echo(f"  Validation enabled: {config.validation.enabled}")
    click.echo(f"  Target library size: {config.optimization.target_library_size}")


def _print_summary(constructs: list) -> None:
    """Print generation summary."""
    summary = get_library_summary(constructs)
    click.echo()
    click.echo("Generation complete!")
    click.echo(f"  Total constructs: {summary['count']}")
    if summary["count"] > 0:
        click.echo(
            f"  Length range: {summary['length_min']}-{summary['length_max']} nt"
        )
        click.echo(f"  Average length: {summary['length_mean']:.1f} nt")
        click.echo(f"  Unique motifs used: {summary['unique_motifs_used']}")
        click.echo(
            f"  Motif usage range: {summary['motif_usage_min']}-{summary['motif_usage_max']} "
            f"(avg: {summary['motif_usage_mean']:.1f})"
        )
        click.echo(f"  Average edit distance: {summary['avg_edit_distance']:.1f}")


if __name__ == "__main__":
    cli()
