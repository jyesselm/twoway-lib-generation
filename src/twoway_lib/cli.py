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
from twoway_lib.helix import random_helix
from twoway_lib.io import get_library_summary, save_library_json
from twoway_lib.motif import Motif, load_motifs
from twoway_lib.validation import fold_sequence


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
@click.option(
    "--no-filter-motifs",
    is_flag=True,
    help="Don't filter motifs by fold test (use all motifs)",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable colored output (for saving to file)",
)
@click.option(
    "--log-json",
    type=click.Path(),
    default=None,
    help="Save log output to JSON file",
)
@click.option(
    "--max-attempts",
    type=int,
    default=None,
    help="Maximum generation attempts (default: num_candidates * 100)",
)
@click.option(
    "--auto-tune",
    is_flag=True,
    help="Auto-tune simulated annealing parameters before optimization",
)
@click.option(
    "--parallel",
    is_flag=True,
    help="Enable parallel candidate generation",
)
@click.option(
    "--workers",
    type=int,
    default=4,
    help="Number of workers for parallel generation (default: 4)",
)
@click.option(
    "--save-motif-results",
    type=click.Path(),
    default=None,
    help="Save motif preprocessing results to JSON",
)
@click.option(
    "--load-motif-results",
    type=click.Path(exists=True),
    default=None,
    help="Load pre-computed motif preprocessing results",
)
@click.option(
    "--detailed-summary",
    type=click.Path(),
    default=None,
    help="Save detailed summary to JSON file",
)
def generate(
    config_path: str,
    motifs_path: str,
    output: str,
    num_candidates: int,
    seed: int | None,
    verbose: bool,
    no_filter_motifs: bool,
    no_color: bool,
    log_json: str | None,
    max_attempts: int | None,
    auto_tune: bool,
    parallel: bool,
    workers: int,
    save_motif_results: str | None,
    load_motif_results: str | None,
    detailed_summary: str | None,
) -> None:
    """
    Generate a two-way junction library.

    CONFIG_PATH: Path to YAML configuration file
    MOTIFS_PATH: Path to CSV file with motifs
    """
    log_collector = _setup_logging(verbose, no_color, log_json)
    log = structlog.get_logger()

    try:
        config = load_config(config_path)
        motifs = load_motifs(motifs_path)
        log.info("Loaded config", path=config_path)
        log.info("Loaded motifs", count=len(motifs), path=motifs_path)

        # Handle motif preprocessing results
        motif_results = None
        if load_motif_results:
            from twoway_lib.preprocessing import (
                load_motif_results as load_results,
            )

            motif_results = load_results(load_motif_results)
            log.info(
                "Loaded motif results",
                path=load_motif_results,
                count=len(motif_results),
            )

        generator = LibraryGenerator(
            config,
            motifs,
            seed=seed,
            filter_motifs=not no_filter_motifs,
        )

        if save_motif_results and not no_filter_motifs:
            from twoway_lib.preprocessing import (
                preprocess_motifs,
            )
            from twoway_lib.preprocessing import (
                save_motif_results as save_results,
            )

            _, _, results = preprocess_motifs(
                motifs,
                helix_length=config.helix_length,
            )
            save_results(results, save_motif_results)
            log.info("Saved motif results", path=save_motif_results)

        constructs = generator.generate(
            num_candidates,
            max_attempts=max_attempts,
            auto_tune=auto_tune,
            parallel=parallel,
            n_workers=workers,
        )

        save_library_json(constructs, output)
        log.info("Saved constructs", count=len(constructs), path=output)

        if detailed_summary:
            from twoway_lib.io import save_detailed_summary

            test_result_dicts = None
            if motif_results:
                from dataclasses import asdict

                test_result_dicts = [asdict(r) for r in motif_results]
            save_detailed_summary(
                constructs,
                detailed_summary,
                motif_test_results=test_result_dicts,
            )
            log.info("Saved detailed summary", path=detailed_summary)

        _print_summary(constructs)
        log_collector.save()

    except Exception as e:
        log.error("Error during generation", error=str(e))
        log_collector.save()
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
    "--validate",
    "validate_path",
    type=click.Path(exists=True),
    help="Validate an existing config file instead of generating",
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


@cli.command("test-motifs")
@click.argument("motifs_path", type=click.Path(exists=True))
@click.option(
    "-r",
    "--repeats",
    default=10,
    help="Number of times to repeat motif in test construct",
)
@click.option("-l", "--helix-length", default=3, help="Length of flanking helices")
@click.option(
    "-s", "--seed", type=int, default=42, help="Random seed for helix generation"
)
@click.option("-v", "--verbose", is_flag=True, help="Show detailed mismatch info")
@click.option(
    "--save-results",
    type=click.Path(),
    default=None,
    help="Save motif test results to JSON file",
)
def test_motifs(
    motifs_path: str,
    repeats: int,
    helix_length: int,
    seed: int,
    verbose: bool,
    save_results: str | None,
) -> None:
    """
    Test each motif for correct folding in a helix context.

    Embeds each motif multiple times in a hairpin construct with random helices
    and checks if Vienna RNA predicts the designed structure.

    MOTIFS_PATH: Path to CSV file with motifs
    """
    from random import Random

    motifs = load_motifs(motifs_path)
    rng = Random(seed)

    click.echo(f"Testing {len(motifs)} motifs with {repeats} repeats each")
    click.echo(f"Helix length: {helix_length} bp")
    click.echo("=" * 70)
    click.echo()

    results = []
    for motif in motifs:
        result = _test_single_motif(motif, repeats, helix_length, rng, verbose)
        results.append(result)

    # Summary
    click.echo()
    click.echo("=" * 70)
    click.echo("SUMMARY")
    click.echo("=" * 70)

    passing = [r for r in results if r["passes"]]
    failing = [r for r in results if not r["passes"]]

    click.echo(f"Passing motifs: {len(passing)}/{len(results)}")
    click.echo(f"Failing motifs: {len(failing)}/{len(results)}")

    if failing:
        click.echo()
        click.echo("Failing motifs (structure mismatch in motif region):")
        for r in failing:
            click.echo(f"  {r['motif']:20s} match: {r['motif_match']:.1%}")

    if save_results:
        import json

        with open(save_results, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to {save_results}")


def _test_single_motif(
    motif: Motif,
    repeats: int,
    helix_length: int,
    rng,
    verbose: bool,
) -> dict:
    """Test a single motif by embedding it in a hairpin construct."""
    # Build test construct: helix + (motif + helix) * repeats + loop
    seq_parts = []
    ss_parts = []

    # Generate helices
    helices = [random_helix(helix_length, rng) for _ in range(repeats + 1)]

    # Build 5' arm: helix + motif_strand1 + helix + ...
    for i in range(repeats):
        seq_parts.append(helices[i].strand1)
        ss_parts.append(helices[i].structure1)
        seq_parts.append(motif.strand1_seq)
        ss_parts.append(motif.strand1_ss)

    # Final helix before loop
    seq_parts.append(helices[repeats].strand1)
    ss_parts.append(helices[repeats].structure1)

    # Hairpin loop
    seq_parts.append("GAAA")
    ss_parts.append("....")

    # Build 3' arm: helix + motif_strand2 + helix + ... (reversed)
    seq_parts.append(helices[repeats].strand2)
    ss_parts.append(helices[repeats].structure2)

    for i in range(repeats - 1, -1, -1):
        seq_parts.append(motif.strand2_seq)
        ss_parts.append(motif.strand2_ss)
        seq_parts.append(helices[i].strand2)
        ss_parts.append(helices[i].structure2)

    sequence = "".join(seq_parts)
    designed_ss = "".join(ss_parts)

    # Fold and compare
    fold_result = fold_sequence(sequence)
    predicted_ss = fold_result.predicted_structure

    # Calculate overall match
    total_match = sum(
        1 for d, p in zip(designed_ss, predicted_ss, strict=True) if d == p
    )
    match_pct = total_match / len(designed_ss)

    # Calculate match per motif instance and collect mismatch info
    motif_positions = _find_motif_positions(helices, motif, repeats, helix_length)

    instances_failed = 0
    motif_matches = 0
    motif_total = 0
    mismatch_examples: list[dict[str, str]] = []

    for _idx, (start, end) in enumerate(motif_positions):
        instance_match = True
        for j in range(start, end):
            motif_total += 1
            if designed_ss[j] == predicted_ss[j]:
                motif_matches += 1
            else:
                instance_match = False

        if not instance_match:
            instances_failed += 1
            if len(mismatch_examples) < 2:
                mismatch_examples.append(
                    {
                        "seq": sequence[start:end],
                        "designed": designed_ss[start:end],
                        "predicted": predicted_ss[start:end],
                    }
                )

    motif_match_pct = motif_matches / motif_total if motif_total > 0 else 1.0
    passes = motif_match_pct == 1.0

    # Build output line
    status = "PASS" if passes else "FAIL"
    if passes:
        click.echo(f"{motif.sequence:15s} {status}")
    else:
        # Show designed vs predicted structure for this motif
        click.echo(
            f"{motif.sequence:15s} {status}  "
            f"failed {instances_failed}/{len(motif_positions)} instances  "
            f"designed: {motif.structure}  predicted: {mismatch_examples[0]['predicted']}"
        )

    if verbose and not passes:
        click.echo(
            f"  Example: seq={mismatch_examples[0]['seq']}  "
            f"designed={mismatch_examples[0]['designed']}  "
            f"predicted={mismatch_examples[0]['predicted']}"
        )
        click.echo()

    return {
        "motif": motif.sequence,
        "passes": passes,
        "overall_match": match_pct,
        "motif_match": motif_match_pct,
        "instances_failed": instances_failed,
        "total_instances": len(motif_positions),
    }


def _find_motif_positions(
    helices: list,
    motif: Motif,
    repeats: int,
    helix_length: int,
) -> list[tuple[int, int]]:
    """Find start/end positions of motif regions in the test construct."""
    positions = []
    motif_s1_len = len(motif.strand1_seq)
    motif_s2_len = len(motif.strand2_seq)

    # 5' arm motif positions
    pos = 0
    for _i in range(repeats):
        pos += helix_length  # helix
        positions.append((pos, pos + motif_s1_len))
        pos += motif_s1_len

    # Skip final helix + loop + final helix
    pos += helix_length + 4 + helix_length

    # 3' arm motif positions (reverse order)
    for _i in range(repeats):
        positions.append((pos, pos + motif_s2_len))
        pos += motif_s2_len + helix_length

    return positions


class LogCollector:
    """Collects log entries for JSON output."""

    def __init__(self, output_path: str | None):
        self.output_path = output_path
        self.entries: list[dict] = []

    def __call__(self, logger, method_name, event_dict):
        """Processor that collects log entries."""
        if self.output_path:
            self.entries.append(dict(event_dict))
        return event_dict

    def save(self):
        """Save collected logs to JSON file."""
        if self.output_path and self.entries:
            import json

            with open(self.output_path, "w") as f:
                json.dump(self.entries, f, indent=2, default=str)


def _setup_logging(
    verbose: bool, no_color: bool = False, log_json: str | None = None
) -> LogCollector:
    """Configure structlog based on verbosity."""
    import logging

    level = logging.DEBUG if verbose else logging.INFO
    collector = LogCollector(log_json)

    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
        collector,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(sort_keys=False, colors=not no_color),
    ]

    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=level,
    )

    return collector


def _print_config_summary(config) -> None:
    """Print configuration summary."""
    click.echo("Configuration summary:")
    click.echo(
        f"  Target length: {config.target_length_min}-{config.target_length_max} nt"
    )
    click.echo(
        f"  Motifs per construct: {config.motifs_per_construct_min}-{config.motifs_per_construct_max}"
    )
    min_h, max_h = config.effective_helix_length_range
    if min_h == max_h:
        click.echo(f"  Helix length: {min_h} bp")
    else:
        click.echo(f"  Helix length: {min_h}-{max_h} bp")
    if config.gu_required_above_length is not None:
        click.echo(f"  GU required above: {config.gu_required_above_length} bp")
    click.echo(f"  Hairpin loop length: {config.hairpin_loop_length} nt")
    click.echo(f"  5' sequence length: {config.p5_length} nt")
    click.echo(f"  3' sequence length: {config.p3_length} nt")
    if config.spacer_5p_length > 0:
        click.echo(f"  5' spacer length: {config.spacer_5p_length} nt")
    if config.spacer_3p_length > 0:
        click.echo(f"  3' spacer length: {config.spacer_3p_length} nt")
    click.echo(f"  Validation enabled: {config.validation.enabled}")
    click.echo(f"  Target library size: {config.optimization.target_library_size}")
    if config.optimization.target_motif_usage is not None:
        click.echo(f"  Target motif usage: {config.optimization.target_motif_usage}")


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
