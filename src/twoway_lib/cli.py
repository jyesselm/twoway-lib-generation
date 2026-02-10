"""Command line interface for two-way junction library generation."""

import sys
from pathlib import Path
from typing import Annotated

import structlog
import typer

from twoway_lib.config import (
    create_example_config,
    list_available_primers,
    load_config,
)
from twoway_lib.generator import LibraryGenerator, estimate_feasible_lengths
from twoway_lib.helix import random_helix
from twoway_lib.io import get_library_summary, save_library_json
from twoway_lib.motif import Motif, load_motifs
from twoway_lib.validation import fold_sequence

app = typer.Typer(help="Two-way junction RNA library generator.")

# ---------------------------------------------------------------------------
# Config subcommand group
# ---------------------------------------------------------------------------
config_app = typer.Typer(help="Configuration file management commands.")
app.add_typer(config_app, name="config")


@config_app.command("init")
def config_init(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output config file path")
    ] = Path("config.yaml"),
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing file")
    ] = False,
) -> None:
    """Create an example config file with documentation."""
    if output.exists() and not force:
        print(f"File already exists: {output}", file=sys.stderr)
        print("Use --force to overwrite.", file=sys.stderr)
        raise typer.Exit(code=1) from None
    try:
        create_example_config(output)
        print(f"Created example config: {output}")
        print("Edit the file to customize your library generation settings.")
    except Exception as e:
        print(f"Error creating config: {e}", file=sys.stderr)
        raise typer.Exit(code=1) from None


@config_app.command("validate")
def config_validate(
    config_path: Annotated[
        Path, typer.Argument(help="Path to YAML config file", exists=True)
    ],
) -> None:
    """Validate an existing configuration file."""
    try:
        cfg = load_config(config_path)
        print(f"Configuration valid: {config_path}")
        _print_config_summary(cfg)
    except Exception as e:
        print(f"Configuration invalid: {e}", file=sys.stderr)
        raise typer.Exit(code=1) from None


@config_app.command("show")
def config_show(
    config_path: Annotated[
        Path, typer.Argument(help="Path to YAML config file", exists=True)
    ],
) -> None:
    """Load a config and display all parsed values."""
    try:
        cfg = load_config(config_path)
        _print_config_summary(cfg)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        raise typer.Exit(code=1) from None


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


@app.command()
def generate(
    config_path: Annotated[
        Path, typer.Argument(help="Path to YAML configuration file", exists=True)
    ],
    motifs_path: Annotated[
        Path, typer.Argument(help="Path to CSV file with motifs", exists=True)
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output JSON file path")
    ] = Path("library.json"),
    num_candidates: Annotated[
        int,
        typer.Option("--num-candidates", "-n", help="Number of candidates to generate"),
    ] = 50000,
    seed: Annotated[
        int | None, typer.Option("--seed", "-s", help="Random seed for reproducibility")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
    no_filter_motifs: Annotated[
        bool,
        typer.Option(
            "--no-filter-motifs",
            help="Don't filter motifs by fold test (use all motifs)",
        ),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option(
            "--no-color",
            help="Disable colored output (for saving to file)",
        ),
    ] = False,
    log_json: Annotated[
        Path | None,
        typer.Option("--log-json", help="Save log output to JSON file"),
    ] = None,
    max_attempts: Annotated[
        int | None,
        typer.Option(
            "--max-attempts",
            help="Maximum generation attempts (default: num_candidates * 100)",
        ),
    ] = None,
    auto_tune: Annotated[
        bool,
        typer.Option(
            "--auto-tune",
            help="Auto-tune simulated annealing parameters before optimization",
        ),
    ] = False,
    parallel: Annotated[
        bool,
        typer.Option("--parallel", help="Enable parallel candidate generation"),
    ] = False,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            help="Number of workers for parallel generation (default: 4)",
        ),
    ] = 4,
    save_motif_results: Annotated[
        Path | None,
        typer.Option(
            "--save-motif-results",
            help="Save motif preprocessing results to JSON",
        ),
    ] = None,
    load_motif_results: Annotated[
        Path | None,
        typer.Option(
            "--load-motif-results",
            help="Load pre-computed motif preprocessing results",
            exists=True,
        ),
    ] = None,
    detailed_summary: Annotated[
        Path | None,
        typer.Option("--detailed-summary", help="Save detailed summary to JSON file"),
    ] = None,
) -> None:
    """
    Generate a two-way junction library.

    CONFIG_PATH: Path to YAML configuration file
    MOTIFS_PATH: Path to CSV file with motifs
    """
    log_json_str = str(log_json) if log_json else None
    log_collector = _setup_logging(verbose, no_color, log_json_str)
    log = structlog.get_logger()

    try:
        config = load_config(config_path)
        motifs = load_motifs(motifs_path)
        log.info("Loaded config", path=str(config_path))
        log.info("Loaded motifs", count=len(motifs), path=str(motifs_path))

        # Handle motif preprocessing results
        motif_results = None
        if load_motif_results:
            from twoway_lib.preprocessing import (
                load_motif_results as load_results,
            )

            motif_results = load_results(str(load_motif_results))
            log.info(
                "Loaded motif results",
                path=str(load_motif_results),
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
            save_results(results, str(save_motif_results))
            log.info("Saved motif results", path=str(save_motif_results))

        constructs = generator.generate(
            num_candidates,
            max_attempts=max_attempts,
            auto_tune=auto_tune,
            parallel=parallel,
            n_workers=workers,
        )

        save_library_json(constructs, str(output))
        log.info("Saved constructs", count=len(constructs), path=str(output))

        if detailed_summary:
            from twoway_lib.io import save_detailed_summary

            test_result_dicts = None
            if motif_results:
                from dataclasses import asdict

                test_result_dicts = [asdict(r) for r in motif_results]
            save_detailed_summary(
                constructs,
                str(detailed_summary),
                motif_test_results=test_result_dicts,
            )
            log.info("Saved detailed summary", path=str(detailed_summary))

        _print_summary(constructs)
        log_collector.save()

    except Exception as e:
        log.error("Error during generation", error=str(e))
        log_collector.save()
        raise typer.Exit(code=1) from None


@app.command()
def check(
    config_path: Annotated[
        Path, typer.Argument(help="Path to YAML configuration file", exists=True)
    ],
    motifs_path: Annotated[
        Path, typer.Argument(help="Path to CSV file with motifs", exists=True)
    ],
) -> None:
    """
    Check configuration and estimate feasible lengths.

    CONFIG_PATH: Path to YAML configuration file
    MOTIFS_PATH: Path to CSV file with motifs
    """
    try:
        config = load_config(config_path)
        motifs = load_motifs(motifs_path)

        print(f"Configuration loaded: {config_path}")
        print(f"Motifs loaded: {len(motifs)} motifs from {motifs_path}")
        print()

        _print_config_summary(config)
        print()

        min_len, max_len = estimate_feasible_lengths(config, motifs)
        print("Estimated feasible length range:")
        print(f"  Minimum: {min_len} nt")
        print(f"  Maximum: {max_len} nt")
        print(f"  Target:  {config.target_length_min}-{config.target_length_max} nt")
        print()

        if min_len > config.target_length_max:
            print("WARNING: Minimum feasible length exceeds target maximum!")
        elif max_len < config.target_length_min:
            print("WARNING: Maximum feasible length is below target minimum!")
        else:
            print("Configuration appears feasible.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise typer.Exit(code=1) from None


@app.command()
def summary(
    library_path: Annotated[
        Path, typer.Argument(help="Path to library JSON file", exists=True)
    ],
) -> None:
    """
    Display summary statistics for a generated library.

    LIBRARY_PATH: Path to library JSON file
    """
    from twoway_lib.io import load_library_json

    try:
        rows = load_library_json(str(library_path))
        print(f"Library: {library_path}")
        print(f"Constructs: {len(rows)}")

        if rows:
            lengths = [r["length"] for r in rows]
            print(f"Length range: {min(lengths)}-{max(lengths)} nt")
            print(f"Average length: {sum(lengths) / len(lengths):.1f} nt")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise typer.Exit(code=1) from None


@app.command()
def primers() -> None:
    """List available p5 and p3 primer sequences."""
    available = list_available_primers()

    print("Available p5 sequences:")
    if available["p5"]:
        for name in available["p5"]:
            print(f"  - {name}")
    else:
        print("  (none available - seq_tools package may not be installed)")

    print()
    print("Available p3 sequences:")
    if available["p3"]:
        for name in available["p3"]:
            print(f"  - {name}")
    else:
        print("  (none available - seq_tools package may not be installed)")

    print()
    print("Use p5_name or p3_name in your config file to reference these by name.")


@app.command("test-motifs")
def test_motifs(
    motifs_path: Annotated[
        Path, typer.Argument(help="Path to CSV file with motifs", exists=True)
    ],
    repeats: Annotated[
        int,
        typer.Option(
            "--repeats",
            "-r",
            help="Number of times to repeat motif in test construct",
        ),
    ] = 10,
    helix_length: Annotated[
        int, typer.Option("--helix-length", "-l", help="Length of flanking helices")
    ] = 3,
    seed: Annotated[
        int,
        typer.Option("--seed", "-s", help="Random seed for helix generation"),
    ] = 42,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed mismatch info")
    ] = False,
    save_results: Annotated[
        Path | None,
        typer.Option("--save-results", help="Save motif test results to JSON file"),
    ] = None,
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

    print(f"Testing {len(motifs)} motifs with {repeats} repeats each")
    print(f"Helix length: {helix_length} bp")
    print("=" * 70)
    print()

    results = []
    for motif in motifs:
        result = _test_single_motif(motif, repeats, helix_length, rng, verbose)
        results.append(result)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passing = [r for r in results if r["passes"]]
    failing = [r for r in results if not r["passes"]]

    print(f"Passing motifs: {len(passing)}/{len(results)}")
    print(f"Failing motifs: {len(failing)}/{len(results)}")

    if failing:
        print()
        print("Failing motifs (structure mismatch in motif region):")
        for r in failing:
            print(f"  {r['motif']:20s} match: {r['motif_match']:.1%}")

    if save_results:
        import json

        with open(save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_results}")


# ---------------------------------------------------------------------------
# Internal helpers (preserved from Click version)
# ---------------------------------------------------------------------------


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
        print(f"{motif.sequence:15s} {status}")
    else:
        # Show designed vs predicted structure for this motif
        print(
            f"{motif.sequence:15s} {status}  "
            f"failed {instances_failed}/{len(motif_positions)} instances  "
            f"designed: {motif.structure}  predicted: {mismatch_examples[0]['predicted']}"
        )

    if verbose and not passes:
        print(
            f"  Example: seq={mismatch_examples[0]['seq']}  "
            f"designed={mismatch_examples[0]['designed']}  "
            f"predicted={mismatch_examples[0]['predicted']}"
        )
        print()

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
    print("Configuration summary:")
    print(f"  Target length: {config.target_length_min}-{config.target_length_max} nt")
    print(
        f"  Motifs per construct: "
        f"{config.motifs_per_construct_min}-{config.motifs_per_construct_max}"
    )
    min_h, max_h = config.effective_helix_length_range
    if min_h == max_h:
        print(f"  Helix length: {min_h} bp")
    else:
        print(f"  Helix length: {min_h}-{max_h} bp")
    if config.gu_required_above_length is not None:
        print(f"  GU required above: {config.gu_required_above_length} bp")
    print(f"  Hairpin loop length: {config.hairpin_loop_length} nt")
    print(f"  5' sequence length: {config.p5_length} nt")
    print(f"  3' sequence length: {config.p3_length} nt")
    if config.spacer_5p_length > 0:
        print(f"  5' spacer length: {config.spacer_5p_length} nt")
    if config.spacer_3p_length > 0:
        print(f"  3' spacer length: {config.spacer_3p_length} nt")
    print(f"  Validation enabled: {config.validation.enabled}")
    print(f"  Target library size: {config.optimization.target_library_size}")
    if config.optimization.target_motif_usage is not None:
        print(f"  Target motif usage: {config.optimization.target_motif_usage}")


def _print_summary(constructs: list) -> None:
    """Print generation summary."""
    lib_summary = get_library_summary(constructs)
    print()
    print("Generation complete!")
    print(f"  Total constructs: {lib_summary['count']}")
    if lib_summary["count"] > 0:
        print(
            f"  Length range: {lib_summary['length_min']}-{lib_summary['length_max']} nt"
        )
        print(f"  Average length: {lib_summary['length_mean']:.1f} nt")
        print(f"  Unique motifs used: {lib_summary['unique_motifs_used']}")
        print(
            f"  Motif usage range: {lib_summary['motif_usage_min']}-"
            f"{lib_summary['motif_usage_max']} "
            f"(avg: {lib_summary['motif_usage_mean']:.1f})"
        )
        print(f"  Average edit distance: {lib_summary['avg_edit_distance']:.1f}")


if __name__ == "__main__":
    app()
