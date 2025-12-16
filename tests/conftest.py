"""Shared test fixtures for twoway_lib tests."""

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest

from twoway_lib.config import LibraryConfig, OptimizationConfig, ValidationConfig
from twoway_lib.construct import Construct
from twoway_lib.hairpin import Hairpin
from twoway_lib.helix import Helix
from twoway_lib.motif import Motif


@pytest.fixture
def sample_motif() -> Motif:
    """A simple two-way junction motif."""
    return Motif.from_string("GAC&GC", "(.(&))")


@pytest.fixture
def sample_motifs() -> list[Motif]:
    """List of sample motifs."""
    return [
        Motif.from_string("GAC&GC", "(.(&))"),
        Motif.from_string("AAG&CUU", "(.(&.))"),
        Motif.from_string("UUG&CAA", "(.(&.))"),
    ]


@pytest.fixture
def sample_helix() -> Helix:
    """A simple 3bp helix."""
    return Helix(
        strand1="AGC",
        strand2="GCU",
        structure1="(((",
        structure2=")))",
    )


@pytest.fixture
def sample_hairpin() -> Hairpin:
    """A simple 4nt hairpin loop."""
    return Hairpin(sequence="GAAA", structure="....")


@pytest.fixture
def sample_config() -> LibraryConfig:
    """Sample library configuration."""
    return LibraryConfig(
        target_length_min=100,
        target_length_max=120,
        motifs_per_construct_min=3,
        motifs_per_construct_max=4,
        p5_sequence="GGAAC",
        p5_structure="(((..",
        p3_sequence="GUUCC",
        p3_structure="..)))",
        helix_length=3,
        hairpin_loop_length=4,
        validation=ValidationConfig(
            enabled=True,
            max_ensemble_defect=5.0,
            allow_structure_differences=True,
            min_structure_match=0.8,
        ),
        optimization=OptimizationConfig(
            iterations=1000,
            initial_temperature=10.0,
            cooling_rate=0.001,
            target_library_size=100,
        ),
    )


@pytest.fixture
def sample_construct(sample_motif: Motif) -> Construct:
    """A simple construct with one motif."""
    return Construct(
        sequence="GGAACAGCGACGCUGCUGUUCCGAAA",
        structure="(((..(((.(.().))).))..)))",
        motifs=[sample_motif],
    )


@pytest.fixture
def config_yaml_content() -> str:
    """YAML content for config file."""
    return """
target_length:
  min: 100
  max: 120

motifs_per_construct:
  min: 3
  max: 4

p5_sequence: "GGAAC"
p5_structure: "(((.."

p3_sequence: "GUUCC"
p3_structure: "..)))"

helix_length: 3
hairpin_loop_length: 4

validation:
  enabled: true
  max_ensemble_defect: 5.0
  allow_structure_differences: true
  min_structure_match: 0.8

optimization:
  iterations: 1000
  initial_temperature: 10.0
  cooling_rate: 0.001
  target_library_size: 100
"""


@pytest.fixture
def motifs_csv_content() -> str:
    """CSV content for motifs file."""
    return """sequence,structure
GAC&GC,(.(&))
AAG&CUU,(.(&.))
UUG&CAA,(.(&.))
"""


@pytest.fixture
def temp_config_file(config_yaml_content: str):
    """Temporary config YAML file."""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml_content)
        f.flush()
        yield Path(f.name)


@pytest.fixture
def temp_motifs_file(motifs_csv_content: str):
    """Temporary motifs CSV file."""
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(motifs_csv_content)
        f.flush()
        yield Path(f.name)


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
