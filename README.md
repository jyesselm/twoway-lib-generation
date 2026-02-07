# twoway-lib

Generate two-way junction RNA hairpin libraries with maximized sequence diversity.

Constructs are assembled from structural motifs embedded between random helices in a hairpin scaffold, then validated by RNA secondary structure prediction and optimized for sequence diversity using simulated annealing.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.11, ViennaRNA (for RNA folding predictions)

## Quick Start

```bash
# Generate a default config file
twoway-lib config -o my_config.yaml

# Check your configuration and feasible length ranges
twoway-lib check my_config.yaml motifs.csv

# Test which motifs fold correctly
twoway-lib test-motifs motifs.csv

# Generate a library
twoway-lib generate my_config.yaml motifs.csv -o library.json -n 50000 -s 42
```

## How It Works

The pipeline has four stages:

1. **Motif preprocessing** -- Each motif is tested in random helix contexts using ViennaRNA to verify it folds correctly. Motifs that fail are filtered out.

2. **Candidate generation** -- Constructs are assembled by randomly selecting motifs and computing exact helix lengths via a length solver (no trial-and-error rejection). Each construct follows the layout:

   ```
   P5 -- [spacer] -- H -- M1_s1 -- H -- M2_s1 -- ... -- H -- hairpin -- H -- ... -- M2_s2 -- H -- M1_s2 -- H -- [spacer] -- P3
   ```

3. **Validation** -- Each candidate is folded with ViennaRNA. Constructs are checked for ensemble defect, structure match, consecutive nucleotide runs, and consecutive GC base pairs.

4. **Optimization** -- Simulated annealing selects a diverse subset from the candidate pool, maximizing pairwise edit distance while balancing motif usage across the library.

## CLI Commands

| Command | Description |
|---|---|
| `generate` | Generate a library from config and motifs |
| `check` | Validate config and estimate feasible construct lengths |
| `test-motifs` | Test each motif for correct folding in helix contexts |
| `config` | Generate a default config file or validate an existing one |
| `summary` | Display summary statistics for a generated library |
| `primers` | List available 5' and 3' primer sequences |

### Generate Options

```bash
twoway-lib generate config.yaml motifs.csv [OPTIONS]
```

| Option | Description |
|---|---|
| `-o, --output PATH` | Output JSON file path |
| `-n, --num-candidates INT` | Number of candidates to generate |
| `-s, --seed INT` | Random seed for reproducibility |
| `--parallel` | Enable parallel candidate generation |
| `--workers INT` | Number of parallel workers (default: 4) |
| `--save-motif-results PATH` | Save motif preprocessing results to JSON |
| `--load-motif-results PATH` | Load pre-computed motif results (skip preprocessing) |
| `--detailed-summary PATH` | Save detailed summary with per-motif usage stats |
| `--auto-tune` | Auto-tune simulated annealing parameters |
| `--no-filter-motifs` | Use all motifs without fold testing |
| `-v, --verbose` | Enable verbose logging |

## Python API

```python
from twoway_lib import LibraryConfig, load_config, load_motifs, LibraryGenerator

# Load configuration and motifs
config = load_config("config.yaml")
motifs = load_motifs("motifs.csv")

# Generate library
generator = LibraryGenerator(config, motifs, seed=42)
constructs = generator.generate(num_candidates=50000)

# Save results
from twoway_lib.io import save_library_json
save_library_json(constructs, "library.json")
```

### Motif Preprocessing

```python
from twoway_lib import preprocess_motifs, MotifTestResult
from twoway_lib.preprocessing import save_motif_results, load_motif_results

# Test motifs and separate passing/failing
passing, failing, results = preprocess_motifs(motifs, helix_length=3, seed=42)

# Save/load results to skip preprocessing on re-runs
save_motif_results(results, "motif_results.json")
results = load_motif_results("motif_results.json")
```

### Length Solver

The length solver computes exact helix lengths to hit a target construct length, eliminating trial-and-error assembly.

```python
from twoway_lib import compute_helix_budget, random_helix_assignment
from random import Random

# Compute total helix base pairs needed
budget = compute_helix_budget(
    target_length=136, motif_lengths=[5, 5, 5, 5, 5, 5],
    p5_len=12, p3_len=19, hairpin_len=6,
)

# Distribute budget across individual helices
assignment = random_helix_assignment(budget, num_helices=7, min_length=2, max_length=5, rng=Random(42))
# e.g. (3, 4, 3, 3, 4, 3, 3)
```

## Configuration

See [`examples/config.yaml`](examples/config.yaml) for a complete example. Key sections:

### Target Length & Motif Count

```yaml
target_length:
  min: 134
  max: 138

motifs_per_construct:
  min: 6
  max: 7
```

### Primer Sequences

```yaml
# Specify directly
p5_sequence: "GGGCGAAAGCCC"
p5_structure: "((((....))))"
p3_sequence: "AAAGAAACAACAACAACAAC"
p3_structure: "...................."

# Or reference by name from seq_tools resources
# p5_name: "cloud_lab_remake_rev_seq"
# p3_name: "tail"
```

### Helix Configuration

```yaml
helix_length: 3                # Default/fixed helix length
allow_wobble_pairs: true       # Allow G-U/U-G wobble pairs

# Variable helix lengths (optional)
# helix_length_min: 2
# helix_length_max: 4

# Require at least one GU pair in longer helices (optional)
# gu_required_above_length: 4
```

### Hairpin

```yaml
# Fixed hairpin (optional -- random 4nt loops if not specified)
hairpin_sequence: "CUUCGG"
hairpin_structure: "(....)"
```

### Spacers (Optional)

```yaml
# Linker sequences between primers and first/last helix
# spacer_5p_sequence: "AA"
# spacer_5p_structure: ".."
# spacer_3p_sequence: "AA"
# spacer_3p_structure: ".."
```

### Validation

```yaml
validation:
  enabled: true
  max_ensemble_defect: 10.0
  allow_structure_differences: true
  min_structure_match: 0.8
  avoid_consecutive_nucleotides: true
  max_consecutive_nucleotides: 5
  avoid_consecutive_gc_pairs: true
  max_consecutive_gc_pairs: 3
```

### Optimization

```yaml
optimization:
  iterations: 10000
  initial_temperature: 10.0
  cooling_rate: 0.00001
  target_library_size: 3000
  # target_motif_usage: 50   # Desired usage count per motif (optional)
```

## Motif Format

Motifs are provided as a CSV file. The `sequence` and `structure` columns are required; `count` and `pdbs` are optional metadata.

```csv
sequence,structure
GAC&GC,(.(&))
AAG&CUU,(.(&.))
CUACC&GAUG,(...(&)..)
```

The `&` separates strand 1 (5' arm) from strand 2 (3' arm). Structure uses dot-bracket notation where `(` pairs with `)` across strands and `.` is unpaired.

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Lint and format
ruff check .
ruff format .

# Type check
mypy src/
```

## License

MIT
