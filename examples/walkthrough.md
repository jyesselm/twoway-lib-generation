# Walkthrough: Generating a Two-Way Junction Library

This walkthrough uses the example files in this directory to generate a library
of RNA constructs, each containing 7 two-way junction motifs embedded in a
hairpin scaffold. All constructs are exactly 149 nucleotides.

## Prerequisites

Install the package:

```bash
pip install -e .
```

You also need ViennaRNA installed for RNA folding predictions.

## Files

- `config.yaml` -- Library generation settings (lengths, helices, validation)
- `sampled.csv` -- Two-way junction motifs from crystal structures (8 motifs)

## Step 1: Inspect the motifs

The motifs file has 8 two-way junctions. The `&` separates strand 1 (5' arm)
from strand 2 (3' arm):

```
sequence,structure,count,pdbs
CG&CAG,((&).),11,...
CAG&CG,(.(&)),11,...
CUACC&GAUG,(...(&)..),13,...
GAUG&CUACC,(..(&)...),13,...
UAA&UG,(.(&)),1,...
UG&UAA,((&).),1,...
AC&GAU,((&).),2,...
GAU&AC,(.(&)),2,...
```

## Step 2: Test which motifs fold correctly

Before generating, test each motif by embedding it in random helix contexts
and checking if ViennaRNA predicts the designed structure:

```bash
twoway-lib test-motifs sampled.csv
```

This will show PASS/FAIL for each motif. Two motifs (`UAA&UG` and `UG&UAA`)
fail because ViennaRNA mispredicts their structure. These are automatically
filtered out during generation.

You can save the results for later reuse:

```bash
twoway-lib test-motifs sampled.csv --save-results motif_results.json
```

## Step 3: Check configuration feasibility

Verify that your config can produce constructs at the target length:

```bash
twoway-lib check config.yaml sampled.csv
```

Expected output:

```
Configuration summary:
  Target length: 149-149 nt
  Motifs per construct: 7-7
  Helix length: 3-5 bp
  Hairpin loop length: 6 nt
  5' sequence length: 12 nt
  3' sequence length: 20 nt

Estimated feasible length range:
  Minimum: 121 nt
  Maximum: 181 nt
  Target:  149-149 nt

Configuration appears feasible.
```

## Step 4: Generate the library

Generate 100 candidate constructs (use more for a real library):

```bash
twoway-lib generate config.yaml sampled.csv \
    -o library.json \
    -n 100 \
    -s 42
```

Expected output:

```
Filtered motifs by fold test    passing=6 excluded=2
Generated valid candidates      count=100
Saved constructs                count=100 path=library.json

Generation complete!
  Total constructs: 100
  Length range: 149-149 nt
  Average length: 149.0 nt
  Unique motifs used: 6
  Motif usage range: 113-126 (avg: 116.7)
  Average edit distance: 49.0
```

Key things to note:
- **All constructs are exactly 149 nt** -- the length solver computes exact
  helix sizes (3-5 bp each) so every motif combination hits the target
- **6 of 8 motifs used** -- 2 were filtered for bad folding
- **Edit distance ~49** -- since all sequences are the same length, this is
  equivalent to Hamming distance (49 positions differ on average between the
  closest pair for each construct)

## Step 5: View the library summary

```bash
twoway-lib summary library.json
```

## Step 6: Generate a larger library with optimization

For a real experiment, generate many candidates and let simulated annealing
select a diverse subset:

```bash
twoway-lib generate config.yaml sampled.csv \
    -o library_large.json \
    -n 50000 \
    -s 42 \
    --detailed-summary summary.json
```

The optimizer selects `target_library_size` constructs (3000 by default) from
the candidate pool, maximizing pairwise edit distance while balancing motif
usage.

The `--detailed-summary` flag saves a JSON with per-motif usage stats and
diversity metrics.

## Understanding the output

Each construct in `library.json` contains:

```json
{
  "index": 0,
  "sequence": "GGGCGAAAGCCCCGAC...",
  "structure": "((((....))))(((((...",
  "length": 149,
  "motifs": [
    {
      "sequence": "CAG&CG",
      "structure": "(.(&))",
      "positions": {
        "strand1": [16, 17, 18],
        "strand2": [123, 124]
      }
    },
    ...
  ]
}
```

The construct layout is:

```
P5 (12nt)  H  M1_s1  H  M2_s1  ...  H  M7_s1  H  hairpin  H  M7_s2  H  ...  M1_s2  H  P3 (20nt)
```

Where:
- **P5/P3** -- fixed primer sequences (12 and 20 nt)
- **H** -- random helices (3-5 bp each, contributing 2x nt to total length)
- **M_s1/M_s2** -- motif strand 1 (5' arm) and strand 2 (3' arm)
- **hairpin** -- CUUCGG loop (6 nt)

## Configuration reference

Key settings in `config.yaml`:

| Setting | Value | Why |
|---|---|---|
| `target_length: 149-149` | Fixed length | All constructs identical length |
| `motifs_per_construct: 7-7` | Fixed count | 7 motifs maximizes feasible combos at 149 nt |
| `helix_length: 3` | Default | Base helix size |
| `helix_length_min/max: 3-5` | Variable | Lets solver adjust helices to hit exact target |
| `hairpin_sequence: CUUCGG` | Fixed loop | Known stable tetraloop with closing pair |
| `allow_wobble_pairs: true` | GU pairs | More helix diversity |
| `max_ensemble_defect: 10.0` | ViennaRNA | Reject poorly-folding constructs |
| `min_structure_match: 0.8` | Tolerance | Allow minor folding differences |
| `max_consecutive_nucleotides: 5` | Synthesis | Avoid homopolymer runs |
| `max_consecutive_gc_pairs: 3` | Structure | Avoid overly stable helix regions |

## Tips

- **Seed** (`-s 42`): Use a fixed seed for reproducibility. Different seeds
  give different libraries.
- **Parallel generation** (`--parallel --workers 4`): Speeds up candidate
  generation for large runs.
- **Pre-computed motif results** (`--load-motif-results`): Skip fold testing
  on re-runs by loading saved results.
- **Choosing target length**: With motifs of total length 5 and 9, only odd
  target lengths work with 7 motifs (even targets need 6 motifs). The length
  solver needs variable helices to have flexibility -- fixed helix sizes
  severely limit which motif combinations are feasible.
