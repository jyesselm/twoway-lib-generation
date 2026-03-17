# Generating a Library from Curated PDB Motifs

This example generates an RNA library using 351 curated two-way junction motifs
extracted from PDB crystal structures. Because the crystal structures often
disagree with ViennaRNA's predictions, we first predict what ViennaRNA actually
thinks each motif's structure is, then use those predicted structures for
library generation.

## The Problem

Motifs from crystal structures have experimentally determined secondary
structures. However, ViennaRNA often predicts a different structure for the
same sequence. If we use the crystal structures directly, most constructs fail
validation because ViennaRNA disagrees. Instead, we:

1. Embed each motif in random helix contexts and fold with ViennaRNA
2. Use the **predicted** structure (what ViennaRNA thinks) instead of the
   crystal structure
3. Filter out motifs that ViennaRNA can't model properly

## Files

- `config.yaml` -- Library generation settings
- `predicted_motifs.csv` -- Pre-computed predicted structures (320 motifs)
- `../../data/curated_twoway.json` -- Source curated motifs (351 entries)

## Step-by-Step Workflow

### Step 1: Predict ViennaRNA structures

This step embeds each motif in 50 random helix contexts, folds with ViennaRNA,
and takes the most common predicted structure as the consensus.

```bash
twoway-lib predict-structures data/curated_twoway.json \
    -o examples/curated_library/predicted_motifs.csv \
    -c 50 -s 42
```

The command automatically removes motifs with bad predictions:

- **Pure helix** (20 removed): ViennaRNA predicts the motif region as fully
  base-paired with no internal loops or bulges. These motifs have no
  distinguishable structure from a regular helix.
- **Unpaired ends** (11 removed): ViennaRNA doesn't predict the flanking
  base pair that connects the motif to adjacent helices. These motifs won't
  embed properly in constructs.

The curated JSON uses `-` as the strand separator and includes flanking base
pairs from the crystal structure. The predict-structures command converts
these to the `&`-separated format used by the generator, keeping the flanking
pairs to better represent the 3D structure.

Result: 320 motifs with ViennaRNA-compatible predicted structures.

### Step 2: Check feasibility

```bash
twoway-lib check examples/curated_library/config.yaml \
    examples/curated_library/predicted_motifs.csv
```

These motifs include flanking base pairs, making them larger (avg 14 nt) than
motifs without flanking pairs (~5 nt). The config accounts for this with a
wider motifs-per-construct range (4-9) and a target length range (147-151 nt).

### Step 3: Generate the library

```bash
twoway-lib generate examples/curated_library/config.yaml \
    examples/curated_library/predicted_motifs.csv \
    -o library.json \
    -n 5000 \
    -s 42 \
    --no-filter-motifs \
    --parallel --workers 8
```

Key flags:
- `--no-filter-motifs` -- Skip fold testing since structures are already
  ViennaRNA-predicted
- `--parallel --workers 8` -- Use multiple processes for candidate generation
- `-n 5000` -- Generate 5000 candidates before selecting a diverse subset

### Step 4: Check results

```bash
twoway-lib summary library.json
```

## Configuration Notes

| Setting | Value | Why |
|---|---|---|
| `target_length: 147-151` | Near-fixed length | Small range for uniform constructs |
| `motifs_per_construct: 4-9` | Wide range | Accommodates variable motif sizes |
| `helix_length_min/max: 3-5` | Variable helices | Length solver adjusts to hit target |
| `allow_wobble_pairs: true` | GU pairs | More helix diversity |
| `max_ensemble_defect: 10.0` | Relaxed | Curated motifs may fold imperfectly |
| `min_structure_match: 0.8` | Tolerant | Allow minor prediction differences |
| `max_consecutive_gc_pairs: 3` | GC limit | Only flags GC runs outside motifs |
| `min_motif_usage: 3` | Floor | Each motif used at least 3 times |
| `target_library_size: 400` | Selection | Optimizer picks 400 from candidates |

The consecutive GC pair check is motif-aware: GC runs that fall entirely
within a motif's positions are allowed (they come from the crystal structure),
only GC runs extending into the randomly generated helix regions are flagged.

## Output Format

The output CSV (`predicted_motifs.csv`) contains:

| Column | Description |
|---|---|
| `sequence` | Motif sequence with `&` separator (includes flanking bp) |
| `structure` | ViennaRNA predicted structure |
| `pdb_structure` | Original crystal structure |
| `motif_id` | Unique identifier from PDB |
| `pdb_id` | Source PDB structure |
| `match_fraction` | Fraction of positions where prediction matches PDB |
| `consensus_agreement` | Fraction of contexts agreeing with consensus |
