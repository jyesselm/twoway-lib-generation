# twoway-lib

Generate two-way junction RNA hairpin libraries with maximized sequence diversity.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### Command Line

Generate a library:
```bash
twoway-lib generate config.yaml motifs.csv -o library.csv -n 50000
```

Check configuration:
```bash
twoway-lib check config.yaml motifs.csv
```

View library summary:
```bash
twoway-lib summary library.csv
```

### Python API

```python
from twoway_lib import LibraryConfig, load_config, load_motifs, LibraryGenerator

# Load configuration and motifs
config = load_config("config.yaml")
motifs = load_motifs("motifs.csv")

# Generate library
generator = LibraryGenerator(config, motifs, seed=42)
constructs = generator.generate(num_candidates=50000)
```

## Configuration

See `examples/config.yaml` for a complete example configuration file.

## Motif Format

Motifs are provided in CSV format with sequence and structure columns:
```csv
sequence,structure
GAC&GC,(.(&))
AAG&CUU,(.(&))
```

## License

MIT
