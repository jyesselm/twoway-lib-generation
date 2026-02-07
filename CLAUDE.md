## Project Overview

This project is to generate rna libraries composed of motifs at the correct number of uses with all constructs being the same size and as distinct as possible from each other.

## Build & Test Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Run single test file
pytest tests/test_example.py -v

# Run single test
pytest tests/test_example.py::test_function_name -v

# Linting and formatting
ruff check .
ruff format .

# Type checking
mypy src/
```
### Indentation & Nesting
- Maximum 3 levels of indentation
- Use early returns to reduce nesting
- Extract nested logic into helper functions

### Function Design
- Target: 30 lines or fewer per function
- One responsibility per function
- Avoid 1-2 line functions unless called frequently
- Prefer simple, readable code over clever solutions

### Module Organization
- Target: 200-300 lines per module
- Prefer many small, focused modules over large files
- Group related functionality into coherent modules

### Documentation
- All functions require docstrings (Google style preferred)
- All functions require type hints for parameters and return values
- Example:
```python
def process_data(items: list[str], limit: int = 10) -> dict[str, int]:
        """Process a list of items and return frequency counts.

            Args:
                items: List of string items to process.
                limit: Maximum number of results to return.

            Returns:
                Dictionary mapping items to their counts.
            """
    ```

### OOP Principles
- Follow SOLID principles
- Favor composition over inheritance
- Keep classes focused on a single responsibility
- Use abstract base classes to define interfaces
- Prefer dependency injection for testability
## Tooling Configuration

### Ruff (Linting & Formatting)
- Max complexity: 10
- Line length: 88 (default)
- Run `ruff check .` before committing
- Run `ruff format .` to auto-format

### Mypy (Type Checking)
- Use minimal strict settings
- Focus on public API type correctness
- Avoid excessive `# type: ignore` comments

### Pytest (Testing)
- Minimum coverage target: 90%
- Write unit tests for all public functions
- Use fixtures for shared test setup
- Use parametrize for testing multiple inputs
## Project Structure

```
src/
    package_name/
        __init__.py
        module.py
tests/
    test_module.py
    conftest.py
pyproject.toml
``````
```
```
"
```
```
```
