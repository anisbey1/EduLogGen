# EduLogGen

Open-source Python framework for **analyzing**, **generating**, **validating**,
and **benchmarking** synthetic educational interaction logs.

EduLogGen helps researchers, universities, and education-technology teams create
privacy-preserving synthetic datasets that reproduce the statistical and
behavioral properties of real learner interaction logs.

> **Status:** early skeleton (v0.1.0). Public APIs for generators and validators
> are under active design. See [`docs/01_VISION.md`](docs/01_VISION.md).

## Features (roadmap)

Version 1.0 targets:

- Data ingestion for educational interaction logs
- Session and statistical analysis
- Markov and Semi-Markov generators
- Validation and benchmarking harnesses
- Visualization utilities
- Command-line interface

## Requirements

- Python 3.11 or newer

## Installation

```bash
# From a local clone (recommended while the package is in early development)
pip install -e ".[dev,docs]"
```

Optional extras:

| Extra | Purpose |
| ----- | ------- |
| `dev` | Ruff, Black, MyPy, PyTest, pre-commit |
| `docs` | MkDocs and Material theme |
| `all` | Development and documentation tools |

## Quick start

```bash
# Editable install (development)
pip install -e .

# Confirm the package is importable
python -c "import eduloggen; print(eduloggen.__version__)"

# Inspect the CLI
eduloggen --help
```

The public version lives in `src/eduloggen/__version__.py`.

## Development

```bash
# Install with development tools
pip install -e ".[dev,docs]"

# Enable git hooks
pre-commit install

# Run the quality suite
ruff check src tests
ruff format --check src tests
black --check src tests
mypy
pytest --cov=eduloggen

# Build documentation
mkdocs serve

# Build installable artifacts
python -m build
```

## Project layout

```text
src/eduloggen/          # Installable package (src layout)
  ingestion/            # Log loading and normalization (planned)
  analysis/             # Session and statistical analysis (planned)
  generators/           # Synthetic generators (planned)
  validation/           # Quality, similarity, privacy checks (planned)
  visualization/        # Plotting utilities (planned)
  cli.py                # Console entry point
tests/                  # PyTest suite
docs/                   # MkDocs sources
examples/               # Runnable examples (planned)
notebooks/              # Exploratory notebooks (planned)
```

## Versioning

EduLogGen follows [Semantic Versioning](https://semver.org/). See
[`CHANGELOG.md`](CHANGELOG.md) for release notes.

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md)
before opening a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
