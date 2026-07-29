# Getting started

This guide installs EduLogGen from a local clone and verifies that the package
and tooling work on your machine.

## Prerequisites

- Python 3.11 or newer
- A virtual environment tool (`venv`, `virtualenv`, or equivalent)
- Git

## Install

```bash
git clone https://github.com/anisbey1/EduLogGen.git
cd EduLogGen
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev,docs]"
```

## Verify the installation

```bash
python -c "import eduloggen; print(eduloggen.__version__)"
eduloggen --help
pytest
```

You should see a semantic version string (for example `0.1.0`), CLI help text,
and a passing test suite.

The version string is defined in `src/eduloggen/__version__.py` and re-exported
from the package root as `eduloggen.__version__`.

## Development tools

| Tool | Role |
| ---- | ---- |
| Ruff | Linting and fast formatting checks |
| Black | Canonical code formatting |
| MyPy | Strict static type checking |
| PyTest | Unit tests and coverage |
| Pre-commit | Git hooks for local quality gates |
| MkDocs | Documentation site |

Enable hooks once:

```bash
pre-commit install
```

Build and preview docs:

```bash
mkdocs serve
```

## Package layout

EduLogGen uses a `src` layout:

```text
src/eduloggen/
  ingestion/       # planned: load and normalize logs
  analysis/        # planned: session and statistical analysis
  generators/      # planned: synthetic generators
  validation/      # planned: quality and privacy validation
  visualization/   # planned: plotting helpers
  cli.py           # console entry point
```

Business logic for these modules is intentionally absent in `0.1.0`. The
namespaces exist so contributors can land features without reshaping the
package later.
