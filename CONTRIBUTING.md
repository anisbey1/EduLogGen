# Contributing to EduLogGen

Thank you for your interest in contributing to EduLogGen.

This document explains how to set up a development environment, the quality
gates every change must pass, and the expectations for pull requests.

## Code of conduct

Be respectful and constructive. Harassment or discrimination of any kind is
not tolerated.

## Development setup

1. Fork and clone the repository.
2. Create a virtual environment with Python 3.11+:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the package in editable mode with development extras:

   ```bash
   pip install -e ".[dev,docs]"
   pre-commit install
   ```

## Quality gates

Before opening a pull request, ensure the following commands succeed:

```bash
ruff check src tests
ruff format --check src tests
black --check src tests
mypy
pytest --cov=eduloggen
mkdocs build --strict
python -m build
```

Pre-commit runs a subset of these checks automatically on `git commit`.

## Coding standards

- Target Python 3.11+ only.
- Prefer explicit type annotations; MyPy runs in strict mode.
- Use Google-style docstrings for public modules, classes, and functions.
- Do not commit secrets, credentials, or private learner data.
- Keep changes focused; avoid unrelated refactors in feature pull requests.

## Semantic Versioning

EduLogGen follows [Semantic Versioning](https://semver.org/):

- **MAJOR** — incompatible API changes
- **MINOR** — backwards-compatible functionality
- **PATCH** — backwards-compatible bug fixes

Update `CHANGELOG.md` for user-visible changes.

## Pull requests

1. Create a topic branch from `main`.
2. Add tests for new behavior when applicable.
3. Update documentation when public APIs or workflows change.
4. Open a pull request with a clear summary and test plan.

## Reporting issues

Please include:

- EduLogGen version (`python -c "import eduloggen; print(eduloggen.__version__)"`)
- Python version and operating system
- Minimal reproduction steps
- Expected versus actual behavior

Do **not** attach real student or learner data to issues.
