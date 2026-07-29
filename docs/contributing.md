# Contributing

Development setup, coding standards, and pull-request expectations are documented
in the repository root file
[`CONTRIBUTING.md`](https://github.com/anisbey1/EduLogGen/blob/main/CONTRIBUTING.md).

## Local checklist

```bash
pip install -e ".[dev,docs]"
pre-commit install
ruff check src tests
black --check src tests
mypy
pytest --cov=eduloggen
mkdocs build --strict
```

## Documentation contributions

- Prefer clear, concise language aimed at researchers and engineers.
- Keep examples free of real learner identifiers or private institutional data.
- Update the changelog for user-visible changes.
