# EduLogGen

[![PyPI](https://img.shields.io/pypi/v/eduloggen)](https://pypi.org/project/eduloggen/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23264161.svg)](https://doi.org/10.5281/zenodo.23264161)

Open-source Python framework for **analyzing**, **generating**, **validating**,
and **benchmarking** synthetic educational interaction logs.

EduLogGen helps researchers, universities, and education-technology teams create
privacy-preserving synthetic datasets that reproduce the statistical and
behavioral properties of real learner interaction logs.

> **Status:** 1.4.2 — the Level 1 log generator is complete: ingestion,
> sessionization, analysis, Markov and Semi-Markov generation, validation,
> benchmarking, visualization, and plugins, from Python and the CLI.
> Level 2, the experimental generator, is under way: labelled anomalies,
> detector scoring, behavioural controls, course calendars, manipulation
> checks, and behavioural profiles are available; outcomes follow
> ([design](https://github.com/anisbey1/EduLogGen/blob/main/docs/04_EXPERIMENTAL_GENERATOR.md)).

## Features

- **Ingest** CSV, TSV, JSON Lines, and Parquet logs through a declarative
  field mapping, with timezone handling, salted ID pseudonymization, and a
  value-free data quality report
- **Sessionize** by explicit session IDs, idle timeout, or both
- **Analyze** event mixes, n-grams, transition structure, timing, and
  navigation graphs; export JSON and Markdown summaries
- **Drill down** per activity, over time (hours, weekdays, weeks, deadlines),
  and per group (course, week, cohort, device…); see exactly where synthetic
  data differs from real data
- **Generate** with `markov` (order-k), `semi_markov` (per-activity timing),
  or the `independent` baseline; seeded, reproducible, with remapped IDs
- **Validate** real versus synthetic data on marginal, structural, temporal,
  sequential, and privacy metrics, with pass/fail thresholds for CI
- **Benchmark** generators fairly with protocol `session_fidelity_v1`
  (learner-level holdout, multiple seeds, real-data reference scores)
- **Plot** real-versus-synthetic distributions, transition heatmaps and
  graphs, Sankey pathway diagrams, session timelines, validation summaries, and benchmark dashboards (PNG/SVG/PDF)
- **Benchmark anomaly detectors** (Level 2): inject five labelled anomaly
  types, keep ground truth in a separate `annotations.csv`, and score
  detectors with `eduloggen evaluate`
- **Run controlled experiments** (Level 2): reweight events, scale time on
  task and session lengths, schedule sessions with a course calendar and
  deadline surges, and get a manipulation check proving each effect
- **Extend** with installable plugins for generators, metrics, readers,
  plots, and benchmark protocols (see [`docs/plugins.md`](https://github.com/anisbey1/EduLogGen/blob/main/docs/plugins.md))
- **Try it instantly** with `eduloggen demo`, a fully synthetic course corpus
- **Configure** runs in YAML/TOML/JSON with documented precedence
  (CLI > `EDULOGGEN_*` environment > file > defaults)

## Case study

On seven modules of the Open University Learning Analytics Dataset (10,143
learners), Markov generators cut the sequence error of an order-blind
baseline by two thirds, the semi-Markov generator also reproduces timing, and
automatic profiles separate low-engagement from successful learners; the
whole study runs in ten minutes on a laptop. See the
[case study](https://anisbey1.github.io/EduLogGen/case-study-oulad/) and
[`studies/oulad/`](https://github.com/anisbey1/EduLogGen/tree/main/studies/oulad).

## Requirements

- Python 3.11 or newer

## Installation

```bash
pip install eduloggen              # core (CSV, TSV, JSON Lines)
pip install "eduloggen[parquet,viz]"  # with Parquet and figures
```

For development, from a clone: `pip install -e ".[dev,docs]"`.

Optional extras:

| Extra | Purpose |
| ----- | ------- |
| `parquet` | Parquet reading and writing (`pyarrow`) |
| `viz` | Figures (`matplotlib`) |
| `dev` | Ruff, Black, MyPy, PyTest, pre-commit (includes `parquet`, `viz`) |
| `docs` | MkDocs and Material theme |
| `all` | Development and documentation tools |

## Quick start

Try it on synthetic demo data:

```bash
eduloggen demo --output demo/
eduloggen benchmark --input demo/
```

On your own data:

```bash
eduloggen ingest --input events.csv --mapping mapping.yaml --output corpus/
eduloggen sessionize --input corpus/ --output sessions/
eduloggen fit --input sessions/ --generator semi_markov --output model/
eduloggen generate --model model/ --n-sessions 1000 --seed 42 --output synthetic/
eduloggen validate --real sessions/ --synthetic synthetic/ \
    --threshold event_type_tvd=0.1 --output report/
eduloggen plot --real sessions/ --synthetic synthetic/ --output figures/
```

`validate` exits with `1` when a threshold fails, so it can gate CI jobs. See
[`examples/configs/`](https://github.com/anisbey1/EduLogGen/tree/main/examples/configs) for a field mapping and a complete
experiment configuration.

From Python:

```python
import eduloggen as elg

real = elg.sessionize(elg.ingest("events.csv", "mapping.yaml").dataset)
model = elg.fit_generator("semi_markov", real, hyperparameters={"order": 2})
synthetic = elg.generate(model, n_sessions=1000, seed=42)
report = elg.validate(real, synthetic, thresholds={"event_type_tvd": 0.1})
print(report.to_markdown())
```

## Privacy

EduLogGen runs locally and never transmits data. Synthetic data reduces but
does not remove re-identification risk: always review the privacy indicators
in the validation report before sharing outputs. See
[`docs/privacy.md`](https://github.com/anisbey1/EduLogGen/blob/main/docs/privacy.md).

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
  api.py                # Workflow façade re-exported as `import eduloggen`
  core/                 # Exceptions, constants, RunContext
  models/               # Canonical events, sessions, datasets, model artifacts
  config/               # Typed configuration and precedence rules
  io/                   # Readers, writers, field mapping, ingest, corpora
  analysis/             # Sessionization and behavioural statistics
  generators/           # Markov, Semi-Markov, baseline; registry; artifacts
  validation/           # Metrics, thresholds, reports
  benchmark/            # Protocols, runner, comparison reports
  datasets/             # Synthetic demo corpus
  plugins/              # Entry-point discovery and registration
  scenarios/            # Level 2: anomaly injection (more to come)
  evaluation/           # Level 2: scoring against ground truth
  privacy/              # ID remapping and metadata stripping
  utils/                # Seeding, hashing, atomic file output
  cli/                  # `eduloggen` command line
  visualization/        # Figures for datasets and reports
tests/                  # PyTest suite
docs/                   # MkDocs sources
examples/               # Example configs (synthetic data only)
notebooks/              # Exploratory notebooks (planned)
```

## Citation

If you use EduLogGen in research, please cite it. GitHub shows a
"Cite this repository" button generated from
[`CITATION.cff`](https://github.com/anisbey1/EduLogGen/blob/main/CITATION.cff);
each release is archived on Zenodo
([doi:10.5281/zenodo.23264161](https://doi.org/10.5281/zenodo.23264161),
which resolves to the latest version).

## Versioning

EduLogGen follows [Semantic Versioning](https://semver.org/). See
[`CHANGELOG.md`](https://github.com/anisbey1/EduLogGen/blob/main/CHANGELOG.md) for release notes.

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](https://github.com/anisbey1/EduLogGen/blob/main/CONTRIBUTING.md)
before opening a pull request.

## License

This project is licensed under the [MIT License](https://github.com/anisbey1/EduLogGen/blob/main/LICENSE).
