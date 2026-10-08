# Changelog

All notable changes to EduLogGen are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `eduloggen.core` foundation package: exception hierarchy with machine-readable
  codes, schema/artifact/report version constants, default `event_type`
  vocabulary, immutable `RunContext`, and shared type aliases
- `eduloggen.models` canonical domain model: immutable, validated `LogRecord`,
  `Session`, and `Participant`; `Dataset` / `SyntheticDataset` snapshots with
  cross-record checks, content fingerprints, and manifests; `EventVocabulary`
  with preserve / map-to-other / reject policies; `Feature`
- `eduloggen.io` ingestion and corpus I/O: streaming CSV/TSV, JSON Lines, and
  Parquet readers and writers; `FieldMapping` with automatic casting,
  timestamp formats and timezone policy, defaults, salted ID pseudonymization,
  and metadata passthrough; `ingest()` with deduplication, sorting, strict
  mode, and a value-free data quality report; atomic corpus directory
  read/write with overwrite protection and fingerprint integrity checks
- `parquet` optional extra (`pyarrow`)
- `eduloggen.config`: strict, immutable `AppConfig` with sections from PRD §16.4;
  YAML/TOML/JSON loading; precedence CLI > `EDULOGGEN_*` env allowlist > file >
  defaults via `resolve_config`; config fingerprints; relative paths resolved
  against the config file
- `FieldMapping.from_file` for YAML/TOML/JSON mapping files
- Example experiment config and field mapping under `examples/configs/`
- `pyyaml` as the first runtime dependency
- `eduloggen.utils`: versioned, order-independent seed derivation
  (`derive_seed`, `make_rng`) and canonical JSON fingerprints
- `eduloggen.privacy`: seeded remapping of learner/session/event ids
  (default for generated data) and metadata key stripping
- Architecture test enforcing the package dependency matrix (SAD §8.2)
- `eduloggen.analysis`: `sessionize()` with explicit / idle-timeout /
  composite strategies and configurable tokenization; order-k transition
  counts with start padding and additive smoothing; n-grams and rare-n-gram
  detection; inter-event, sojourn, and duration statistics; session feature
  rows, learner profiles, corpus features, and a navigation graph;
  `analyze()` returning an `AnalysisResult` with JSON and Markdown exports
- `eduloggen.generators`: `BaseGenerator` contract (fit / generate / describe /
  save / load) with shared population modelling (sessions per learner,
  inter-session gaps, companion fields, courses) and seeded, id-remapped
  output; `markov` (order-k with backoff and additive smoothing; empirical,
  Poisson, or fixed lengths), `semi_markov` (per-token sojourn timing:
  empirical, lognormal, gamma, exponential), and `independent` baseline;
  generator registry; inspectable JSON model artifacts with integrity checks
- `GeneratorModel` in `eduloggen.models`
- `eduloggen.validation`: `validate()` comparing real and synthetic datasets
  with 11 built-in metrics — `event_type_tvd`, `activity_jsd`,
  `session_length_ks`, `session_duration_w1`, `interevent_time_ks`,
  `bigram_tvd`, `transition_jsd`, `topn_path_overlap`, and privacy indicators
  `exact_session_dup_rate`, `rare_ngram_replay_rate`, `nn_distance_p05`;
  thresholds with pass / fail / info / skip statuses; `ValidationReport` with
  JSON and Markdown output; metric registry for custom metrics
- `eduloggen.datasets.demo_dataset`: seeded, fully synthetic demo course corpus
- `eduloggen.benchmark`: protocol `session_fidelity_v1` with a learner-level
  holdout split, multi-seed runs, mean/std aggregation, fit/generate timings,
  a real-vs-real reference row, failure isolation per generator, and
  `BenchmarkReport` JSON/Markdown output; `run_benchmark` in the API
- CLI commands `benchmark` and `demo`
- CI smoke job running the full pipeline and benchmark on the demo corpus
- `eduloggen.api` workflow façade (`load_config`, `ingest`, `load_dataset`,
  `save_dataset`, `sessionize`, `analyze`, `fit_generator`, `generate`,
  `validate`), re-exported from `import eduloggen`
- CLI commands `ingest`, `sessionize`, `analyze`, `fit`, `generate`,
  `validate`, `info`, `plugins` with global `--config`, `--seed`, `-v`,
  `--quiet`, `--json-logs`, `--force`, `--run-id`; exit codes 0/1/2;
  `run_manifest.json` in every output directory; `python -m eduloggen`
- Documentation: privacy and responsible-use statement, end-to-end
  getting-started guide, API reference pages for every package

### Changed

- Atomic output-directory writing moved to `eduloggen.utils.fs` and shared by
  corpus and model artifacts
- SAD §8.2: `validation` may depend on `analysis` (documented in the matrix)
- Coverage gate raised from 80% to 95% (ADR-011)
- `eduloggen.cli` is now a package; the `eduloggen` console script entry
  point is unchanged

### Changed

- `eduloggen.ingestion` now re-exports the ingest API from `eduloggen.io`

### Planned

- Data ingestion APIs for educational interaction logs
- Session and statistical analysis modules
- Markov and Semi-Markov generators
- Validation and benchmarking framework
- Visualization utilities and expanded CLI

## [0.1.0] - 2026-07-29

### Added

- Initial production-quality package skeleton (`src` layout)
- Package metadata and build configuration via `pyproject.toml`
- Tooling: Ruff, Black, MyPy, PyTest, pre-commit, MkDocs
- GitHub Actions continuous integration workflow
- Documented subpackage namespaces for ingestion, analysis, generators,
  validation, visualization, and CLI
- MIT license and Semantic Versioning policy

[Unreleased]: https://github.com/anisbey1/EduLogGen/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/anisbey1/EduLogGen/releases/tag/v0.1.0
