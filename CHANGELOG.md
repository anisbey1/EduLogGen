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
