# Changelog

All notable changes to EduLogGen are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Fine-grained analysis
- Per-activity profiles (`activity_profiles`): share, session and learner
  reach, start/end shares, repeat share, dwell-time distribution,
  predecessors and successors, success rate, mean score, companions
- Temporal analysis (`temporal_profile`, `deadline_effects`): timezone-aware
  activity by hour and weekday, weekday x hour table, weekly trends from the
  first session, daily counts, gaps between a learner's sessions, activity
  ratio before deadlines
- Stratified analysis (FR-A.7; `analyze_by`, `split_by`, `session_groups`):
  by `course`, `week`, `weekday`, `hour`, `learner_group`, or
  `metadata:<key>`, with each group's mix compared to the overall mix
- Fine-grained validation (`compare_detailed`): ranked per-token,
  per-transition (including invented and never-produced transitions),
  per-length, per-hour, per-weekday, and per-group differences
- CLI: `analyze --detail --by --groups --timezone --deadlines`,
  `validate --detailed --by --groups --timezone`; plot `activity_heatmap`

## [1.2.0] - 2026-10-09

Level 2, milestone M2: run controlled experiments and prove each control
took effect.

### Added

- Level 2, milestone M2: behavioural controls, course calendars, and
  manipulation checks
- `eduloggen.scenarios.Controls` / `apply_controls`: `event_weights`
  (reweight transitions into a token; `0` removes it), `dwell_scale`,
  `session_length` (scale or fixed), `sessions_per_learner` (mean or fixed);
  controls produce a new model whose parameters record them
- `eduloggen.generators.SessionCalendar`: period, timezone-aware hour and
  weekday weights, deadline surges; `generate(..., calendar=...)` for every
  generator; a learner's sessions never overlap and never start in
  zero-weight hours
- `manipulation_check`: compares a controlled sample with an uncontrolled
  baseline (same model, seed, size) for every control, the calendar, and
  anomalies; sample-size-aware tolerances for distributional checks
- `ExperimentSettings` / `run_experiment` and CLI
  `generate --experiment FILE` (controls, calendar, anomalies), writing
  `manipulation_check.json` / `.md`
- Generator capability tag `supports_event_weights`
- Example `examples/configs/experiment.yaml`; experiment guide

## [1.1.0] - 2026-10-09

Level 2, milestone M1: benchmark anomaly detectors on synthetic data with
labelled ground truth.

### Added

- Level 2, milestone M1 (`docs/04_EXPERIMENTAL_GENERATOR.md`): labelled
  anomaly injection for benchmarking detectors
- `eduloggen.scenarios`: five anomaly injectors — `event_frequency`,
  `abnormal_timing`, `repetition`, `inactivity` (unusual but valid) and
  `unexpected_transition` (invalid workflow); `inject_anomalies` with seeded,
  disjoint session selection, one anomaly per session, no overlapping
  sessions, and full id remapping so ids never reveal injected records;
  `register_anomaly` for custom injectors
- `Annotations` ground-truth table in `eduloggen.models`, stored as
  `annotations.csv` in corpora with its own fingerprint (`read_annotations`)
- `eduloggen.evaluation.evaluate_detection`: precision, recall, F1, false
  positive rate, ROC-AUC, average precision, and recall per anomaly type and
  per category
- CLI: `generate --anomalies FILE` and `evaluate --corpus --predictions`
- `privacy.remap_ids_with_mapping` returning the old-to-new id mapping
- Example `examples/configs/anomalies.yaml`; detector-testing guide

## [1.0.0] - 2026-10-09

First stable release: the complete Level 1 log generator. Learn from real
educational interaction logs, generate privacy-conscious synthetic sessions,
validate them, compare generators, and plot the results, from Python or the
`eduloggen` command line.

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
- `eduloggen.visualization` (optional `viz` extra): event/activity frequency,
  session length and duration, inter-event time (histogram and ECDF), and
  transition heatmap comparisons; sampled session timelines; validation and
  benchmark charts; colour-blind-safe palette; reproducible PNG/SVG/PDF export;
  `plot_datasets` for the standard set; CLI `plot` command
- `eduloggen.plugins`: entry-point discovery (`eduloggen.plugins.<kind>`) for
  generators, metrics, readers, visualizers, and benchmark suites; contract
  checks before registration; built-ins protected; broken plugins reported
  without crashing (`strict=True` to fail); `register_plugin`,
  `unregister_plugin`, `list_plugins`; CLI discovers at startup, `plugins`
  shows sources, versions, and errors; run manifests record plugin versions
- Registration hooks: `register_reader` (with file suffixes),
  `register_plot`, `register_protocol`; `BUILTIN_*` name sets
- Plugin author guide (`docs/plugins.md`)
- Node-link transition graph (`plot_transition_graph`) and multi-step Sankey
  pathway diagram (`plot_sankey`, `sankey_flows`), matplotlib only
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
- `io.format` and `ingest(format=...)` accept plugin reader names
- `eduloggen.cli` is now a package; the `eduloggen` console script entry
  point is unchanged
- `eduloggen.ingestion` now re-exports the ingest API from `eduloggen.io`

## [0.1.0] - 2026-07-29

### Added

- Initial production-quality package skeleton (`src` layout)
- Package metadata and build configuration via `pyproject.toml`
- Tooling: Ruff, Black, MyPy, PyTest, pre-commit, MkDocs
- GitHub Actions continuous integration workflow
- Documented subpackage namespaces for ingestion, analysis, generators,
  validation, visualization, and CLI
- MIT license and Semantic Versioning policy

[Unreleased]: https://github.com/anisbey1/EduLogGen/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/anisbey1/EduLogGen/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/anisbey1/EduLogGen/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/anisbey1/EduLogGen/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/anisbey1/EduLogGen/releases/tag/v0.1.0
