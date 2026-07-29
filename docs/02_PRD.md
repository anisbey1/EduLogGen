# EduLogGen Product Requirements Document

| Field | Value |
| ----- | ----- |
| **Document** | Product Requirements Document (PRD) |
| **Product** | EduLogGen |
| **Version** | 1.0 |
| **Status** | Draft |
| **Authors** | EduLogGen Contributors |
| **Related** | [`01_VISION.md`](01_VISION.md) |
| **Package version baseline** | 0.1.0 (skeleton); PRD targets **1.0.0** product scope |
| **License** | MIT |

---

## Table of contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Target Users](#3-target-users)
4. [User Personas](#4-user-personas)
5. [Functional Requirements](#5-functional-requirements)
6. [Non-functional Requirements](#6-non-functional-requirements)
7. [System Modules](#7-system-modules)
8. [Package Structure](#8-package-structure)
9. [Data Models](#9-data-models)
10. [Plugin Architecture](#10-plugin-architecture)
11. [Generator Framework](#11-generator-framework)
12. [Validation Framework](#12-validation-framework)
13. [Visualization Framework](#13-visualization-framework)
14. [CLI Specification](#14-cli-specification)
15. [Python API Specification](#15-python-api-specification)
16. [Configuration System](#16-configuration-system)
17. [Logging](#17-logging)
18. [Benchmark Framework](#18-benchmark-framework)
19. [Privacy Requirements](#19-privacy-requirements)
20. [Testing Strategy](#20-testing-strategy)
21. [Documentation Strategy](#21-documentation-strategy)
22. [Release Strategy](#22-release-strategy)
23. [Future Roadmap](#23-future-roadmap)
24. [Risks](#24-risks)
25. [Success Metrics](#25-success-metrics)

---

## 1. Executive Summary

EduLogGen is an open-source Python framework for **analyzing**, **generating**,
**validating**, and **benchmarking** synthetic educational interaction logs. It
addresses a structural barrier in Educational Data Mining (EDM) and Learning
Analytics (LA): real learner logs are privacy-sensitive and rarely shareable,
which blocks reproducible research and fair method comparison.

Unlike general-purpose synthetic-data toolkits aimed at tabular, medical, or
financial domains, EduLogGen treats educational logs as **sessionized event
sequences** with temporal dynamics, navigation structure, and educational
context. The product is a **modular platform**, not a single algorithm.
Statistical, probabilistic, and (in later releases) deep-learning generators
share one data model, one training/generation contract, and one evaluation
harness.

**Version 1.0** delivers:

- Ingestion and normalization of educational interaction logs
- Session and statistical analysis
- Markov and Semi-Markov generators
- A validation and privacy-oriented evaluation suite
- Visualization utilities
- A documented Python API and CLI
- A reproducible benchmark framework

**Long-term goal:** become the reference open-source framework for synthetic
educational interaction logs—analogous to the role Scikit-learn plays in
classical machine learning.

This PRD defines product scope, requirements, architecture boundaries, public
interfaces, quality bars, and success criteria **before implementation**. It is
the authoritative product contract for contributors and reviewers.

---

## 2. Problem Statement

### 2.1 Context

Educational institutions and platforms collect large volumes of interaction data
from Learning Management Systems (LMS), Intelligent Tutoring Systems (ITS),
MOOCs, mobile learning apps, assessment platforms, and educational games. These
logs support research on engagement, navigation, mastery, dropout, and adaptive
instruction.

### 2.2 Core problem

Real educational logs typically contain personally identifiable or sensitive
learner information. Institutions therefore restrict sharing. As a result:

1. **Reproducibility suffers** — published experiments cannot be re-run on the
   same data.
2. **Comparability suffers** — methods are evaluated on private corpora that
   others cannot access.
3. **Onboarding is slow** — graduate students and new labs lack safe datasets
   for prototyping.
4. **Privacy risk remains** — even “anonymized” releases can be re-identifiable
   when sequences and rare events are distinctive.

### 2.3 Gap in existing tools

Current synthetic-data generators primarily target:

- Tabular datasets
- Medical records
- Financial transactions
- Generic time-series

Educational interaction logs differ because they encode:

| Characteristic | Why it matters |
| -------------- | -------------- |
| Event sequences | Order of actions carries pedagogical meaning |
| Navigation behavior | Paths through content, assessments, forums |
| Temporal dynamics | Inter-event delays, session duration, time-of-day |
| Learning behavior | Attempts, hints, correctness, revisits |
| Session structure | Login-to-logout (or idle-bounded) episodes |
| Educational context | Course, activity type, item difficulty, modality |

Existing tools do not explicitly model this combination. Researchers therefore
either (a) cannot share data, (b) invent one-off simulators that are not
comparable, or (c) use generic generators that destroy educational structure.

### 2.4 Product response

EduLogGen must provide a **unified, extensible, privacy-aware** framework to:

1. Analyze real educational logs under local institutional control.
2. Learn behavioral and statistical patterns from those logs.
3. Generate synthetic sessions that preserve utility for research.
4. Validate statistical similarity, behavioral realism, and privacy risk.
5. Benchmark generators under standardized protocols.

---

## 3. Target Users

Primary audiences (aligned with the project vision):

| Audience | Typical need |
| -------- | ------------ |
| Educational Data Mining researchers | Comparable generators and published benchmarks |
| Learning Analytics researchers | Session-aware synthetic corpora for analytics pipelines |
| AI in Education researchers | Train/evaluate models without private learner data |
| Universities / research labs | Institutional synthetic-data capability with privacy review |
| EdTech companies | Safe internal datasets for QA, demos, and model development |
| Graduate students | Reproducible tutorials and starter datasets |

Secondary audiences:

- Open-source contributors extending generators or validators
- Privacy / compliance officers reviewing release risk
- Course instructors creating teaching materials for EDM/LA courses

Out of scope as primary users for v1.0: K–12 classroom teachers using the tool
day-to-day without research or engineering support; non-technical administrators
expecting a fully no-code SaaS product.

---

## 4. User Personas

### 4.1 Persona A — “Dr. Maya Chen” (EDM researcher)

- **Role:** Assistant professor in Educational Data Mining
- **Goals:** Publish generator comparisons; share synthetic corpora with papers
- **Pain points:** Cannot redistribute institutional LMS logs; reviewers ask for
  data availability statements she cannot satisfy
- **Usage:** Python API for analysis → fit Markov/Semi-Markov → validate →
  export synthetic CSV/Parquet → cite EduLogGen version and config
- **Success:** A paper supplement with synthetic data + benchmark table that
  other labs can reproduce

### 4.2 Persona B — “Luis Ortega” (Learning Analytics engineer)

- **Role:** Data engineer at a university LA team
- **Goals:** Build dashboards and models on realistic session structure without
  exposing students
- **Pain points:** De-identification pipelines are brittle; stakeholders fear
  residual disclosure
- **Usage:** CLI pipelines on secured servers; configuration files checked into
  internal Git; privacy reports for governance
- **Success:** Synthetic logs that preserve session-length and activity-mix
  distributions for staging environments

### 4.3 Persona C — “Aisha Rahman” (PhD student)

- **Role:** First-year PhD student in AI in Education
- **Goals:** Prototype sequence models quickly with trustworthy toy-to-realistic
  data
- **Pain points:** Steep setup; unclear evaluation; fear of mishandling real data
- **Usage:** Tutorials, notebooks, small public sample schemas, CLI
  `eduloggen generate` / `validate`
- **Success:** Completes a course assignment or pilot experiment without ever
  loading identifiable student records onto a laptop

### 4.4 Persona D — “Kenji Sato” (EdTech ML lead)

- **Role:** ML lead at an adaptive learning startup
- **Goals:** Stress-test recommenders and content navigators with diverse
  synthetic behaviors
- **Pain points:** Production logs cannot leave the VPC; vendors propose opaque
  black-box synthesizers
- **Usage:** Plugin API to register proprietary generators; standard validators;
  CI benchmark jobs
- **Success:** Nightly synthetic regression suite with fixed seeds and tracked
  metrics

### 4.5 Persona E — “Sofia Martins” (Open-source contributor)

- **Role:** Graduate researcher contributing a new generator
- **Goals:** Land a well-scoped PR with tests and docs
- **Pain points:** Unclear interfaces; inconsistent evaluation
- **Usage:** Plugin contracts, contribution guide, benchmark harness
- **Success:** New generator passes interface tests and appears in benchmark
  reports with documented hyperparameters

---

## 5. Functional Requirements

Requirements use IDs of the form `FR-X.Y`. Priority: **P0** (must for v1.0),
**P1** (should), **P2** (nice-to-have / post-1.0 if deferred).

### 5.1 Ingestion

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-I.1 | Load tabular interaction logs from CSV and Parquet | P0 |
| FR-I.2 | Map source columns to the canonical event schema via explicit field mapping | P0 |
| FR-I.3 | Validate required fields and types; emit actionable error reports | P0 |
| FR-I.4 | Support optional fields (grades, correctness, item IDs, device, etc.) | P0 |
| FR-I.5 | Deduplicate and sort events by learner and timestamp | P0 |
| FR-I.6 | Ingest JSON Lines event streams | P1 |
| FR-I.7 | Provide adapters/documentation for common LMS export shapes (generic, not vendor-locked) | P1 |
| FR-I.8 | Stream / chunk large files without requiring full in-memory load | P1 |

### 5.2 Sessionization and analysis

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-A.1 | Segment events into sessions using configurable idle timeout and/or explicit session IDs | P0 |
| FR-A.2 | Compute session-level features (length, duration, unique activities, event mix) | P0 |
| FR-A.3 | Compute corpus-level descriptive statistics (event frequencies, transition counts, inter-event time summaries) | P0 |
| FR-A.4 | Estimate n-gram / Markov transition structures from session sequences | P0 |
| FR-A.5 | Estimate sojourn / inter-event time distributions for Semi-Markov modeling | P0 |
| FR-A.6 | Export analysis summaries as structured reports (JSON/YAML/Markdown tables) | P0 |
| FR-A.7 | Support stratified analysis by course, activity type, or cohort label when present | P1 |

### 5.3 Generation

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-G.1 | Fit a first-order (and configurable-order) Markov generator on session sequences | P0 |
| FR-G.2 | Fit a Semi-Markov generator combining transitions with time distributions | P0 |
| FR-G.3 | Generate a configurable number of synthetic sessions with a fixed random seed | P0 |
| FR-G.4 | Persist and reload fitted generator artifacts | P0 |
| FR-G.5 | Export synthetic events in the canonical schema (CSV/Parquet) | P0 |
| FR-G.6 | Allow control of session-length distribution (empirical, parametric, or fixed) | P0 |
| FR-G.7 | Support start-state / absorbing-state configuration for session boundaries | P1 |
| FR-G.8 | Register third-party generators via the plugin API without core forks | P0 |

### 5.4 Validation

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-V.1 | Compare real vs synthetic marginal event distributions | P0 |
| FR-V.2 | Compare transition / bigram distributions | P0 |
| FR-V.3 | Compare session-length and session-duration distributions | P0 |
| FR-V.4 | Compare inter-event time distributions | P0 |
| FR-V.5 | Produce a machine-readable validation report and a human-readable summary | P0 |
| FR-V.6 | Support configurable pass/fail thresholds for CI use | P1 |
| FR-V.7 | Include at least one privacy-oriented risk indicator (see §19) | P0 |
| FR-V.8 | Allow custom metric plugins | P0 |

### 5.5 Visualization

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-Z.1 | Plot event frequency comparison (real vs synthetic) | P0 |
| FR-Z.2 | Plot session-length / duration histograms | P0 |
| FR-Z.3 | Plot transition heatmaps or network summaries | P0 |
| FR-Z.4 | Plot inter-event time distributions | P0 |
| FR-Z.5 | Save figures to common image formats (PNG/SVG/PDF) | P0 |
| FR-Z.6 | Optional interactive backends for notebooks | P1 |

### 5.6 Benchmarking

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-B.1 | Run multiple generators on the same dataset under identical protocol | P0 |
| FR-B.2 | Record metrics, seeds, versions, and configs in a benchmark result artifact | P0 |
| FR-B.3 | Produce comparative tables suitable for papers and CI artifacts | P0 |
| FR-B.4 | Ship at least one public synthetic/demo corpus schema for smoke benchmarks | P0 |
| FR-B.5 | Support holdout / train-eval split protocols | P1 |

### 5.7 Interfaces

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-C.1 | Provide a CLI covering ingest, analyze, fit, generate, validate, visualize, benchmark | P0 |
| FR-C.2 | Provide a stable Python API for the same workflows | P0 |
| FR-C.3 | Support configuration files (YAML/TOML) for non-interactive runs | P0 |
| FR-C.4 | Exit non-zero on validation/benchmark failure when thresholds are set | P0 |

### 5.8 Explicit non-goals (v1.0)

- Hosting or transmitting real learner data to EduLogGen servers (there are none)
- Guaranteeing differential privacy ε-bounds for all generators without user
  configuration (DP is a roadmap item; v1.0 focuses on risk indicators and safe
  workflow defaults)
- Vendor-specific LMS reverse engineering beyond documented generic mappings
- A graphical desktop application
- Real-time streaming generation for production tutoring systems

---

## 6. Non-functional Requirements

### 6.1 Quality attributes

| ID | Attribute | Requirement |
| -- | --------- | ----------- |
| NFR-1 | Reproducibility | Same inputs + config + seed ⇒ bit-stable or documented-stochastic outputs across runs on the same platform |
| NFR-2 | Extensibility | New generators/metrics installable via plugins without modifying core packages |
| NFR-3 | Scientific rigor | Metrics and protocols documented with assumptions and citations where applicable |
| NFR-4 | Privacy preservation | Default workflows discourage raw PII; privacy checks are first-class |
| NFR-5 | Portability | Python 3.11+ on Linux, macOS, and Windows |
| NFR-6 | Typed API | Public API is fully type-annotated; package ships `py.typed` |
| NFR-7 | Performance | Analyze and fit Markov models on ≥1e6 events on a modern laptop within practical interactive time (target: minutes, not hours); document scaling guidance |
| NFR-8 | Memory | Provide chunked ingestion path so datasets larger than RAM can be processed for core Markov workflows (P1 complete if streaming fit is deferred with clear limits) |
| NFR-9 | Usability | A new user can run ingest→fit→generate→validate on the demo corpus within 30 minutes using docs alone |
| NFR-10 | Observability | Structured logging with configurable verbosity; no silent failures on data errors |

### 6.2 Engineering standards

| ID | Requirement |
| -- | ----------- |
| NFR-11 | Source layout uses `src/eduloggen` |
| NFR-12 | Quality gates: Ruff, Black, MyPy (strict), PyTest, pre-commit, GitHub Actions |
| NFR-13 | Semantic Versioning for public releases |
| NFR-14 | MIT license retained |
| NFR-15 | Documentation built with MkDocs; API reference generated from docstrings |
| NFR-16 | Public functions/classes use Google-style docstrings |
| NFR-17 | No real learner PII in repository, issues, tests, or examples |

### 6.3 Compatibility and dependencies

- Core v1.0 dependencies should remain minimal and well-known in the scientific
  Python ecosystem (e.g., NumPy/Pandas for tabular work; plotting library for
  visualization extras).
- Heavy ML frameworks (PyTorch, TensorFlow, etc.) are **optional extras** for
  post-1.0 deep generators, not required for core install.
- Optional dependency groups: `dev`, `docs`, `viz`, `bench`, and later `dl`.

### 6.4 Security posture (software)

- No execution of untrusted pickled objects from the network by default.
- Generator artifact formats prefer versioned, inspectable serialization
  (documented schema) over opaque blobs where practical.
- CLI paths are validated; configuration cannot silently overwrite user files
  without explicit flags.

---

## 7. System Modules

EduLogGen is organized into cohesive modules. Each module has a clear
responsibility and published interfaces.

```text
┌─────────────────────────────────────────────────────────────────┐
│                         Interfaces                               │
│              CLI  ·  Python API  ·  Config files                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                        Orchestration                             │
│         Pipelines · Benchmark runner · Plugin registry           │
└──┬──────────┬──────────┬──────────┬──────────┬──────────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
Ingestion  Analysis  Generators  Validation  Visualization
   │          │          │          │          │
   └──────────┴─────► Canonical data model ◄───┴──────────┘
                             │
                             ▼
                      Privacy utilities
```

| Module | Responsibility |
| ------ | -------------- |
| **Ingestion** | Load, map, validate, and normalize external logs into the canonical model |
| **Analysis** | Sessionization, descriptive statistics, transition and timing estimates |
| **Generators** | Fit and sample synthetic sessions; persist models |
| **Validation** | Statistical, behavioral, and privacy metrics; reporting |
| **Visualization** | Plots and figure export for analysis and validation |
| **Benchmark** | Multi-generator evaluation protocols and result aggregation |
| **Plugins** | Discovery, registration, and lifecycle of extensions |
| **Config** | Typed configuration loading and validation |
| **Logging** | Process-wide logging policy and helpers |
| **CLI** | User-facing command surface over the Python API |

Cross-cutting rules:

1. Generators never read vendor-specific files directly; they consume the
   canonical model produced by ingestion/analysis.
2. Validation never depends on a specific generator implementation—only on
   real vs synthetic event/session tables and metadata.
3. Visualization is optional at runtime; core pipelines must succeed without
   plotting backends installed when `viz` extra is absent (degrade with a clear
   error if a plot command is invoked).

---

## 8. Package Structure

The installable package follows the existing `src` layout and extends it for
v1.0 without breaking the documented namespaces.

```text
src/eduloggen/
  __init__.py                 # Public version and selective re-exports
  py.typed
  cli/                        # CLI package (evolve from cli.py as surface grows)
    __init__.py
    main.py                   # Entry point
    commands/                 # ingest, analyze, fit, generate, validate, ...
  core/                       # Shared types, exceptions, constants
  models/                     # Canonical data models and schemas
  ingestion/
  analysis/
  generators/
    base.py                   # Abstract generator contract
    markov.py
    semi_markov.py
    registry.py
  validation/
    base.py
    metrics/
    reports.py
  visualization/
  benchmark/
  privacy/
  config/
  logging.py                  # or logging/ package
  plugins/
tests/
  unit/
  integration/
  fixtures/                   # Synthetic-only fixtures
docs/
examples/
notebooks/
```

### 8.1 Packaging rules

- Import path root: `eduloggen`
- Public API imported from stable modules; internal modules may be prefixed
  with `_` when not part of the supported surface
- Console script remains `eduloggen`
- Tests mirror package areas (`tests/unit/generators`, etc.)
- Examples remain outside the installable wheel except when explicitly packaged
  as package data for demo corpora (preferred: downloadable or `tests/fixtures`)

### 8.2 Dependency boundaries

| Package area | May depend on |
| ------------ | ------------- |
| `models`, `core` | Minimal third parties |
| `ingestion`, `analysis` | `models`, `core` |
| `generators` | `models`, `analysis` outputs, `core` |
| `validation`, `privacy` | `models`, `core` |
| `visualization` | `models`; optional plotting libs |
| `benchmark` | generators, validation, config |
| `cli` | all of the above via API façades |

Circular imports between generators and validation are forbidden.

---

## 9. Data Models

Canonical models are the interoperability contract across modules. Field names
below are normative for v1.0; physical storage may be Pandas DataFrame,
Polars DataFrame, or Arrow tables as decided in design docs—but **logical
schema** remains stable.

### 9.1 Event record

One row per recorded interaction.

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `event_id` | string | yes | Unique event identifier within a corpus |
| `learner_id` | string | yes | Pseudonymous learner key (never real names/emails in shared artifacts) |
| `timestamp` | datetime (UTC, timezone-aware) | yes | Event time |
| `session_id` | string | conditional | Required after sessionization if not provided upstream |
| `activity_id` | string | yes | Activity / resource / item identifier |
| `event_type` | string | yes | Controlled vocabulary (e.g., `view`, `attempt`, `submit`, `navigate`, `forum_post`, `video_play`, `other`) |
| `course_id` | string | no | Course or offering identifier |
| `score` | float | no | Numeric score if applicable |
| `success` | bool | no | Correctness / success flag |
| `duration_ms` | int | no | Event or action duration |
| `metadata` | mapping | no | Extension bag for non-core attributes |

Constraints:

- Timestamps must be sortable; naive datetimes are rejected or normalized per
  config with an explicit timezone policy.
- `learner_id` in **published** synthetic outputs should be synthetic keys,
  never copied real IDs, unless the user explicitly opts into ID-preserving mode
  for private local experiments (default: regenerate synthetic IDs).

### 9.2 Session record

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `session_id` | string | yes | Session key |
| `learner_id` | string | yes | Owner learner |
| `start_time` | datetime | yes | First event time |
| `end_time` | datetime | yes | Last event time |
| `n_events` | int | yes | Event count |
| `duration_s` | float | yes | Session duration in seconds |
| `event_sequence` | list[string] | yes | Ordered activity or event-type tokens used by generators |
| `course_id` | string | no | Dominant or declared course |

### 9.3 Corpus

A corpus is a named collection with:

- Events table
- Sessions table (optional until sessionized)
- Corpus metadata: schema version, source description, creation time,
  EduLogGen version, privacy notes, random seeds used for derived artifacts

### 9.4 Generator artifact

| Field | Description |
| ----- | ----------- |
| `generator_id` | Registered name (e.g., `markov`, `semi_markov`) |
| `generator_version` | Implementation version |
| `eduloggen_version` | Framework version |
| `hyperparameters` | JSON-serializable config |
| `vocabulary` | Token set / mapping |
| `parameters` | Model-specific fitted parameters |
| `training_fingerprint` | Hash of training config + data fingerprint (not raw data) |
| `fitted_at` | Timestamp |

### 9.5 Validation report

Structured document containing:

- Dataset fingerprints (real/synthetic counts, schema version)
- Metric results with names, values, confidence intervals if applicable
- Threshold evaluations (pass/fail/skip)
- Privacy indicator results
- Runtime metadata (duration, versions, seed)

### 9.6 Controlled vocabularies

v1.0 ships a default `event_type` vocabulary and allows extension via config.
Unknown types may be preserved as opaque tokens or mapped to `other` according
to a configured policy (default: preserve).

---

## 10. Plugin Architecture

### 10.1 Goals

- Allow third parties to add generators, metrics, ingestion adapters, and
  plotters without forking EduLogGen
- Keep discovery deterministic and explicit (no surprise imports)
- Enforce interface contracts via abstract base classes and registry checks

### 10.2 Plugin kinds

| Kind | Extension point |
| ---- | --------------- |
| `generator` | Fit/sample synthetic sessions |
| `metric` | Validation metric |
| `ingestor` | Source-format adapter |
| `sessionizer` | Alternative session segmentation strategy |
| `visualizer` | Named plot routine |
| `benchmark_suite` | Predefined evaluation protocol |

### 10.3 Registration

Plugins register through one or more of:

1. **Built-in registry** — core implementations registered at import
2. **Entry points** — Python packaging entry points under
   `eduloggen.plugins.<kind>`
3. **Explicit API** — `register_plugin(kind, name, factory)` for notebooks and
   proprietary code

Each plugin metadata record includes: name, version, kind, summary, and
optional capability tags (e.g., `supports_timing`, `gpu`).

### 10.4 Lifecycle

1. Discover → 2. Validate contract → 3. Instantiate with config → 4. Execute →
5. Dispose resources

Failed contract validation must raise a typed error before execution.

### 10.5 Stability rules

- Plugin interface versions are SemVer’d independently from package minors when
  breaking plugin ABIs
- Deprecated interfaces emit warnings for at least one minor release
- Plugins must not write outside user-specified output directories

---

## 11. Generator Framework

### 11.1 Design principles

1. **Unified contract** — all generators implement the same abstract interface
2. **Fit once, sample many** — separation of training and generation
3. **Seeded sampling** — reproducibility is mandatory
4. **Session-native** — generators produce sessions that expand to events
5. **Inspectable** — fitted parameters can be summarized for papers/logs

### 11.2 Abstract contract (behavioral)

Every generator must support:

| Operation | Description |
| --------- | ----------- |
| `fit(corpus, config)` | Estimate parameters from real sessions/events |
| `sample(n_sessions, seed, config)` | Draw synthetic sessions/events |
| `save(path)` / `load(path)` | Persist and restore artifacts |
| `describe()` | Human/machine summary of hyperparameters and size stats |

Optional capabilities (advertised via tags):

- Conditional generation given `course_id` or initial activity
- Timing generation (required for Semi-Markov)
- Online/partial fit (post-1.0)

### 11.3 Built-in generators (v1.0)

#### Markov generator

- Models token transitions within sessions (tokens = activity IDs or event
  types per config)
- Supports order `k ≥ 1` (default 1)
- Session length drawn from empirical or configured distribution
- Handles unseen contexts via configurable smoothing / backoff

#### Semi-Markov generator

- Extends Markov transitions with sojourn or inter-event time distributions
- Parametric families (e.g., log-normal, gamma, exponential) and/or empirical
  resampling selectable by config
- Ensures timestamps are strictly increasing within a session

### 11.4 Generation outputs

Default output: event table in canonical schema with synthetic `learner_id` /
`session_id` / `event_id` values, plus optional sessions table.

### 11.5 Failure modes

- Empty corpus → typed error
- Vocabulary size 1 with no self-transition policy → typed error or documented
  degenerate behavior
- Invalid seed type → validation error
- Insufficient data for requested Markov order → warning + automatic backoff or
  hard fail per config (default: fail with guidance)

---

## 12. Validation Framework

### 12.1 Purpose

Quantify how well synthetic logs match real logs on **utility** dimensions and
estimate **privacy risk**, producing reproducible reports.

### 12.2 Metric categories (v1.0)

| Category | Examples |
| -------- | -------- |
| Marginal similarity | Event-type / activity frequency divergence (e.g., TVD, JS divergence, χ²) |
| Sequential similarity | Bigram/transition divergence; optional longer n-grams |
| Session structure | Length and duration distribution distances (KS, Wasserstein) |
| Temporal | Inter-event time distribution distances |
| Privacy indicators | Nearest-neighbor distance heuristics; exact-session duplication rate; rare-sequence replay rate |
| Utility smoke (optional P1) | Train a simple next-event predictor on real vs synthetic and compare accuracy gap |

Exact metric set is finalized in a metrics design note; this PRD mandates
**categories** and at least one metric per P0 category.

### 12.3 Report contract

- JSON (machine) + Markdown/HTML summary (human)
- Each metric entry: `name`, `value`, `direction` (lower/higher better),
  `threshold`, `status` (`pass`/`fail`/`warn`/`skip`), `details`
- Global status is fail if any P0 threshold fails

### 12.4 Comparison protocol

Validation always declares:

- Which real split was used
- Which synthetic sample size was used
- Seed
- Whether synthetic IDs were remapped

### 12.5 Extensibility

Custom metrics implement a `compute(real, synthetic, context) -> MetricResult`
contract and register via the plugin system.

---

## 13. Visualization Framework

### 13.1 Role

Visualization supports exploratory analysis and paper-ready figures. It is not
required for headless CI validation (tables suffice), but CLI/API plot commands
are P0 for researcher UX.

### 13.2 Required plot types (v1.0)

1. Event frequency bar/compare plots
2. Session length histogram overlay
3. Session duration histogram overlay
4. Transition heatmap (top-N activities)
5. Inter-event time distribution overlay
6. Validation metric summary chart (optional convenience)

### 13.3 API expectations

- Functions accept corpus objects or metric reports
- Return a figure object and/or write to a path
- Apply accessible defaults (colorblind-friendly palettes where practical)
- Never embed learner PII in titles/labels (use counts and synthetic IDs)

### 13.4 Backends

- Default static backend via a mainstream plotting library
- Optional notebook interactivity as P1
- Missing dependency → clear install hint (`pip install eduloggen[viz]`)

---

## 14. CLI Specification

### 14.1 Invocation

```text
eduloggen <command> [options]
```

Global options:

| Option | Purpose |
| ------ | ------- |
| `--version` | Print package version |
| `--verbose` / `-v` | Increase log verbosity (repeatable) |
| `--quiet` | Errors only |
| `--config PATH` | Load YAML/TOML configuration |
| `--seed INT` | Global default seed where applicable |
| `--json-logs` | Structured JSON logging to stderr/stdout per config |

### 14.2 Commands (v1.0)

| Command | Purpose |
| ------- | ------- |
| `ingest` | Load and normalize source logs to a corpus artifact |
| `sessionize` | Build sessions from events |
| `analyze` | Compute descriptive and transition statistics |
| `fit` | Fit a named generator |
| `generate` | Sample synthetic sessions/events from a fitted artifact |
| `validate` | Compare real vs synthetic corpora |
| `plot` | Produce visualization artifacts |
| `benchmark` | Run multi-generator evaluation protocol |
| `plugins` | List discovered plugins |
| `info` | Show environment, version, and optional corpus summary |

### 14.3 Command semantics (summary)

**ingest**

- Inputs: source path(s), mapping config
- Outputs: corpus directory or file package
- Validates schema; writes data quality report

**fit**

- Inputs: corpus, generator name, hyperparameters
- Outputs: generator artifact path

**generate**

- Inputs: artifact, `n_sessions`, seed, output path
- Outputs: synthetic corpus

**validate**

- Inputs: real corpus, synthetic corpus, metric set, thresholds
- Outputs: report; exit code `0` pass, `1` fail, `2` usage/runtime error

**benchmark**

- Inputs: corpus, generator list, protocol name/config
- Outputs: comparative result table + raw metric JSON

### 14.4 UX rules

- Help text must include examples
- Destructive overwrites require `--force`
- Paths in errors should be absolute-normalized for clarity
- Progress indicators for long jobs when attached to a TTY

---

## 15. Python API Specification

### 15.1 Design goals

- Prefer explicit functions/classes over hidden globals
- Mirror CLI capabilities
- Be notebook-friendly (few lines to first synthetic sample)
- Remain fully typed

### 15.2 Public workflow surface (conceptual)

The v1.0 public API must enable approximately:

1. `load_events(path, mapping) -> Corpus`
2. `sessionize(corpus, policy) -> Corpus`
3. `analyze(corpus) -> AnalysisResult`
4. `fit_generator(name, corpus, config) -> Generator`
5. `generator.sample(n, seed) -> Corpus`
6. `validate(real, synthetic, metrics, thresholds) -> ValidationReport`
7. `plot_*(...)` helpers
8. `run_benchmark(protocol, generators, corpus) -> BenchmarkResult`

Exact names are fixed in an API design note prior to implementation freeze;
this PRD requires the **capabilities**, stability policy, and layering.

### 15.3 Stability policy

| Symbol | Stability |
| ------ | --------- |
| Documented in API reference without leading underscore | Public SemVer |
| `eduloggen.experimental` (if introduced) | May break in minor versions with warnings |
| `_` prefixed | Private; may change anytime |

### 15.4 Error model

Typed exceptions rooted at `EduLogGenError`, including at least:

- `SchemaError`
- `ConfigError`
- `IngestionError`
- `FitError`
- `GenerationError`
- `ValidationError`
- `PluginError`

Exceptions include machine-readable `code` strings where useful for CLI/UI.

### 15.5 Interop

- DataFrame round-trip helpers for events/sessions
- Export to CSV/Parquet
- Avoid requiring users to depend on internal classes for common tasks

---

## 16. Configuration System

### 16.1 Goals

- Make non-interactive, reproducible experiments natural
- Validate early with clear messages
- Allow CLI flags to override file config with documented precedence

### 16.2 Formats

- YAML and TOML supported for v1.0
- JSON accepted for machine-generated configs (P1 if not P0)

### 16.3 Precedence (highest wins)

1. Explicit CLI flags
2. Environment variables prefixed with `EDULOGGEN_` (documented allowlist)
3. Config file values
4. Built-in defaults

### 16.4 Config sections (normative outline)

- `project` — name, output directories
- `logging` — level, json mode
- `ingestion` — paths, mapping, dtype coercion
- `sessionization` — idle timeout, id strategy
- `analysis` — which statistics to compute
- `generator` — name + hyperparameters
- `generation` — n_sessions, seed, id strategy
- `validation` — metrics, thresholds
- `visualization` — backend, style, export paths
- `benchmark` — protocol, generator list, repeats
- `privacy` — policies and checks enabled

### 16.5 Validation

Configs are parsed into typed objects (e.g., Pydantic models or equivalent).
Unknown keys fail by default (strict mode) with an opt-in `allow_unknown` for
forward compatibility experiments.

### 16.6 Sharing configs

Configs must be safe to publish: no file-system secrets, no embedded PII.
Data fingerprints and paths may be local; published appendices should use
relative demo paths.

---

## 17. Logging

### 17.1 Requirements

- Use the standard `logging` framework
- Library code logs through named loggers under `eduloggen.*` and **does not**
  configure root handlers on import (CLI/apps configure handlers)
- CLI configures consistent formatting by default

### 17.2 Levels

| Level | Usage |
| ----- | ----- |
| DEBUG | Detailed trace for developers |
| INFO | Major pipeline stages, counts, timings |
| WARNING | Recoverable issues, backoff, deprecated features |
| ERROR | Failures that abort a command/API call |
| CRITICAL | Unexpected fatal conditions |

### 17.3 Content rules

- Log record counts, durations, seeds, generator names, metric summaries
- **Never** log raw personally identifying fields (names, emails, IPs, free-text
  posts). Prefer counts and hashed/fingerprinted identifiers if debugging ID
  issues
- Support `--json-logs` for ingestion into institutional log systems

### 17.4 Correlation

Long jobs emit a `run_id` in logs and reports to join CLI output with artifacts.

---

## 18. Benchmark Framework

### 18.1 Purpose

Provide a **standardized protocol** so generators can be compared fairly across
papers and CI builds—addressing the vision’s “benchmark-driven evaluation”
pillar.

### 18.2 Protocol elements

A benchmark protocol specifies:

1. Dataset (or dataset adapter) and split strategy
2. Preprocessing / sessionization settings
3. Generators and hyperparameter grids (or fixed defaults)
4. Sample sizes and seeds (multiple seeds for variance)
5. Metrics and aggregation (mean/std)
6. Runtime and memory capture (best-effort)
7. Output artifact schema

### 18.3 Built-in protocol (v1.0)

**`session_fidelity_v1`** (name illustrative):

- Fit on train sessions
- Generate N synthetic sessions
- Evaluate P0 validation metrics against holdout or full real distribution as
  documented
- Report wall-clock fit/sample times

### 18.4 Result artifact

Must include enough metadata for reproduction: EduLogGen version, platform,
dependency versions (optional lock digest), seeds, configs, metric values.

### 18.5 CI usage

- Smoke benchmark on tiny synthetic fixture runs on every PR
- Heavier benchmarks optional via scheduled workflows or labels

### 18.6 Leaderboard stance

EduLogGen may publish **example results** on public demo data. It will not claim
a universal ranking on private institutional data. Documentation must discourage
over-generalizing demo scores.

---

## 19. Privacy Requirements

### 19.1 Principles

1. **Local-first** — real data remains under user control; EduLogGen does not
   phone home
2. **Minimize** — examples and tests use synthetic data only
3. **Discourage identifiers** — default synthetic outputs use generated IDs
4. **Measure residual risk** — privacy indicators are part of validation
5. **No false guarantees** — docs must not claim “anonymous” or “zero risk”
   without qualification

### 19.2 Concrete requirements

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| PR-1 | Repository and CI contain no real learner data | P0 |
| PR-2 | Docs include a privacy & responsible use statement | P0 |
| PR-3 | Default generation remaps learner/session IDs | P0 |
| PR-4 | Validation includes duplication / nearest-neighbor style indicators | P0 |
| PR-5 | Config option to strip or hash optional free-text metadata fields on ingest | P1 |
| PR-6 | Guidance for institutional review / data governance checklists | P1 |
| PR-7 | Experimental differential privacy mechanisms | P2 (roadmap) |

### 19.3 Threats in scope (informal)

- Accidental publication of real logs via examples or issue trackers
- Synthetic data that memorizes rare real sessions
- Re-identification via unique pathways when synthetic data is too close to real

### 19.4 Threats out of scope (v1.0)

- Formal adversarial reconstruction attacks as a complete product feature
- Legal certification (GDPR/FERPA compliance stamps)

EduLogGen provides **engineering controls and metrics**, not legal advice.

---

## 20. Testing Strategy

### 20.1 Layers

| Layer | Purpose |
| ----- | ------- |
| Unit | Models, metrics math, sessionization edge cases, config parsing |
| Integration | ingest→sessionize→fit→generate→validate happy paths |
| Contract | Plugin interface compliance for built-ins and example plugins |
| Property / fuzz (selective) | Schema validators; sorting/session boundaries |
| Snapshot | Report structure (not floating metrics) where useful |
| Docs smoke | MkDocs build; notebook/example execution in CI where feasible |

### 20.2 Quality gates

- PyTest in CI on Python 3.11–3.13
- Coverage threshold maintained (baseline already ≥80% on skeleton; v1.0 target
  ≥85% for core modules, with pragmatic excludes for pure plotting GUIs)
- Ruff, Black, MyPy strict on `src`
- Pre-commit required for contributors

### 20.3 Fixtures

- Only synthetic fixtures
- Include degenerate cases: empty file, single event, single learner, unordered
  timestamps, missing optional columns
- Golden small corpus for end-to-end demos

### 20.4 Non-functional tests

- Seed reproducibility tests for Markov/Semi-Markov sampling
- CLI exit code tests for validation pass/fail
- Performance smoke tests with time budgets on tiny/medium fixtures (informational
  on PR, enforced on main if stable)

### 20.5 Security/privacy tests

- Scanners or custom checks ensuring fixtures lack email-like / PII-like patterns
- Tests that default ID remapping does not echo input learner IDs

---

## 21. Documentation Strategy

### 21.1 Audiences and artifacts

| Audience | Artifacts |
| -------- | --------- |
| New users | Getting started, quickstart, demo corpus walkthrough |
| Researchers | Concepts (sessions, generators, metrics), benchmark protocol, how to cite |
| Engineers | API reference, CLI reference, config reference, plugins guide |
| Contributors | CONTRIBUTING, architecture overview, ADRs for major decisions |
| Governance | Privacy & responsible use, security notes |

### 21.2 Tooling

- MkDocs Material site (already bootstrapped)
- `mkdocstrings` for API reference from Google-style docstrings
- Changelog via Keep a Changelog + SemVer
- Architecture Decision Records under `docs/adr/` as decisions harden

### 21.3 Documentation requirements for features

No P0 feature merges without:

1. User-facing explanation
2. API/CLI reference updates
3. At least one example or test doubling as documentation
4. Changelog entry

### 21.4 Scientific communication

- Prefer precise metric definitions over marketing language
- Cite foundational literature where algorithms are standard (Markov models,
  distribution distances, etc.)
- Provide a BibTeX citation snippet once a release/citable version exists

---

## 22. Release Strategy

### 22.1 Versioning

Semantic Versioning:

- **MAJOR** — incompatible API/schema/plugin ABI changes
- **MINOR** — backward-compatible features
- **PATCH** — backward-compatible fixes

Schema versions for corpora/artifacts are versioned explicitly and may advance
independently with migration notes.

### 22.2 Release channels

| Channel | Cadence | Audience |
| ------- | ------- | -------- |
| `0.x` | Frequent | Early adopters; breaking changes allowed with notes |
| `1.0.0` | Milestone | First stable API for vision scope |
| Post-1.0 minors | As features land | Additive generators/metrics |
| Pre-releases | `rc`, `beta` | Community testing |

### 22.3 Release checklist

1. Tests/docs green on CI
2. Changelog updated
3. Version bump in package metadata
4. Tagged release on GitHub
5. Publish wheel/sdist to PyPI
6. Deploy documentation site matching the tag
7. Smoke install from PyPI in a clean environment

### 22.4 Support policy

- Latest minor of the current major: actively supported
- Previous major: critical fixes for a limited window (documented at 1.0)
- Security issues handled via responsible disclosure process (document in
  `SECURITY.md` before 1.0)

### 22.5 Compatibility promises (1.x)

- Canonical event/session field names
- CLI command names listed in §14.2
- Documented Python API symbols
- Plugin entry-point group names

---

## 23. Future Roadmap

Aligned with the vision; ordered roughly by dependency.

### 23.1 Toward 1.0

- Complete ingestion, analysis, Markov, Semi-Markov
- Validation + privacy indicators
- Visualization + CLI/API parity
- Benchmark protocol `session_fidelity_v1`
- Demo corpus and tutorials

### 23.2 Post-1.0 (deep and advanced generators)

- CTGAN-based tabular session-feature synthesis (where applicable)
- TVAE
- TimeGAN
- Sequence VAE
- Transformer-based generators
- Hybrid pipelines (structure from Markov, richness from neural models)

### 23.3 Platform evolution

- Stronger differential privacy options
- Distributed/batched fitting for very large logs
- Catalog of community plugins
- Optional cloud-neutral workflow templates (still local-data by default)
- Cross-language export (e.g., Arrow) for non-Python consumers

### 23.4 Research program

- Public challenge track on demo datasets
- Metric standardization working notes with the EDM/LA community
- Replication packages for landmark educational datasets **only** when
  redistribution rights exist

---

## 24. Risks

| ID | Risk | Impact | Likelihood | Mitigation |
| -- | ---- | ------ | ---------- | ---------- |
| R-1 | Synthetic data oversold as privacy-safe | Legal/ethical harm; reputational | Medium | Strong docs language; privacy metrics; no absolute claims |
| R-2 | Metrics disagree with pedagogical utility | Low adoption by domain experts | Medium | Include behavioral/session metrics; invite LA reviewers; allow custom metrics |
| R-3 | Scope creep into full LMS integration | Delayed 1.0 | High | Strict non-goals; generic adapters only |
| R-4 | Deep learning deps explode core install | User friction | Medium | Optional extras; core stays statistical |
| R-5 | Non-reproducible floating-point / OS variance | Benchmark disputes | Medium | Document platform sensitivity; prefer distribution distances with tolerances |
| R-6 | Contributor plugins break silently | Trust erosion | Medium | Contract tests; registry validation; versioned plugin ABI |
| R-7 | Lack of redistributable real-world public logs | Weak demos | High | Curate synthetic demos; partner for licensed open datasets; clear schema examples |
| R-8 | Institutional security review blocks use | Slow uptake | Medium | Offline-first design; SBOMs optional; minimal deps |
| R-9 | Maintainer bandwidth | Stalled roadmap | Medium | Thin core; plugin ecosystem; clear contribution ladder |
| R-10 | Schema churn before 1.0 | Downstream breakage | Medium | Freeze candidate schema early; migration guides |

---

## 25. Success Metrics

Success is measured at product, community, and scientific levels. Targets are
for the **12 months following 1.0.0** unless noted.

### 25.1 Product readiness (gate to call “1.0 done”)

| Metric | Target |
| ------ | ------ |
| P0 functional requirements implemented and tested | 100% |
| End-to-end demo (ingest→validate) documented runtime on demo corpus | ≤ 30 minutes for new users |
| CLI commands in §14.2 available | 100% |
| Public API capabilities in §15.2 available | 100% |
| CI green on Python 3.11–3.13 | Required |
| Privacy indicators present in default validation | ≥ 1 P0 metric |

### 25.2 Adoption

| Metric | Target (indicative) |
| ------ | ------------------- |
| PyPI downloads | Track trend; aspire to sustained monthly growth post-1.0 |
| GitHub stars / dependents | Community signal (non-vanity; correlate with issues/PRs) |
| External plugins or citations | ≥ 3 independent uses (plugin, paper, or course) within 12 months |

### 25.3 Scientific utility

| Metric | Target |
| ------ | ------ |
| Benchmark protocol used in at least one external comparison | ≥ 1 |
| Issues requesting “data availability” resolved via EduLogGen synthetic release pattern | Qualitative success stories documented |
| Reproduction of tutorial results by a third party | At least one confirmed reproduction |

### 25.4 Quality

| Metric | Target |
| ------ | ------ |
| Open critical defects older than 30 days | 0 |
| Mean time to first response on issues | ≤ 7 days (best effort for volunteer project) |
| Documentation build + example smoke in CI | Required |

### 25.5 Privacy & trust

| Metric | Target |
| ------ | ------ |
| Incidents of real PII in repo/CI | 0 |
| User-facing privacy documentation completeness | Reviewed each minor release |

### 25.6 North-star outcome

EduLogGen is cited or depended upon as the **default open toolkit** for
synthetic educational interaction logs in EDM/LA research workflows—measured by
independent papers, courses, and EdTech pipelines using its schema, generators,
and benchmark protocol rather than one-off simulators.

---

## Appendix A — Requirement traceability to vision

| Vision objective | PRD coverage |
| ---------------- | ------------ |
| Analyze educational interaction logs | §§5.2, 7, 9, 14–15 |
| Learn learner behavioral patterns | §§5.2, 11 |
| Generate realistic synthetic sessions | §§5.3, 11 |
| Preserve statistical similarity | §§5.4, 12, 18 |
| Preserve behavioral realism | §§5.4, 12, 13 |
| Protect learner privacy | §§5.4, 19, 20.5 |
| Benchmark methods | §§5.6, 18, 25 |

## Appendix B — Document control

| Version | Date | Notes |
| ------- | ---- | ----- |
| 1.0 | 2026-07-29 | Initial comprehensive PRD for EduLogGen 1.0 scope |

Changes to this PRD that alter P0 scope require an explicit decision record and
changelog note prior to implementation milestones.
