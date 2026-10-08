# EduLogGen Software Architecture Document

| Field | Value |
| ----- | ----- |
| **Document** | Software Architecture Document (SAD) |
| **Product** | EduLogGen |
| **Document version** | 1.0 |
| **Status** | Normative draft |
| **Authors** | EduLogGen Contributors (Chief Software Architect) |
| **Governing standards** | ISO/IEC/IEEE 42010:2011 (architecture description concepts) |
| **Related documents** | [`01_VISION.md`](01_VISION.md), [`02_PRD.md`](02_PRD.md) |
| **Baseline package** | `eduloggen` 0.1.0 skeleton; architecture targets **1.0.0** |
| **License** | MIT |

---

## Document control

| Version | Date | Description |
| ------- | ---- | ----------- |
| 1.0 | 2026-07-29 | Initial authoritative Software Architecture Document |

This document is the **single source of truth** for EduLogGen system architecture.
Where this SAD and earlier sketches diverge, **this SAD prevails** for
implementation. Product intent remains governed by the Vision and PRD; this SAD
refines those intents into architectural structure, views, and decisions.

---

## Table of contents

1. [Introduction](#1-introduction)
2. [Architectural Goals](#2-architectural-goals)
3. [Architectural Principles](#3-architectural-principles)
4. [System Context](#4-system-context)
5. [High-Level Architecture](#5-high-level-architecture)
6. [Repository Structure](#6-repository-structure)
7. [Python Package Structure](#7-python-package-structure)
8. [Dependency Rules](#8-dependency-rules)
9. [Domain Model](#9-domain-model)
10. [Data Flow](#10-data-flow)
11. [Analysis Architecture](#11-analysis-architecture)
12. [Generator Architecture](#12-generator-architecture)
13. [Plugin System](#13-plugin-system)
14. [Validation Architecture](#14-validation-architecture)
15. [Visualization Architecture](#15-visualization-architecture)
16. [Configuration Architecture](#16-configuration-architecture)
17. [Logging Architecture](#17-logging-architecture)
18. [Error Handling](#18-error-handling)
19. [CLI Architecture](#19-cli-architecture)
20. [Python API](#20-python-api)
21. [Testing Architecture](#21-testing-architecture)
22. [Documentation Architecture](#22-documentation-architecture)
23. [CI/CD Architecture](#23-cicd-architecture)
24. [Extension Points](#24-extension-points)
25. [Security](#25-security)
26. [Performance](#26-performance)
27. [Future Architecture](#27-future-architecture)
28. [Design Patterns](#28-design-patterns)
29. [Architecture Decision Records](#29-architecture-decision-records)
30. [Conclusion](#30-conclusion)

### Detailed annexes (normative elaborations)

Sections **29A–29X** elaborate interfaces, state machines, metrics, CLI/API
contracts, file plans, scenarios, and additional ADRs. They are part of the
authoritative architecture description.

---

## 1. Introduction

### 1.1 Purpose

This Software Architecture Document (SAD) describes the architecture of
**EduLogGen**, an open-source Python framework for analyzing, generating,
validating, and benchmarking synthetic educational interaction logs.

In the sense of ISO/IEC/IEEE 42010, this document is an *architecture
description*: it identifies stakeholders and concerns, presents architecture
views that address those concerns, and records decisions with rationale. Its
purpose is to:

1. Provide a normative blueprint from which all subsequent implementation,
   tests, and documentation can be derived.
2. Bound module responsibilities and dependency directions so contributors can
   extend the system without eroding cohesion.
3. Align engineering structure with the scientific goals of reproducibility,
   extensibility, privacy preservation, and benchmark-driven evaluation stated
   in the Vision and PRD.
4. Serve as onboarding material for researchers, developers, and maintainers.

### 1.2 Scope

**In scope**

- Logical and physical structure of the `eduloggen` Python package
- Canonical domain model and data lifecycle
- Analysis, generation, validation, visualization, benchmark, CLI, and API
  architectures for product version **1.0**
- Plugin and configuration architectures
- Testing, documentation, CI/CD, security, and performance architectures
- Extension mechanisms and future-facing seams (deep learning, distributed
  execution, services) without mandating their immediate implementation

**Out of scope**

- Algorithmic derivations and numerical proofs (belong in research notes)
- Vendor-specific LMS reverse-engineering beyond generic adapters
- Legal compliance certification (GDPR/FERPA stamps)
- Hosted multi-tenant SaaS operations (EduLogGen is local-first software)

### 1.3 Intended audience

| Audience | Primary concerns addressed |
| -------- | -------------------------- |
| Core maintainers | Module boundaries, SemVer, ADRs |
| Contributors | Extension points, dependency rules, patterns |
| Researchers | Pipeline semantics, reproducibility, benchmarks |
| Data scientists | API/CLI workflows, configuration, reports |
| Institutional engineers | Security, privacy, deployment constraints |
| Reviewers of PRs | Whether changes respect architecture |

### 1.4 Definitions

| Term | Definition |
| ---- | ---------- |
| **Architecture** | Fundamental concepts or properties of a system in its environment, embodied in its elements, relationships, and principles of design and evolution (IEEE 42010) |
| **Concern** | Interest in a system relevant to one or more stakeholders |
| **View** | Representation of a system from the perspective of related concerns |
| **Corpus / Dataset** | Named collection of events (and optionally sessions) plus metadata |
| **Event / LogRecord** | Atomic interaction observation at a point in time |
| **Session** | Contiguous episode of learner activity bounded by policy |
| **Participant** | Learner entity referenced by a pseudonymous identifier |
| **Generator** | Component that fits a behavioral model and samples synthetic sessions |
| **Validator / Metric** | Component that scores similarity, utility, or privacy risk |
| **Plugin** | Externally or internally registered extension implementing a contract |
| **Artifact** | Persisted, versioned object (corpus package, fitted model, report) |
| **Run ID** | Correlation identifier joining logs, configs, and output artifacts |
| **P0 / P1 / P2** | Requirement priorities from the PRD |

### 1.5 References

| ID | Reference |
| -- | --------- |
| REF-1 | EduLogGen Vision (`docs/01_VISION.md`) |
| REF-2 | EduLogGen PRD (`docs/02_PRD.md`) |
| REF-3 | ISO/IEC/IEEE 42010:2011 — Architecture description |
| REF-4 | Semantic Versioning 2.0.0 |
| REF-5 | PEP 517/518/621 — Python packaging |
| REF-6 | Keep a Changelog |
| REF-7 | MIT License |

---

## 2. Architectural Goals

Architectural goals express *quality attributes* (concerns) that shape structure.
Each goal includes a definition, architectural response, and measurable signal.

### 2.1 Modularity

**Definition.** The system is decomposed into cohesive packages with narrow
interfaces and replaceable implementations.

**Architectural response.** Layered packages (`models`/`core` at the center;
`io`, `analysis`, `generators`, `validation`, `benchmark`, `visualization` as
peers; `cli` as an adapter). Plugin registries isolate extension code from core
algorithms.

**Signal.** A new generator lands as a new module + registry entry without
editing validators.

### 2.2 Extensibility

**Definition.** Third parties add readers, generators, metrics, plots, and
benchmarks without forking.

**Architectural response.** Strategy + Registry + Entry-point discovery; stable
abstract contracts (`BaseGenerator`, `BaseMetric`, `BaseReader`, etc.).

**Signal.** External package registers via `eduloggen.plugins.*` entry points.

### 2.3 Maintainability

**Definition.** Contributors locate change sites quickly; coupling remains low.

**Architectural response.** Dependency rules (§8), explicit public API (§20),
ADR discipline (§29), Google-style docstrings, strict MyPy.

**Signal.** Mean time to implement a scoped PR remains small; circular-import
CI check stays green.

### 2.4 Reproducibility

**Definition.** Same inputs, configuration, and seeds yield documented-stable
outputs on a given platform.

**Architectural response.** Seed plumbing through fit/sample; config
fingerprints in artifacts; run IDs; benchmark protocols with recorded metadata.

**Signal.** Reproducibility tests fail the build when seeds diverge.

### 2.5 Testability

**Definition.** Units are isolatable; pipelines have clear seams for fakes.

**Architectural response.** Dependency inversion at plugin boundaries; pure
domain objects; fixtures that are synthetic-only; contract tests for plugins.

**Signal.** Target line coverage **>95%** for core logic packages (see §21).

### 2.6 Performance

**Definition.** Practical interactive runtimes for million-event Markov workflows
on a modern laptop; scalable paths for larger corpora.

**Architectural response.** Chunked/streaming readers; vectorized analysis where
safe; optional parallelism for independent sessions; caching of analysis
artifacts; deep-learning deps kept optional.

**Signal.** Performance smoke tests with budgets; documented scaling guidance.

### 2.7 Reliability

**Definition.** Failures are explicit, typed, and recoverable at clear
boundaries; no silent data corruption.

**Architectural response.** Schema validation on ingest; typed exception
hierarchy; strict config parsing; non-zero CLI exits on validation failure.

**Signal.** Integration tests cover degenerate corpora and assert error codes.

### 2.8 Usability

**Definition.** Researchers reach first synthetic dataset quickly via CLI or
API; messages are actionable.

**Architectural response.** Parity between CLI and Python API; opinionated
defaults; demo corpus; progressive disclosure of advanced config.

**Signal.** New-user path ≤ 30 minutes per PRD NFR-9.

### 2.9 Documentation

**Definition.** Architecture, API, and research protocols remain discoverable
and synchronized with code.

**Architectural response.** MkDocs + mkdocstrings; this SAD; ADRs; examples and
tutorials as first-class artifacts; docs CI `--strict`.

**Signal.** P0 features cannot merge without docs updates (PRD rule).

### 2.10 Open-source friendliness

**Definition.** Contribution friction is low; licensing is clear; governance is
transparent.

**Architectural response.** MIT license; `src` layout; pre-commit; CONTRIBUTING;
plugin ladder; no telemetry; no mandatory accounts.

**Signal.** External plugins and citations emerge without private forks.

### 2.11 Quality-attribute summary matrix

| Attribute | Priority (v1.0) | Primary architectural lever |
| --------- | --------------- | --------------------------- |
| Modularity | Critical | Package boundaries + DIP |
| Extensibility | Critical | Plugins + contracts |
| Reproducibility | Critical | Seeds + artifacts + protocols |
| Privacy (from Vision) | Critical | Defaults + validators |
| Maintainability | High | Dependency rules + ADRs |
| Testability | High | Seams + coverage target |
| Performance | High | Streaming + caching |
| Reliability | High | Validation + errors |
| Usability | High | CLI/API parity |
| Documentation | High | MkDocs + SAD |
| Open-source friendliness | High | MIT + contribution UX |

---

## 3. Architectural Principles

### 3.1 SOLID

| Principle | Application in EduLogGen |
| --------- | ------------------------ |
| **S**ingle Responsibility | Each package owns one concern (I/O ≠ generation ≠ validation) |
| **O**pen/Closed | New generators/metrics open via plugins; closed to core edits |
| **L**iskov Substitution | Any `BaseGenerator` may replace another in benchmark harness |
| **I**nterface Segregation | Separate reader/metric/plotter contracts; no god interfaces |
| **D**ependency Inversion | High-level pipelines depend on abstractions, not concretes |

### 3.2 Dependency inversion

Orchestration (CLI, benchmark runner, façade services) depends on abstract
ports. Concrete Markov/Semi-Markov classes are adapters registered in the
registry. Validation depends on `Dataset` abstractions, never on generator
internals.

### 3.3 Separation of concerns

Distinct axes:

- **Data shape** (`models`) vs **algorithms** (`analysis`, `generators`)
- **Scoring** (`validation`) vs **rendering** (`visualization`)
- **User I/O** (`cli`, config files) vs **domain services** (API façades)

### 3.4 Single responsibility

A module that both fits a model *and* writes HTML dashboards violates SRP.
Reports are data; visualization consumes reports.

### 3.5 Composition over inheritance

Prefer composing timing models, transition models, and ID strategies into a
Semi-Markov generator over deep inheritance trees. Inheritance is reserved for
shallow template-method bases (`BaseGenerator`).

### 3.6 Immutable domain objects where appropriate

Value objects such as `LogRecord`, metric results, and configuration snapshots
are immutable (e.g., frozen dataclasses / attrs) after construction. Mutable
builders exist only during ingest/sessionization assembly, then yield immutable
`Dataset` snapshots for downstream stages.

### 3.7 Explicit interfaces

Public behaviors are declared via typing `Protocol` or ABC classes with
documented pre/postconditions. Duck typing alone is insufficient for plugin
validation.

### 3.8 Plugin architecture

Extension is a first-class architectural style: discovery, contract check,
instantiation, execution, disposal.

### 3.9 Configuration over hard coding

Thresholds, idle timeouts, Markov order, metric sets, and paths come from the
configuration hierarchy (§16), not literals scattered in algorithms.

### 3.10 Additional principles

- **Local-first privacy** — no network exfiltration of learner data by design
- **Fail fast** — schema and config errors abort before silent wrong science
- **Scientific honesty** — architecture supports metrics and caveats, not
  overclaimed anonymity
- **Thin core, optional weight** — deep learning stacks are extras, not core

---

## 4. System Context

### 4.1 Context overview (IEEE 42010 system-in-environment)

EduLogGen executes on user-controlled infrastructure. Real educational logs
remain inside institutional boundaries. The framework reads local files,
writes local artifacts, and optionally publishes *synthetic* outputs chosen by
the user.

```text
                         ┌──────────────────────────────────────┐
                         │     Educational Institution / Lab     │
                         │  (data governance, private storage)   │
                         └───────────────┬──────────────────────┘
                                         │ authorizes access
                                         ▼
┌──────────────┐   raw logs    ┌─────────────────────┐   synthetic   ┌────────────┐
│ LMS / ITS /  │──────────────►│                     │──────────────►│ Shared     │
│ MOOC / App   │               │     EduLogGen       │               │ research / │
│ event stores │               │  (local process)    │               │ staging    │
└──────────────┘               │                     │──────────────►│ datasets   │
                               │  CLI │ Python API   │               └────────────┘
┌──────────────┐  configs/     │  Benchmark harness  │ reports/figs
│ Researcher / │◄─────────────►│                     │──────────────► papers, CI
│ Data Scientist│  seeds       └──────────▲──────────┘
└──────────────┘                          │
┌──────────────┐                          │ plugins / PRs
│ Developer /  │──────────────────────────┘
│ Contributor  │
└──────────────┘
```

### 4.2 Actors and external systems

| Actor | Type | Interactions |
| ----- | ---- | ------------ |
| **Researcher** | Human | Designs experiments; runs analyze/fit/generate/validate; cites versions |
| **Developer** | Human | Extends plugins; maintains core; reviews ADRs |
| **Data Scientist** | Human | Notebooks/API; feature exploration; model prototyping on synthetic data |
| **Educational Institution** | Organization | Owns real data; sets policy; consumes privacy reports |
| **CLI** | Interface | Non-interactive pipelines and batch jobs |
| **Python API** | Interface | Programmatic embedding in research code |
| **Benchmark Framework** | Subsystem | Multi-generator evaluation protocols |

### 4.3 Context diagram — interface focus

```text
                 ┌─────────────┐
                 │  Config     │
                 │ YAML/TOML/  │
                 │ Env/CLI     │
                 └──────┬──────┘
                        │
     ┌──────────────────┼──────────────────┐
     │                  ▼                  │
     │         ┌─────────────────┐         │
     │         │   EduLogGen     │         │
CLI ─┼────────►│   Application   │◄────────┼─ Python API
     │         │   Core          │         │
     │         └────────┬────────┘         │
     │                  │                  │
     │         ┌────────▼────────┐         │
     │         │ Plugin Registry │◄────────┼─ External plugins
     │         └────────┬────────┘         │
     │                  │                  │
     └──────────────────┼──────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     Filesystem    Artifact store   Logs/Reports
     (raw/synth)   (models)         (JSON/MD/PNG)
```

### 4.4 Stakeholder concerns map

| Stakeholder | Concerns | Views in this SAD |
| ----------- | -------- | ----------------- |
| Researcher | Fidelity, privacy, reproducibility | §§5, 11–14, 18 |
| Developer | Modularity, extension, CI | §§6–8, 13, 21, 23–24 |
| Data Scientist | API clarity, viz, notebooks | §§15, 19–20, 22 |
| Institution | Security, governance | §§17, 25 |
| Benchmark Framework | Fair protocol, artifacts | §§14, 18 |

---

## 5. High-Level Architecture

### 5.1 End-to-end processing pipeline

The normative pipeline realizes the Vision objectives as a staged dataflow.
Each stage has typed inputs/outputs and may persist artifacts for
reproducibility.

```text
 Raw Logs
    │
    ▼
 ┌──────────┐
 │  Reader  │  (format adapters: CSV, Parquet, JSONL, …)
 └────┬─────┘
      │ mapped records
      ▼
 ┌─────────────────────┐
 │ Schema Validation   │  (required fields, dtypes, timezone policy)
 └────┬────────────────┘
      │ canonical LogRecords
      ▼
 ┌─────────────────────┐
 │  Session Builder    │  (idle timeout / explicit session_id)
 └────┬────────────────┘
      │ Dataset (events + sessions)
      ▼
 ┌─────────────────────┐
 │  Analysis           │  (features, transitions, timing, graphs, profiles)
 └────┬────────────────┘
      │ AnalysisResult + Behavior Model inputs
      ▼
 ┌─────────────────────┐
 │  Behavior Model     │  (estimated structures consumed by generators)
 └────┬────────────────┘
      │ fitted structures / GeneratorModel
      ▼
 ┌─────────────────────┐
 │  Generator          │  (fit → sample; seeded)
 └────┬────────────────┘
      │ SyntheticDataset
      ▼
 ┌─────────────────────┐
 │  Validation         │  (statistical, behavioral, privacy, utility)
 └────┬────────────────┘
      │ ValidationReport
      ▼
 ┌─────────────────────┐
 │  Benchmark          │  (multi-generator protocol aggregation)
 └────┬────────────────┘
      │ BenchmarkReport
      ▼
 ┌─────────────────────┐
 │  Reports / Plots   │  (JSON, Markdown, figures, dashboards)
 └─────────────────────┘
```

### 5.2 Stage explanations

#### 5.2.1 Raw logs

Source event exports from LMS/ITS/MOOC/apps. EduLogGen does not assume a single
vendor schema; mapping configuration projects source columns onto the canonical
model.

#### 5.2.2 Reader

`io` readers stream or load source files, apply column maps, coerce types, and
emit provisional records. Readers never perform scientific analysis.

#### 5.2.3 Validation (ingest-time)

Distinct from *scientific* validation later: this stage enforces schema
integrity, monotonicity policies, and duplicate handling. Failures raise
`SchemaError` / `IngestionError`.

#### 5.2.4 Session builder

Groups events into sessions using explicit `session_id` or idle-gap
sessionization. Produces `Session` entities and attaches `session_id` on events.

#### 5.2.5 Analysis

Computes descriptive statistics, transition structures, temporal summaries,
navigation graphs, and optional user profiles. Outputs are immutable analysis
artifacts usable by generators and visualizations.

#### 5.2.6 Behavior model

The estimated behavioral representation (e.g., transition matrix + length
distribution + timing laws). May be an intermediate object or embedded inside a
fitted `GeneratorModel`. Architecturally it is the bridge between analysis and
generation.

#### 5.2.7 Generator

Implements `fit` / `generate` (also named `sample` in API) / `save` / `load`.
Produces `SyntheticDataset` with remapped IDs by default.

#### 5.2.8 Validation (scientific)

Compares real vs synthetic datasets across metric categories; emits
`ValidationReport` with pass/fail thresholds.

#### 5.2.9 Benchmark

Runs multiple generators under a fixed protocol, aggregates metrics and
timings into `BenchmarkReport`.

#### 5.2.10 Reports

Serializes machine-readable JSON and human-readable Markdown/HTML; optional
plots and dashboards via visualization package.

### 5.3 Logical layered view

```text
┌────────────────────────────────────────────────────────────┐
│ Presentation / Adaptation                                  │
│   cli/   ·   notebooks (external)   ·   future REST        │
├────────────────────────────────────────────────────────────┤
│ Application Services (façades)                             │
│   ingest · analyze · fit · generate · validate · benchmark │
├────────────────────────────────────────────────────────────┤
│ Domain / Plugins                                           │
│   analysis · generators · validation · visualization       │
├────────────────────────────────────────────────────────────┤
│ Domain Model & Policies                                    │
│   models · privacy policies · config schemas               │
├────────────────────────────────────────────────────────────┤
│ Infrastructure                                             │
│   io · logging · serialization · filesystem                │
└────────────────────────────────────────────────────────────┘
```

### 5.4 Runtime deployment view (v1.0)

Single process, single host, local filesystem. No mandatory server. Optional
future deployments (§27) reuse the same domain core behind other adapters.

---

## 6. Repository Structure

```text
EduLogGen/
├── .github/                 # CI/CD workflows, templates
├── .pre-commit-config.yaml  # Local quality hooks
├── CHANGELOG.md             # SemVer release notes
├── CONTRIBUTING.md          # Contributor process
├── LICENSE                  # MIT
├── README.md                # Landing overview
├── mkdocs.yml               # Documentation site config
├── pyproject.toml           # Package metadata & tool config
├── docs/                    # Authoritative product & architecture docs
├── src/eduloggen/           # Installable Python package
├── tests/                   # PyTest suite
├── examples/                # Runnable scripts (synthetic data only)
└── notebooks/               # Exploratory tutorials
```

| Directory / file | Why it exists |
| ---------------- | ------------- |
| `.github/` | Automates lint, typecheck, test, build, docs deploy, releases |
| `.pre-commit-config.yaml` | Shifts quality left before CI |
| `CHANGELOG.md` | Human history aligned with SemVer |
| `CONTRIBUTING.md` | Reduces contribution ambiguity |
| `LICENSE` | Legal clarity for open-source adoption |
| `README.md` | First-contact orientation |
| `mkdocs.yml` | Binds documentation IA to site generation |
| `pyproject.toml` | Single packaging & tool configuration surface |
| `docs/` | Vision, PRD, SAD, guides — governance artifacts |
| `src/eduloggen/` | PEP 517 src layout prevents accidental editable hacks |
| `tests/` | Executable specification of behavior |
| `examples/` | Copy-paste research workflows |
| `notebooks/` | Pedagogical exploration without bloating the wheel |

** Normative rule:** Real learner data must never be committed under any
directory.

---

## 7. Python Package Structure

### 7.1 Target package tree (normative for 1.0)

The architecture expands the 0.1.0 skeleton namespaces into the following
packages. The PRD’s `ingestion` concern is realized primarily by **`io`**
(readers/writers) plus schema validation in **`models`/`core`**. A thin
compatibility alias may re-export ingest helpers during migration.

```text
src/eduloggen/
├── __init__.py              # version + stable re-exports
├── py.typed
├── core/                    # exceptions, constants, run context
├── config/                  # typed settings, loaders, precedence
├── models/                  # domain entities & schemas
├── io/                      # readers, writers, corpus packages
├── analysis/                # sessionization, features, behavior stats
├── generators/              # BaseGenerator + families + registry
├── validation/              # metrics, reports, privacy indicators
├── benchmark/               # protocols, runner, aggregation
├── visualization/           # plotters, dashboards
├── cli/                     # argparse/typer commands
├── plugins/                 # discovery & registration
├── privacy/                 # ID remapping, stripping policies
└── utils/                   # pure helpers (hashing, time, seeding)
```

### 7.2 Package responsibilities

#### 7.2.1 `core`

**Responsibilities:** shared exceptions hierarchy roots, global constants
(schema version strings), `RunContext` (run_id, seed, clocks), typing aliases.

**Depends on:** stdlib; minimal third parties.

**Public interfaces:** `EduLogGenError` tree roots; `RunContext`; schema version
constants.

#### 7.2.2 `config`

**Responsibilities:** parse YAML/TOML/JSON; merge precedence; validate into
typed settings objects; emit config fingerprints.

**Depends on:** `core`, `models` (for enum/vocabulary references only).

**Public interfaces:** `load_config(path) -> AppConfig`; `AppConfig` sections;
`ConfigError`.

#### 7.2.3 `models`

**Responsibilities:** canonical domain entities (`Dataset`, `LogRecord`,
`Session`, …), schema validation primitives, vocabulary registries.

**Depends on:** `core`.

**Public interfaces:** entity types; `validate_schema`; conversion helpers to/from
tabular rows.

#### 7.2.4 `io`

**Responsibilities:** format readers/writers; corpus directory layout; streaming
and chunk APIs; field mapping application.

**Depends on:** `core`, `models`, `config` (mapping fragments).

**Public interfaces:** `BaseReader`, `BaseWriter`, `read_corpus`, `write_corpus`,
built-in CSV/Parquet/(JSONL) readers.

#### 7.2.5 `analysis`

**Responsibilities:** session builder orchestration hooks; feature extraction;
transition matrices; time statistics; sequence analysis; navigation graphs;
user profiles; serialization of `AnalysisResult`.

**Depends on:** `core`, `models`, `utils`.

**Public interfaces:** `sessionize`, `analyze`, `AnalysisResult`, feature
extractors.

#### 7.2.6 `generators`

**Responsibilities:** `BaseGenerator` contract; Markov / Semi-Markov; registry;
artifact save/load; family-specific modules (statistical, probabilistic, …).

**Depends on:** `core`, `models`, `analysis` (results / behavior structures),
`privacy` (ID strategy), `utils` (RNG).

**Must not depend on:** `validation`, `visualization`, `cli`.

**Public interfaces:** `BaseGenerator`, `fit_generator`, `get_generator`,
concrete built-ins, artifact I/O.

#### 7.2.7 `validation`

**Responsibilities:** metric contracts; statistical/behavioral/privacy/utility
metrics; report assembly; threshold evaluation.

**Depends on:** `core`, `models`, `utils`.

**Must not depend on:** `generators` (compares datasets only).

**Public interfaces:** `BaseMetric`, `validate`, `ValidationReport`.

#### 7.2.8 `benchmark`

**Responsibilities:** protocol definitions; multi-seed runner; result
aggregation; comparison tables.

**Depends on:** `core`, `config`, `models`, `generators` (via abstract registry),
`validation`, `utils`.

**Public interfaces:** `run_benchmark`, `BenchmarkProtocol`, `BenchmarkReport`.

#### 7.2.9 `visualization`

**Responsibilities:** plot modules; export to PNG/SVG/PDF; optional interactive
backends; dashboard assembly from reports.

**Depends on:** `core`, `models`; optional viz third parties.

**Must not be required** to import for headless validate/benchmark success.

**Public interfaces:** `plot_*` functions; `BaseVisualizer`.

#### 7.2.10 `cli`

**Responsibilities:** argument parsing; mapping CLI to application services;
exit codes; progress UX; logging configuration for processes.

**Depends on:** application façades and packages above; never embeds algorithms.

**Public interfaces:** `main(argv) -> int`; command modules.

#### 7.2.11 `utils`

**Responsibilities:** seeding helpers, hashing/fingerprints, time normalization
utilities, small pure functions.

**Depends on:** stdlib / `core` only.

**Rule:** no hidden business workflows in `utils`.

#### 7.2.12 Supporting packages

| Package | Role |
| ------- | ---- |
| `plugins` | Entry-point discovery, registration API |
| `privacy` | Default ID remapping, metadata stripping policies |

### 7.3 Public interface examples (non-normative signatures)

Interface sketches guide implementers; exact names freeze before 1.0 API lock.

```text
class BaseGenerator(ABC):
    def fit(self, dataset: Dataset, config: GeneratorConfig) -> GeneratorModel: ...
    def generate(self, model: GeneratorModel, n_sessions: int, seed: int,
                 config: GenerationConfig) -> SyntheticDataset: ...
    def save(self, model: GeneratorModel, path: Path) -> None: ...
    def load(self, path: Path) -> GeneratorModel: ...
    def describe(self, model: GeneratorModel) -> dict[str, Any]: ...

class BaseMetric(ABC):
    name: str
    def compute(self, real: Dataset, synthetic: Dataset,
                context: ValidationContext) -> MetricResult: ...

class BaseReader(ABC):
    def read(self, source: SourceRef, mapping: FieldMapping) -> Iterable[LogRecord]: ...
```

### 7.4 Mapping from 0.1.0 skeleton

| Skeleton today | Architecture target |
| -------------- | ------------------- |
| `ingestion/` | `io/` (+ mapping validation) |
| `analysis/` | `analysis/` (expanded) |
| `generators/` | `generators/` (expanded) |
| `validation/` | `validation/` |
| `visualization/` | `visualization/` |
| `cli.py` | `cli/` package |
| *(absent)* | `core/`, `config/`, `models/`, `benchmark/`, `plugins/`, `privacy/`, `utils/` |

---

## 8. Dependency Rules

### 8.1 Allowed dependency direction

```text
cli ──────────────► benchmark ──► generators ──► analysis ──► models
 │                     │              │              │          ▲
 │                     ▼              │              ▼          │
 │                 validation ────────┴──────────► privacy      │
 │                     │                             │          │
 ▼                     ▼                             ▼          │
config ◄──────────── utils ◄────────────────────── core ───────┘
 │                                                          ▲
 └──────────────────────────► io ───────────────────────────┘
 visualization ──► models (and reports types from validation)
 plugins ──► (registers factories; no upward domain deps required)
```

### 8.2 Dependency matrix

| From \ To | core | config | models | io | analysis | generators | validation | benchmark | viz | cli | utils | privacy |
| --------- | ---- | ------ | ------ | -- | -------- | ---------- | ---------- | --------- | --- | --- | ----- | ------- |
| core | — | | | | | | | | | | | |
| config | ✓ | — | ✓ | | | | | | | | | |
| models | ✓ | | — | | | | | | | | | |
| io | ✓ | ✓ | ✓ | — | | | | | | | ✓ | ✓* |
| analysis | ✓ | | ✓ | | — | | | | | | ✓ | |
| generators | ✓ | ✓ | ✓ | | ✓ | — | ✗ | | | | ✓ | ✓ |
| validation | ✓ | ✓ | ✓ | | ✓† | ✗ | — | | | | ✓ | ✓ |
| benchmark | ✓ | ✓ | ✓ | | ✓ | ✓ | ✓ | — | | | ✓ | |
| visualization | ✓ | | ✓ | | ✓* | | ✓* | ✓* | — | | | |
| cli | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| utils | ✓ | | | | | | | | | | — | |
| privacy | ✓ | | ✓ | | | | | | | | ✓ | — |

\* optional/read-only consumption of analysis or report DTOs  
† validation reuses analysis statistics (sequences, n-grams, timing) instead
of re-implementing them; analysis never imports generators, so ADR-005 holds  
✗ **forbidden**

### 8.3 Forbidden dependencies (normative)

1. `generators` → `validation` (prevents evaluation leakage into training code)
2. `validation` → `generators` (metrics must be generator-agnostic)
3. `models` → any algorithmic package
4. `utils` → `cli` / `generators` / `validation`
5. Any package → `cli` (CLI is a leaf adapter)
6. Circular imports across packages (enforced in CI)

### 8.4 Import hygiene

- Prefer importing from package `__init__` public exports, not deep private modules
- `_internal` modules may change without SemVer notice
- Third-party heavy imports inside `visualization` and future `generators.dl`
  should be lazy to keep core import time low

---

## 9. Domain Model

### 9.1 Entity-relationship overview

```text
┌─────────────┐       contains        ┌─────────────┐
│   Dataset   │──────────────────────►│  LogRecord  │
│             │                       │  (Event)    │
│  metadata   │       contains        └──────┬──────┘
│  schema_ver │──────────────────────►┌──────▼──────┐
└──────┬──────┘                       │   Session   │
       │                              └──────┬──────┘
       │ owned-by conceptually               │
       ▼                                     ▼
┌─────────────┐                       ┌─────────────┐
│ Participant │◄──────────────────────│  (learner)  │
└─────────────┘                       └─────────────┘

Analysis produces: Feature, Statistics, navigation structures
Generation produces: SyntheticDataset (subtype/role of Dataset)
Evaluation produces: ValidationReport, BenchmarkReport
Control plane: Configuration, GeneratorModel
```

### 9.2 Dataset

| Aspect | Specification |
| ------ | ------------- |
| **Attributes** | `dataset_id`, `schema_version`, `events`, `sessions?`, `metadata` (source description, privacy notes, fingerprints, eduloggen_version, created_at, seeds) |
| **Responsibilities** | Aggregate events/sessions; expose counts; provide immutable snapshot semantics; carry provenance |
| **Relationships** | Contains many `LogRecord`; optionally many `Session`; references `Configuration` fingerprint used to build it |

### 9.3 LogRecord (Event)

| Aspect | Specification |
| ------ | ------------- |
| **Attributes** | `event_id`, `learner_id`, `timestamp` (UTC aware), `session_id?`, `activity_id`, `event_type`, `course_id?`, `score?`, `success?`, `duration_ms?`, `metadata?` |
| **Responsibilities** | Represent one interaction; remain schema-valid; support ordering by `(learner_id, timestamp)` |
| **Relationships** | Belongs to one `Participant`; optionally one `Session`; belongs to one `Dataset` |

### 9.4 Session

| Aspect | Specification |
| ------ | ------------- |
| **Attributes** | `session_id`, `learner_id`, `start_time`, `end_time`, `n_events`, `duration_s`, `event_sequence`, `course_id?` |
| **Responsibilities** | Bound an episode; provide token sequence for generators; derive duration/length features |
| **Relationships** | Owned by `Participant`; composed of ordered `LogRecord`s |

### 9.5 Event

In EduLogGen, **Event** is synonymous with **LogRecord** at the domain layer.
APIs may use either name; documentation prefers `LogRecord` for ingested rows
and `event` in narrative about sequences.

### 9.6 Participant

| Aspect | Specification |
| ------ | ------------- |
| **Attributes** | `learner_id` (pseudonymous), optional cohort/course labels, aggregate profile pointers |
| **Responsibilities** | Identity boundary for sessionization and profile analysis; never stores raw PII fields in canonical model |
| **Relationships** | Has many `Session`s and `LogRecord`s |

### 9.7 Feature

| Aspect | Specification |
| ------ | ------------- |
| **Attributes** | `name`, `scope` (`event`\|`session`\|`learner`\|`corpus`), `value`, `dtype`, optional `params` |
| **Responsibilities** | Carry derived measurements used by analysis summaries or ML tabular generators later |
| **Relationships** | Produced by analysis extractors from `Dataset`/`Session` |

### 9.8 Statistics

| Aspect | Specification |
| ------ | ------------- |
| **Attributes** | Named summary objects: marginal counts, transition matrices, timing summaries, length histograms, graph metrics |
| **Responsibilities** | Compact, serializable description of corpus behavior for reports and model fitting |
| **Relationships** | Contained in `AnalysisResult`; consumed by generators and visualizers |

### 9.9 SyntheticDataset

| Aspect | Specification |
| ------ | ------------- |
| **Attributes** | Same structural shape as `Dataset`, plus `generation_metadata` (generator_id, model fingerprint, seed, n_sessions, id_strategy) |
| **Responsibilities** | Mark synthetic provenance; default remapped IDs; remain schema-compatible with real datasets for validation |
| **Relationships** | Produced by `GeneratorModel.generate`; compared against real `Dataset` |

### 9.10 ValidationReport

| Aspect | Specification |
| ------ | ------------- |
| **Attributes** | `run_id`, dataset fingerprints, list of `MetricResult`, thresholds, global `status`, timings, versions |
| **Responsibilities** | Persist evaluation outcome; drive CLI exit codes; feed dashboards |
| **Relationships** | References real & synthetic dataset fingerprints; aggregates metrics |

### 9.11 BenchmarkReport

| Aspect | Specification |
| ------ | ------------- |
| **Attributes** | protocol name/version, per-generator metric tables, seeds, runtime/memory, environment metadata |
| **Responsibilities** | Fair comparison artifact for papers and CI |
| **Relationships** | Contains multiple `ValidationReport` slices or equivalent metric matrices |

### 9.12 Configuration

| Aspect | Specification |
| ------ | ------------- |
| **Attributes** | Nested sections: project, logging, io/ingestion, sessionization, analysis, generator, generation, validation, visualization, benchmark, privacy |
| **Responsibilities** | Fully describe a run; be fingerprintable; be publishable without secrets |
| **Relationships** | Inputs to every application service; stored alongside artifacts |

### 9.13 GeneratorModel

| Aspect | Specification |
| ------ | ------------- |
| **Attributes** | `generator_id`, versions, hyperparameters, vocabulary, parameters, training_fingerprint, fitted_at |
| **Responsibilities** | Persist fitted behavior; support `describe`; enable reloadable sampling |
| **Relationships** | Produced by `fit`; consumed by `generate`; independent of raw training rows after fit |

---

## 10. Data Flow

### 10.1 Lifecycle diagram

```text
 [Source files]
       │  FieldMapping
       ▼
  Reader.read ──► [LogRecord stream]
       │
       ▼
  Schema validate / sort / dedupe
       │
       ▼
  SessionBuilder ──► [Dataset: events+sessions]
       │
       ├──────────────► Writer (corpus artifact)
       │
       ▼
  Analyzer ──► [AnalysisResult / Statistics / Features]
       │
       ▼
  Generator.fit ──► [GeneratorModel artifact]
       │
       ▼
  Generator.generate(seed) ──► [SyntheticDataset]
       │
       ├──────────────► Writer (synthetic corpus)
       │
       ▼
  Validator.compute ──► [ValidationReport]
       │
       ▼
  Benchmark.aggregate ──► [BenchmarkReport]
       │
       ▼
  Visualizer / Reporter ──► figures + Markdown/JSON
```

### 10.2 Transformations

| Stage | Transformation |
| ----- | -------------- |
| Map | Source columns → canonical fields |
| Coerce | Strings/numbers → typed values; timestamps → UTC |
| Clean | Drop/repair duplicates per policy |
| Sessionize | Events → sessions + sequences |
| Featurize | Sessions/events → Feature sets |
| Estimate | Sequences/timings → Statistics / behavior model |
| Fit | Statistics + data → GeneratorModel |
| Sample | GeneratorModel + seed → SyntheticDataset |
| Score | (real, synthetic) → MetricResult[] |
| Aggregate | Many runs → BenchmarkReport |

### 10.3 Validation steps (gates)

```text
Gate A (ingest):   schema + dtype + timezone + required fields
Gate B (session):  non-empty sessions policy; ordering; duration ≥ 0
Gate C (fit):      minimum support for order-k; vocabulary size checks
Gate D (generate): seed present; n_sessions > 0; ID strategy applied
Gate E (science):  metric thresholds → pass/fail
Gate F (privacy):  duplication / NN indicators evaluated
```

Failure at Gates A–D aborts the pipeline with typed errors. Gate E/F failures
produce reports and non-zero CLI status without corrupting artifacts already
written (unless `--atomic` output mode is enabled).

### 10.4 Artifact store layout (recommended)

```text
runs/<run_id>/
  config.fingerprint.json
  corpus/                 # real normalized
  analysis/
  models/<generator_id>/
  synthetic/
  reports/
    validation.json
    validation.md
    benchmark.json
  figures/
  logs/
```

---

## 11. Analysis Architecture

### 11.1 Overview

Analysis transforms a sessionized `Dataset` into scientific structures required
by generators, validators, and visualizations—without performing sampling.

```text
 Dataset
    │
    ├─► Session extraction (if needed)
    ├─► Feature extraction
    ├─► Transition matrices / n-grams
    ├─► Time statistics
    ├─► Sequence analysis
    ├─► Navigation graphs
    └─► User profiles
           │
           ▼
     AnalysisResult
```

### 11.2 Session extraction

**Policies**

- Explicit `session_id` passthrough
- Idle-timeout segmentation per learner (configurable gap, default documented)
- Hybrid: trust explicit IDs when present, else timeout

**Outputs:** `Session` table; event `session_id` backfill; rejection report for
orphans if configured.

### 11.3 Feature extraction

Scoped features:

| Scope | Examples |
| ----- | -------- |
| Event | event_type one-hots (analysis only), duration |
| Session | length, duration, unique activities, success rate |
| Learner | sessions per learner, mean session length |
| Corpus | vocabulary size, density of transitions |

Features are named, typed, and serializable for later CTGAN-like tabular
generators without redesign.

### 11.4 Behavior modeling (analysis-side)

Produces intermediate behavioral descriptors:

- Start-state distribution
- Transition structure
- Length distribution
- Timing laws (empirical or parametric fits)

These descriptors may be consumed directly by generators or recomputed inside
`fit` for encapsulation; architecture allows sharing via `AnalysisResult` to
avoid double work in benchmarks.

### 11.5 Transition matrices

- Order-`k` count tensors / sparse maps
- Smoothing options recorded in config
- Tokenization mode: `activity_id` vs `event_type` vs composite tokens

### 11.6 Time statistics

- Inter-event deltas within sessions
- Sojourn times per token (optional)
- Session duration distribution
- Optional diurnal features (P1)

### 11.7 Sequence analysis

- n-gram frequencies
- Session motif mining (optional P1)
- Rare sequence identification (feeds privacy indicators)

### 11.8 Navigation graphs

Directed graph where nodes are activities (or types) and weighted edges are
transitions. Used for visualization (Sankey/heatmap) and behavioral metrics
(path entropy, reciprocity—metric-dependent).

### 11.9 User profiles

Aggregates per `Participant` for stratification and future conditional
generators. Profiles must use pseudonymous IDs only and avoid free-text PII.

---

## 12. Generator Architecture

### 12.1 Abstract framework

All generators implement a uniform lifecycle so benchmarks and CLI remain
generator-agnostic.

```text
                    «abstract»
                  BaseGenerator
                        △
          ┌─────────────┼─────────────────────────────┐
          │             │                             │
   Statistical    Probabilistic                 DeepLearning
   Family         /Behavioral Family            Family (post-1.0)
          │             │                             │
     Frequency     Markov / Semi-Markov         CTGAN/TVAE/...
     baselines     Hybrid wrappers              Transformers
```

### 12.2 BaseGenerator lifecycle

| Method | Responsibility |
| ------ | -------------- |
| `fit(dataset, config) -> GeneratorModel` | Estimate parameters; validate support; set fingerprints |
| `generate(model, n_sessions, seed, config) -> SyntheticDataset` | Seeded sampling; ID strategy; schema-compliant output |
| `save(model, path)` | Persist versioned artifact (inspectable format preferred) |
| `load(path) -> GeneratorModel` | Restore with schema/version checks |
| `describe(model)` | Summarize hyperparameters and sizes for logs/papers |

API naming note: PRD also uses `sample`; architecture treats `generate` as the
canonical method name, with `sample` as an allowed alias re-export for
ergonomics.

### 12.3 Lifecycle sequence

```text
User/CLI
  │ fit(dataset)
  ▼
BaseGenerator.fit
  │ builds GeneratorModel
  ▼
save(path) ──► artifact store
  │
  │ generate(n, seed)
  ▼
SyntheticDataset ──► validate / export
```

### 12.4 Generator families

#### Statistical

Non-sequential or lightly structured baselines (e.g., i.i.d. event-type
sampling, independent length draws). Useful as benchmark lower bounds.

#### Probabilistic

Explicit probabilistic sequence models: Markov chains of order `k`,
Semi-Markov models with timing distributions. **v1.0 flagship family.**

#### Behavioral

Models emphasizing pedagogical/behavioral constructs (hint usage, attempt
loops, navigation intents). May wrap probabilistic cores with behavioral
constraints or post-filters. Partially overlapping with probabilistic family;
tagged `behavioral` when constraints are first-class.

#### Deep learning

Neural generators (CTGAN, TVAE, TimeGAN, Sequence VAE, Transformers, Diffusion).
**Post-1.0**, optional `[dl]` extra, same `BaseGenerator` contract.

#### Hybrid

Pipelines composing families (e.g., Markov structure + neural timing; or
tabular CTGAN on session features + sequence expander). Hybrids are
orchestrators implementing `BaseGenerator` by composing child strategies.

### 12.5 v1.0 built-ins

| Generator | Family | Notes |
| --------- | ------ | ----- |
| `markov` | Probabilistic | Order-k transitions + length model |
| `semi_markov` | Probabilistic | Transitions + sojourn/inter-event times |

### 12.6 Extension mechanism

1. Subclass `BaseGenerator` (or implement Protocol)
2. Advertise capability tags (`supports_timing`, `conditional`, …)
3. Register via built-in registry, entry point, or `register_generator(name, factory)`
4. Pass contract tests in CI template for external plugins

Generators must not read vendor files; they consume `Dataset` / analysis
artifacts only.

---

## 13. Plugin System

### 13.1 Goals

Dynamic, deterministic, contract-checked extension without core modification.

### 13.2 Plugin kinds

| Kind | Contract |
| ---- | -------- |
| `reader` / `ingestor` | `BaseReader` |
| `writer` | `BaseWriter` |
| `sessionizer` | session policy strategy |
| `generator` | `BaseGenerator` |
| `metric` | `BaseMetric` |
| `visualizer` | `BaseVisualizer` |
| `benchmark_suite` | `BenchmarkProtocol` |

### 13.3 Discovery

```text
Startup / first use
    │
    ├─ load built-in registrations
    ├─ scan importlib.metadata entry points: eduloggen.plugins.<kind>
    ├─ apply explicit register_* calls (notebooks, proprietary)
    ▼
Registry(name → factory, metadata)
    │
    ▼
Contract validation (issubclass / structural checks)
```

Discovery order is documented and stable. Name collisions raise `PluginError`
unless explicitly overridden with `--force-plugin` style opts (discouraged).

### 13.4 Registration API (conceptual)

```text
register_plugin(kind: PluginKind, name: str, factory: Callable, metadata: PluginMeta) -> None
list_plugins(kind: PluginKind | None) -> list[PluginMeta]
create_plugin(kind: PluginKind, name: str, **kwargs) -> Any
```

### 13.5 Future external plugins

External packages publish entry points in their `pyproject.toml` without
depending on EduLogGen internals beyond public contracts. EduLogGen may host a
plugin catalog in docs; it does not vendor third-party plugins into core.

### 13.6 Safety

- Plugins execute in-process (v1.0); trust boundaries are packaging/supply-chain
- Plugins may only write to user-specified output directories
- No automatic download of plugin code at runtime

---

## 14. Validation Architecture

### 14.1 Layers of validation

| Layer | Purpose |
| ----- | ------- |
| Statistical | Marginal & distributional similarity |
| Behavioral | Sequential/session/navigation realism |
| Privacy | Memorization / linkage risk indicators |
| Utility | Downstream task smoke tests (P1) |

```text
 Real Dataset ──┐
                ├──► Metric.set.compute ──► MetricResult[] ──► ValidationReport
 Synthetic ─────┘                              │
                                        thresholds
                                               ▼
                                         status pass/fail
```

### 14.2 Statistical validation

Examples: total variation distance, Jensen–Shannon divergence, χ² tests on
event/activity frequencies; KS/Wasserstein on continuous timing/length
distributions. Exact metric list lives in metrics catalog; architecture
requires at least one P0 metric per PRD category.

### 14.3 Behavioral validation

Transition/bigram divergence; session length/duration similarity; optional
path-level metrics on navigation graphs; rare-motif frequency deltas.

### 14.4 Privacy validation

- Exact session duplication rate against real train set
- Nearest-neighbor distance heuristics on sequence embeddings/hash space
- Rare-sequence replay rate

Privacy metrics never claim zero risk; they inform governance.

### 14.5 Utility validation

Train a simple next-event predictor (or similar) on real vs synthetic and
compare accuracy/calibration gaps (P1). Kept optional to avoid heavy deps in
default validate.

### 14.6 Evaluation metrics contract

```text
MetricResult:
  name, value, direction, ci?, details, status?, threshold?
```

### 14.7 Benchmark methodology

Protocol `session_fidelity_v1`:

1. Split or declare evaluation reference set
2. Fit each generator on train portion with recorded config
3. Generate N sessions × R seeds
4. Compute metric matrix
5. Aggregate mean/std; record runtime/memory best-effort
6. Emit `BenchmarkReport`

Fairness rules: identical sessionization, tokenization, and sample sizes across
candidates unless the protocol explicitly varies them.

---

## 15. Visualization Architecture

### 15.1 Role

Optional rendering layer over datasets, analysis results, and reports. Core
pipelines remain functional if `[viz]` extra is absent.

### 15.2 Modules

| Module | Description |
| ------ | ----------- |
| **Timeline** | Event timelines per session (sampled, privacy-safe) |
| **Transition Graph** | Directed graph of activity transitions |
| **Sankey** | Flow across top-N activities / event types |
| **Heatmaps** | Transition matrices; temporal heatmaps |
| **Distributions** | Overlaid real vs synthetic histograms/ECDFs |
| **Session Statistics** | Length/duration/activity-mix panels |
| **Benchmark Dashboard** | Multi-generator metric comparison charts |

### 15.3 Architecture

```text
 Dataset / AnalysisResult / ValidationReport / BenchmarkReport
                         │
                         ▼
                 BaseVisualizer
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   static backend   interactive     export adapters
   (default)        (optional)      PNG/SVG/PDF
```

### 15.4 Constraints

- No raw PII in labels/titles
- Colorblind-friendly defaults where practical
- Deterministic styling configs for paper reproducibility

---

## 16. Configuration Architecture

### 16.1 Hierarchy (highest wins)

```text
 CLI flags
    │ overrides
 Env vars (EDULOGGEN_*)
    │ overrides
 Config file (YAML | TOML | JSON)
    │ overrides
 Built-in defaults
```

### 16.2 Sources

| Source | Use |
| ------ | --- |
| YAML | Human-authored experiments |
| TOML | Packaged with pyproject-adjacent workflows |
| JSON | Machine-generated pipelines |
| Environment | CI secrets-free toggles (paths, log level)—allowlisted keys only |
| CLI | One-off overrides |

### 16.3 Typed config model

Parsed into immutable `AppConfig` with sections matching PRD §16.4. Strict
unknown-key rejection by default.

### 16.4 Fingerprinting

Canonical JSON serialization of resolved config (with path normalization
policy) hashed into artifacts for reproducibility.

### 16.5 Example logical sections

`project`, `logging`, `io`, `sessionization`, `analysis`, `generator`,
`generation`, `validation`, `visualization`, `benchmark`, `privacy`.

---

## 17. Logging Architecture

### 17.1 Principles

- Standard library `logging`
- Library loggers under `eduloggen.*` **do not** configure root handlers on import
- CLI/apps attach handlers

### 17.2 Structured logging

When enabled (`--json-logs` / config), each record includes: `time`, `level`,
`logger`, `message`, `run_id`, optional `event` key, and safe fields (counts,
durations, names)—never emails/names/free-text posts.

### 17.3 Debug logging

DEBUG traces algorithm stages, chunk progress, plugin discovery. Guardged so
default INFO remains readable.

### 17.4 Research experiment logging

Experiments log: seed, generator_id, dataset fingerprint, metric summaries,
artifact paths. Encourages paper appendices without ad-hoc print statements.

### 17.5 Benchmark logging

Benchmark runner emits per-generator start/end, seed indices, metric
completion, and resource snapshots. Benchmark logs are attachable to
`BenchmarkReport` metadata.

### 17.6 Logger topology

```text
eduloggen
├── eduloggen.io
├── eduloggen.analysis
├── eduloggen.generators
├── eduloggen.validation
├── eduloggen.benchmark
├── eduloggen.visualization
├── eduloggen.plugins
└── eduloggen.cli
```

---

## 18. Error Handling

### 18.1 Exception hierarchy

```text
EduLogGenError
├── ConfigError
├── SchemaError
├── IoError (IngestionError / ExportError)
├── AnalysisError
├── FitError
├── GenerationError
├── ValidationError          # metric/threshold domain failures may be soft
├── BenchmarkError
├── PluginError
└── InternalError            # unexpected invariants
```

### 18.2 Category semantics

| Error | Typical cause | User action |
| ----- | ------------- | ----------- |
| ConfigError | Bad YAML/types/unknown keys | Fix config |
| SchemaError | Missing fields/dtypes | Fix mapping/source |
| IoError | Missing files/permissions/format | Fix paths/format |
| FitError | Insufficient data/order too high | Adjust params/data |
| GenerationError | Bad seed/n/model incompatible | Fix generation config |
| ValidationError | API misuse of validate; not merely fail thresholds | Fix call |
| PluginError | Contract/discovery failure | Fix plugin packaging |

### 18.3 Soft vs hard failures

Threshold failures in scientific validation are **report statuses**, not
necessarily thrown exceptions. CLI translates global fail → exit code 1.
Programming misuse throws.

### 18.4 Error content

Exceptions expose: human message, machine `code`, optional `context` dict
(paths, field names)—never raw PII payloads.

---

## 19. CLI Architecture

### 19.1 Command hierarchy

```text
eduloggen
├── ingest
├── sessionize
├── analyze
├── fit
├── generate
├── validate
├── benchmark
├── visualize | plot
├── plugins
└── info
```

Global options: `--version`, `-v/--verbose`, `--quiet`, `--config`, `--seed`,
`--json-logs`, `--force` (overwrite).

### 19.2 Command specifications

#### analyze

- **Args:** `--input` corpus, `--config`, `--output` analysis dir
- **Out:** `AnalysisResult` artifact + summary Markdown

#### generate

- **Args:** `--model`, `--n-sessions`, `--seed`, `--output`, ID strategy flags
- **Out:** `SyntheticDataset` corpus package

#### validate

- **Args:** `--real`, `--synthetic`, `--metrics`, `--thresholds`, `--output`
- **Out:** `ValidationReport` JSON/MD; exit 0/1/2

#### benchmark

- **Args:** `--protocol`, `--generators`, `--input`, `--output`, repeats/seeds
- **Out:** `BenchmarkReport` + tables

#### visualize

- **Args:** `--input` (dataset/report), `--plot` names, `--output` dir
- **Out:** figure files

### 19.3 CLI layering

```text
argv → parser → AppConfig merge → application service → domain packages → exit code
```

No business logic in parsers beyond validation of argument shapes.

---

## 20. Python API

### 20.1 Public services (façade)

Stable, notebook-friendly functions (names normative intent for 1.0 freeze):

| Service | Responsibility |
| ------- | -------------- |
| `load_dataset` / `read_events` | IO + schema validation |
| `sessionize` | Session builder |
| `analyze` | AnalysisResult |
| `fit_generator` | name + dataset → GeneratorModel |
| `generate` | model + seed → SyntheticDataset |
| `validate` | real vs synthetic → ValidationReport |
| `run_benchmark` | protocol execution → BenchmarkReport |
| `plot_*` | visualization helpers |
| `load_config` | configuration |
| `list_plugins` / `register_plugin` | extension |

### 20.2 Stability

| Tier | SemVer rule |
| ---- | ----------- |
| Documented public API | Breaking changes require MAJOR |
| `eduloggen.experimental` | May break in MINOR with warnings |
| `_private` | Unstable |

### 20.3 API stability process

1. Propose in ADR/PR
2. Document in MkDocs API reference
3. Provide deprecation window when possible
4. Update changelog

### 20.4 Interop

DataFrame export/import helpers for events/sessions; Parquet preferred for
large corpora; CSV for accessibility.

---

## 21. Testing Architecture

### 21.1 Objectives

Testing is an architectural concern: seams exist so behavior is verifiable.
**Target coverage: >95%** of core logic packages (`core`, `models`, `io`,
`analysis`, `generators`, `validation`, `benchmark`, `config`, `privacy`,
`utils`). Visualization backends may exclude pure rendering glue with
documented pragmas; logic that selects data for plots remains covered.

### 21.2 Test kinds

| Kind | Scope | Examples |
| ---- | ----- | -------- |
| **Unit** | Functions/classes in isolation | session gap splits; metric math; config merge |
| **Integration** | Multi-package pipelines | ingest→sessionize→fit→generate→validate |
| **Property** | Invariants via hypothesis/settings | sorted timestamps; durations ≥ 0; seed stability |
| **Regression** | Golden behaviors | known transition counts on fixtures |
| **Performance** | Budgets | fit time on medium synthetic corpus |
| **Benchmark tests** | Protocol smoke | `session_fidelity_v1` on tiny fixture |
| **Contract** | Plugin ABI | built-in generators satisfy BaseGenerator |
| **CLI** | Exit codes/args | validate fail → exit 1 |

### 21.3 Layout

```text
tests/
├── unit/
├── integration/
├── property/
├── performance/
├── contract/
├── cli/
└── fixtures/          # synthetic only
```

### 21.4 Coverage architecture

- `pytest-cov` in CI with fail-under calibrated to **>95%** for core paths
- Coverage reports uploaded as artifacts
- Forbidden to lower threshold silently; requires ADR

### 21.5 Privacy tests

Fixtures scanned for email-like patterns; assertions that default generation
does not echo real learner IDs.

### 21.6 Reproducibility tests

Identical seeds ⇒ identical synthetic token sequences on same platform/version.

---

## 22. Documentation Architecture

### 22.1 Documentation system

```text
docs/ (markdown sources)
   │
   ▼
MkDocs Material + mkdocstrings
   │
   ▼
site/ (generated, not committed)
   │
   ▼
GitHub Pages / equivalent deploy
```

### 22.2 Doc types

| Type | Content |
| ---- | ------- |
| **MkDocs site** | User-facing navigation |
| **API docs** | Generated from Google-style docstrings |
| **Examples** | `examples/` scripts |
| **Tutorials** | `notebooks/` + narrative guides |
| **Developer guide** | Contribution, architecture summary, dependency rules |
| **Research guide** | Metrics definitions, benchmark protocols, citation |

### 22.3 Governance docs

Vision, PRD, and this SAD remain canonical. ADRs under `docs/adr/` capture
evolving decisions; significant ADRs are summarized in §29.

### 22.4 Definition of done (docs)

P0 features require: narrative docs, API/CLI references, example or test, changelog.

---

## 23. CI/CD Architecture

### 23.1 Pipeline overview

```text
PR / push to main
    │
    ├─► quality matrix (py 3.11–3.13)
    │     Ruff · Black · MyPy · PyTest+coverage
    ├─► build (sdist/wheel + twine check)
    ├─► docs (mkdocs build --strict)
    └─► (optional) performance smoke
                │
tag vX.Y.Z ─────► release workflow
                ├─ build & publish PyPI
                └─ deploy docs for tag
```

### 23.2 GitHub Actions responsibilities

| Job | Purpose |
| --- | ------- |
| Lint/format | Ruff + Black |
| Typing | MyPy strict |
| Tests | PyTest + coverage gate |
| Build | Packaging integrity |
| Docs | Prevent broken references |
| Release | Trusted publishing / tokens as configured |

### 23.3 Local parity

Pre-commit mirrors a subset of CI to reduce round-trips.

### 23.4 Documentation deployment

Deploy only from trusted branches/tags; published docs versioned to match
releases where feasible.

### 23.5 Release workflow gates

Changelog updated; CI green; version bump; tag; PyPI; docs; smoke install.

---

## 24. Extension Points

Contributors extend EduLogGen **without modifying existing core modules** by
implementing contracts and registering plugins.

### 24.1 New readers

1. Implement `BaseReader`
2. Register `eduloggen.plugins.reader` entry point
3. Provide mapping examples + contract tests
4. Document source assumptions

### 24.2 New generators

1. Implement `BaseGenerator` lifecycle
2. Register plugin + capability tags
3. Add unit + reproducibility tests
4. Optionally add benchmark suite entry

### 24.3 New validators (metrics)

1. Implement `BaseMetric.compute`
2. Declare direction and default thresholds (optional)
3. Register `eduloggen.plugins.metric`
4. Document statistical assumptions

### 24.4 New plots

1. Implement `BaseVisualizer` or plot function
2. Register visualizer plugin
3. Ensure PII-safe labels
4. Add smoke test with synthetic fixture

### 24.5 New benchmarks

1. Define `BenchmarkProtocol` (splits, metrics, generators list)
2. Register `benchmark_suite`
3. Provide tiny fixture path for CI smoke

### 24.6 Extension diagram

```text
External package
   │ entry points
   ▼
Plugin Registry ──► Factories
   │
   ▼
Pipelines (unchanged) use abstractions
```

---

## 25. Security

### 25.1 Threat model (software)

EduLogGen runs locally with access to user-selected files. Threats include:
accidental PII leakage into logs/artifacts/repos; path traversal via crafted
configs; unsafe deserialization; supply-chain compromise of dependencies.

### 25.2 Secure file handling

- Resolve and normalize paths; reject writes outside declared output roots when
  sandbox mode enabled
- Require `--force` to overwrite
- Do not follow untrusted symlinks in output trees when mode enabled (platform
  best effort)

### 25.3 Dependency management

- Pin ranges thoughtfully in packaging; prefer minimal core deps
- Optional heavy stacks isolated in extras
- CI Dependabot/renovate recommended
- No install-time arbitrary script execution beyond PEP 517 builds

### 25.4 Unsafe serialization avoidance

- **Default artifact format:** versioned JSON/JSON-lines + array storage
  (e.g., NumPy `.npz` only when necessary) or Parquet for tables
- **Pickle:** disabled by default for `load`; if ever offered, require explicit
  `allow_pickle=True` and local trusted paths only
- Deep learning checkpoints confined to `[dl]` extras with documented risks

### 25.5 Reproducibility as security-adjacent property

Fingerprints and seeds make silent tampering of configs more detectable in
scientific workflows; they are not cryptographic attestation.

### 25.6 Privacy engineering

Aligns with PRD privacy requirements: local-first, synthetic fixtures, default
ID remapping, privacy metrics, responsible-use documentation.

---

## 26. Performance

### 26.1 Memory strategy

- Prefer columnar tables for events
- Session sequences may be stored compactly (token ids, not repeated strings)
- Release references after stage boundaries in long pipelines
- Document peak-memory expectations per corpus size

### 26.2 Streaming readers

`BaseReader` supports iterator mode yielding records/chunks so Gate A can run
without full materialization. Sessionization may still require per-learner
windows; architecture allows spill-to-disk strategies later.

### 26.3 Chunk processing

Analysis of transitions can aggregate counts chunk-wise (sufficient statistics)
for Markov fits—key to scaling beyond RAM for v1 probabilistic models.

### 26.4 Parallelism

| Opportunity | Approach |
| ----------- | -------- |
| Independent learners sessionization | optional process/thread pool |
| Multi-seed generation | parallel map with distinct seeds |
| Benchmark generators | sequential by default for fair timing; optional parallel with caveats |

Default v1.0 favors determinism over aggressive parallelism.

### 26.5 Caching

Cache `AnalysisResult` and fitted models keyed by `(dataset_fingerprint, config_fingerprint)`.
CLI `--recompute` bypasses caches.

### 26.6 Performance diagram

```text
Large file ── stream chunks ── accumulate counts ── fit model
                 │
                 └── optional spill / cache
```

---

## 27. Future Architecture

Seams reserved without implementing now:

| Capability | Architectural seam |
| ---------- | ------------------ |
| **CTGAN / TVAE** | Tabular feature generators behind `BaseGenerator`; `[dl]` extra |
| **TimeGAN / Sequence VAE / Transformer / Diffusion** | Sequence DL generators; shared training callbacks interface |
| **Federated learning** | Replace local `fit` data access with federated client protocol adapter; core metrics unchanged |
| **Distributed benchmarking** | Benchmark runner orchestrates workers; report aggregation remains central |
| **REST API** | HTTP adapter calling the same application services as CLI |
| **Web dashboard** | Consumes `BenchmarkReport` / figures; no domain logic duplication |
| **Cloud execution** | Container entrypoint invoking CLI; still local-data mounts by default |

```text
Future adapters                 Stable core
┌──────────┐                   ┌─────────────────────┐
│ REST     │──┐                │ Application services│
│ Web UI   │──┼───────────────►│ Domain packages     │
│ Cloud job│──┘                │ Plugins             │
└──────────┘                   └─────────────────────┘
```

Principle: **adapters change; domain contracts endure.**

---

## 28. Design Patterns

| Pattern | Where applied | Purpose |
| ------- | ------------- | ------- |
| **Factory** | Plugin `create_plugin`, generator factories | Name → instance without hardcoding classes in CLI |
| **Strategy** | Sessionizers, metrics, generators, readers | Interchangeable algorithms behind one interface |
| **Registry** | Central plugin maps | Discovery and lookup |
| **Builder** | Dataset/session assembly during ingest | Incremental construction then freeze |
| **Adapter** | CLI/REST/future UI; LMS column maps | Integrate external shapes to canonical model |
| **Observer** | Optional progress callbacks / logging events | Decouple long-running jobs from UX |
| **Template Method** | `BaseGenerator` lifecycle | Enforce fit/generate/save/load structure |
| **Dependency Injection** | Pass configs, RNG, registries into services | Testability and explicit dependencies |

### 28.1 Pattern collaboration

```text
CLI Adapter → Service (DI: config, registry) → Strategy(Generator)
                                 │
                                 └─► Registry/Factory resolves Strategy
```

---

## 29. Architecture Decision Records

The following ADRs are **accepted** for EduLogGen 1.0 architecture. Future
changes to these decisions require a superseding ADR.

### ADR-001 — Src layout and single package name

- **Context:** Need installable library with clean imports and test isolation.
- **Decision:** Use `src/eduloggen` with Hatchling; package name `eduloggen`.
- **Alternatives:** Flat layout; multiple packages (`eduloggen-core`, etc.).
- **Consequences:** Standard tooling; clearer packaging; slightly more verbose
  paths for newcomers.

### ADR-002 — Canonical sessionized event model

- **Context:** Educational logs vary wildly by vendor.
- **Decision:** Define a canonical `LogRecord`/`Session`/`Dataset` model; map
  sources via configuration.
- **Alternatives:** Keep vendor schemas end-to-end; RDF/graph-native store.
- **Consequences:** One validation/generation stack; mapping burden upfront.

### ADR-003 — Local-first, no hosted data plane

- **Context:** Privacy sensitivity of learner logs.
- **Decision:** Architecture assumes local process + filesystem; no telemetry.
- **Alternatives:** Managed SaaS; automatic cloud uploads.
- **Consequences:** Trust improved; collaboration features deferred to user.

### ADR-004 — Plugin registry with entry points

- **Context:** Extensibility without forks.
- **Decision:** Built-ins + `importlib.metadata` entry points + explicit API.
- **Alternatives:** Convention-based folder plugins; dynamic remote install.
- **Consequences:** Familiar packaging UX; supply-chain responsibility on users.

### ADR-005 — Generators isolated from validators

- **Context:** Risk of leakage and circular deps.
- **Decision:** Forbid `generators ↔ validation` dependencies; both use
  `Dataset`.
- **Alternatives:** Generators embed evaluation hooks.
- **Consequences:** Cleaner science; slightly more plumbing in benchmarks.

### ADR-006 — Markov & Semi-Markov as v1 flagships

- **Context:** Vision scope for 1.0; need interpretable baselines.
- **Decision:** Ship Markov and Semi-Markov first; DL later as extras.
- **Alternatives:** Jump to transformers immediately.
- **Consequences:** Faster reliable 1.0; DL users wait; better benchmarks.

### ADR-007 — Configuration precedence CLI > env > file > defaults

- **Context:** Research reproducibility vs convenience.
- **Decision:** Documented precedence with typed strict configs.
- **Alternatives:** File-only; env-only 12-factor pure.
- **Consequences:** Predictable overrides; must teach precedence in docs.

### ADR-008 — Immutable snapshots after stage gates

- **Context:** Accidental mutation corrupts experiments.
- **Decision:** Freeze domain objects after ingest/sessionize; copy-on-write
  for transforms.
- **Alternatives:** Fully mutable dataframes everywhere.
- **Consequences:** Safer reasoning; possible copy costs (mitigate with
  columnar sharing).

### ADR-009 — Artifact format prefers inspectable serialization

- **Context:** Pickle security and opacity issues.
- **Decision:** Default to versioned inspectable formats; pickle opt-in only.
- **Alternatives:** Pickle-first; opaque binary blobs.
- **Consequences:** Slightly more schema work; better auditability.

### ADR-010 — Visualization is an optional extra

- **Context:** Headless CI and minimal installs matter.
- **Decision:** Core pipelines run without viz deps; `visualize` command
  requires `[viz]`.
- **Alternatives:** Hard-depend on matplotlib/plotly.
- **Consequences:** Clearer extras story; need good error messages.

### ADR-011 — Coverage target >95% for core logic

- **Context:** Scientific software correctness expectations.
- **Decision:** Enforce >95% coverage on core packages in CI.
- **Alternatives:** 80% pragmatic gate forever.
- **Consequences:** Higher quality; more test maintenance; viz exclusions
  documented.

### ADR-012 — Seeded RNG plumbing as architectural requirement

- **Context:** Reproducibility is a Vision pillar.
- **Decision:** Every generate/benchmark path requires explicit seed policy.
- **Alternatives:** Nondeterministic defaults.
- **Consequences:** Verbose APIs; trustworthy experiments.

### ADR-013 — Privacy indicators in default validation

- **Context:** Synthetic data can memorize.
- **Decision:** Include at least duplication/NN-style indicators by default.
- **Alternatives:** Utility-only metrics.
- **Consequences:** More compute; better governance narratives.

### ADR-014 — Application façades over deep imports

- **Context:** Users otherwise couple to internals.
- **Decision:** Public API services wrap packages; CLI calls façades.
- **Alternatives:** Expose all submodules as public.
- **Consequences:** Stable surface; discipline required in `__init__` exports.

### ADR-015 — Benchmark protocols as versioned plugins

- **Context:** Fair comparison needs frozen recipes.
- **Decision:** Protocols are versioned objects (`session_fidelity_v1`).
- **Alternatives:** Ad-hoc scripts per paper only.
- **Consequences:** Enabling shared science; protocol evolution needs care.

### ADR-016 — Error taxonomy with machine codes

- **Context:** CLI/automation needs stable handling.
- **Decision:** Typed exceptions with `code` strings; soft validation statuses
  separate from throws.
- **Alternatives:** Bare `Exception` / boolean returns only.
- **Consequences:** Better UX; more types to maintain.

### ADR-017 — `io` package naming over `ingestion` alone

- **Context:** Need symmetric read/write and corpus packages.
- **Decision:** Primary package name `io`; ingest is a workflow over `io`.
- **Alternatives:** Keep only `ingestion/` forever.
- **Consequences:** Aligns with SAD; migration aliases may be provided.

### ADR-018 — Strict MyPy and Ruff/Black as quality architecture

- **Context:** Open-source consistency across contributors.
- **Decision:** Strict typing + lint/format gates in pre-commit and CI.
- **Alternatives:** Lenient typing; formatter wars.
- **Consequences:** Upfront friction; long-term maintainability.

---


## 29A. Detailed Interface Catalog (Normative Intent)

This section elaborates interfaces so implementers can derive modules without
guesswork. Signatures are architectural contracts, not final source code.

### 29A.1 RunContext

| Field | Type intent | Notes |
| ----- | ----------- | ----- |
| `run_id` | UUID/string | Correlates logs and artifacts |
| `seed` | int \| None | Global default seed |
| `started_at` | datetime | UTC |
| `eduloggen_version` | string | Package version |
| `platform` | mapping | python, os, machine |

### 29A.2 FieldMapping

Maps source column names to canonical fields; supports transforms:

- `rename`
- `parse_datetime(format, timezone_policy)`
- `cast`
- `default`
- `hash` (privacy)
- `drop`

Validation fails if required canonical fields remain unmapped.

### 29A.3 Dataset persistence layout

```text
corpus_dir/
  manifest.json          # schema_version, counts, fingerprints, privacy_notes
  events.parquet         # or events.csv
  sessions.parquet       # optional
  mapping.used.yaml      # optional provenance of FieldMapping
  quality_report.json    # ingest Gate A findings
```

### 29A.4 GeneratorModel persistence layout

```text
model_dir/
  manifest.json          # generator_id, versions, fingerprints
  hyperparameters.json
  vocabulary.json
  parameters/            # family-specific (transition counts, dist params)
  DESCRIPTION.md         # optional human summary from describe()
```

### 29A.5 Report schemas (logical)

**MetricResult**

| Field | Meaning |
| ----- | ------- |
| `name` | Stable metric id |
| `value` | float or structured |
| `direction` | `lower_better` \| `higher_better` |
| `threshold` | optional |
| `status` | `pass`\|`fail`\|`warn`\|`skip` |
| `details` | free-form JSON-safe mapping |

**ValidationReport.status** is `fail` if any non-skipped P0 metric fails.

---

## 29B. Pipeline State Machine

```text
                 ┌────────────┐
                 │  EMPTY     │
                 └─────┬──────┘
                       │ ingest OK
                       ▼
                 ┌────────────┐
                 │ RAW_CANON  │  events only
                 └─────┬──────┘
                       │ sessionize OK
                       ▼
                 ┌────────────┐
                 │ SESSIONIZED│
                 └─────┬──────┘
                       │ analyze OK
                       ▼
                 ┌────────────┐
                 │ ANALYZED   │
                 └─────┬──────┘
                       │ fit OK
                       ▼
                 ┌────────────┐
                 │ MODEL_READY│
                 └─────┬──────┘
                       │ generate OK
                       ▼
                 ┌────────────┐
                 │ SYNTHETIC  │
                 └─────┬──────┘
                       │ validate
                       ▼
                 ┌────────────┐
                 │ EVALUATED  │
                 └────────────┘
```

Illegal transitions raise `InternalError` or service-level errors. Benchmark
orchestrates multiple MODEL_READY → SYNTHETIC → EVALUATED branches.

---

## 29C. Analysis Component Details

### 29C.1 Session extraction algorithms

**IdleTimeoutSessionizer**

1. Partition events by `learner_id`
2. Sort by timestamp
3. Start new session when gap > `idle_timeout`
4. Assign monotonic `session_id` per learner or globally unique IDs

**ExplicitSessionizer**

1. Require non-null `session_id`
2. Validate that sessions do not interleave inconsistently per policy
3. Compute start/end/duration/n_events/sequence

**CompositeSessionizer**

Uses explicit IDs when present for a learner; falls back to idle timeout
otherwise. Policy recorded in config fingerprint.

### 29C.2 Feature extractor catalog (v1)

| Feature name | Scope | Description |
| ------------ | ----- | ----------- |
| `session_length` | session | Number of events |
| `session_duration_s` | session | end-start seconds |
| `n_unique_activities` | session | Distinct activity_id |
| `event_type_entropy` | session | Shannon entropy of types |
| `success_rate` | session | Mean of success when present |
| `learner_n_sessions` | learner | Count of sessions |
| `corpus_vocab_size` | corpus | Distinct tokens |
| `corpus_transition_density` | corpus | Nonzero transitions / possible |

### 29C.3 Transition estimation

```text
sequences → count contexts → optional smooth → normalize rows → TransitionModel
```

Backoff policy when order-k unsupported: fail (default) or reduce k with WARNING
if `on_insufficient_data: backoff`.

### 29C.4 Timing estimation

For each sample of Δt:

- Fit parametric family selected in config, or
- Store empirical resampling reservoir

Store goodness-of-fit diagnostics in AnalysisResult details (not necessarily
P0 for generation).

### 29C.5 Navigation graph builder

```text
nodes = tokens
edges[(a,b)] += count
prune to top-N nodes for viz (analysis keeps full sparse graph)
```

### 29C.6 User profile builder

Aggregates learner-scope features; optional clustering hooks are post-1.0 and
must not be required for Markov fit.

---

## 29D. Generator Family Specifications

### 29D.1 Statistical family

**IndependentEventGenerator** (optional baseline)

- Draw session length from empirical distribution
- Draw each token i.i.d. from unigram frequencies
- Purpose: lower-bound benchmark reference

### 29D.2 Markov generator (normative v1)

**Fit**

1. Tokenize sessions per config
2. Estimate start distribution and order-k transitions
3. Estimate length distribution
4. Persist vocabulary and matrices

**Generate**

1. Draw length L
2. Draw start token from start distribution
3. For i in 2..L: draw next from transition row (with smoothing/unseen policy)
4. Materialize timestamps if timing disabled via constant Δt or external timing
5. Apply ID remapping strategy
6. Emit SyntheticDataset

### 29D.3 Semi-Markov generator (normative v1)

Extends Markov with:

1. Draw Δt from timing model after each token (or sojourn before transition)
2. Enforce strictly increasing timestamps
3. Session duration consistency checks (warn if extreme outliers)

### 29D.4 Behavioral family (extension guidance)

Examples of constraints applied as filters or constrained sampling:

- Maximum consecutive failures before forced navigate-away
- Hint-before-resubmit patterns
- Resource prerequisite edges from a course graph plugin

Behavioral generators still expose `BaseGenerator` and record constraints in
hyperparameters.

### 29D.5 Deep learning family (future)

Shared concerns to design now:

- Train/val split hooks
- Early stopping callbacks (Observer pattern)
- Device selection config
- Checkpoint directories under model artifact
- Capability tag `gpu`

They must still export SyntheticDataset in canonical schema.

### 29D.6 Hybrid family (future)

```text
HybridGenerator.fit:
  child_a.fit → model_a
  child_b.fit → model_b
  return HybridModel(model_a, model_b, glue_config)

HybridGenerator.generate:
  structure = child_a.generate_structure(...)
  enrich = child_b.enrich(structure, ...)
  return assemble(enrich)
```

---

## 29E. Validation Metric Catalog (Architectural)

Exact formulas belong in research notes; architecture freezes **metric IDs**
and categories.

### 29E.1 Statistical

| Metric ID | Category | Direction |
| --------- | -------- | --------- |
| `event_type_tvd` | marginal | lower_better |
| `activity_jsd` | marginal | lower_better |
| `session_length_ks` | session structure | lower_better |
| `session_duration_w1` | session structure | lower_better |
| `interevent_time_ks` | temporal | lower_better |

### 29E.2 Behavioral

| Metric ID | Category | Direction |
| --------- | -------- | --------- |
| `bigram_tvd` | sequential | lower_better |
| `transition_jsd` | sequential | lower_better |
| `topn_path_overlap` | navigation | higher_better |

### 29E.3 Privacy

| Metric ID | Category | Direction |
| --------- | -------- | --------- |
| `exact_session_dup_rate` | privacy | lower_better |
| `rare_ngram_replay_rate` | privacy | lower_better |
| `nn_distance_p05` | privacy | higher_better |

### 29E.4 Utility (P1)

| Metric ID | Category | Direction |
| --------- | -------- | --------- |
| `next_event_acc_gap` | utility | lower_better |

---

## 29F. CLI Argument Reference (Architectural)

### 29F.1 Global

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `--config` | path | YAML/TOML/JSON |
| `--seed` | int | default seed |
| `-v/-vv` | flag | INFO/DEBUG |
| `--quiet` | flag | ERROR only |
| `--json-logs` | flag | structured logs |
| `--force` | flag | overwrite outputs |
| `--run-id` | str | optional override |

### 29F.2 ingest

| Argument | Description |
| -------- | ----------- |
| `--input` | source file(s) |
| `--mapping` | FieldMapping file |
| `--output` | corpus dir |
| `--format` | csv\|parquet\|jsonl\|auto |
| `--strict` | schema strictness |

### 29F.3 fit

| Argument | Description |
| -------- | ----------- |
| `--input` | corpus |
| `--generator` | registry name |
| `--output` | model dir |
| `--set KEY=VAL` | hyperparameter overrides |

### 29F.4 generate

| Argument | Description |
| -------- | ----------- |
| `--model` | model dir |
| `--n-sessions` | int |
| `--seed` | int |
| `--output` | synthetic corpus |
| `--id-strategy` | remap\|preserve (default remap) |

### 29F.5 validate

| Argument | Description |
| -------- | ----------- |
| `--real` | corpus |
| `--synthetic` | corpus |
| `--metrics` | list or `@file` |
| `--thresholds` | file |
| `--output` | report dir |

Exit codes: `0` pass, `1` fail thresholds, `2` usage/runtime error.

### 29F.6 benchmark

| Argument | Description |
| -------- | ----------- |
| `--protocol` | e.g. session_fidelity_v1 |
| `--input` | corpus |
| `--generators` | list |
| `--seeds` | list or count |
| `--output` | report dir |

### 29F.7 visualize

| Argument | Description |
| -------- | ----------- |
| `--input` | corpus or report |
| `--plot` | timeline\|transitions\|sankey\|heatmap\|distributions\|sessions\|dashboard |
| `--output` | figures dir |

---

## 29G. Python API Service Contracts

### 29G.1 Recommended import surface

```text
import eduloggen as elg

cfg = elg.load_config("experiment.yaml")
ds = elg.load_dataset("data/corpus", config=cfg)
ds = elg.sessionize(ds, config=cfg)
analysis = elg.analyze(ds, config=cfg)
model = elg.fit_generator("semi_markov", ds, config=cfg)
synth = elg.generate(model, n_sessions=1000, seed=7, config=cfg)
report = elg.validate(ds, synth, config=cfg)
bench = elg.run_benchmark("session_fidelity_v1", ds, generators=["markov", "semi_markov"], config=cfg)
elg.plot_distributions(ds, synth, path="fig/dist.png")
```

The above is illustrative pseudo-API for implementers and docs writers.

### 29G.2 Thread-safety

Public services are **not** required to be thread-safe on shared mutable
registries during registration. After discovery freeze, read-only generate may
be parallelized across processes with separate RNG streams.

### 29G.3 Versioning of schemas

| Schema | Version field | Compatibility rule |
| ------ | ------------- | ------------------ |
| Dataset manifest | `schema_version` | Readers accept declared compatible range |
| GeneratorModel | `artifact_version` | load() rejects incompatible majors |
| ValidationReport | `report_version` | Consumers tolerate additive fields |

---

## 29H. Operational Scenarios

### 29H.1 Scenario — Paper experiment

1. Researcher prepares mapping for institutional extract on secure machine
2. Runs ingest/sessionize/analyze
3. Fits markov and semi_markov with fixed seeds
4. Generates synthetic corpora with remapped IDs
5. Validates; attaches reports to paper supplement
6. Publishes synthetic data + config fingerprint + EduLogGen version

### 29H.2 Scenario — EdTech CI

1. Nightly job pulls pseudonymized logs inside VPC
2. `eduloggen benchmark --protocol session_fidelity_v1 ...`
3. Fails CI if privacy dup rate exceeds threshold
4. Publishes only metric JSON internally

### 29H.3 Scenario — External plugin author

1. Implements `BaseGenerator` in separate package
2. Declares entry point
3. Users `pip install eduloggen_mygen`
4. Appears in `eduloggen plugins`

### 29H.4 Sequence diagram — fit/generate/validate

```text
Researcher    CLI/API    IO    Analysis    Generator    Validation
    │            │        │        │            │             │
    │ input path │        │        │            │             │
    │───────────►│ read   │        │            │             │
    │            │───────►│        │            │             │
    │            │ dataset│        │            │             │
    │            │────────────────►│            │             │
    │            │ analysis result │            │             │
    │            │─────────────────────────────►│             │
    │            │            model artifact    │             │
    │            │◄─────────────────────────────│             │
    │            │ generate(seed)               │             │
    │            │─────────────────────────────►│             │
    │            │            synthetic         │             │
    │            │◄─────────────────────────────│             │
    │            │ validate(real, synth)        │             │
    │            │───────────────────────────────────────────►│
    │            │            report            │             │
    │◄───────────│◄───────────────────────────────────────────│
```

---

## 29I. Quality Attribute Scenarios (ATAM-style)

### 29I.1 Modularity scenario

**Stimulus:** Contributor adds a new metric.  
**Response:** New module + registry entry; no generator files changed.  
**Measure:** Diff touches ≤3 paths besides tests/docs.

### 29I.2 Reproducibility scenario

**Stimulus:** Two runs with identical config/seed/data fingerprint.  
**Response:** Identical synthetic token sequences on same platform/version.  
**Measure:** Byte-identical events parquet or documented tolerance policy.

### 29I.3 Privacy scenario

**Stimulus:** User runs default generate.  
**Response:** Output learner_ids ∩ real learner_ids = ∅.  
**Measure:** Automated test enforces.

### 29I.4 Performance scenario

**Stimulus:** 1e6 events Markov fit on laptop-class CPU.  
**Response:** Completes within interactive timeframe (minutes).  
**Measure:** Performance smoke records duration artifact.

### 29I.5 Reliability scenario

**Stimulus:** Missing required column on ingest.  
**Response:** SchemaError with field name; no partial corrupt corpus without
quality report.  
**Measure:** Integration test.

---

## 29J. Cross-Cutting Privacy Architecture

```text
Ingest ──► optional strip/hash metadata
   │
Sessionize/Analyze ──► pseudonymous IDs only in canonical model
   │
Generate ──► default remap IDs + new event_ids
   │
Validate ──► privacy metrics
   │
Export ──► user decides sharing; docs warn residual risk
```

Privacy is not a single module call; it is a **pipeline policy** coordinated by
`privacy` package helpers and validation metrics.

---

## 29K. Mapping Skeleton Packages to Target Architecture

During implementation from 0.1.0 → 1.0.0:

1. Introduce `core`, `config`, `models`, `utils`, `privacy`, `plugins`, `benchmark`
2. Evolve `ingestion` → `io` (keep temporary shim re-exports)
3. Expand `cli.py` → `cli/` package
4. Keep existing documented namespaces importable until next MAJOR after shims
   deprecate

Migration must be covered by ADR superseding ADR-017 only if reversed.

---


## 29L. Module File Plans (Implementation Derivation)

The following file plans are normative *targets* for 1.0. Exact filenames may
vary slightly, but responsibilities must land in the indicated packages.

### 29L.1 `core`

```text
core/
  __init__.py
  exceptions.py      # hierarchy in §18
  constants.py       # schema versions, default vocab
  context.py         # RunContext
  typing.py          # shared aliases/protocols
```

### 29L.2 `config`

```text
config/
  __init__.py
  schema.py          # AppConfig models
  loader.py          # YAML/TOML/JSON load
  merge.py           # precedence merge
  fingerprint.py     # canonical hash
  env.py             # EDULOGGEN_* allowlist
```

### 29L.3 `models`

```text
models/
  __init__.py
  records.py         # LogRecord, Session, Participant
  dataset.py         # Dataset, SyntheticDataset
  features.py        # Feature
  statistics.py      # Statistics containers
  reports.py         # ValidationReport, BenchmarkReport DTOs (or thin)
  generator_model.py # GeneratorModel manifest structs
  vocab.py           # event_type vocabulary
  schema_validate.py
```

### 29L.4 `io`

```text
io/
  __init__.py
  base.py            # BaseReader, BaseWriter
  csv_reader.py
  parquet_reader.py
  jsonl_reader.py
  writers.py
  corpus.py          # corpus directory package I/O
  mapping.py         # FieldMapping application
  streaming.py       # chunk iterators
```

### 29L.5 `analysis`

```text
analysis/
  __init__.py
  sessionize.py
  features.py
  transitions.py
  timing.py
  sequences.py
  graphs.py
  profiles.py
  service.py         # analyze() façade internals
  result.py          # AnalysisResult
```

### 29L.6 `generators`

```text
generators/
  __init__.py
  base.py
  registry.py
  markov.py
  semi_markov.py
  statistical/
    independent.py
  artifact.py        # save/load
  tokenization.py
```

### 29L.7 `validation`

```text
validation/
  __init__.py
  base.py
  context.py
  report.py
  thresholds.py
  metrics/
    marginal.py
    sequential.py
    temporal.py
    privacy.py
    utility.py
  service.py
```

### 29L.8 `benchmark`

```text
benchmark/
  __init__.py
  protocol.py
  runner.py
  aggregate.py
  protocols/
    session_fidelity_v1.py
```

### 29L.9 `visualization`

```text
visualization/
  __init__.py
  base.py
  timeline.py
  transitions.py
  sankey.py
  heatmaps.py
  distributions.py
  sessions.py
  dashboard.py
  export.py
```

### 29L.10 `cli`

```text
cli/
  __init__.py
  main.py
  parsing.py
  output.py
  commands/
    ingest.py
    sessionize.py
    analyze.py
    fit.py
    generate.py
    validate.py
    benchmark.py
    visualize.py
    plugins.py
    info.py
```

---

## 29M. Configuration Examples (Documentation Specimens)

These examples are **specimens for implementers and users**, not executable
product code. They define expected keys.

### 29M.1 Minimal experiment (YAML)

```yaml
project:
  name: demo-markov
  output_dir: runs/demo

logging:
  level: INFO
  json: false

io:
  input: data/demo/events.csv
  format: csv
  mapping: configs/demo_mapping.yaml

sessionization:
  strategy: idle_timeout
  idle_timeout_s: 1800

generator:
  name: markov
  order: 1
  tokenization: activity_id
  length_model: empirical
  smoothing: laplace
  smoothing_alpha: 0.1

generation:
  n_sessions: 1000
  seed: 42
  id_strategy: remap

validation:
  metrics:
    - event_type_tvd
    - bigram_tvd
    - session_length_ks
    - interevent_time_ks
    - exact_session_dup_rate
  thresholds:
    event_type_tvd: 0.25
    exact_session_dup_rate: 0.01

privacy:
  strip_metadata_keys: ["ip", "email", "raw_message"]
```

### 29M.2 Environment allowlist (normative keys)

| Variable | Maps to |
| -------- | ------- |
| `EDULOGGEN_LOG_LEVEL` | logging.level |
| `EDULOGGEN_OUTPUT_DIR` | project.output_dir |
| `EDULOGGEN_SEED` | generation.seed / global seed |
| `EDULOGGEN_CONFIG` | default config path if CLI omits |

Unknown `EDULOGGEN_*` variables are ignored with DEBUG log (or WARN in strict
env mode).

---

## 29N. Data Quality Report Architecture

Ingest Gate A produces `quality_report.json`:

| Section | Contents |
| ------- | -------- |
| `summary` | rows read, rows kept, rows dropped |
| `field_coverage` | null rates per canonical field |
| `timestamp_issues` | naive timestamps, out-of-order counts |
| `duplicate_events` | counts by policy |
| `warnings` | recoverable issues |
| `errors` | fatal issues if any |

Writers should refuse to mark corpus `READY` if fatal errors exist.

---

## 29O. RNG and Seeding Architecture

```text
global_seed (config/CLI)
    │
    ├─ derive fit_seed     = hash(global_seed, "fit", generator_id)
    ├─ derive gen_seed     = hash(global_seed, "generate", i)
    └─ derive bench_seeds  = hash(global_seed, "bench", generator_id, rep)
```

Derivation must be stable across versions within a MAJOR release or versioned
explicitly (`seed_scheme_version`). Using Python `random` and NumPy generators
must be documented per component; mixing without coordination is forbidden.

---

## 29P. Observability and Run Manifest

Each run writes `run_manifest.json`:

- run_id, timestamps, argv/API entry
- resolved config fingerprint
- plugin versions discovered
- input fingerprints
- output paths
- exit status

This supports research audit trails and institutional reviews.

---

## 29Q. Compatibility and Deprecation Architecture

```text
Announce deprecation (WARNING) → keep ≥1 MINOR → remove in next MAJOR
```

Shims for `eduloggen.ingestion` → `eduloggen.io` follow this policy.

Plugin ABI versions appear in `PluginMeta.abi_version`. Host refuses plugins
with incompatible major ABI.

---

## 29R. Additional Architecture Decision Records

### ADR-019 — Tokenization modes are configuration, not hard-coded

- **Context:** Some studies need activity paths; others event types.
- **Decision:** `tokenization: activity_id | event_type | composite` in config.
- **Alternatives:** Separate generator classes per token type.
- **Consequences:** More flexible science; metrics must declare tokenization.

### ADR-020 — Soft validation vs hard exceptions

- **Context:** CI needs fail statuses without stack traces for threshold fails.
- **Decision:** Threshold failures are report statuses; CLI maps to exit codes.
- **Alternatives:** Raise ValidationError on any fail.
- **Consequences:** Cleaner automation; callers must check status.

### ADR-021 — Parquet preferred, CSV supported

- **Context:** Large educational logs.
- **Decision:** Prefer Parquet for corpus packages; CSV first-class for
  accessibility.
- **Alternatives:** SQL store; HDF5-only.
- **Consequences:** Dependency on parquet engine in io extra or core (decide at
  implementation; architecture allows optional engine with clear error).

### ADR-022 — Benchmark fairness over maximal parallelism

- **Context:** Parallel generator runs distort runtime metrics.
- **Decision:** Default sequential generator timing in protocols; parallel opt-in.
- **Alternatives:** Always parallel.
- **Consequences:** Slower CI benches; fairer papers.

### ADR-023 — No network calls in core runtime

- **Context:** Institutional firewalls and privacy.
- **Decision:** Core library performs no outbound network I/O.
- **Alternatives:** Auto-download models/datasets.
- **Consequences:** Users supply data/models; trust improved.

### ADR-024 — AnalysisResult cache keyed by fingerprints

- **Context:** Repeated fits in benchmarks recompute transitions.
- **Decision:** Optional on-disk cache keyed by dataset+config fingerprints.
- **Alternatives:** Always recompute.
- **Consequences:** Faster iteration; cache invalidation must be correct.

### ADR-025 — Public demo corpus is synthetic

- **Context:** Need tutorials without legal risk.
- **Decision:** Ship/generate synthetic demo corpora only.
- **Alternatives:** Bundle real open datasets with complex licenses.
- **Consequences:** Weaker external validity demos; safer project.

---

## 29S. Risk-Driven Architectural Mitigations

| Risk (from PRD) | Architectural mitigation |
| --------------- | ------------------------ |
| Overselling privacy | Privacy metrics + docs warnings + no “anonymous” defaults in UI copy |
| Scope creep LMS | `io` adapters generic; vendor packs as external plugins |
| DL dependency bloat | extras `[dl]`; lazy imports |
| Plugin breakage | contract tests + abi_version |
| Schema churn | schema_version + migration notes + freeze candidate before 1.0 |

---

## 29T. Contributor Architecture Checklist

Before opening a PR that touches structure:

1. Does it violate §8 dependency rules?
2. Does it introduce a new public symbol without docs?
3. Are seeds threaded?
4. Are fixtures synthetic?
5. Is an ADR needed?
6. Do CLI and API both expose the capability (if user-facing)?
7. Are metrics generator-agnostic?
8. Will coverage remain >95% for core?

---


## 29U. Visualization Module Specifications

### 29U.1 Timeline

**Input:** Dataset (optionally sampled sessions).  
**Behavior:** Plot events along a time axis for a small sample of sessions.  
**Privacy:** Cap sessions shown; label with synthetic session_id only.  
**Output:** PNG/SVG.

### 29U.2 Transition Graph

**Input:** Navigation graph / transition counts.  
**Behavior:** Node-link diagram with edge weights; filter top-N nodes.  
**Output:** Static figure; optional Graphviz/networkx backend.

### 29U.3 Sankey

**Input:** Aggregated flows between stages or top activities.  
**Behavior:** Multi-level flow for interpretability of major pathways.  
**Output:** SVG preferred for papers.

### 29U.4 Heatmaps

**Input:** Transition matrix or binned temporal activity.  
**Behavior:** Ordered by frequency; shared color scale for real vs synthetic
when comparing pairs.

### 29U.5 Distributions

**Input:** Real and synthetic feature arrays.  
**Behavior:** Overlay histograms/KDE/ECDF for length, duration, Δt.

### 29U.6 Session Statistics

**Input:** AnalysisResult or Dataset.  
**Behavior:** Small-multiple panels summarizing session-level features.

### 29U.7 Benchmark Dashboard

**Input:** BenchmarkReport.  
**Behavior:** Grouped bar/radar/table rendering of metric means with error bars
from multi-seed runs. Designed for CI artifact browsing and paper drafts.

---

## 29V. Testing Matrix by Package

| Package | Unit | Integration | Property | Perf | Contract |
| ------- | ---- | ----------- | -------- | ---- | -------- |
| core | ✓ | | | | |
| config | ✓ | ✓ | ✓ | | |
| models | ✓ | | ✓ | | |
| io | ✓ | ✓ | ✓ | ✓ | ✓ (readers) |
| analysis | ✓ | ✓ | ✓ | ✓ | |
| generators | ✓ | ✓ | ✓ | ✓ | ✓ |
| validation | ✓ | ✓ | ✓ | | ✓ |
| benchmark | ✓ | ✓ | | ✓ | ✓ |
| visualization | ✓ smoke | | | | ✓ |
| cli | | ✓ | | | |
| privacy | ✓ | ✓ | | | |
| plugins | ✓ | ✓ | | | ✓ |
| utils | ✓ | | ✓ | | |

---

## 29W. End-to-End Acceptance Criteria (Architecture-Level)

A build may be called **architecturally complete for 1.0** when:

1. All packages in §7 exist with responsibilities respected
2. Pipeline stages in §5 are callable via CLI and Python API
3. Markov and Semi-Markov pass contract + reproducibility tests
4. Validation emits reports including privacy indicators
5. Benchmark protocol `session_fidelity_v1` runs on demo corpus in CI
6. Dependency rules enforced (import linter or equivalent test)
7. Coverage >95% on core logic packages
8. Docs site includes Vision, PRD, SAD, tutorials, API reference
9. No real PII in repository
10. ADRs 001–025 remain consistent with code

---

## 29X. Glossary Expansion

| Term | Notes |
| ---- | ----- |
| **Behavior Model** | Estimated structure prior to or inside GeneratorModel |
| **Corpus Package** | Directory layout of Dataset on disk |
| **Fingerprint** | Hash of data and/or config for provenance |
| **Gate** | Validation checkpoint in data flow |
| **Idle timeout** | Max inter-event gap before new session |
| **Remap** | Replace real IDs with synthetic IDs |
| **Sufficient statistics** | Aggregates enabling fit without raw replay |
| **Token** | Atomic symbol in sequences (activity/type/composite) |

---


## 29Y. Deployment and Environment View

### 29Y.1 Supported runtime topology (v1.0)

```text
┌─────────────────────────────────────────┐
│ Researcher workstation / institutional  │
│ compute node / CI runner                │
│                                         │
│  Python 3.11+                           │
│    └─ eduloggen (library + CLI)         │
│         └─ reads/writes local filesystem│
│                                         │
│  Optional: Jupyter kernel importing API │
└─────────────────────────────────────────┘
```

No database server, message broker, or cloud account is required.

### 29Y.2 Containerization (optional operational pattern)

Institutions may wrap the CLI in a container for locked dependencies. The
architecture treats containers as deployment adapters: mount data volumes,
invoke `eduloggen`, export artifacts. Containerization must not introduce
network exfiltration defaults.

### 29Y.3 Multi-user shared filesystem

When multiple analysts share a corpus store:

- Treat corpus packages as immutable after READY
- Use distinct `runs/<run_id>/` output trees
- Avoid concurrent writes to the same model directory without locking

### 29Y.4 Resource classes

| Class | Event scale | Guidance |
| ----- | ----------- | -------- |
| S | ≤1e5 | In-memory fine |
| M | ≤1e6 | Prefer Parquet; cache analysis |
| L | ≤1e7 | Chunked sufficient statistics; spill policies |
| XL | >1e7 | Post-1.0 distributed seams |

---

## 29Z. Scientific Integrity Architecture

EduLogGen’s architecture deliberately supports **honest science**:

1. Metrics and protocols are versioned and citable
2. Config fingerprints travel with artifacts
3. Demo leaderboards are labeled as non-universal
4. Privacy metrics discourage false safety claims
5. Generator limitations must be expressible via `describe()` and docs

Contributors proposing new metrics must document assumptions, failure modes,
and directionality before registration as built-ins.

---

## 30. Conclusion

### 30.1 Architecture summary

EduLogGen is architected as a **modular, local-first, plugin-extensible**
scientific framework centered on a **canonical sessionized event model**. A
clear pipeline—read → schema-validate → sessionize → analyze → fit/generate →
scientifically validate → benchmark → report—maps directly to Vision objectives
and PRD requirements.

Dependency rules keep generators and validators decoupled, preserve a thin
core, and push environmental concerns (CLI, future REST/UI/cloud) to adapters.
Quality attributes—especially reproducibility, extensibility, privacy
awareness, and testability—are enforced through configuration fingerprints,
seeded sampling, privacy metrics, contract tests, and a **>95%** coverage
target for core logic.

### 30.2 Guidelines for future contributors

1. **Read Vision → PRD → this SAD** before proposing structural changes.
2. **Respect dependency rules**; if you need a forbidden edge, write an ADR.
3. **Extend via plugins** rather than editing closed modules.
4. **Keep domain objects schema-stable**; migrate with versioned schema bumps.
5. **Never commit real learner data**; use synthetic fixtures only.
6. **Prefer inspectable artifacts** and explicit seeds.
7. **Match CLI and Python API capabilities** when adding workflows.
8. **Document metrics and assumptions** with scientific precision.
9. **Add tests at the appropriate layer** until coverage gates pass.
10. **Record decisions** in ADRs when changing principles in §3 or §29.

### 30.3 Closing statement

This Software Architecture Document is the authoritative structural contract
for EduLogGen. Implementations that conform to these views, package
boundaries, domain models, and ADRs will remain composable as the project
grows from Markov baselines to deep generative models and beyond—without
sacrificing the scientific and ethical commitments that define the project.

---

## Appendix A — Traceability (PRD → Architecture)

| PRD area | SAD sections |
| -------- | ------------ |
| Functional requirements | §§5, 11–15, 19–20 |
| Non-functional requirements | §§2, 21, 25–26 |
| System modules / package structure | §§6–8 |
| Data models | §9 |
| Plugins / generators / validation / viz | §§12–15 |
| Config / logging | §§16–17 |
| Benchmark / privacy | §§14, 18, 25 |
| Testing / docs / release | §§21–23 |

## Appendix B — Viewpoint catalog (IEEE 42010)

| Viewpoint | Concerns | Primary sections |
| --------- | -------- | ---------------- |
| Context | Actors, environment | §4 |
| Composition | Packages, repos | §§6–8 |
| Information | Domain entities | §9 |
| Functional / dataflow | Pipeline | §§5, 10–15 |
| Interface | CLI/API/plugins | §§13, 19–20, 24 |
| Operations | Config, logging, CI | §§16–17, 23 |
| Quality | Security, perf, test | §§21, 25–26 |
| Evolution | Future, ADRs | §§27, 29 |

## Appendix C — Glossary quick reference

See §1.4 Definitions. Additional synonyms: **Corpus ≈ Dataset**; **Event ≈
LogRecord**; **sample ≈ generate** (API alias).

---

*End of Software Architecture Document v1.0*
