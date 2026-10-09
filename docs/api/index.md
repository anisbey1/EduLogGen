# API reference

`import eduloggen` exposes the workflow functions from
[`eduloggen.api`](workflow.md). Subpackages hold the building blocks.

| Module | Purpose |
| ------ | ------- |
| [`eduloggen`](eduloggen.md) | Package root, version, workflow re-exports |
| [`eduloggen.api`](workflow.md) | `ingest`, `sessionize`, `analyze`, `fit_generator`, `generate`, `validate`, `run_benchmark` |
| [`eduloggen.core`](core.md) | Exceptions, constants, `RunContext` |
| [`eduloggen.models`](models.md) | `LogRecord`, `Session`, `Dataset`, `GeneratorModel` |
| [`eduloggen.config`](config.md) | `AppConfig`, `resolve_config` |
| [`eduloggen.io`](io.md) | Readers, writers, `FieldMapping`, `ingest`, corpora |
| [`eduloggen.analysis`](analysis.md) | `sessionize`, `analyze`, statistics |
| [`eduloggen.generators`](generators.md) | `BaseGenerator`, built-ins, registry |
| [`eduloggen.validation`](validation.md) | Metrics, `validate`, `ValidationReport` |
| [`eduloggen.benchmark`](benchmark.md) | `run_benchmark`, protocols, `BenchmarkReport` |
| [`eduloggen.datasets`](datasets.md) | Synthetic demo corpus |
| [`eduloggen.visualization`](visualization.md) | Figures (requires the `viz` extra) |
| [`eduloggen.scenarios`](scenarios.md) | Level 2: labelled anomaly injection |
| [`eduloggen.evaluation`](evaluation.md) | Level 2: scoring methods against ground truth |
| [`eduloggen.plugins`](plugins.md) | Plugin discovery and registration |
| [`eduloggen.privacy`](privacy.md) | ID remapping, metadata stripping |
| [`eduloggen.cli`](cli.md) | Command-line entry point |
