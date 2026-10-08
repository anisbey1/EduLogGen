# API reference

`import eduloggen` exposes the workflow functions from
[`eduloggen.api`](workflow.md). Subpackages hold the building blocks.

| Module | Purpose |
| ------ | ------- |
| [`eduloggen`](eduloggen.md) | Package root, version, workflow re-exports |
| [`eduloggen.api`](workflow.md) | `ingest`, `sessionize`, `analyze`, `fit_generator`, `generate`, `validate` |
| [`eduloggen.core`](core.md) | Exceptions, constants, `RunContext` |
| [`eduloggen.models`](models.md) | `LogRecord`, `Session`, `Dataset`, `GeneratorModel` |
| [`eduloggen.config`](config.md) | `AppConfig`, `resolve_config` |
| [`eduloggen.io`](io.md) | Readers, writers, `FieldMapping`, `ingest`, corpora |
| [`eduloggen.analysis`](analysis.md) | `sessionize`, `analyze`, statistics |
| [`eduloggen.generators`](generators.md) | `BaseGenerator`, built-ins, registry |
| [`eduloggen.validation`](validation.md) | Metrics, `validate`, `ValidationReport` |
| [`eduloggen.privacy`](privacy.md) | ID remapping, metadata stripping |
| [`eduloggen.cli`](cli.md) | Command-line entry point |
