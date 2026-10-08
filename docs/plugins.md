# Writing plugins

EduLogGen can be extended without forking it. A plugin is an ordinary Python
package that declares **entry points**; once it is installed, EduLogGen finds
it automatically.

## Extension points

| Kind | Entry point group | Provide |
| ---- | ----------------- | ------- |
| `generator` | `eduloggen.plugins.generator` | A `BaseGenerator` subclass (or factory) whose `name` equals the entry point name |
| `metric` | `eduloggen.plugins.metric` | A `BaseMetric` subclass or instance whose `name` equals the entry point name |
| `reader` | `eduloggen.plugins.reader` | A `BaseReader` subclass (or factory); an optional `suffixes` class attribute (e.g. `(".xes",)`) enables `format: auto` |
| `visualizer` | `eduloggen.plugins.visualizer` | A function `(real, synthetic_or_None) -> matplotlib Figure` |
| `benchmark_suite` | `eduloggen.plugins.benchmark_suite` | A `BenchmarkProtocol` whose `name` equals the entry point name |

Session segmentation strategies and writers are not pluggable yet.

## Example: a generator package

```python
# my_generators/hmm.py
from typing import Any, ClassVar
from eduloggen.generators import BaseGenerator

class HmmGenerator(BaseGenerator):
    name: ClassVar[str] = "hmm"
    tags: ClassVar[frozenset[str]] = frozenset({"probabilistic", "latent_states"})
    defaults: ClassVar[dict[str, Any]] = {"n_states": 4}

    def _fit_family(self, dataset, sequences, hyperparameters):
        ...  # return JSON-compatible parameters

    def _sequence_sampler(self, model):
        ...  # return a function (rng, length) -> (tokens, gaps_in_seconds)
```

```toml
# pyproject.toml of your package
[project.entry-points."eduloggen.plugins.generator"]
hmm = "my_generators.hmm:HmmGenerator"
```

After `pip install my-generators`:

```bash
eduloggen plugins            # lists "hmm  [...; from my-generators 0.1.0]"
eduloggen fit --input sessions/ --generator hmm --output model/
eduloggen benchmark --input sessions/ --generators markov,hmm
```

`BaseGenerator` handles hyperparameter validation, session lengths, learners,
timestamps, ID remapping, saving, and loading; a plugin only models token
sequences and the gaps between events. Fitted parameters must be JSON data so
model artifacts stay inspectable.

## How discovery works

- The CLI discovers plugins at startup. In Python, call
  `eduloggen.plugins.discover_plugins()` explicitly; importing EduLogGen never
  imports third-party code on its own.
- Entry points are processed in a fixed order (kind, then name).
- Each plugin is checked against its contract **before** it is registered.
- Plugins can never replace built-ins; a name clash is an error.
- A broken plugin is skipped and reported: `eduloggen plugins` lists it under
  `errors:` and exits with code 1. Use `discover_plugins(strict=True)` to
  fail instead.
- Every `run_manifest.json` records the external plugins and their versions.

## Registering without packaging

In notebooks or private code, register directly:

```python
from eduloggen.plugins import register_plugin, unregister_plugin

register_plugin("generator", "hmm", HmmGenerator)
register_plugin("metric", "my_metric", MyMetric())
```

## Safety

Plugins run in the same process with the same permissions as EduLogGen.
Install only packages you trust. Plugins must write only to output
directories the user chose, and EduLogGen never downloads plugin code.
