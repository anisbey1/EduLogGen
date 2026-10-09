"""EduLogGen: synthetic educational interaction log framework.

EduLogGen is an open-source Python framework for analyzing, generating,
validating, and benchmarking synthetic educational interaction logs.

The package is organized into focused subpackages:

- ``core``: exceptions, constants, run context, and shared types
- ``models``: canonical events, sessions, datasets, and vocabulary
- ``config``: typed settings, file loading, and precedence rules
- ``io``: readers, writers, field mapping, ingest, and corpus directories
- ``privacy``: id remapping and metadata stripping
- ``utils``: seed derivation and hashing helpers
- ``ingestion``: compatibility alias for the ingest API in ``io``
- ``analysis``: session and statistical analysis of learner behavior
- ``generators``: statistical and learning-based synthetic generators
- ``validation``: quality, similarity, and privacy validation
- ``visualization``: plotting and exploratory visualization utilities
- ``cli``: command-line interface entry points

The top-level namespace re-exports the workflow façade from
:mod:`eduloggen.api`::

    import eduloggen as elg

    real = elg.sessionize(elg.ingest("events.csv", "mapping.yaml").dataset)
    model = elg.fit_generator("semi_markov", real)
    synthetic = elg.generate(model, n_sessions=1000, seed=7)
    report = elg.validate(real, synthetic)

See the project vision document (``docs/01_VISION.md``) for goals and scope.
"""

from __future__ import annotations

from eduloggen.__version__ import __version__
from eduloggen.api import (
    analyze,
    demo_dataset,
    evaluate_detection,
    fit_generator,
    generate,
    ingest,
    inject_anomalies,
    load_config,
    load_dataset,
    run_benchmark,
    save_dataset,
    sessionize,
    validate,
)

__all__ = [
    "__version__",
    "analyze",
    "demo_dataset",
    "evaluate_detection",
    "fit_generator",
    "generate",
    "ingest",
    "inject_anomalies",
    "load_config",
    "load_dataset",
    "run_benchmark",
    "save_dataset",
    "sessionize",
    "validate",
]
