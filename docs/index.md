# EduLogGen documentation

EduLogGen is an open-source Python framework for analyzing, generating,
validating, and benchmarking synthetic educational interaction logs.

Its mission is to provide researchers, universities, and educational technology
developers with a reliable framework for creating privacy-preserving synthetic
datasets that faithfully reproduce the statistical and behavioral properties of
real learner interaction logs.

## Why EduLogGen?

Educational interaction logs differ from generic tabular or time-series data:
they encode event sequences, navigation behavior, temporal dynamics, session
structure, and educational context. Existing synthetic-data tools rarely model
these characteristics explicitly.

EduLogGen is designed as a **modular platform** rather than a single algorithm.
Researchers can compare statistical, probabilistic, and deep learning approaches
using a unified API.

## Current status

The project is in an early scaffolding phase (`0.1.0`). The installable package
layout, tooling, documentation site, and continuous integration pipeline are in
place. Generator and validator implementations will follow the roadmap described
in the [Vision](01_VISION.md) document.

## Next steps

- [Getting Started](getting-started.md) — install and verify the package
- [Vision](01_VISION.md) — goals, scope, and scientific contributions
- [API Reference](api/index.md) — package surface area
- [Contributing](contributing.md) — how to contribute
