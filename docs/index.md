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

**1.0.0** completes Level 1, the log generator: ingest real logs, build
sessions, analyze behaviour, fit Markov and Semi-Markov generators, generate
synthetic data, validate fidelity and privacy, benchmark generators, and plot
the results. Next is Level 2, an experimental generator with controllable
profiles, temporal patterns, and labelled anomalies (see the design note).

## Next steps

- [Getting Started](getting-started.md) — install and run the full workflow
- [Vision](01_VISION.md) — goals, scope, and scientific contributions
- [API Reference](api/index.md) — package surface area
- [Contributing](contributing.md) — how to contribute
