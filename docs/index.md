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

Install with `pip install eduloggen`. Level 1, the log generator, is
complete: ingest real logs, build sessions, analyze behaviour, fit Markov and
Semi-Markov generators, generate synthetic data, validate fidelity and
privacy, benchmark generators, and plot the results.

Level 2, the experimental generator, is under way: labelled anomalies with
detector scoring, behavioural controls and course calendars with a
manipulation check, and behavioural profiles (learned, provided, or defined
by hand) are available; simulated outcomes follow (see the
[design note](04_EXPERIMENTAL_GENERATOR.md)). The
[OULAD case study](case-study-oulad.md) shows the package on real data.

To cite EduLogGen, use the "Cite this repository" button on GitHub or the
Zenodo DOI [10.5281/zenodo.23264161](https://doi.org/10.5281/zenodo.23264161).

## Next steps

- [Getting Started](getting-started.md) — install and run the full workflow
- [Case study: OULAD](case-study-oulad.md) — results on real data
- [Vision](01_VISION.md) — goals, scope, and scientific contributions
- [API Reference](api/index.md) — package surface area
- [Contributing](contributing.md) — how to contribute
