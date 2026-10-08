"""Benchmark protocols for comparing generators fairly (PRD §18, SAD §14.7).

Typical use::

    from eduloggen.benchmark import run_benchmark

    report = run_benchmark(sessionized, ["markov", "semi_markov", "independent"])
    print(report.to_markdown())
"""

from __future__ import annotations

from eduloggen.benchmark.protocol import (
    PROTOCOLS,
    BenchmarkProtocol,
    get_protocol,
    split_by_learner,
)
from eduloggen.benchmark.report import (
    BenchmarkReport,
    GeneratorResult,
    MetricSummary,
    summarize,
)
from eduloggen.benchmark.runner import DEFAULT_GENERATORS, run_benchmark

__all__ = [
    "DEFAULT_GENERATORS",
    "PROTOCOLS",
    "BenchmarkProtocol",
    "BenchmarkReport",
    "GeneratorResult",
    "MetricSummary",
    "get_protocol",
    "run_benchmark",
    "split_by_learner",
    "summarize",
]
