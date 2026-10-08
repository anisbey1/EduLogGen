"""Benchmark protocols for comparing generators fairly (PRD §18, SAD §14.7).

Typical use::

    from eduloggen.benchmark import run_benchmark

    report = run_benchmark(sessionized, ["markov", "semi_markov", "independent"])
    print(report.to_markdown())
"""

from __future__ import annotations

from eduloggen.benchmark.protocol import (
    BUILTIN_PROTOCOLS,
    PROTOCOLS,
    BenchmarkProtocol,
    get_protocol,
    register_protocol,
    split_by_learner,
    unregister_protocol,
)
from eduloggen.benchmark.report import (
    BenchmarkReport,
    GeneratorResult,
    MetricSummary,
    summarize,
)
from eduloggen.benchmark.runner import DEFAULT_GENERATORS, run_benchmark

__all__ = [
    "BUILTIN_PROTOCOLS",
    "DEFAULT_GENERATORS",
    "PROTOCOLS",
    "BenchmarkProtocol",
    "BenchmarkReport",
    "GeneratorResult",
    "MetricSummary",
    "get_protocol",
    "register_protocol",
    "run_benchmark",
    "split_by_learner",
    "summarize",
    "unregister_protocol",
]
