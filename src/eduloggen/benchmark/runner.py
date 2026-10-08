"""Run a benchmark protocol over several generators (SAD §14.7, ADR-022).

Generators run sequentially so their timings are comparable. A generator
that fails to fit or generate is recorded with its error instead of
aborting the whole benchmark.
"""

from __future__ import annotations

import logging
import platform
import time
import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from eduloggen.__version__ import __version__
from eduloggen.benchmark.protocol import get_protocol, split_by_learner
from eduloggen.benchmark.report import BenchmarkReport, GeneratorResult, summarize
from eduloggen.core import BenchmarkError, EduLogGenError
from eduloggen.generators import get_generator
from eduloggen.models import Dataset
from eduloggen.utils import derive_seed
from eduloggen.validation import get_metric, validate

__all__ = ["run_benchmark"]

logger = logging.getLogger(__name__)

DEFAULT_GENERATORS = ("markov", "semi_markov", "independent")


def run_benchmark(
    dataset: Dataset,
    generators: Iterable[str] = DEFAULT_GENERATORS,
    *,
    protocol: str = "session_fidelity_v1",
    repeats: int | None = None,
    n_sessions: int | None = None,
    seed: int = 0,
    hyperparameters: Mapping[str, Mapping[str, Any]] | None = None,
    run_id: str | None = None,
) -> BenchmarkReport:
    """Compare generators under a protocol.

    Args:
        dataset: Sessionized real data.
        generators: Registered generator names, compared in this order.
        protocol: Protocol name.
        repeats: Override the protocol's number of seeds per generator.
        n_sessions: Override the protocol's sample size.
        seed: Global seed; split and generation seeds derive from it.
        hyperparameters: Generator name to hyperparameters.
        run_id: Run identifier; random by default.

    Returns:
        The benchmark report.

    Raises:
        BenchmarkError: If arguments are invalid or the data cannot be split.
        ConfigError: If the protocol or a generator name is unknown.
    """
    spec = get_protocol(protocol)
    names = list(generators)
    if not names or len(set(names)) != len(names):
        raise BenchmarkError(
            "generators must be a non-empty list without duplicates",
            code="benchmark_invalid_argument",
        )
    for value, label in ((repeats, "repeats"), (n_sessions, "n_sessions")):
        if value is not None and (isinstance(value, bool) or value < 1):
            raise BenchmarkError(
                f"{label} must be a positive integer", code="benchmark_invalid_argument"
            )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise BenchmarkError(
            "seed must be a non-negative integer", code="benchmark_invalid_argument"
        )
    for name in names:
        get_generator(name)

    started = time.perf_counter()
    train, holdout = split_by_learner(dataset, spec.holdout_fraction, seed)
    n_repeats = repeats or spec.repeats
    size = n_sessions or spec.n_sessions or holdout.n_sessions
    metrics = list(spec.metrics)
    directions = {m: get_metric(m).direction for m in metrics}

    reference_report = validate(holdout, train, metrics=metrics, seed=seed)
    reference = {m.name: m.value for m in reference_report.metrics}

    results = []
    for name in names:
        params = dict((hyperparameters or {}).get(name, {}))
        logger.info("benchmarking %s", name)
        results.append(
            _run_one(
                name, params, train, holdout, metrics, directions, size, n_repeats, seed
            )
        )

    return BenchmarkReport(
        run_id=run_id or uuid.uuid4().hex,
        created_at=datetime.now(UTC),
        protocol=MappingProxyType(
            spec.to_dict() | {"repeats": n_repeats, "n_sessions": size}
        ),
        seed=seed,
        dataset=MappingProxyType(
            {
                "dataset_id": dataset.dataset_id,
                "fingerprint": dataset.fingerprint(),
                "train": _summary(train),
                "holdout": _summary(holdout),
            }
        ),
        reference=MappingProxyType(reference),
        results=tuple(results),
        environment=MappingProxyType(
            {
                "eduloggen_version": __version__,
                "python": platform.python_version(),
                "os": platform.system(),
                "machine": platform.machine(),
            }
        ),
        duration_s=time.perf_counter() - started,
    )


def _run_one(
    name: str,
    params: dict[str, Any],
    train: Dataset,
    holdout: Dataset,
    metrics: list[str],
    directions: Mapping[str, str],
    size: int,
    repeats: int,
    seed: int,
) -> GeneratorResult:
    generator = get_generator(name)
    seeds = tuple(derive_seed(seed, "benchmark", name, r) or 0 for r in range(repeats))
    values: dict[str, list[float | None]] = {m: [] for m in metrics}
    fit_s: float | None = None
    durations: list[float] = []
    fingerprint: str | None = None
    try:
        clock = time.perf_counter()
        model = generator.fit(train, params)
        fit_s = time.perf_counter() - clock
        fingerprint = model.fingerprint()
        for generation_seed in seeds:
            clock = time.perf_counter()
            synthetic = generator.generate(model, size, generation_seed)
            durations.append(time.perf_counter() - clock)
            report = validate(holdout, synthetic, metrics=metrics, seed=generation_seed)
            for result in report.metrics:
                values[result.name].append(result.value)
    except EduLogGenError as exc:
        logger.warning("%s failed: %s", name, exc)
        return GeneratorResult(
            name=name,
            hyperparameters=MappingProxyType(params),
            model_fingerprint=fingerprint,
            fit_s=fit_s,
            generate_s=None,
            seeds=seeds,
            metrics=MappingProxyType({}),
            error=str(exc),
        )
    return GeneratorResult(
        name=name,
        hyperparameters=MappingProxyType(params),
        model_fingerprint=fingerprint,
        fit_s=fit_s,
        generate_s=sum(durations) / len(durations),
        seeds=seeds,
        metrics=MappingProxyType(
            {m: summarize(directions[m], values[m]) for m in metrics}
        ),
    )


def _summary(dataset: Dataset) -> dict[str, Any]:
    return {
        "fingerprint": dataset.fingerprint(),
        "n_events": dataset.n_events,
        "n_sessions": dataset.n_sessions,
        "n_learners": len(dataset.learner_ids),
    }
