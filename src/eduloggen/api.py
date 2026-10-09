"""Notebook-friendly façade over the pipeline (PRD §15, SAD §20, §29G.1).

Every function takes an optional :class:`~eduloggen.config.AppConfig`;
explicit keyword arguments override config values, which override built-in
defaults. The CLI is a thin layer over these functions.

Example::

    import eduloggen as elg

    cfg = elg.load_config("experiment.yaml")
    real = elg.sessionize(elg.ingest("events.csv", "mapping.yaml").dataset, config=cfg)
    model = elg.fit_generator("semi_markov", real, config=cfg)
    synthetic = elg.generate(model, n_sessions=1000, seed=7)
    report = elg.validate(real, synthetic, config=cfg)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from eduloggen.analysis import (
    AnalysisResult,
    SessionStrategy,
    activity_profiles,
    analyze_by,
    temporal_profile,
)
from eduloggen.analysis import analyze as _analyze
from eduloggen.analysis import sessionize as _sessionize
from eduloggen.benchmark import BenchmarkReport
from eduloggen.benchmark import run_benchmark as _run_benchmark
from eduloggen.config import AppConfig, resolve_config
from eduloggen.core import GenerationError, PathLike
from eduloggen.datasets import demo_dataset
from eduloggen.evaluation import evaluate_detection
from eduloggen.generators import SessionCalendar, get_generator
from eduloggen.io import (
    FieldMapping,
    IngestResult,
    read_corpus,
    write_corpus,
)
from eduloggen.io import ingest as _ingest
from eduloggen.io.base import Format
from eduloggen.models import (
    Dataset,
    EventVocabulary,
    GeneratorModel,
    SyntheticDataset,
    TokenField,
    UnknownEventPolicy,
)
from eduloggen.privacy import IdStrategy, strip_metadata
from eduloggen.scenarios import inject_anomalies
from eduloggen.validation import ValidationReport, compare_detailed
from eduloggen.validation import validate as _validate

__all__ = [
    "activity_profiles",
    "analyze",
    "analyze_by",
    "compare_detailed",
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
    "temporal_profile",
    "validate",
]


def load_config(
    path: PathLike | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    allow_unknown: bool = False,
) -> AppConfig:
    """Resolve configuration from file, ``EDULOGGEN_*`` variables, and overrides.

    See :func:`eduloggen.config.resolve_config`.
    """
    return resolve_config(path, overrides=overrides, allow_unknown=allow_unknown)


def ingest(
    source: PathLike,
    mapping: FieldMapping | PathLike,
    *,
    config: AppConfig | None = None,
    format: str | None = None,
    strict: bool | None = None,
    dataset_id: str | None = None,
) -> IngestResult:
    """Read a source log into a validated dataset.

    ``mapping`` may be a :class:`FieldMapping` or a mapping file path.
    Metadata keys listed in ``privacy.strip_metadata_keys`` are removed.
    """
    cfg = config or AppConfig()
    if not isinstance(mapping, FieldMapping):
        mapping = FieldMapping.from_file(mapping)
    vocabulary = EventVocabulary(
        unknown_policy=UnknownEventPolicy(cfg.io.unknown_event_policy)
    ).extend(cfg.io.event_types)
    result = _ingest(
        source,
        mapping,
        format=format or cfg.io.format,
        strict=cfg.io.strict if strict is None else strict,
        on_duplicate=cfg.io.on_duplicate,
        vocabulary=vocabulary,
        dataset_id=dataset_id,
    )
    stripped = strip_metadata(result.dataset, cfg.privacy.strip_metadata_keys)
    return IngestResult(dataset=stripped, report=result.report, mapping=result.mapping)


def load_dataset(path: PathLike) -> Dataset:
    """Load a corpus directory (see :func:`eduloggen.io.read_corpus`)."""
    return read_corpus(path)


def save_dataset(
    dataset: Dataset,
    path: PathLike,
    *,
    config: AppConfig | None = None,
    format: Format | None = None,
    force: bool = False,
) -> Path:
    """Write a corpus directory (see :func:`eduloggen.io.write_corpus`)."""
    cfg = config or AppConfig()
    return write_corpus(
        dataset, path, format=format or cfg.io.output_format, force=force
    )


def sessionize(
    dataset: Dataset,
    *,
    config: AppConfig | None = None,
    strategy: SessionStrategy | None = None,
    idle_timeout_s: float | None = None,
    tokenization: TokenField | None = None,
) -> Dataset:
    """Group events into sessions using the ``sessionization`` settings."""
    section = (config or AppConfig()).sessionization
    return _sessionize(
        dataset,
        strategy=strategy or section.strategy,
        idle_timeout_s=idle_timeout_s or section.idle_timeout_s,
        tokenization=tokenization or section.tokenization,
    )


def analyze(
    dataset: Dataset, *, config: AppConfig | None = None, ngram_order: int | None = None
) -> AnalysisResult:
    """Compute descriptive statistics of a sessionized dataset."""
    section = (config or AppConfig()).analysis
    return _analyze(dataset, ngram_order=ngram_order or section.ngram_order)


def fit_generator(
    name: str | None,
    dataset: Dataset,
    *,
    config: AppConfig | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
) -> GeneratorModel:
    """Fit a registered generator.

    The generator defaults to ``generator.name``; hyperparameters are the
    config's ``generator`` keys updated with ``hyperparameters``. Config
    keys only apply when the configured generator is the one being fitted.
    """
    section = (config or AppConfig()).generator
    generator_name = name or section.name
    base = dict(section.params) if generator_name == section.name else {}
    generator = get_generator(generator_name)
    return generator.fit(dataset, {**base, **(hyperparameters or {})})


def generate(
    model: GeneratorModel,
    *,
    config: AppConfig | None = None,
    n_sessions: int | None = None,
    seed: int | None = None,
    id_strategy: IdStrategy | None = None,
    calendar: SessionCalendar | None = None,
) -> SyntheticDataset:
    """Sample synthetic data from a fitted model.

    ``calendar`` (Level 2) draws session start times from a course calendar.

    Raises:
        GenerationError: If no seed is given here or in ``generation.seed``.
    """
    section = (config or AppConfig()).generation
    chosen_seed = section.seed if seed is None else seed
    if chosen_seed is None:
        raise GenerationError(
            "generation needs a seed: pass seed= or set generation.seed",
            code="generation_missing_seed",
        )
    generator = get_generator(model.generator_id)
    return generator.generate(
        model,
        n_sessions or section.n_sessions,
        chosen_seed,
        id_strategy=id_strategy or section.id_strategy,
        calendar=calendar,
    )


def validate(
    real: Dataset,
    synthetic: Dataset,
    *,
    config: AppConfig | None = None,
    metrics: Iterable[str] | None = None,
    thresholds: Mapping[str, float] | None = None,
    seed: int | None = None,
    real_split: str = "full",
) -> ValidationReport:
    """Compare real and synthetic data using the ``validation`` settings.

    Explicit ``thresholds`` are merged over configured ones; configured
    thresholds for metrics not being computed are ignored, explicit ones are
    an error. The seed falls back to ``generation.seed`` and then 0.
    """
    cfg = config or AppConfig()
    chosen = list(metrics) if metrics is not None else list(cfg.validation.metrics)
    configured = {
        k: v for k, v in cfg.validation.thresholds.items() if not chosen or k in chosen
    }
    limits = {**configured, **(thresholds or {})}
    return _validate(
        real,
        synthetic,
        metrics=chosen or None,
        thresholds=limits,
        seed=seed if seed is not None else (cfg.generation.seed or 0),
        real_split=real_split,
    )


def run_benchmark(
    dataset: Dataset,
    *,
    config: AppConfig | None = None,
    generators: Iterable[str] | None = None,
    protocol: str | None = None,
    repeats: int | None = None,
    n_sessions: int | None = None,
    seed: int | None = None,
) -> BenchmarkReport:
    """Compare generators using the ``benchmark`` settings.

    Unsessionized data is sessionized with the config first. Hyperparameters
    from the ``generator`` section apply to the configured generator only.
    """
    cfg = config or AppConfig()
    if dataset.sessions is None:
        dataset = sessionize(dataset, config=cfg)
    return _run_benchmark(
        dataset,
        list(generators) if generators is not None else list(cfg.benchmark.generators),
        protocol=protocol or cfg.benchmark.protocol,
        repeats=repeats or cfg.benchmark.repeats,
        n_sessions=n_sessions,
        seed=seed if seed is not None else (cfg.generation.seed or 0),
        hyperparameters={cfg.generator.name: dict(cfg.generator.params)},
    )
