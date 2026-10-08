"""Benchmark results and their aggregation (PRD §18.4, SAD §9.11)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any

from eduloggen.core import REPORT_VERSION

__all__ = ["BenchmarkReport", "GeneratorResult", "MetricSummary", "summarize"]

_DISCLAIMER = (
    "Scores describe these generators on this dataset only. Do not read them "
    "as a general ranking, especially for demo data."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricSummary:
    """One metric across repeats.

    ``std`` is the sample standard deviation (0 for a single value). All
    statistics are ``None`` when no repeat produced a value.
    """

    direction: str
    values: tuple[float, ...]
    mean: float | None
    std: float | None
    min: float | None
    max: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "direction": self.direction,
            "values": list(self.values),
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricSummary:
        """Restore a summary serialized with :meth:`to_dict`."""
        return cls(
            direction=data["direction"],
            values=tuple(data["values"]),
            mean=data["mean"],
            std=data["std"],
            min=data["min"],
            max=data["max"],
        )


def summarize(direction: str, values: Sequence[float | None]) -> MetricSummary:
    """Aggregate per-repeat values, ignoring repeats where the metric skipped."""
    data = [float(v) for v in values if v is not None]
    if not data:
        return MetricSummary(
            direction=direction, values=(), mean=None, std=None, min=None, max=None
        )
    mean = math.fsum(data) / len(data)
    std = (
        math.sqrt(math.fsum((v - mean) ** 2 for v in data) / (len(data) - 1))
        if len(data) > 1
        else 0.0
    )
    return MetricSummary(
        direction=direction,
        values=tuple(data),
        mean=mean,
        std=std,
        min=min(data),
        max=max(data),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratorResult:
    """Outcome for one generator.

    Attributes:
        name: Generator name.
        hyperparameters: Hyperparameters used for fitting.
        model_fingerprint: Fitted model fingerprint (``None`` if fit failed).
        fit_s: Fit wall-clock seconds.
        generate_s: Mean generation wall-clock seconds per repeat.
        seeds: Generation seeds, one per repeat.
        metrics: Metric name to summary across repeats.
        error: Error message if the generator failed, else ``None``.
    """

    name: str
    hyperparameters: Mapping[str, Any] = field(hash=False)
    model_fingerprint: str | None
    fit_s: float | None
    generate_s: float | None
    seeds: tuple[int, ...]
    metrics: Mapping[str, MetricSummary] = field(hash=False)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "name": self.name,
            "hyperparameters": dict(self.hyperparameters),
            "model_fingerprint": self.model_fingerprint,
            "fit_s": self.fit_s,
            "generate_s": self.generate_s,
            "seeds": list(self.seeds),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GeneratorResult:
        """Restore a result serialized with :meth:`to_dict`."""
        return cls(
            name=data["name"],
            hyperparameters=MappingProxyType(dict(data["hyperparameters"])),
            model_fingerprint=data["model_fingerprint"],
            fit_s=data["fit_s"],
            generate_s=data["generate_s"],
            seeds=tuple(data["seeds"]),
            metrics=MappingProxyType(
                {k: MetricSummary.from_dict(v) for k, v in data["metrics"].items()}
            ),
            error=data.get("error"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkReport:
    """A complete, reproducible benchmark result.

    Attributes:
        run_id: Run identifier.
        created_at: UTC creation time.
        protocol: The protocol settings used.
        seed: Global benchmark seed.
        dataset: Fingerprints and sizes of the full, train, and holdout data.
        reference: Metric values of the real train set against the holdout:
            the score of "a perfect generator", i.e. the noise floor.
        results: One entry per generator, in requested order.
        environment: Framework version and platform.
        duration_s: Total wall-clock seconds.
        report_version: Report layout version.
    """

    run_id: str
    created_at: datetime
    protocol: Mapping[str, Any] = field(hash=False)
    seed: int
    dataset: Mapping[str, Any] = field(hash=False)
    reference: Mapping[str, float | None] = field(hash=False)
    results: tuple[GeneratorResult, ...]
    environment: Mapping[str, Any] = field(hash=False)
    duration_s: float
    report_version: str = REPORT_VERSION

    def result(self, name: str) -> GeneratorResult:
        """Look up a generator's result.

        Raises:
            KeyError: If the generator was not benchmarked.
        """
        for result in self.results:
            if result.name == name:
                return result
        raise KeyError(name)

    def best(self, metric: str) -> str | None:
        """Generator with the best mean for ``metric`` (``None`` if no values)."""
        scored = [
            (r.metrics[metric].mean, r.name, r.metrics[metric].direction)
            for r in self.results
            if metric in r.metrics and r.metrics[metric].mean is not None
        ]
        if not scored:
            return None
        lower = scored[0][2] == "lower_better"
        return (min if lower else max)(scored, key=lambda item: item[0] or 0.0)[1]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "report_version": self.report_version,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "protocol": dict(self.protocol),
            "seed": self.seed,
            "dataset": dict(self.dataset),
            "reference": dict(self.reference),
            "results": [r.to_dict() for r in self.results],
            "environment": dict(self.environment),
            "duration_s": self.duration_s,
            "notes": [_DISCLAIMER],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkReport:
        """Restore a report serialized with :meth:`to_dict`."""
        return cls(
            run_id=data["run_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            protocol=MappingProxyType(dict(data["protocol"])),
            seed=data["seed"],
            dataset=MappingProxyType(dict(data["dataset"])),
            reference=MappingProxyType(dict(data["reference"])),
            results=tuple(GeneratorResult.from_dict(r) for r in data["results"]),
            environment=MappingProxyType(dict(data["environment"])),
            duration_s=float(data["duration_s"]),
            report_version=data.get("report_version", REPORT_VERSION),
        )

    def to_markdown(self) -> str:
        """Comparison table: mean ± std per generator, best in bold."""
        names = [r.name for r in self.results]
        holdout = self.dataset.get("holdout", {})
        lines = [
            f"# Benchmark `{self.protocol.get('name')}`",
            "",
            f"- Dataset `{self.dataset.get('dataset_id')}`: train "
            f"{self.dataset.get('train', {}).get('n_sessions')} sessions, holdout "
            f"{holdout.get('n_sessions')} sessions "
            f"({holdout.get('n_learners')} unseen learners)",
            f"- Seed {self.seed}, {self.protocol.get('repeats')} repeats; "
            f"run `{self.run_id}`, {self.duration_s:.1f}s",
            "- Reference = real train data scored against the holdout "
            "(what a perfect generator could reach)",
            "",
            "| Metric | Better | Reference | " + " | ".join(names) + " |",
            "| --- | --- | --- | " + " | ".join("---" for _ in names) + " |",
        ]
        for metric in self.protocol.get("metrics", []):
            direction = next(
                (
                    r.metrics[metric].direction
                    for r in self.results
                    if metric in r.metrics
                ),
                "lower_better",
            )
            best = self.best(metric)
            cells = []
            for r in self.results:
                summary = r.metrics.get(metric)
                if r.error or summary is None or summary.mean is None:
                    cells.append("-")
                    continue
                text = f"{summary.mean:.3g} ± {summary.std or 0:.2g}"
                cells.append(f"**{text}**" if r.name == best else text)
            better = "lower" if direction == "lower_better" else "higher"
            reference = self.reference.get(metric)
            ref = "-" if reference is None else f"{reference:.3g}"
            lines.append(f"| {metric} | {better} | {ref} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "| Generator | Fit (s) | Generate (s) | Status |",
            "| --- | --- | --- | --- |",
        ]
        for r in self.results:
            fit = "-" if r.fit_s is None else f"{r.fit_s:.2f}"
            gen = "-" if r.generate_s is None else f"{r.generate_s:.2f}"
            lines.append(f"| {r.name} | {fit} | {gen} | {r.error or 'ok'} |")
        lines += ["", f"_{_DISCLAIMER}_"]
        return "\n".join(lines) + "\n"
