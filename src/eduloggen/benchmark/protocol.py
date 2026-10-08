"""Versioned benchmark protocols and the learner-level split (PRD §18, ADR-015).

A protocol freezes everything that must be identical across candidates:
split, repeats, sample size, and metrics. Changing any of them means a new
protocol version, so published numbers stay comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from eduloggen.core import BenchmarkError, ConfigError, PluginError
from eduloggen.models import Dataset
from eduloggen.utils import make_rng
from eduloggen.validation import DEFAULT_METRICS, available_metrics

__all__ = [
    "BUILTIN_PROTOCOLS",
    "PROTOCOLS",
    "BenchmarkProtocol",
    "get_protocol",
    "register_protocol",
    "split_by_learner",
    "unregister_protocol",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkProtocol:
    """A frozen evaluation recipe.

    Attributes:
        name: Protocol id, including its version (e.g. ``session_fidelity_v1``).
        description: What the protocol measures.
        holdout_fraction: Share of learners held out as the reference set.
        repeats: Generation seeds per generator.
        n_sessions: Sessions to generate; ``None`` matches the holdout size.
        metrics: Validation metrics computed against the holdout.
    """

    name: str
    description: str
    holdout_fraction: float = 0.3
    repeats: int = 3
    n_sessions: int | None = None
    metrics: tuple[str, ...] = DEFAULT_METRICS

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "name": self.name,
            "description": self.description,
            "holdout_fraction": self.holdout_fraction,
            "repeats": self.repeats,
            "n_sessions": self.n_sessions,
            "metrics": list(self.metrics),
        }


PROTOCOLS: dict[str, BenchmarkProtocol] = {
    "session_fidelity_v1": BenchmarkProtocol(
        name="session_fidelity_v1",
        description=(
            "Fit on 70% of learners, generate as many sessions as the 30% "
            "holdout has, and compare each sample with the holdout on all "
            "built-in metrics; 3 seeds per generator."
        ),
    ),
}


BUILTIN_PROTOCOLS = frozenset(PROTOCOLS)
"""Names of the protocols shipped with EduLogGen."""


def register_protocol(protocol: BenchmarkProtocol, *, replace: bool = False) -> None:
    """Add a benchmark protocol (e.g. from a plugin).

    Raises:
        PluginError: If the object is not a protocol, its name is invalid,
            built in, or taken (without ``replace``), or a metric is unknown.
    """
    if not isinstance(protocol, BenchmarkProtocol):
        raise PluginError(
            "benchmark suites must be BenchmarkProtocol instances",
            code="plugin_contract_violation",
        )
    if not protocol.name.isidentifier():
        raise PluginError(
            "protocol names must be identifiers",
            code="plugin_invalid_name",
            context={"name": protocol.name},
        )
    if protocol.name in BUILTIN_PROTOCOLS or (
        protocol.name in PROTOCOLS and not replace
    ):
        raise PluginError(
            f"protocol {protocol.name!r} is already registered",
            code="plugin_duplicate",
            context={"name": protocol.name},
        )
    unknown = sorted(set(protocol.metrics) - set(available_metrics()))
    if unknown or not 0 < protocol.holdout_fraction < 1 or protocol.repeats < 1:
        raise PluginError(
            "protocol has unknown metrics or invalid settings",
            code="plugin_contract_violation",
            context={"name": protocol.name, "unknown_metrics": unknown},
        )
    PROTOCOLS[protocol.name] = protocol


def unregister_protocol(name: str) -> None:
    """Remove a non-built-in protocol (no-op if absent)."""
    if name not in BUILTIN_PROTOCOLS:
        PROTOCOLS.pop(name, None)


def get_protocol(name: str) -> BenchmarkProtocol:
    """Look up a protocol by name.

    Raises:
        ConfigError: If the protocol is unknown.
    """
    try:
        return PROTOCOLS[name]
    except KeyError:
        raise ConfigError(
            f"unknown benchmark protocol {name!r}",
            code="benchmark_unknown_protocol",
            context={"name": name, "available": sorted(PROTOCOLS)},
        ) from None


def split_by_learner(
    dataset: Dataset, holdout_fraction: float, seed: int
) -> tuple[Dataset, Dataset]:
    """Split a sessionized dataset into train and holdout by learner.

    Every learner lands wholly on one side, so the holdout measures
    generalization to unseen learners rather than recall of seen ones.

    Raises:
        BenchmarkError: If the dataset is not sessionized, has fewer than two
            learners, or the fraction is not strictly between 0 and 1.
    """
    if dataset.sessions is None:
        raise BenchmarkError(
            "dataset must be sessionized", code="benchmark_not_sessionized"
        )
    if not 0 < holdout_fraction < 1:
        raise BenchmarkError(
            "holdout_fraction must be between 0 and 1",
            code="benchmark_invalid_argument",
        )
    learners = sorted(dataset.learner_ids)
    if len(learners) < 2:
        raise BenchmarkError(
            "a benchmark needs at least two learners",
            code="benchmark_insufficient_data",
        )
    make_rng(seed, "benchmark", "split").shuffle(learners)
    n_holdout = min(
        max(1, math.ceil(len(learners) * holdout_fraction)), len(learners) - 1
    )
    holdout_ids = frozenset(learners[:n_holdout])

    def part(keep: bool, suffix: str) -> Dataset:
        return Dataset(
            dataset_id=f"{dataset.dataset_id}-{suffix}",
            events=tuple(
                e for e in dataset.events if (e.learner_id in holdout_ids) == keep
            ),
            sessions=tuple(
                s
                for s in dataset.sessions or ()
                if (s.learner_id in holdout_ids) == keep
            ),
            metadata=dataset.metadata,
        )

    return part(False, "train"), part(True, "holdout")
