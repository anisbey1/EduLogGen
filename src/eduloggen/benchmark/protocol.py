"""Versioned benchmark protocols and the learner-level split (PRD §18, ADR-015).

A protocol freezes everything that must be identical across candidates:
split, repeats, sample size, and metrics. Changing any of them means a new
protocol version, so published numbers stay comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from eduloggen.core import BenchmarkError, ConfigError
from eduloggen.models import Dataset
from eduloggen.utils import make_rng
from eduloggen.validation import DEFAULT_METRICS

__all__ = ["PROTOCOLS", "BenchmarkProtocol", "get_protocol", "split_by_learner"]


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
