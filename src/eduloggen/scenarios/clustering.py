"""Learner features and seeded k-means++ for automatic profiles (M3).

Pure Python: features are standardised (z-scores) with statistics from the
training data only, so holdout learners can later be assigned to the nearest
learned profile without influencing it.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from eduloggen.analysis import learner_profiles, session_sequences
from eduloggen.core import ConfigError
from eduloggen.models import Dataset

__all__ = [
    "KMeansResult",
    "Standardizer",
    "feature_means",
    "kmeans",
    "learner_features",
    "nearest",
]


def learner_features(
    dataset: Dataset, tokens: Sequence[str] | None = None
) -> tuple[list[str], dict[str, list[float]]]:
    """Behavioural features per learner.

    Features: number of sessions, mean session length, log mean session
    duration, success rate (0.5 when unknown), and the share of each token.

    Args:
        dataset: Sessionized dataset.
        tokens: Token columns to use (defaults to the dataset's tokens, sorted);
            pass the training tokens when featurising new data.

    Returns:
        Feature names and learner id to feature vector.
    """
    sequences = session_sequences(dataset)
    vocabulary = (
        list(tokens)
        if tokens is not None
        else sorted({t for s in sequences for t in s})
    )
    token_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for session in dataset.sessions or ():
        token_counts[session.learner_id].update(session.event_sequence)
    success: defaultdict[str, list[bool]] = defaultdict(list)
    for event in dataset.events:
        if event.success is not None:
            success[event.learner_id].append(event.success)
    names = ["n_sessions", "mean_session_length", "log_mean_duration_s", "success_rate"]
    names += [f"share:{t}" for t in vocabulary]
    vectors = {}
    for row in learner_profiles(dataset):
        learner = row["learner_id"]
        if not row["learner_n_sessions"]:
            continue
        counts = token_counts[learner]
        total = sum(counts.values()) or 1
        flags = success[learner]
        vectors[learner] = [
            float(row["learner_n_sessions"]),
            float(row["mean_session_length"]),
            math.log1p(float(row["mean_session_duration_s"])),
            sum(flags) / len(flags) if flags else 0.5,
            *(counts.get(t, 0) / total for t in vocabulary),
        ]
    return names, vectors


@dataclass(frozen=True, slots=True)
class Standardizer:
    """Per-feature mean and standard deviation (1 where a feature is constant)."""

    means: tuple[float, ...]
    stds: tuple[float, ...]

    @classmethod
    def fit(cls, rows: Sequence[Sequence[float]]) -> Standardizer:
        """Estimate from training rows."""
        n = len(rows)
        means = [sum(col) / n for col in zip(*rows, strict=True)]
        stds = [
            math.sqrt(sum((v - m) ** 2 for v in col) / n) or 1.0
            for col, m in zip(zip(*rows, strict=True), means, strict=True)
        ]
        return cls(tuple(means), tuple(stds))

    def transform(self, row: Sequence[float]) -> list[float]:
        """Z-scores of one row."""
        return [(v - m) / s for v, m, s in zip(row, self.means, self.stds, strict=True)]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {"means": list(self.means), "stds": list(self.stds)}


@dataclass(frozen=True, slots=True)
class KMeansResult:
    """Clustering of standardised rows.

    Attributes:
        centroids: One centroid per cluster (standardised space).
        labels: Cluster index per input row.
        inertia: Sum of squared distances to the assigned centroid.
    """

    centroids: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...] = field(hash=False)
    inertia: float


def kmeans(
    rows: Sequence[Sequence[float]],
    k: int,
    *,
    rng: random.Random,
    n_init: int = 10,
    max_iter: int = 100,
) -> KMeansResult:
    """Seeded k-means++ with several restarts; the lowest inertia wins.

    Raises:
        ConfigError: If ``k`` is not between 1 and the number of rows.
    """
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(rows):
        raise ConfigError(
            f"n_profiles must be between 1 and the number of learners ({len(rows)})",
            code="config_invalid_value",
            context={"key": "profiles.n_profiles"},
        )
    best: KMeansResult | None = None
    for _ in range(n_init):
        centroids = _init(rows, k, rng)
        labels: list[int] = []
        for _ in range(max_iter):
            new_labels = [_nearest(row, centroids)[0] for row in rows]
            if new_labels == labels:
                break
            labels = new_labels
            centroids = _update(rows, labels, centroids)
        inertia = sum(
            _sq(row, centroids[label]) for row, label in zip(rows, labels, strict=True)
        )
        if best is None or inertia < best.inertia - 1e-12:
            best = KMeansResult(
                tuple(tuple(c) for c in centroids), tuple(labels), inertia
            )
    assert best is not None  # n_init >= 1
    return best


def nearest(row: Sequence[float], centroids: Sequence[Sequence[float]]) -> int:
    """Index of the closest centroid."""
    return _nearest(row, centroids)[0]


def _sq(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def _nearest(
    row: Sequence[float], centroids: Sequence[Sequence[float]]
) -> tuple[int, float]:
    distances = [_sq(row, c) for c in centroids]
    index = min(range(len(centroids)), key=lambda i: (distances[i], i))
    return index, distances[index]


def _init(
    rows: Sequence[Sequence[float]], k: int, rng: random.Random
) -> list[list[float]]:
    centroids = [list(rows[rng.randrange(len(rows))])]
    while len(centroids) < k:
        weights = [_nearest(row, centroids)[1] for row in rows]
        if not any(weights):
            centroids.append(list(rows[rng.randrange(len(rows))]))
            continue
        centroids.append(list(rng.choices(rows, weights)[0]))
    return centroids


def _update(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    centroids: Sequence[Sequence[float]],
) -> list[list[float]]:
    members: defaultdict[int, list[Sequence[float]]] = defaultdict(list)
    for row, label in zip(rows, labels, strict=True):
        members[label].append(row)
    updated = []
    for index in range(len(centroids)):
        group = members.get(index)
        if not group:
            # empty cluster: move it to the point farthest from its centroid
            far = max(rows, key=lambda r: _sq(r, centroids[_nearest(r, centroids)[0]]))
            updated.append(list(far))
            continue
        updated.append([sum(col) / len(group) for col in zip(*group, strict=True)])
    return updated


def feature_means(
    rows: Mapping[str, Sequence[float]], ids: Sequence[str]
) -> list[float]:
    """Mean raw feature vector of the given learners."""
    group = [rows[i] for i in ids]
    return [sum(col) / len(group) for col in zip(*group, strict=True)]
