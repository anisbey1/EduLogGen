"""Score profile recovery against ground-truth profile annotations (M3).

Compares a predicted ``learner_id -> cluster`` mapping with the ``profile``
annotations written by :func:`eduloggen.scenarios.generate_profiles`. The
metrics are invariant to cluster names: adjusted Rand index (ARI),
normalised mutual information (NMI, arithmetic normalisation), and purity.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from eduloggen.core import ConfigError
from eduloggen.models import Annotations

__all__ = [
    "ClusteringReport",
    "adjusted_rand_index",
    "evaluate_clustering",
    "normalized_mutual_information",
    "purity",
]


@dataclass(frozen=True, slots=True)
class ClusteringReport:
    """Agreement between true profiles and predicted clusters.

    Attributes:
        n_learners: Learners scored (annotated and predicted).
        missing: Annotated learners without a prediction.
        ari: Adjusted Rand index (1 perfect, about 0 random).
        nmi: Normalised mutual information in [0, 1].
        purity: Share of learners in their cluster's majority profile.
        contingency: True profile to predicted cluster to count.
    """

    n_learners: int
    missing: int
    ari: float
    nmi: float
    purity: float
    contingency: Mapping[str, Mapping[str, int]] = field(hash=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible summary."""
        return {
            "n_learners": self.n_learners,
            "missing": self.missing,
            "ari": self.ari,
            "nmi": self.nmi,
            "purity": self.purity,
            "contingency": {k: dict(v) for k, v in self.contingency.items()},
        }

    def to_markdown(self) -> str:
        """Short Markdown report."""
        return (
            "# Profile recovery\n\n"
            f"Learners: {self.n_learners} (missing predictions: {self.missing})\n\n"
            "| ARI | NMI | Purity |\n| --- | --- | ------ |\n"
            f"| {self.ari:.3f} | {self.nmi:.3f} | {self.purity:.3f} |\n"
        )


def evaluate_clustering(
    annotations: Annotations, predictions: Mapping[str, str]
) -> ClusteringReport:
    """Score predicted clusters against learner ``profile`` annotations.

    Raises:
        ConfigError: If there are no profile annotations or no overlap.
    """
    truth = {
        row.id: str(row.value)
        for row in annotations.rows
        if row.level == "learner" and row.type == "profile"
    }
    if not truth:
        raise ConfigError(
            "annotations have no learner profile rows", code="config_invalid_value"
        )
    shared = sorted(set(truth) & set(predictions))
    if not shared:
        raise ConfigError(
            "no predicted learner has a profile annotation",
            code="config_invalid_value",
        )
    true = [truth[i] for i in shared]
    pred = [str(predictions[i]) for i in shared]
    table: dict[str, Counter[str]] = {}
    for t, p in zip(true, pred, strict=True):
        table.setdefault(t, Counter())[p] += 1
    return ClusteringReport(
        n_learners=len(shared),
        missing=len(truth) - len(shared),
        ari=adjusted_rand_index(true, pred),
        nmi=normalized_mutual_information(true, pred),
        purity=purity(true, pred),
        contingency={k: dict(sorted(v.items())) for k, v in sorted(table.items())},
    )


def _pairs(n: float) -> float:
    return n * (n - 1) / 2


def adjusted_rand_index(true: list[str], pred: list[str]) -> float:
    """Adjusted Rand index of two labelings of the same items."""
    n = len(true)
    joint = Counter(zip(true, pred, strict=True))
    index = sum(_pairs(v) for v in joint.values())
    a = sum(_pairs(v) for v in Counter(true).values())
    b = sum(_pairs(v) for v in Counter(pred).values())
    expected = a * b / _pairs(n) if n > 1 else 0.0
    top = (a + b) / 2
    if top == expected:
        return 1.0
    return (index - expected) / (top - expected)


def _entropy(counts: Counter[Any], n: int) -> float:
    return -sum(c / n * math.log(c / n) for c in counts.values())


def normalized_mutual_information(true: list[str], pred: list[str]) -> float:
    """Mutual information divided by the mean of the two entropies."""
    n = len(true)
    ct, cp = Counter(true), Counter(pred)
    h = (_entropy(ct, n) + _entropy(cp, n)) / 2
    if h == 0:
        return 1.0
    mi = sum(
        c / n * math.log(c * n / (ct[t] * cp[p]))
        for (t, p), c in Counter(zip(true, pred, strict=True)).items()
    )
    return max(0.0, min(1.0, mi / h))


def purity(true: list[str], pred: list[str]) -> float:
    """Share of items that belong to their cluster's majority class."""
    by_cluster: dict[str, Counter[str]] = {}
    for t, p in zip(true, pred, strict=True):
        by_cluster.setdefault(p, Counter())[t] += 1
    return sum(c.most_common(1)[0][1] for c in by_cluster.values()) / len(true)
