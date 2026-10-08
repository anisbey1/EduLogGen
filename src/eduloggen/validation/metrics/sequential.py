"""Sequential and navigation metrics (SAD §29E.2)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, ClassVar

from eduloggen.analysis import ngram_counts, session_sequences
from eduloggen.models import Dataset
from eduloggen.validation.base import BaseMetric, Category, Direction, ValidationContext
from eduloggen.validation.distances import jensen_shannon, total_variation

__all__ = ["BigramTVD", "TopNPathOverlap", "TransitionJSD"]

Result = tuple[float | None, dict[str, Any]]


class BigramTVD(BaseMetric):
    """Total variation distance between within-session bigram frequencies."""

    name: ClassVar[str] = "bigram_tvd"
    category: ClassVar[Category] = "sequential"
    direction: ClassVar[Direction] = "lower_better"
    description = "Share of bigram probability mass that must move (0 = same)"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> Result:
        p = ngram_counts(session_sequences(real), 2)
        q = ngram_counts(session_sequences(synthetic), 2)
        if not p or not q:
            return None, {"reason": "a dataset has no multi-event sessions"}
        return total_variation(p, q), {
            "real_bigrams": len(p),
            "synthetic_bigrams": len(q),
            "synthetic_bigrams_unseen_in_real": len(q.keys() - p.keys()),
        }


class TransitionJSD(BaseMetric):
    """Mean Jensen-Shannon divergence between next-token distributions.

    For each token observed as a predecessor in the real data, compare
    ``P(next | token)`` in real and synthetic data; average the divergences
    weighted by how often the token precedes another in real data. A token
    never followed by anything in synthetic data counts as maximal divergence
    (1.0).
    """

    name: ClassVar[str] = "transition_jsd"
    category: ClassVar[Category] = "sequential"
    direction: ClassVar[Direction] = "lower_better"
    description = "Real-frequency-weighted JSD of next-token distributions"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> Result:
        real_rows = _rows(ngram_counts(session_sequences(real), 2))
        if not real_rows:
            return None, {"reason": "real data has no multi-event sessions"}
        synthetic_rows = _rows(ngram_counts(session_sequences(synthetic), 2))
        total = sum(sum(row.values()) for row in real_rows.values())
        divergence = 0.0
        missing = 0
        for token, row in real_rows.items():
            weight = sum(row.values()) / total
            other = synthetic_rows.get(token)
            if other is None:
                missing += 1
                divergence += weight
            else:
                divergence += weight * jensen_shannon(row, other)
        return divergence, {
            "real_states": len(real_rows),
            "states_missing_in_synthetic": missing,
        }


def _rows(bigrams: Counter[tuple[str, ...]]) -> dict[str, Counter[str]]:
    rows: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for (first, second), count in bigrams.items():
        rows[first][second] += count
    return dict(rows)


class TopNPathOverlap(BaseMetric):
    """Share of the real top-N paths (n-grams) also in the synthetic top-N.

    Parameters: ``n`` (most frequent paths compared, default 10) and
    ``path_length`` (tokens per path, default 3).
    """

    name: ClassVar[str] = "topn_path_overlap"
    category: ClassVar[Category] = "navigation"
    direction: ClassVar[Direction] = "higher_better"
    description = "Overlap of the most frequent navigation paths (1 = identical)"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> Result:
        params = context.params_for(self.name)
        n = int(params.get("n", 10))
        length = int(params.get("path_length", 3))
        real_top = _top(ngram_counts(session_sequences(real), length), n)
        if not real_top:
            return None, {"reason": f"real data has no paths of length {length}"}
        synthetic_top = _top(ngram_counts(session_sequences(synthetic), length), n)
        shared = real_top & synthetic_top
        return len(shared) / len(real_top), {
            "n": n,
            "path_length": length,
            "compared": len(real_top),
            "shared": len(shared),
        }


def _top(counts: Counter[tuple[str, ...]], n: int) -> set[tuple[str, ...]]:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {gram for gram, _ in ranked[:n]}
