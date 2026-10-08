"""Privacy risk indicators (PRD §19, SAD §14.4, §29E.3).

These indicators flag memorisation of real sessions. They estimate risk; a
good score is not a guarantee of anonymity.
"""

from __future__ import annotations

import random
from typing import Any, ClassVar, TypeVar

from eduloggen.analysis import ngram_counts, quantile, rare_ngrams, session_sequences
from eduloggen.models import Dataset
from eduloggen.utils import make_rng
from eduloggen.validation.base import BaseMetric, Category, Direction, ValidationContext
from eduloggen.validation.distances import normalized_edit_distance

__all__ = [
    "ExactSessionDuplicateRate",
    "NearestNeighbourDistance",
    "RareNgramReplayRate",
]

Result = tuple[float | None, dict[str, Any]]
T = TypeVar("T")


class ExactSessionDuplicateRate(BaseMetric):
    """Share of synthetic sessions that copy a real session token for token.

    Only sessions with at least ``min_length`` events (default 3) count:
    very short sessions coincide by chance and say little about memorisation.
    """

    name: ClassVar[str] = "exact_session_dup_rate"
    category: ClassVar[Category] = "privacy"
    direction: ClassVar[Direction] = "lower_better"
    description = "Share of synthetic sessions identical to a real session"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> Result:
        min_length = int(context.params_for(self.name).get("min_length", 3))
        real_set = {s for s in session_sequences(real) if len(s) >= min_length}
        candidates = [s for s in session_sequences(synthetic) if len(s) >= min_length]
        if not candidates:
            return None, {
                "reason": f"no synthetic sessions with at least {min_length} events",
                "min_length": min_length,
            }
        copies = sum(1 for s in candidates if s in real_set)
        return copies / len(candidates), {
            "min_length": min_length,
            "synthetic_sessions_checked": len(candidates),
            "copies": copies,
        }


class RareNgramReplayRate(BaseMetric):
    """Share of real n-grams seen exactly once that synthetic data reproduces.

    Rare pathways are the most identifying. Parameter ``n`` (default 3).
    """

    name: ClassVar[str] = "rare_ngram_replay_rate"
    category: ClassVar[Category] = "privacy"
    direction: ClassVar[Direction] = "lower_better"
    description = "Share of once-seen real n-grams that appear in synthetic data"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> Result:
        n = int(context.params_for(self.name).get("n", 3))
        rare = rare_ngrams(ngram_counts(session_sequences(real), n), max_count=1)
        if not rare:
            return None, {"reason": f"real data has no rare {n}-grams", "n": n}
        produced = set(ngram_counts(session_sequences(synthetic), n))
        replayed = len(rare & produced)
        return replayed / len(rare), {
            "n": n,
            "rare_real_ngrams": len(rare),
            "replayed": replayed,
        }


class NearestNeighbourDistance(BaseMetric):
    """5th percentile of each synthetic session's distance to its closest real one.

    Distance is the normalized edit distance between token sequences, so 0
    means an exact copy and 1 means nothing in common. A low percentile means
    a noticeable share of synthetic sessions sit very close to real ones.
    Sessions are subsampled (seeded) to bound cost: parameters
    ``max_synthetic`` (default 200) and ``max_real`` (default 1000).
    """

    name: ClassVar[str] = "nn_distance_p05"
    category: ClassVar[Category] = "privacy"
    direction: ClassVar[Direction] = "higher_better"
    description = "5th percentile nearest-real-session edit distance (0 = copies)"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> Result:
        params = context.params_for(self.name)
        rng = make_rng(context.seed, "validation", self.name)
        real_seqs = list(session_sequences(real))
        synthetic_seqs = list(session_sequences(synthetic))
        if not real_seqs or not synthetic_seqs:
            return None, {"reason": "a dataset has no sessions"}
        real_seqs = _subsample(real_seqs, int(params.get("max_real", 1000)), rng)
        synthetic_seqs = _subsample(
            synthetic_seqs, int(params.get("max_synthetic", 200)), rng
        )
        exact = set(real_seqs)
        distances = sorted(
            0.0 if seq in exact else _nearest(seq, real_seqs) for seq in synthetic_seqs
        )
        return quantile(distances, 0.05), {
            "synthetic_sessions_checked": len(synthetic_seqs),
            "real_sessions_compared": len(real_seqs),
            "median": quantile(distances, 0.5),
            "share_exact": sum(d == 0 for d in distances) / len(distances),
        }


def _subsample(items: list[T], limit: int, rng: random.Random) -> list[T]:
    if len(items) <= limit:
        return items
    return rng.sample(items, limit)


def _nearest(sequence: tuple[str, ...], candidates: list[tuple[str, ...]]) -> float:
    best = 1.0
    length = len(sequence)
    for other in candidates:
        longest = max(length, len(other))
        if longest and abs(length - len(other)) / longest >= best:
            continue
        best = min(best, normalized_edit_distance(sequence, other))
        if best == 0.0:
            break
    return best
