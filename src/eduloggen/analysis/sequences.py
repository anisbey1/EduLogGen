"""Token sequences and n-gram statistics (SAD §11.7)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from eduloggen.core import AnalysisError
from eduloggen.models import Dataset

__all__ = ["ngram_counts", "rare_ngrams", "session_sequences"]

NGram = tuple[str, ...]


def session_sequences(dataset: Dataset) -> tuple[tuple[str, ...], ...]:
    """Token sequences of all sessions, ordered by ``session_id``.

    Raises:
        AnalysisError: If the dataset has not been sessionized.
    """
    if dataset.sessions is None:
        raise AnalysisError(
            "dataset must be sessionized first", code="analysis_not_sessionized"
        )
    ordered = sorted(dataset.sessions, key=lambda session: session.session_id)
    return tuple(session.event_sequence for session in ordered)


def ngram_counts(sequences: Iterable[Sequence[str]], n: int) -> Counter[NGram]:
    """Count contiguous n-grams within each sequence (never across them).

    Raises:
        AnalysisError: If ``n`` is less than 1.
    """
    if n < 1:
        raise AnalysisError("n must be at least 1", code="analysis_invalid_parameter")
    counts: Counter[NGram] = Counter()
    for sequence in sequences:
        tokens = tuple(sequence)
        for start in range(len(tokens) - n + 1):
            counts[tokens[start : start + n]] += 1
    return counts


def rare_ngrams(counts: Counter[NGram], max_count: int = 1) -> frozenset[NGram]:
    """N-grams seen at most ``max_count`` times.

    Rare pathways are the ones most likely to identify a learner; validation
    checks whether synthetic data replays them.
    """
    return frozenset(gram for gram, count in counts.items() if count <= max_count)
