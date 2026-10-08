"""The :func:`analyze` façade (FR-A.3-A.6)."""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType

from eduloggen.analysis.features import corpus_features
from eduloggen.analysis.graphs import navigation_graph
from eduloggen.analysis.result import AnalysisResult
from eduloggen.analysis.sequences import ngram_counts, session_sequences
from eduloggen.analysis.stats import describe
from eduloggen.analysis.timing import (
    interevent_times,
    session_durations,
    sojourn_times,
)
from eduloggen.analysis.transitions import count_transitions
from eduloggen.core import AnalysisError
from eduloggen.models import Dataset

__all__ = ["analyze"]


def analyze(dataset: Dataset, *, ngram_order: int = 2) -> AnalysisResult:
    """Compute descriptive and behavioural statistics.

    Args:
        dataset: A sessionized dataset (see :func:`sessionize`).
        ngram_order: Longest n-gram to count (at least 1).

    Returns:
        The analysis result.

    Raises:
        AnalysisError: If the dataset is not sessionized or ``ngram_order``
            is less than 1.
    """
    if ngram_order < 1:
        raise AnalysisError(
            "ngram_order must be at least 1", code="analysis_invalid_parameter"
        )
    sequences = session_sequences(dataset)
    transitions = count_transitions(sequences, order=1)
    ngrams = {n: ngram_counts(sequences, n) for n in range(2, ngram_order + 1)}
    return AnalysisResult(
        dataset_id=dataset.dataset_id,
        dataset_fingerprint=dataset.fingerprint(),
        ngram_order=ngram_order,
        event_type_counts=MappingProxyType(
            dict(Counter(event.event_type for event in dataset.events))
        ),
        token_counts=MappingProxyType(
            dict(Counter(token for sequence in sequences for token in sequence))
        ),
        transitions=transitions,
        ngram_counts=MappingProxyType(
            {n: MappingProxyType(dict(counts)) for n, counts in ngrams.items()}
        ),
        n_rare_ngrams=MappingProxyType(
            {
                n: sum(1 for c in counts.values() if c == 1)
                for n, counts in ngrams.items()
            }
        ),
        session_length=describe(len(sequence) for sequence in sequences),
        session_duration_s=describe(session_durations(dataset)),
        interevent_s=describe(interevent_times(dataset)),
        sojourn_s=MappingProxyType(
            {
                token: describe(values)
                for token, values in sojourn_times(dataset).items()
            }
        ),
        corpus_features=corpus_features(dataset, transitions),
        graph=navigation_graph(sequences),
    )
