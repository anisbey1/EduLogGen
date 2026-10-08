"""Session- and corpus-level features (FR-A.2, SAD §29C.2).

Session features are returned as table rows (one mapping per session) so they
can be exported or fed to tabular models. Corpus features are
:class:`~eduloggen.models.Feature` objects.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from eduloggen.analysis.sequences import session_sequences
from eduloggen.analysis.stats import entropy
from eduloggen.analysis.transitions import TransitionCounts
from eduloggen.models import Dataset, Feature, FeatureScope, LogRecord

__all__ = ["corpus_features", "session_feature_rows"]


def session_feature_rows(dataset: Dataset) -> list[dict[str, Any]]:
    """One row of features per session, ordered by ``session_id``.

    Columns: ``session_id``, ``learner_id``, ``course_id``,
    ``session_length``, ``session_duration_s``, ``n_unique_activities``,
    ``event_type_entropy`` (bits), and ``success_rate`` (``None`` when no
    event of the session has a success flag).

    Raises:
        AnalysisError: If the dataset has not been sessionized.
    """
    session_sequences(dataset)
    members: defaultdict[str, list[LogRecord]] = defaultdict(list)
    for event in dataset.events:
        if event.session_id is not None:
            members[event.session_id].append(event)
    rows = []
    for session in sorted(dataset.sessions or (), key=lambda s: s.session_id):
        events = members[session.session_id]
        flags = [event.success for event in events if event.success is not None]
        rows.append(
            {
                "session_id": session.session_id,
                "learner_id": session.learner_id,
                "course_id": session.course_id,
                "session_length": session.n_events,
                "session_duration_s": session.duration_s,
                "n_unique_activities": len({event.activity_id for event in events}),
                "event_type_entropy": entropy(Counter(e.event_type for e in events)),
                "success_rate": sum(flags) / len(flags) if flags else None,
            }
        )
    return rows


def corpus_features(
    dataset: Dataset, transitions: TransitionCounts
) -> tuple[Feature, ...]:
    """Corpus-wide size and structure features."""
    corpus = FeatureScope.CORPUS
    return (
        Feature(name="n_events", scope=corpus, value=dataset.n_events, dtype="int"),
        Feature(name="n_sessions", scope=corpus, value=dataset.n_sessions, dtype="int"),
        Feature(
            name="n_learners",
            scope=corpus,
            value=len(dataset.learner_ids),
            dtype="int",
        ),
        Feature(
            name="corpus_vocab_size",
            scope=corpus,
            value=len(transitions.vocabulary),
            dtype="int",
        ),
        Feature(
            name="corpus_transition_density",
            scope=corpus,
            value=transitions.density(),
            dtype="float",
            params={"order": transitions.order},
        ),
    )
