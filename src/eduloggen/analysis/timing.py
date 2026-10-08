"""Time statistics: inter-event gaps and per-token sojourns (FR-A.5, §11.6).

Within a session, the *sojourn* of an event is the time until the next event
of the same session. It is attributed to the event's token, which is what a
semi-Markov model needs: how long a learner stays in a state before moving
on. The last event of a session has no sojourn.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from itertools import pairwise

from eduloggen.analysis.sequences import session_sequences
from eduloggen.core import AnalysisError
from eduloggen.models import Dataset, LogRecord

__all__ = ["interevent_times", "session_durations", "sojourn_times"]


def _session_events(
    dataset: Dataset,
) -> Iterator[tuple[tuple[str, ...], list[LogRecord]]]:
    session_sequences(dataset)  # raises if not sessionized
    members: defaultdict[str, list[LogRecord]] = defaultdict(list)
    for event in dataset.sorted_events():
        if event.session_id is not None:
            members[event.session_id].append(event)
    for session in sorted(dataset.sessions or (), key=lambda s: s.session_id):
        events = members[session.session_id]
        if len(events) != session.n_events:  # pragma: no cover - Dataset invariant
            raise AnalysisError(
                "session events are inconsistent", code="analysis_invalid_data"
            )
        yield session.event_sequence, events


def interevent_times(dataset: Dataset) -> list[float]:
    """Seconds between consecutive events within each session.

    Raises:
        AnalysisError: If the dataset has not been sessionized.
    """
    gaps: list[float] = []
    for _, events in _session_events(dataset):
        gaps.extend(
            (later.timestamp - earlier.timestamp).total_seconds()
            for earlier, later in pairwise(events)
        )
    return gaps


def sojourn_times(dataset: Dataset) -> dict[str, list[float]]:
    """Seconds spent on each token before the next event, grouped by token.

    Raises:
        AnalysisError: If the dataset has not been sessionized.
    """
    result: defaultdict[str, list[float]] = defaultdict(list)
    for tokens, events in _session_events(dataset):
        for token, earlier, later in zip(tokens, events, events[1:], strict=False):
            result[token].append((later.timestamp - earlier.timestamp).total_seconds())
    return dict(sorted(result.items()))


def session_durations(dataset: Dataset) -> list[float]:
    """Duration in seconds of every session.

    Raises:
        AnalysisError: If the dataset has not been sessionized.
    """
    session_sequences(dataset)
    return [session.duration_s for session in dataset.sessions or ()]
