"""Relate session tokens back to event fields (ADR-019).

Session sequences hold one token per event, taken from either ``event_type``
or ``activity_id`` at sessionization time. Generators sample tokens, so to
emit full events they must know which field the token fills and how the
other field is distributed for each token (its *companion*).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from eduloggen.analysis import session_sequences
from eduloggen.core import FitError
from eduloggen.models import Dataset, LogRecord, TokenField

__all__ = ["companion_counts", "companion_field", "detect_tokenization"]


def detect_tokenization(dataset: Dataset) -> TokenField:
    """Infer which event field the session tokens were taken from.

    ``event_type`` wins when both fields match (e.g. identical columns).

    Raises:
        FitError: If the tokens match neither field.
    """
    sequences = dict(
        zip(
            sorted(s.session_id for s in dataset.sessions or ()),
            session_sequences(dataset),
            strict=True,
        )
    )
    members = _members(dataset)
    for field in ("event_type", "activity_id"):
        if all(
            tuple(getattr(event, field) for event in members[sid]) == tokens
            for sid, tokens in sequences.items()
        ):
            return field
    raise FitError(
        "session sequences match neither event_type nor activity_id; "
        "re-run sessionize on this dataset",
        code="fit_unknown_tokenization",
    )


def companion_field(tokenization: TokenField) -> TokenField:
    """The event field not covered by the tokens."""
    return "activity_id" if tokenization == "event_type" else "event_type"


def companion_counts(
    dataset: Dataset, tokenization: TokenField
) -> dict[str, dict[str, int]]:
    """For each token, how often each companion value occurs with it."""
    other = companion_field(tokenization)
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for events in _members(dataset).values():
        for event in events:
            counts[getattr(event, tokenization)][getattr(event, other)] += 1
    return {token: dict(sorted(c.items())) for token, c in sorted(counts.items())}


def _members(dataset: Dataset) -> dict[str, list[LogRecord]]:
    members: defaultdict[str, list[LogRecord]] = defaultdict(list)
    for event in dataset.sorted_events():
        if event.session_id is not None:
            members[event.session_id].append(event)
    return members
