"""Group events into sessions (FR-A.1, SAD §11.2, §29C.1).

Three strategies:

- ``explicit``: use the source ``session_id`` of every event.
- ``idle_timeout``: per learner, start a new session whenever the gap since
  the previous event exceeds the timeout. Source session ids are ignored.
- ``composite`` (default): learners whose events all carry a session id keep
  them; other learners are split by idle timeout.

Generated session ids have the form ``"<learner_id>:s<k>"`` with ``k``
counting from 1 in time order per learner.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from typing import Final, Literal, TypeVar

from eduloggen.core import AnalysisError, SchemaError
from eduloggen.models import Dataset, LogRecord, Session, TokenField

__all__ = ["DEFAULT_IDLE_TIMEOUT_S", "SessionStrategy", "sessionize"]

SessionStrategy = Literal["explicit", "idle_timeout", "composite"]

DEFAULT_IDLE_TIMEOUT_S: Final = 1800.0
"""Default inactivity gap (30 minutes), a common choice in web analytics."""

D = TypeVar("D", bound=Dataset)


def sessionize(
    dataset: D,
    *,
    strategy: SessionStrategy = "composite",
    idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
    tokenization: TokenField = "event_type",
) -> D:
    """Build sessions and backfill each event's ``session_id``.

    Args:
        dataset: Events to group; existing sessions are replaced.
        strategy: Grouping strategy (see module docs).
        idle_timeout_s: Gap in seconds that starts a new session.
        tokenization: Event attribute used for session token sequences.

    Returns:
        A new dataset of the same type with every event assigned to a
        session and ``sessions`` populated.

    Raises:
        AnalysisError: If ``strategy`` is ``explicit`` and an event has no
            session id, an explicit session spans several learners, the
            timeout is not positive, or a generated id collides with an
            explicit one.
    """
    if idle_timeout_s <= 0:
        raise AnalysisError(
            "idle_timeout_s must be positive", code="analysis_invalid_parameter"
        )
    if strategy not in ("explicit", "idle_timeout", "composite"):
        raise AnalysisError(
            "unknown sessionization strategy",
            code="analysis_invalid_parameter",
            context={"strategy": strategy},
        )

    by_learner: defaultdict[str, list[LogRecord]] = defaultdict(list)
    for event in dataset.sorted_events():
        by_learner[event.learner_id].append(event)

    assigned: list[LogRecord] = []
    explicit_ids: set[str] = set()
    generated_ids: set[str] = set()
    for learner_id, events in by_learner.items():
        has_ids = all(event.session_id is not None for event in events)
        if strategy == "explicit" and not has_ids:
            raise AnalysisError(
                "explicit sessionization needs a session_id on every event",
                code="analysis_missing_session_id",
            )
        if strategy == "idle_timeout" or (strategy == "composite" and not has_ids):
            split = _split_by_timeout(learner_id, events, idle_timeout_s)
            generated_ids.update(event.session_id or "" for event in split)
            assigned.extend(split)
        else:
            explicit_ids.update(event.session_id or "" for event in events)
            assigned.extend(events)

    collisions = explicit_ids & generated_ids
    if collisions:
        raise AnalysisError(
            "generated session ids collide with explicit ones",
            code="analysis_session_id_collision",
            context={"count": len(collisions)},
        )
    sessions = _build_sessions(assigned, tokenization)
    return replace(dataset, events=tuple(assigned), sessions=sessions)


def _split_by_timeout(
    learner_id: str, events: list[LogRecord], idle_timeout_s: float
) -> list[LogRecord]:
    result: list[LogRecord] = []
    index = 0
    previous: LogRecord | None = None
    for event in events:
        gap = (
            None
            if previous is None
            else (event.timestamp - previous.timestamp).total_seconds()
        )
        if gap is None or gap > idle_timeout_s:
            index += 1
        result.append(replace(event, session_id=f"{learner_id}:s{index}"))
        previous = event
    return result


def _build_sessions(
    events: Iterable[LogRecord], tokenization: TokenField
) -> tuple[Session, ...]:
    groups: defaultdict[str, list[LogRecord]] = defaultdict(list)
    for event in events:
        groups[event.session_id or ""].append(event)
    try:
        return tuple(
            Session.from_events(session_id, members, token_field=tokenization)
            for session_id, members in sorted(groups.items())
        )
    except SchemaError as exc:
        raise AnalysisError(
            "an explicit session contains events of several learners",
            code="analysis_mixed_session",
        ) from exc
