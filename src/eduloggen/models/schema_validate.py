"""Cross-record schema checks for datasets (ingest Gate A, session Gate B).

Single records validate themselves on construction. The checks here cover
rules that span records: identifier uniqueness, session references, and
schema version compatibility.
"""

from __future__ import annotations

from collections.abc import Sequence

from eduloggen.core import SCHEMA_VERSION, SchemaError
from eduloggen.models import _checks as chk
from eduloggen.models.records import LogRecord, Session

__all__ = [
    "check_schema_version",
    "validate_events",
    "validate_sessions",
]


def check_schema_version(version: str) -> None:
    """Ensure a declared schema version is readable by this release.

    Versions are compatible when their major component matches
    :data:`~eduloggen.core.SCHEMA_VERSION`.

    Args:
        version: Declared ``schema_version`` (``"MAJOR.MINOR"``).

    Raises:
        SchemaError: If the version is malformed or has a different major.
    """
    major, _, minor = version.partition(".")
    if not (major.isdigit() and minor.isdigit()):
        raise SchemaError(
            "schema_version must look like 'MAJOR.MINOR'",
            code=chk.UNSUPPORTED_VERSION,
            context={"schema_version": version},
        )
    if major != SCHEMA_VERSION.partition(".")[0]:
        raise SchemaError(
            f"schema_version {version} is incompatible with {SCHEMA_VERSION}",
            code=chk.UNSUPPORTED_VERSION,
            context={"schema_version": version, "supported": SCHEMA_VERSION},
        )


def validate_events(events: Sequence[LogRecord]) -> None:
    """Check rules spanning several events.

    Args:
        events: Events of one dataset.

    Raises:
        SchemaError: If an item is not a :class:`LogRecord` or an
            ``event_id`` repeats.
    """
    seen: set[str] = set()
    for event in events:
        if not isinstance(event, LogRecord):
            raise chk.invalid("Dataset", "events", "must contain LogRecord items")
        if event.event_id in seen:
            raise _duplicate("event_id", event.event_id)
        seen.add(event.event_id)


def validate_sessions(sessions: Sequence[Session], events: Sequence[LogRecord]) -> None:
    """Check sessions against each other and against the events they cover.

    Every event with a ``session_id`` must reference a listed session owned by
    the same learner, and every session must have exactly ``n_events`` events.

    Args:
        sessions: Sessions of one dataset.
        events: Events of the same dataset.

    Raises:
        SchemaError: If sessions repeat, an event references an unknown
            session or another learner's session, or event counts disagree.
    """
    by_id: dict[str, Session] = {}
    for session in sessions:
        if not isinstance(session, Session):
            raise chk.invalid("Dataset", "sessions", "must contain Session items")
        if session.session_id in by_id:
            raise _duplicate("session_id", session.session_id)
        by_id[session.session_id] = session

    counts = dict.fromkeys(by_id, 0)
    for event in events:
        if event.session_id is None:
            continue
        owner = by_id.get(event.session_id)
        if owner is None:
            raise chk.inconsistent(
                "LogRecord", "session_id", "references an unknown session"
            )
        if owner.learner_id != event.learner_id:
            raise chk.inconsistent(
                "LogRecord", "learner_id", "differs from its session's learner"
            )
        counts[event.session_id] += 1

    for session_id, count in counts.items():
        if count != by_id[session_id].n_events:
            raise SchemaError(
                "Session.n_events does not match the number of its events",
                code=chk.INCONSISTENT,
                context={
                    "record": "Session",
                    "field": "n_events",
                    "session_id": session_id,
                    "expected": by_id[session_id].n_events,
                    "actual": count,
                },
            )


def _duplicate(field: str, value: str) -> SchemaError:
    return SchemaError(
        f"duplicate {field} in dataset",
        code=chk.DUPLICATE_ID,
        context={"field": field, "id": value},
    )
