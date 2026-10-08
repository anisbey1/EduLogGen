"""Canonical event, session, and participant records (PRD §9.1-9.2, SAD §9).

Records are immutable value objects validated on construction. Timestamps are
always stored as timezone-aware UTC datetimes; aware datetimes in other zones
are converted, naive datetimes are rejected.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

from eduloggen.core import EVENT_REQUIRED_FIELDS, SESSION_REQUIRED_FIELDS
from eduloggen.models import _checks as chk

__all__ = ["LogRecord", "Participant", "Session", "TokenField"]

TokenField = Literal["event_type", "activity_id"]
"""Event attribute used as the token in a session's ``event_sequence``."""

_EVENT_FIELDS = frozenset(
    {
        *EVENT_REQUIRED_FIELDS,
        "session_id",
        "course_id",
        "score",
        "success",
        "duration_ms",
        "metadata",
    }
)
_SESSION_FIELDS = frozenset({*SESSION_REQUIRED_FIELDS, "course_id"})


@dataclass(frozen=True, slots=True, kw_only=True)
class LogRecord:
    """One recorded learner interaction (an *event*).

    Attributes:
        event_id: Unique identifier within a corpus.
        learner_id: Pseudonymous learner key; never a real name or email.
        timestamp: Event time as a timezone-aware UTC datetime.
        activity_id: Activity, resource, or item identifier.
        event_type: Event type token (see :class:`EventVocabulary`).
        session_id: Session key, if known.
        course_id: Course or offering identifier.
        score: Numeric score, if applicable.
        success: Correctness or success flag.
        duration_ms: Action duration in milliseconds.
        metadata: Read-only extension bag for non-core attributes.
    """

    event_id: str
    learner_id: str
    timestamp: datetime
    activity_id: str
    event_type: str
    session_id: str | None = None
    course_id: str | None = None
    score: float | None = None
    success: bool | None = None
    duration_ms: int | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), hash=False
    )

    def __post_init__(self) -> None:
        """Validate and normalize fields.

        Raises:
            SchemaError: If any field has an invalid type or value.
        """
        name = "LogRecord"
        chk.require_str(name, "event_id", self.event_id)
        chk.require_str(name, "learner_id", self.learner_id)
        chk.require_str(name, "activity_id", self.activity_id)
        chk.require_str(name, "event_type", self.event_type)
        chk.optional_str(name, "session_id", self.session_id)
        chk.optional_str(name, "course_id", self.course_id)
        chk.optional_bool(name, "success", self.success)
        if self.duration_ms is not None:
            chk.non_negative_int(name, "duration_ms", self.duration_ms)
        _set(self, "timestamp", chk.require_utc(name, "timestamp", self.timestamp))
        _set(self, "score", chk.optional_float(name, "score", self.score))
        _set(self, "metadata", chk.freeze_mapping(name, "metadata", self.metadata))

    @property
    def sort_key(self) -> tuple[str, datetime, str]:
        """Canonical ordering key ``(learner_id, timestamp, event_id)``."""
        return (self.learner_id, self.timestamp, self.event_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible row; timestamps become ISO strings."""
        return {
            "event_id": self.event_id,
            "learner_id": self.learner_id,
            "timestamp": self.timestamp.isoformat(),
            "activity_id": self.activity_id,
            "event_type": self.event_type,
            "session_id": self.session_id,
            "course_id": self.course_id,
            "score": self.score,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LogRecord:
        """Build a record from a canonical row.

        Args:
            data: Mapping with canonical field names. ``timestamp`` may be a
                datetime or an ISO 8601 string. ``None`` values for optional
                fields are treated as absent.

        Returns:
            The validated record.

        Raises:
            SchemaError: If required fields are missing, unknown fields are
                present, or values are invalid.
        """
        chk.check_keys("LogRecord", data, EVENT_REQUIRED_FIELDS, _EVENT_FIELDS)
        values = dict(data)
        values["timestamp"] = chk.parse_utc(
            "LogRecord", "timestamp", values["timestamp"]
        )
        if values.get("metadata") is None:
            values.pop("metadata", None)
        return cls(**values)


@dataclass(frozen=True, slots=True, kw_only=True)
class Session:
    """A bounded episode of one learner's activity (PRD §9.2).

    ``n_events`` and ``duration_s`` are derived from ``event_sequence`` and the
    time bounds, so they can never disagree with them.

    Attributes:
        session_id: Session key.
        learner_id: Owning learner.
        start_time: Time of the first event (UTC).
        end_time: Time of the last event (UTC); never before ``start_time``.
        event_sequence: Ordered tokens, one per event, used by generators.
        course_id: Dominant or declared course.
    """

    session_id: str
    learner_id: str
    start_time: datetime
    end_time: datetime
    event_sequence: tuple[str, ...]
    course_id: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize fields.

        Raises:
            SchemaError: If any field is invalid or the time bounds are reversed.
        """
        name = "Session"
        chk.require_str(name, "session_id", self.session_id)
        chk.require_str(name, "learner_id", self.learner_id)
        chk.optional_str(name, "course_id", self.course_id)
        start = chk.require_utc(name, "start_time", self.start_time)
        end = chk.require_utc(name, "end_time", self.end_time)
        if end < start:
            raise chk.invalid(name, "end_time", "must not be before start_time")
        raw: object = self.event_sequence
        if isinstance(raw, str) or not isinstance(raw, Iterable):
            raise chk.invalid(name, "event_sequence", "must be a sequence of tokens")
        sequence = tuple(raw)
        if not sequence:
            raise chk.invalid(name, "event_sequence", "must not be empty")
        for token in sequence:
            chk.require_str(name, "event_sequence", token)
        _set(self, "start_time", start)
        _set(self, "end_time", end)
        _set(self, "event_sequence", sequence)

    @property
    def n_events(self) -> int:
        """Number of events in the session."""
        return len(self.event_sequence)

    @property
    def duration_s(self) -> float:
        """Session duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible row including derived fields."""
        return {
            "session_id": self.session_id,
            "learner_id": self.learner_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "n_events": self.n_events,
            "duration_s": self.duration_s,
            "event_sequence": list(self.event_sequence),
            "course_id": self.course_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Session:
        """Build a session from a canonical row.

        Derived fields ``n_events`` and ``duration_s`` are required by the
        schema and must agree with the sequence and time bounds.

        Args:
            data: Mapping with canonical session field names.

        Returns:
            The validated session.

        Raises:
            SchemaError: If fields are missing, unknown, invalid, or derived
                fields disagree with the data.
        """
        name = "Session"
        chk.check_keys(name, data, SESSION_REQUIRED_FIELDS, _SESSION_FIELDS)
        session = cls(
            session_id=data["session_id"],
            learner_id=data["learner_id"],
            start_time=chk.parse_utc(name, "start_time", data["start_time"]),
            end_time=chk.parse_utc(name, "end_time", data["end_time"]),
            event_sequence=data["event_sequence"],
            course_id=data.get("course_id"),
        )
        if data["n_events"] != session.n_events:
            raise chk.inconsistent(
                "Session", "n_events", "does not match event_sequence length"
            )
        duration = chk.optional_float(name, "duration_s", data["duration_s"])
        if duration is None or abs(duration - session.duration_s) > 1e-6:
            raise chk.inconsistent(
                "Session", "duration_s", "does not match start/end times"
            )
        return session

    @classmethod
    def from_events(
        cls,
        session_id: str,
        events: Iterable[LogRecord],
        *,
        token_field: TokenField = "event_type",
    ) -> Session:
        """Summarize one learner's events into a session.

        Events are ordered by timestamp (ties broken by ``event_id``). The
        session's ``course_id`` is the most common non-empty course, with ties
        resolved by first occurrence.

        Args:
            session_id: Key for the new session.
            events: Events belonging to the session.
            token_field: Event attribute used as the sequence token.

        Returns:
            The session spanning the given events.

        Raises:
            SchemaError: If ``events`` is empty or spans several learners.
        """
        ordered = sorted(events, key=lambda event: event.sort_key)
        if not ordered:
            raise chk.invalid("Session", "event_sequence", "must not be empty")
        learners = {event.learner_id for event in ordered}
        if len(learners) > 1:
            raise chk.inconsistent(
                "Session", "learner_id", "events belong to several learners"
            )
        courses = Counter(e.course_id for e in ordered if e.course_id is not None)
        return cls(
            session_id=session_id,
            learner_id=ordered[0].learner_id,
            start_time=ordered[0].timestamp,
            end_time=ordered[-1].timestamp,
            event_sequence=tuple(getattr(e, token_field) for e in ordered),
            course_id=courses.most_common(1)[0][0] if courses else None,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Participant:
    """A pseudonymous learner, the identity boundary for sessions and profiles.

    Attributes:
        learner_id: Pseudonymous learner key.
        cohort: Optional cohort label.
        course_ids: Courses the learner has events in.
    """

    learner_id: str
    cohort: str | None = None
    course_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Validate fields.

        Raises:
            SchemaError: If any field is invalid.
        """
        chk.require_str("Participant", "learner_id", self.learner_id)
        chk.optional_str("Participant", "cohort", self.cohort)
        courses = frozenset(self.course_ids)
        for course in courses:
            chk.require_str("Participant", "course_ids", course)
        _set(self, "course_ids", courses)


def _set(instance: object, name: str, value: object) -> None:
    """Assign a normalized value on a frozen dataclass during construction."""
    object.__setattr__(instance, name, value)
