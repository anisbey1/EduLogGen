"""Tests for LogRecord, Session, and Participant."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from eduloggen.core import SchemaError
from eduloggen.models import Event, LogRecord, Participant, Session

from .conftest import T0, MakeEvent

# --------------------------------------------------------------------------
# LogRecord
# --------------------------------------------------------------------------


def test_event_alias() -> None:
    assert Event is LogRecord


def test_log_record_defaults(make_event: MakeEvent) -> None:
    event = make_event()
    assert event.session_id is None
    assert event.score is None
    assert dict(event.metadata) == {}


def test_log_record_normalizes_timezone_and_score(make_event: MakeEvent) -> None:
    paris = timezone(timedelta(hours=2))
    event = make_event(timestamp=datetime(2026, 3, 1, 11, 0, tzinfo=paris), score=3)
    assert event.timestamp == T0
    assert event.timestamp.tzinfo is UTC
    assert isinstance(event.score, float)


def test_log_record_is_immutable_and_hashable(make_event: MakeEvent) -> None:
    event = make_event(metadata={"device": "mobile"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.event_id = "x"  # type: ignore[misc]
    with pytest.raises(TypeError):
        event.metadata["device"] = "desktop"  # type: ignore[index]
    assert hash(event) == hash(make_event(metadata={"other": 1}))


def test_log_record_metadata_is_copied(make_event: MakeEvent) -> None:
    source = {"k": 1}
    event = make_event(metadata=source)
    source["k"] = 2
    assert event.metadata["k"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_id", ""),
        ("learner_id", 5),
        ("activity_id", None),
        ("event_type", ""),
        ("session_id", ""),
        ("course_id", 1),
        ("timestamp", datetime(2026, 3, 1)),
        ("timestamp", "2026-03-01T09:00:00+00:00"),
        ("score", math.nan),
        ("score", math.inf),
        ("score", True),
        ("score", "1"),
        ("success", 1),
        ("duration_ms", -1),
        ("duration_ms", 1.5),
        ("duration_ms", True),
        ("metadata", ["a"]),
        ("metadata", {1: "a"}),
    ],
)
def test_log_record_rejects_invalid(
    make_event: MakeEvent, field: str, value: Any
) -> None:
    with pytest.raises(SchemaError) as info:
        make_event(**{field: value})
    assert info.value.code == "schema_invalid_value"
    assert info.value.context["field"] == field


def test_log_record_error_never_contains_value(make_event: MakeEvent) -> None:
    with pytest.raises(SchemaError) as info:
        make_event(learner_id=["alice@example.org"])
    assert "alice" not in str(info.value)
    assert "alice" not in repr(info.value.context)


def test_log_record_round_trip(make_event: MakeEvent) -> None:
    event = make_event(
        session_id="s1",
        course_id="c1",
        score=0.5,
        success=True,
        duration_ms=1200,
        metadata={"device": "mobile"},
    )
    row = json.loads(json.dumps(event.to_dict()))
    assert LogRecord.from_dict(row) == event


def test_log_record_from_dict_accepts_z_suffix_and_null_metadata() -> None:
    event = LogRecord.from_dict(
        {
            "event_id": "e1",
            "learner_id": "L1",
            "timestamp": "2026-03-01T09:00:00Z",
            "activity_id": "a",
            "event_type": "view",
            "metadata": None,
            "score": None,
        }
    )
    assert event.timestamp == T0


def test_log_record_from_dict_missing_field() -> None:
    with pytest.raises(SchemaError) as info:
        LogRecord.from_dict({"event_id": "e1"})
    assert info.value.code == "schema_missing_field"
    assert info.value.context["field"] == "learner_id"


def test_log_record_from_dict_unknown_field(make_event: MakeEvent) -> None:
    row = make_event().to_dict() | {"email": "x", "name": "y"}
    with pytest.raises(SchemaError) as info:
        LogRecord.from_dict(row)
    assert info.value.code == "schema_unknown_field"
    assert info.value.context["fields"] == ["email", "name"]


def test_log_record_from_dict_bad_timestamp(make_event: MakeEvent) -> None:
    row = make_event().to_dict() | {"timestamp": "yesterday"}
    with pytest.raises(SchemaError) as info:
        LogRecord.from_dict(row)
    assert info.value.context["field"] == "timestamp"


def test_sort_key(make_event: MakeEvent) -> None:
    late = make_event("e2", minutes=5)
    early = make_event("e1", minutes=1)
    other = make_event("e0", learner_id="L0", minutes=9)
    assert sorted([late, early, other], key=lambda e: e.sort_key) == [
        other,
        early,
        late,
    ]


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------


def _session(**overrides: Any) -> Session:
    values: dict[str, Any] = {
        "session_id": "s1",
        "learner_id": "L1",
        "start_time": T0,
        "end_time": T0 + timedelta(minutes=10),
        "event_sequence": ["view", "attempt", "submit"],
    }
    values.update(overrides)
    return Session(**values)


def test_session_derived_fields() -> None:
    session = _session()
    assert session.n_events == 3
    assert session.duration_s == 600.0
    assert session.event_sequence == ("view", "attempt", "submit")


def test_session_zero_duration_allowed() -> None:
    assert _session(end_time=T0, event_sequence=["view"]).duration_s == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", ""),
        ("learner_id", None),
        ("course_id", ""),
        ("start_time", datetime(2026, 3, 1)),
        ("end_time", T0 - timedelta(seconds=1)),
        ("event_sequence", []),
        ("event_sequence", "view"),
        ("event_sequence", 3),
        ("event_sequence", ["view", ""]),
    ],
)
def test_session_rejects_invalid(field: str, value: Any) -> None:
    with pytest.raises(SchemaError) as info:
        _session(**{field: value})
    assert info.value.context["field"] == field


def test_session_round_trip() -> None:
    session = _session(course_id="c1")
    row = json.loads(json.dumps(session.to_dict()))
    assert row["n_events"] == 3
    assert row["duration_s"] == 600.0
    assert Session.from_dict(row) == session


@pytest.mark.parametrize(
    ("field", "value"),
    [("n_events", 4), ("duration_s", 1.0), ("duration_s", "600")],
)
def test_session_from_dict_rejects_inconsistent(field: str, value: Any) -> None:
    row = _session().to_dict() | {field: value}
    with pytest.raises(SchemaError) as info:
        Session.from_dict(row)
    assert info.value.context["field"] == field


def test_session_from_events(make_event: MakeEvent) -> None:
    events = [
        make_event("e3", minutes=7, event_type="submit", course_id="c2"),
        make_event("e1", minutes=0, activity_id="intro", course_id="c1"),
        make_event("e2", minutes=3, event_type="attempt", course_id="c2"),
    ]
    session = Session.from_events("s1", events)
    assert session.event_sequence == ("view", "attempt", "submit")
    assert session.start_time == T0
    assert session.duration_s == 420.0
    assert session.course_id == "c2"

    by_activity = Session.from_events("s1", events, token_field="activity_id")
    assert by_activity.event_sequence == ("intro", "quiz-1", "quiz-1")


def test_session_from_events_without_course(make_event: MakeEvent) -> None:
    assert Session.from_events("s1", [make_event()]).course_id is None


def test_session_from_events_rejects_empty_and_mixed(make_event: MakeEvent) -> None:
    with pytest.raises(SchemaError):
        Session.from_events("s1", [])
    with pytest.raises(SchemaError) as info:
        Session.from_events("s1", [make_event("e1"), make_event("e2", learner_id="L2")])
    assert info.value.code == "schema_inconsistent"


# --------------------------------------------------------------------------
# Participant
# --------------------------------------------------------------------------


def test_participant() -> None:
    participant = Participant(learner_id="L1", course_ids={"c1"})  # type: ignore[arg-type]
    assert participant.course_ids == frozenset({"c1"})
    assert participant.cohort is None
    with pytest.raises(SchemaError):
        Participant(learner_id="")
    with pytest.raises(SchemaError):
        Participant(learner_id="L1", cohort="")
    with pytest.raises(SchemaError):
        Participant(learner_id="L1", course_ids=frozenset({""}))
