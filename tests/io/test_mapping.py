"""Tests for FieldMapping."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from eduloggen.core import ConfigError
from eduloggen.io import FieldMapping, FieldSpec, RowIssue, pseudonymize

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)

BASIC: dict[str, Any] = {
    "fields": {
        "learner_id": "user",
        "timestamp": "time",
        "activity_id": "item",
        "event_type": "action",
    }
}


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "user": "u1",
        "time": "2026-03-01T09:00:00Z",
        "item": "quiz",
        "action": "view",
    }
    row.update(overrides)
    return row


def test_minimal_mapping_generates_event_ids() -> None:
    mapping = FieldMapping.from_dict(BASIC)
    mapped = mapping.apply(_row(), 7)
    assert mapped.issues == ()
    assert mapped.values == {
        "learner_id": "u1",
        "timestamp": T0,
        "activity_id": "quiz",
        "event_type": "view",
        "event_id": "row-7",
    }


def test_casts_optional_fields_and_metadata() -> None:
    mapping = FieldMapping.from_dict(
        {
            "fields": {
                **BASIC["fields"],
                "event_id": "id",
                "score": "grade",
                "success": "correct",
                "duration_ms": "ms",
                "course_id": "course",
            },
            "metadata": ["device", "lang"],
        }
    )
    row = _row(
        id=12, grade="0.75", correct="yes", ms="1500", course=101, device="mobile"
    )
    values = mapping.apply(row, 1).values
    assert values is not None
    assert values["event_id"] == "12"
    assert values["score"] == 0.75
    assert values["success"] is True
    assert values["duration_ms"] == 1500
    assert values["course_id"] == "101"
    assert values["metadata"] == {"device": "mobile"}


def test_defaults_fill_missing_values() -> None:
    mapping = FieldMapping.from_dict(
        {
            "fields": {
                **BASIC["fields"],
                "event_type": {"source": "action", "default": "other"},
            }
        }
    )
    values = mapping.apply(_row(action=""), 1).values
    assert values is not None
    assert values["event_type"] == "other"
    assert mapping.missing_columns(["user", "time", "item"]) == []


def test_missing_optional_values_are_omitted() -> None:
    mapping = FieldMapping.from_dict({"fields": {**BASIC["fields"], "score": "grade"}})
    values = mapping.apply(_row(grade=None), 1).values
    assert values is not None
    assert "score" not in values


def test_row_issues_are_collected_without_values() -> None:
    mapping = FieldMapping.from_dict({"fields": {**BASIC["fields"], "score": "grade"}})
    mapped = mapping.apply(_row(user="", time="2026-03-01 09:00", grade="high"), 4)
    assert mapped.values is None
    assert mapped.issues == (
        RowIssue(4, "learner_id", "missing_required"),
        RowIssue(4, "timestamp", "naive_timestamp"),
        RowIssue(4, "score", "not_a_number"),
    )


def test_timezone_policy() -> None:
    mapping = FieldMapping.from_dict({**BASIC, "timezone": "Europe/Paris"})
    mapped = mapping.apply(_row(time="2026-03-01 10:00:00"), 1)
    assert mapped.values is not None
    assert mapped.values["timestamp"] == T0
    assert mapped.assumed_timezone

    explicit = mapping.apply(_row(), 1)
    assert not explicit.assumed_timezone


def test_field_timezone_and_format_override() -> None:
    mapping = FieldMapping.from_dict(
        {
            "fields": {
                **BASIC["fields"],
                "timestamp": {
                    "source": "time",
                    "format": "%d/%m/%Y %H:%M",
                    "timezone": "Asia/Tokyo",
                },
            },
            "timezone": "Europe/Paris",
        }
    )
    values = mapping.apply(_row(time="01/03/2026 18:00"), 1).values
    assert values is not None
    assert values["timestamp"] == T0


def test_epoch_format() -> None:
    mapping = FieldMapping.from_dict(
        {
            "fields": {
                **BASIC["fields"],
                "timestamp": {"source": "time", "format": "epoch_ms"},
            }
        }
    )
    values = mapping.apply(_row(time=T0.timestamp() * 1000), 1).values
    assert values is not None
    assert values["timestamp"] == T0


def test_hashing_is_salted_and_stable() -> None:
    mapping = FieldMapping.from_dict(
        {
            "fields": {
                **BASIC["fields"],
                "learner_id": {"source": "user", "hash": {"salt": "s1"}},
            }
        }
    )
    first = mapping.apply(_row(user="alice@example.org"), 1).values
    second = mapping.apply(_row(user="alice@example.org"), 2).values
    assert first is not None and second is not None
    assert first["learner_id"] == second["learner_id"]
    assert first["learner_id"] == pseudonymize("alice@example.org", "s1")
    assert first["learner_id"].startswith("h_")
    assert "alice" not in first["learner_id"]
    assert pseudonymize("alice@example.org", "s2") != first["learner_id"]


def test_to_dict_redacts_salt_and_round_trips() -> None:
    raw: dict[str, Any] = {
        "fields": {
            **BASIC["fields"],
            "learner_id": {"source": "user", "hash": {"salt": "secret"}},
            "timestamp": {"source": "time", "format": "epoch_s", "timezone": "UTC"},
            "event_type": {"source": "action", "default": "other"},
        },
        "metadata": ["device"],
        "drop": ["email"],
        "timezone": "Europe/Paris",
    }
    mapping = FieldMapping.from_dict(raw)
    dumped = mapping.to_dict()
    assert "secret" not in str(dumped)
    assert dumped["fields"]["learner_id"]["hash"] == {"salt": "<redacted>"}
    assert dumped["fields"]["timestamp"] == {
        "source": "time",
        "format": "epoch_s",
        "timezone": "UTC",
    }
    assert dumped["fields"]["event_type"]["default"] == "other"
    assert dumped["drop"] == ["email"]


def test_identity_mapping() -> None:
    mapping = FieldMapping.identity()
    assert mapping.sources()["score"] == "score"
    assert mapping.known_columns() >= {"event_id", "timestamp", "duration_ms"}


def test_known_and_missing_columns() -> None:
    mapping = FieldMapping.from_dict(
        {**BASIC, "metadata": ["device"], "drop": ["email"]}
    )
    assert mapping.known_columns() == {
        "user",
        "time",
        "item",
        "action",
        "device",
        "email",
    }
    assert mapping.missing_columns(["user", "item"]) == ["action", "time"]


def test_mapping_is_immutable() -> None:
    mapping = FieldMapping.from_dict(BASIC)
    with pytest.raises(TypeError):
        mapping.fields["score"] = FieldSpec(source="g")  # type: ignore[index]


@pytest.mark.parametrize(
    ("data", "code"),
    [
        ("not a mapping", "mapping_invalid"),
        ({"fields": {}}, "mapping_invalid"),
        ({"fields": BASIC["fields"], "extra": 1}, "mapping_unknown_key"),
        ({"fields": {**BASIC["fields"], "email": "mail"}}, "mapping_unknown_field"),
        ({"fields": {"learner_id": "user"}}, "mapping_missing_required"),
        ({"fields": {**BASIC["fields"], "score": 3}}, "mapping_invalid"),
        ({"fields": {**BASIC["fields"], "score": {"src": "g"}}}, "mapping_unknown_key"),
        (
            {"fields": {**BASIC["fields"], "score": {"source": "g", "format": "iso8"}}},
            "mapping_invalid",
        ),
        (
            {
                "fields": {
                    **BASIC["fields"],
                    "score": {"source": "g", "hash": {"salt": "x"}},
                }
            },
            "mapping_invalid",
        ),
        (
            {
                "fields": {
                    **BASIC["fields"],
                    "learner_id": {"source": "u", "hash": {"salt": ""}},
                }
            },
            "mapping_invalid",
        ),
        (
            {"fields": {**BASIC["fields"], "learner_id": {"source": "u", "hash": "x"}}},
            "mapping_invalid",
        ),
        (
            {"fields": {**BASIC["fields"], "learner_id": {"source": ""}}},
            "mapping_invalid",
        ),
        (
            {"fields": {**BASIC["fields"], "learner_id": {"source": 5}}},
            "mapping_invalid",
        ),
        ({**BASIC, "timezone": "Mars/Olympus"}, "mapping_invalid"),
        ({**BASIC, "metadata": "device"}, "mapping_invalid"),
        ({**BASIC, "metadata": ["", "x"]}, "mapping_invalid"),
        ({**BASIC, "metadata": ["user"]}, "mapping_conflict"),
        ({**BASIC, "metadata": ["x"], "drop": ["x"]}, "mapping_conflict"),
    ],
)
def test_invalid_mappings(data: Any, code: str) -> None:
    with pytest.raises(ConfigError) as info:
        FieldMapping.from_dict(data)
    assert info.value.code == code


def test_constructor_rejects_non_spec() -> None:
    with pytest.raises(ConfigError):
        fields: dict[str, Any] = {
            name: FieldSpec(source=source) for name, source in BASIC["fields"].items()
        }
        FieldMapping(fields=fields | {"score": "g"})
