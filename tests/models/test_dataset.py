"""Tests for Dataset, SyntheticDataset, and cross-record validation."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from eduloggen.core import SCHEMA_VERSION, SchemaError
from eduloggen.models import (
    Dataset,
    DatasetMetadata,
    GenerationMetadata,
    LogRecord,
    Participant,
    Session,
    SyntheticDataset,
    check_schema_version,
)

from .conftest import MakeEvent


@pytest.fixture
def events(make_event: MakeEvent) -> list[LogRecord]:
    return [
        make_event("e2", minutes=5, event_type="submit", session_id="s1"),
        make_event("e1", minutes=0, session_id="s1", course_id="c1"),
        make_event("e3", minutes=1, learner_id="L2", session_id="s2"),
    ]


@pytest.fixture
def sessions(events: list[LogRecord]) -> list[Session]:
    return [
        Session.from_events("s1", events[:2]),
        Session.from_events("s2", events[2:]),
    ]


def test_dataset_basics(events: list[LogRecord]) -> None:
    dataset = Dataset(dataset_id="demo", events=events)  # type: ignore[arg-type]
    assert isinstance(dataset.events, tuple)
    assert dataset.n_events == 3
    assert dataset.n_sessions == 0
    assert not dataset.is_sessionized
    assert dataset.learner_ids == {"L1", "L2"}
    assert dataset.schema_version == SCHEMA_VERSION
    assert [e.event_id for e in dataset.sorted_events()] == ["e1", "e2", "e3"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        dataset.events = ()  # type: ignore[misc]


def test_empty_dataset_is_valid() -> None:
    dataset = Dataset(dataset_id="empty", events=())
    assert dataset.n_events == 0
    assert dataset.learner_ids == frozenset()


def test_with_sessions_returns_new_snapshot(
    events: list[LogRecord], sessions: list[Session]
) -> None:
    raw = Dataset(dataset_id="demo", events=tuple(events))
    sessionized = raw.with_sessions(sessions)
    assert raw.sessions is None
    assert sessionized.is_sessionized
    assert sessionized.n_sessions == 2
    assert sessionized.events is raw.events


def test_participants(events: list[LogRecord], sessions: list[Session]) -> None:
    dataset = Dataset(dataset_id="d", events=tuple(events), sessions=tuple(sessions))
    assert dataset.participants() == (
        Participant(learner_id="L1", course_ids=frozenset({"c1"})),
        Participant(learner_id="L2"),
    )


def test_rejects_duplicate_event_ids(make_event: MakeEvent) -> None:
    with pytest.raises(SchemaError) as info:
        Dataset(dataset_id="d", events=(make_event("e1"), make_event("e1")))
    assert info.value.code == "schema_duplicate_id"
    assert info.value.context == {"field": "event_id", "id": "e1"}


def test_rejects_duplicate_session_ids(
    events: list[LogRecord], sessions: list[Session]
) -> None:
    with pytest.raises(SchemaError) as info:
        Dataset(
            dataset_id="d",
            events=tuple(events),
            sessions=(sessions[0], sessions[0]),
        )
    assert info.value.code == "schema_duplicate_id"


def test_rejects_unknown_session_reference(
    events: list[LogRecord], sessions: list[Session]
) -> None:
    with pytest.raises(SchemaError) as info:
        Dataset(dataset_id="d", events=tuple(events), sessions=(sessions[0],))
    assert info.value.context["field"] == "session_id"


def test_rejects_session_owned_by_other_learner(
    make_event: MakeEvent, sessions: list[Session]
) -> None:
    event = make_event("e9", learner_id="L2", session_id="s1")
    with pytest.raises(SchemaError) as info:
        Dataset(dataset_id="d", events=(event,), sessions=tuple(sessions))
    assert info.value.context["field"] == "learner_id"


def test_rejects_session_event_count_mismatch(
    events: list[LogRecord], sessions: list[Session]
) -> None:
    with pytest.raises(SchemaError) as info:
        Dataset(dataset_id="d", events=tuple(events[1:]), sessions=tuple(sessions))
    assert info.value.context["session_id"] == "s1"
    assert info.value.context["expected"] == 2
    assert info.value.context["actual"] == 1


def test_events_without_session_id_are_allowed(
    make_event: MakeEvent, events: list[LogRecord], sessions: list[Session]
) -> None:
    orphan = make_event("e4", minutes=30)
    dataset = Dataset(
        dataset_id="d", events=(*events, orphan), sessions=tuple(sessions)
    )
    assert dataset.n_events == 4


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"dataset_id": ""}, "dataset_id"),
        ({"events": "abc"}, "events"),
        ({"events": 5}, "events"),
        ({"events": ("x",)}, "events"),
        ({"sessions": {"a": 1}}, "sessions"),
        ({"sessions": ("x",)}, "sessions"),
        ({"metadata": {}}, "metadata"),
    ],
)
def test_rejects_invalid_fields(overrides: dict[str, Any], field: str) -> None:
    values: dict[str, Any] = {"dataset_id": "d", "events": ()}
    values.update(overrides)
    with pytest.raises(SchemaError) as info:
        Dataset(**values)
    assert info.value.context["field"] == field


@pytest.mark.parametrize("version", ["1.0", "1.7"])
def test_accepts_compatible_schema_versions(version: str) -> None:
    check_schema_version(version)
    assert Dataset(dataset_id="d", events=(), schema_version=version)


@pytest.mark.parametrize("version", ["2.0", "0.9", "1", "v1.0", "1.x"])
def test_rejects_incompatible_schema_versions(version: str) -> None:
    with pytest.raises(SchemaError) as info:
        check_schema_version(version)
    assert info.value.code == "schema_unsupported_version"


def test_fingerprint_ignores_order_id_and_metadata(
    events: list[LogRecord], sessions: list[Session]
) -> None:
    a = Dataset(dataset_id="a", events=tuple(events), sessions=tuple(sessions))
    b = Dataset(
        dataset_id="b",
        events=tuple(reversed(events)),
        sessions=tuple(reversed(sessions)),
        metadata=DatasetMetadata(source_description="other"),
    )
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint().startswith("sha256:")
    assert len(a.fingerprint()) == len("sha256:") + 64


def test_fingerprint_changes_with_content(
    make_event: MakeEvent, events: list[LogRecord], sessions: list[Session]
) -> None:
    base = Dataset(dataset_id="d", events=tuple(events))
    changed = Dataset(dataset_id="d", events=(*events, make_event("e9")))
    assert base.fingerprint() != changed.fingerprint()
    assert base.fingerprint() != base.with_sessions(sessions).fingerprint()


def test_fingerprint_rejects_non_json_metadata(make_event: MakeEvent) -> None:
    dataset = Dataset(dataset_id="d", events=(make_event(metadata={"x": object()}),))
    with pytest.raises(SchemaError) as info:
        dataset.fingerprint()
    assert info.value.context["field"] == "metadata"


def test_manifest_is_json_and_has_no_rows(
    events: list[LogRecord], sessions: list[Session]
) -> None:
    dataset = Dataset(dataset_id="demo", events=tuple(events), sessions=tuple(sessions))
    manifest = json.loads(json.dumps(dataset.to_manifest()))
    assert manifest["synthetic"] is False
    assert manifest["counts"] == {"events": 3, "sessions": 2, "learners": 2}
    assert manifest["fingerprint"] == dataset.fingerprint()
    assert "events" not in manifest


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def test_dataset_metadata_round_trip() -> None:
    metadata = DatasetMetadata(
        source_description="demo",
        privacy_notes="synthetic",
        seeds={"generate": 7},
        config_fingerprint="sha256:abc",
    )
    restored = DatasetMetadata.from_dict(json.loads(json.dumps(metadata.to_dict())))
    assert restored == metadata
    assert metadata.created_at.tzinfo is UTC


def test_dataset_metadata_seeds_are_read_only() -> None:
    metadata = DatasetMetadata(seeds={"a": 1})
    with pytest.raises(TypeError):
        metadata.seeds["a"] = 2  # type: ignore[index]


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"source_description": None}, "source_description"),
        ({"privacy_notes": 1}, "privacy_notes"),
        ({"eduloggen_version": ""}, "eduloggen_version"),
        ({"config_fingerprint": ""}, "config_fingerprint"),
        ({"seeds": {"a": -1}}, "seeds"),
        ({"created_at": datetime(2026, 1, 1)}, "created_at"),
    ],
)
def test_dataset_metadata_rejects_invalid(
    overrides: dict[str, Any], field: str
) -> None:
    with pytest.raises(SchemaError) as info:
        DatasetMetadata(**overrides)
    assert info.value.context["field"] == field


def test_dataset_metadata_from_dict_rejects_unknown_keys() -> None:
    row = DatasetMetadata().to_dict() | {"owner": "x"}
    with pytest.raises(SchemaError) as info:
        DatasetMetadata.from_dict(row)
    assert info.value.code == "schema_unknown_field"


# --------------------------------------------------------------------------
# SyntheticDataset
# --------------------------------------------------------------------------


def _generation(**overrides: Any) -> GenerationMetadata:
    values: dict[str, Any] = {
        "generator_id": "markov",
        "model_fingerprint": "sha256:abc",
        "seed": 42,
        "n_sessions": 2,
    }
    values.update(overrides)
    return GenerationMetadata(**values)


def test_synthetic_dataset(events: list[LogRecord], sessions: list[Session]) -> None:
    synthetic = SyntheticDataset(
        dataset_id="syn",
        events=tuple(events),
        generation=_generation(),
    )
    assert isinstance(synthetic, Dataset)
    attached = synthetic.with_sessions(sessions)
    assert isinstance(attached, SyntheticDataset)
    assert attached.generation == synthetic.generation

    manifest = json.loads(json.dumps(attached.to_manifest()))
    assert manifest["synthetic"] is True
    assert manifest["generation"]["id_strategy"] == "remap"


def test_synthetic_dataset_still_validates_records(make_event: MakeEvent) -> None:
    with pytest.raises(SchemaError):
        SyntheticDataset(
            dataset_id="syn",
            events=(make_event("e1"), make_event("e1")),
            generation=_generation(),
        )


def test_synthetic_dataset_requires_generation_metadata() -> None:
    with pytest.raises(SchemaError) as info:
        SyntheticDataset(dataset_id="syn", events=(), generation={})  # type: ignore[arg-type]
    assert info.value.context["field"] == "generation"


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"generator_id": ""}, "generator_id"),
        ({"model_fingerprint": None}, "model_fingerprint"),
        ({"id_strategy": ""}, "id_strategy"),
        ({"seed": -1}, "seed"),
        ({"n_sessions": 0}, "n_sessions"),
        ({"n_sessions": 1.5}, "n_sessions"),
    ],
)
def test_generation_metadata_rejects_invalid(
    overrides: dict[str, Any], field: str
) -> None:
    with pytest.raises(SchemaError) as info:
        _generation(**overrides)
    assert info.value.context["field"] == field
