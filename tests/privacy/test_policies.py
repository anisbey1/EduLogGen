"""Tests for id remapping and metadata stripping."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pytest

from eduloggen.core import ConfigError
from eduloggen.models import (
    Dataset,
    GenerationMetadata,
    LogRecord,
    Session,
    SyntheticDataset,
)
from eduloggen.privacy import apply_id_strategy, remap_ids, strip_metadata

T0 = datetime(2026, 3, 1, 9, tzinfo=UTC)


def _event(i: int, learner: str, session: str | None, **extra: object) -> LogRecord:
    return LogRecord(
        event_id=f"real-{i}",
        learner_id=learner,
        timestamp=T0 + timedelta(minutes=i),
        activity_id=f"a{i}",
        event_type="view",
        session_id=session,
        **extra,  # type: ignore[arg-type]
    )


@pytest.fixture
def dataset() -> Dataset:
    events = (
        _event(0, "alice", "alice-s1", metadata={"ip": "10.0.0.1", "device": "m"}),
        _event(1, "alice", "alice-s1"),
        _event(2, "bob", "bob-s1", metadata={"email": "b@x.org"}),
        _event(3, "carol", None),
    )
    sessions = (
        Session.from_events("alice-s1", events[:2]),
        Session.from_events("bob-s1", events[2:3]),
    )
    return Dataset(dataset_id="d", events=events, sessions=sessions)


def test_remap_replaces_all_ids(dataset: Dataset) -> None:
    remapped = remap_ids(dataset, seed=1)
    assert remapped.learner_ids == {"L1", "L2", "L3"}
    assert {e.event_id for e in remapped.events} == {"E1", "E2", "E3", "E4"}
    assert remapped.sessions is not None
    assert {s.session_id for s in remapped.sessions} == {"S1", "S2"}
    text = repr(remapped)
    for original in ("alice", "bob", "carol", "real-"):
        assert original not in text


def test_remap_preserves_structure(dataset: Dataset) -> None:
    remapped = remap_ids(dataset, seed=1)
    assert remapped.n_events == dataset.n_events
    assert remapped.events == remapped.sorted_events()
    assert sorted(e.activity_id for e in remapped.events) == ["a0", "a1", "a2", "a3"]
    assert sorted(len(s.event_sequence) for s in remapped.sessions or ()) == [1, 2]
    two_event = next(s for s in remapped.sessions or () if s.n_events == 2)
    members = [e for e in remapped.events if e.session_id == two_event.session_id]
    assert {e.learner_id for e in members} == {two_event.learner_id}
    assert dict(next(e for e in remapped.events if e.activity_id == "a0").metadata) == {
        "ip": "10.0.0.1",
        "device": "m",
    }
    assert next(e for e in remapped.events if e.activity_id == "a3").session_id is None


def test_remap_is_seeded(dataset: Dataset) -> None:
    assert remap_ids(dataset, seed=5) == remap_ids(dataset, seed=5)
    mappings = {
        tuple(
            e.learner_id
            for e in sorted(
                remap_ids(dataset, seed=s).events, key=lambda e: e.activity_id
            )
        )
        for s in range(20)
    }
    assert len(mappings) > 1


def test_remap_keeps_dataset_type_and_handles_unsessionized(dataset: Dataset) -> None:
    synthetic = SyntheticDataset(
        dataset_id="syn",
        events=dataset.events,
        sessions=dataset.sessions,
        generation=GenerationMetadata(
            generator_id="markov", model_fingerprint="sha256:x", seed=1, n_sessions=2
        ),
    )
    remapped = remap_ids(synthetic, seed=0)
    assert isinstance(remapped, SyntheticDataset)
    assert remapped.generation == synthetic.generation

    raw = Dataset(dataset_id="raw", events=dataset.events)
    remapped_raw = remap_ids(raw, seed=0)
    assert remapped_raw.sessions is None
    assert {e.session_id for e in remapped_raw.events} == {"S1", "S2", None}


def test_remap_pads_ids(dataset: Dataset) -> None:
    events = tuple(_event(i, f"u{i}", None) for i in range(12))
    remapped = remap_ids(Dataset(dataset_id="d", events=events), seed=0)
    assert "L01" in remapped.learner_ids
    assert remapped.events[0].event_id == "E01"


def test_strip_metadata(dataset: Dataset) -> None:
    stripped = strip_metadata(dataset, ["ip", "email"])
    assert [dict(e.metadata) for e in stripped.events] == [{"device": "m"}, {}, {}, {}]
    assert stripped.sessions == dataset.sessions
    assert strip_metadata(dataset, []) is dataset
    assert strip_metadata(dataset, ["absent"]) is dataset


def test_apply_id_strategy(dataset: Dataset, caplog: pytest.LogCaptureFixture) -> None:
    assert apply_id_strategy(dataset, "remap", 1) == remap_ids(dataset, 1)
    with caplog.at_level(logging.WARNING, logger="eduloggen.privacy"):
        assert apply_id_strategy(dataset, "preserve", 1) is dataset
    assert "do not share" in caplog.text
    with pytest.raises(ConfigError):
        apply_id_strategy(dataset, "hash", 1)  # type: ignore[arg-type]
