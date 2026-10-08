"""Synthetic training data for generator tests."""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from eduloggen.analysis import sessionize
from eduloggen.models import Dataset, LogRecord

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)


def build(
    sessions: Sequence[Sequence[str]],
    *,
    gap_after: dict[str, float] | None = None,
    activities: dict[str, str] | None = None,
    learners: int = 3,
) -> Dataset:
    """Sessionized dataset with the given event-type sequences.

    Sessions rotate over ``learners`` learners and are 6 hours apart, so the
    default 30 minute idle timeout reproduces them exactly. ``gap_after``
    sets the seconds after each event type (default 30).
    """
    gaps = gap_after or {}
    acts = activities or {}
    events: list[LogRecord] = []
    for index, tokens in enumerate(sessions):
        clock = T0 + timedelta(hours=6 * index)
        for token in tokens:
            events.append(
                LogRecord(
                    event_id=f"e{len(events)}",
                    learner_id=f"u{index % learners}",
                    timestamp=clock,
                    activity_id=acts.get(token, f"act-{token}"),
                    event_type=token,
                    course_id="c1" if index % 2 else "c2",
                )
            )
            clock += timedelta(seconds=gaps.get(token, 30.0))
    return sessionize(Dataset(dataset_id="train", events=tuple(events)))


@pytest.fixture
def abc() -> Dataset:
    """Ten identical sessions ``a b c``."""
    return build([("a", "b", "c")] * 10)


@pytest.fixture
def varied() -> Dataset:
    """Sixty random sessions with token-dependent timing."""
    rng = random.Random(0)
    tokens = ["view", "attempt", "submit", "video_play", "navigate"]
    sessions = [
        [rng.choice(tokens) for _ in range(rng.randint(1, 8))] for _ in range(60)
    ]
    return build(
        sessions,
        gap_after={"video_play": 300.0, "navigate": 5.0},
        learners=20,
    )
